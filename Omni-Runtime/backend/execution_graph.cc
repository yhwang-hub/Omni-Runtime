#include "Omni-Runtime/backend/backend.h"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace backend {
namespace {

bool IsSameInput(const Linear &first, const Linear &second) {
  return first.rows == second.rows && first.inner == second.inner &&
         first.input.data == second.input.data &&
         first.input.offset_bytes == second.input.offset_bytes &&
         first.input.backend_kind == second.input.backend_kind;
}

bool HasOverlap(const DeviceView &first, const DeviceView &second) {
  if (!first.IsValid() || !second.IsValid() || first.backend_kind != second.backend_kind) {
    return false;
  }
  if (first.backend_kind == BackendKind::kCuda) {
    const uintptr_t first_base = reinterpret_cast<uintptr_t>(first.data);
    const uintptr_t second_base = reinterpret_cast<uintptr_t>(second.data);
    if (first_base > std::numeric_limits<uintptr_t>::max() - first.offset_bytes ||
        second_base > std::numeric_limits<uintptr_t>::max() - second.offset_bytes) {
      return true;
    }
    const uintptr_t first_start = first_base + first.offset_bytes;
    const uintptr_t second_start = second_base + second.offset_bytes;
    return first_start <= second_start ? second_start - first_start < first.size_bytes
                                       : first_start - second_start < second.size_bytes;
  }
  return first.data == second.data &&
         first.offset_bytes < second.offset_bytes + second.size_bytes &&
         second.offset_bytes < first.offset_bytes + first.size_bytes;
}

bool HasStorage(const DeviceView &view, const BackendKind kind, const uint64_t elements,
                const size_t element_bytes) {
  return view.IsValid() && view.backend_kind == kind && view.offset_bytes % element_bytes == 0U &&
         elements <= view.size_bytes / element_bytes;
}

bool GetElementBytes(const ScalarType type, size_t *const bytes) {
  OMNI_RETURN_VAL_IF_LOG(bytes == nullptr, false, ERROR, "Scalar byte count output is null");
  switch (type) {
  case ScalarType::kFloat32:
    *bytes = sizeof(float);
    return true;
  case ScalarType::kFloat16:
  case ScalarType::kBFloat16:
    *bytes = sizeof(uint16_t);
    return true;
  }
  LOG_ERROR("Invalid parallel projection scalar type");
  return false;
}

class GraphCompiler final {
public:
  bool Compile(const ScalarType type, const BackendKind kind, const std::vector<ComputeNode> &nodes,
               std::vector<ComputeOperation> *const operations) {
    OMNI_RETURN_VAL_IF_LOG(operations == nullptr || nodes.empty() ||
                               nodes.size() > std::numeric_limits<uint32_t>::max(),
                           false, ERROR, "Invalid compute graph inputs");
    size_t element_bytes = 0U;
    OMNI_RETURN_VAL_IF(!GetElementBytes(type, &element_bytes), false);
    OMNI_RETURN_VAL_IF_LOG(kind != BackendKind::kCuda && kind != BackendKind::kOpenCL, false, ERROR,
                           "Compute graph requires a concrete backend");
    const size_t count = nodes.size();
    ancestors_.assign(count, std::vector<bool>(count, false));
    std::vector<bool> is_emitted(count, false);
    std::vector<uint32_t> order;
    for (size_t step = 0U; step < count; ++step) {
      size_t next = count;
      for (size_t node = 0U; node < count; ++node) {
        if (is_emitted[node]) {
          continue;
        }
        bool is_ready = true;
        for (const uint32_t dependency : nodes[node].dependencies) {
          OMNI_RETURN_VAL_IF_LOG(dependency >= count || dependency == node, false, ERROR,
                                 "Invalid compute graph dependency");
          is_ready = is_ready && is_emitted[dependency];
        }
        if (is_ready) {
          next = node;
          break;
        }
      }
      OMNI_RETURN_VAL_IF_LOG(next == count, false, ERROR, "Compute graph has a cycle");
      for (const uint32_t dependency : nodes[next].dependencies) {
        ancestors_[next][dependency] = true;
        for (size_t node = 0U; node < count; ++node) {
          ancestors_[next][node] = ancestors_[next][node] || ancestors_[dependency][node];
        }
      }
      is_emitted[next] = true;
      order.push_back(static_cast<uint32_t>(next));
    }
    std::vector<Access> accesses(count);
    for (size_t node = 0U; node < count; ++node) {
      if (const auto *const group = std::get_if<ParallelLinear>(&nodes[node].operation)) {
        OMNI_RETURN_VAL_IF(!ValidateParallelLinear(type, kind, *group), false);
      }
      Collect(nodes[node].operation, &accesses[node]);
      for (const DeviceView &view : accesses[node].views) {
        OMNI_RETURN_VAL_IF_LOG(view.backend_kind != kind, false, ERROR,
                               "Compute graph contains memory from another backend");
      }
      for (size_t previous = 0U; previous < node; ++previous) {
        OMNI_RETURN_VAL_IF_LOG(!ancestors_[node][previous] && !ancestors_[previous][node] &&
                                   HasConflict(accesses[node], accesses[previous]),
                               false, ERROR, "Compute graph has an unordered memory conflict");
      }
    }
    std::vector<ComputeOperation> compiled;
    is_emitted.assign(count, false);
    for (const uint32_t node : order) {
      if (is_emitted[node]) {
        continue;
      }
      const Linear *const first = std::get_if<Linear>(&nodes[node].operation);
      ParallelLinear group;
      std::vector<uint32_t> branches;
      if (first != nullptr) {
        group.projections.push_back(*first);
        for (const uint32_t candidate : order) {
          if (candidate == node || is_emitted[candidate] || group.projections.size() == 3U) {
            continue;
          }
          const Linear *const projection = std::get_if<Linear>(&nodes[candidate].operation);
          const bool is_ready = std::all_of(
              nodes[candidate].dependencies.begin(), nodes[candidate].dependencies.end(),
              [&is_emitted](const uint32_t dependency) { return is_emitted[dependency]; });
          if (is_ready && projection != nullptr && IsSameInput(*first, *projection)) {
            group.projections.push_back(*projection);
            branches.push_back(candidate);
          }
        }
      }
      if (group.projections.size() > 1U) {
        OMNI_RETURN_VAL_IF(!ValidateParallelLinear(type, kind, group), false);
        compiled.emplace_back(std::move(group));
      } else {
        compiled.push_back(nodes[node].operation);
      }
      is_emitted[node] = true;
      for (const uint32_t branch : branches) {
        is_emitted[branch] = true;
      }
    }
    *operations = std::move(compiled);
    return true;
  }

private:
  struct Access final {
    std::vector<DeviceView> views;
    std::vector<DeviceView> writes;
  };

  static void Collect(const ComputeOperation &operation, Access *const access) {
    std::visit(
        [access](const auto &args) {
          using Operation = std::decay_t<decltype(args)>;
          if constexpr (std::is_same_v<Operation, ParallelLinear>) {
            for (const Linear &projection : args.projections) {
              Collect(ComputeOperation(projection), access);
            }
          } else {
#define OMNI_COLLECT_VIEW(field)                                                                   \
  if constexpr (requires { args.field; }) {                                                        \
    if (args.field.IsValid()) {                                                                    \
      access->views.push_back(args.field);                                                         \
    }                                                                                              \
  }
            OMNI_COLLECT_VIEW(input)
            OMNI_COLLECT_VIEW(weight)
            OMNI_COLLECT_VIEW(output)
            OMNI_COLLECT_VIEW(indices)
            OMNI_COLLECT_VIEW(query)
            OMNI_COLLECT_VIEW(key)
            OMNI_COLLECT_VIEW(value)
            OMNI_COLLECT_VIEW(query_weight)
            OMNI_COLLECT_VIEW(key_weight)
            OMNI_COLLECT_VIEW(position_ids)
            OMNI_COLLECT_VIEW(positions)
            OMNI_COLLECT_VIEW(cache_position)
            OMNI_COLLECT_VIEW(query_output)
            OMNI_COLLECT_VIEW(key_output)
            OMNI_COLLECT_VIEW(key_cache)
            OMNI_COLLECT_VIEW(value_cache)
            OMNI_COLLECT_VIEW(decode_partials)
            OMNI_COLLECT_VIEW(residual)
            OMNI_COLLECT_VIEW(residual_output)
            OMNI_COLLECT_VIEW(lhs)
            OMNI_COLLECT_VIEW(rhs)
            OMNI_COLLECT_VIEW(gate)
            OMNI_COLLECT_VIEW(up)
            OMNI_COLLECT_VIEW(logits)
            OMNI_COLLECT_VIEW(source)
            OMNI_COLLECT_VIEW(destination)
            OMNI_COLLECT_VIEW(first_weight)
            OMNI_COLLECT_VIEW(second_weight)
            OMNI_COLLECT_VIEW(bias)
            OMNI_COLLECT_VIEW(position_weight)
            OMNI_COLLECT_VIEW(qkv)
            OMNI_COLLECT_VIEW(scale)
            OMNI_COLLECT_VIEW(shift)
            OMNI_COLLECT_VIEW(state)
            OMNI_COLLECT_VIEW(velocity)
            OMNI_COLLECT_VIEW(query_gate)
            OMNI_COLLECT_VIEW(fc1_weight)
            OMNI_COLLECT_VIEW(fc1_bias)
            OMNI_COLLECT_VIEW(fc2_weight)
            OMNI_COLLECT_VIEW(fc2_bias)
#undef OMNI_COLLECT_VIEW
#define OMNI_COLLECT_WRITE(field)                                                                  \
  if constexpr (requires { args.field; }) {                                                        \
    if (args.field.IsValid()) {                                                                    \
      access->writes.push_back(args.field);                                                        \
    }                                                                                              \
  }
            OMNI_COLLECT_WRITE(output)
            OMNI_COLLECT_WRITE(query_output)
            OMNI_COLLECT_WRITE(key_output)
            OMNI_COLLECT_WRITE(residual_output)
            OMNI_COLLECT_WRITE(destination)
            if constexpr (std::is_same_v<Operation, StoreKvCache> ||
                          std::is_same_v<Operation, NormalizeRopeCache>) {
              OMNI_COLLECT_WRITE(key_cache)
              OMNI_COLLECT_WRITE(value_cache)
            }
            if constexpr (std::is_same_v<Operation, AdvanceSequence>) {
              OMNI_COLLECT_WRITE(positions)
              OMNI_COLLECT_WRITE(cache_position)
            }
            if constexpr (std::is_same_v<Operation, Attention>) {
              OMNI_COLLECT_WRITE(decode_partials)
            }
            if constexpr (std::is_same_v<Operation, GridQkvRope> ||
                          std::is_same_v<Operation, SplitGatedQuery>) {
              OMNI_COLLECT_WRITE(query)
              OMNI_COLLECT_WRITE(key)
              OMNI_COLLECT_WRITE(value)
              OMNI_COLLECT_WRITE(gate)
            }
#undef OMNI_COLLECT_WRITE
          }
        },
        operation);
  }

  static bool HasConflict(const Access &first, const Access &second) {
    for (const DeviceView &write : first.writes) {
      for (const DeviceView &view : second.views) {
        if (HasOverlap(write, view)) {
          return true;
        }
      }
    }
    for (const DeviceView &write : second.writes) {
      for (const DeviceView &view : first.views) {
        if (HasOverlap(write, view)) {
          return true;
        }
      }
    }
    return false;
  }

  std::vector<std::vector<bool>> ancestors_;
};

} // namespace

bool ValidateParallelLinear(const ScalarType type, const BackendKind kind,
                            const ParallelLinear &operation) {
  OMNI_RETURN_VAL_IF_LOG((kind != BackendKind::kCuda && kind != BackendKind::kOpenCL) ||
                             operation.projections.size() < 2U || operation.projections.size() > 3U,
                         false, ERROR, "Parallel projection requires two or three branches");
  size_t bytes = 0U;
  OMNI_RETURN_VAL_IF(!GetElementBytes(type, &bytes), false);
  const Linear &first = operation.projections.front();
  for (const Linear &projection : operation.projections) {
    OMNI_RETURN_VAL_IF_LOG(projection.rows <= 0 || projection.inner <= 0 ||
                               projection.columns <= 0 || !IsSameInput(first, projection) ||
                               !HasStorage(projection.input, kind,
                                           uint64_t(projection.rows) * projection.inner, bytes) ||
                               !HasStorage(projection.weight, kind,
                                           uint64_t(projection.columns) * projection.inner,
                                           bytes) ||
                               !HasStorage(projection.output, kind,
                                           uint64_t(projection.rows) * projection.columns, bytes),
                           false, ERROR, "Invalid parallel projection shape, input or storage");
    for (const Linear &other : operation.projections) {
      OMNI_RETURN_VAL_IF_LOG(
          HasOverlap(projection.output, other.input) ||
              HasOverlap(projection.output, other.weight) ||
              (&projection != &other && HasOverlap(projection.output, other.output)),
          false, ERROR, "Parallel projection outputs overlap another branch");
    }
  }
  return true;
}

bool CompileComputeGraph(const ScalarType type, const BackendKind kind,
                         const std::vector<ComputeNode> &nodes,
                         std::vector<ComputeOperation> *const operations) {
  GraphCompiler compiler;
  return compiler.Compile(type, kind, nodes, operations);
}

} // namespace backend
} // namespace omni_runtime
