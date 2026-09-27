#include "Omni-Runtime/graph/element/adapter.h"

#include <algorithm>

#include "Omni-Runtime/utils/status_macros.h"

namespace omni_runtime {
namespace graph {

Function::Function() : Adapter(ElementType::kFunction) {
}

bool Function::SetFunction(const FunctionType type, const std::function<Status()> &function) {
  OMNI_RETURN_VAL_IF(!function || IsInitialized(), false);
  if (type == FunctionType::kInit) {
    init_function_ = function;
  } else if (type == FunctionType::kRun) {
    run_function_ = function;
  } else if (type == FunctionType::kDestroy) {
    destroy_function_ = function;
  } else {
    return false;
  }
  return true;
}

Status Function::Init() {
  return init_function_ ? init_function_() : Status();
}

Status Function::Run() {
  return run_function_ ? run_function_() : Status();
}

Status Function::Destroy() {
  return destroy_function_ ? destroy_function_() : Status();
}

Fence::Fence() : Adapter(ElementType::kFence) {
}

Status Fence::WaitElement(const std::shared_ptr<Element> &element) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr, InvalidArgumentStatus("fence element is null"), ERROR,
                    "fence element is null");
  OMNI_RETURN_VAL_IF_LOG(element->timeout_ms() <= kNoTimeoutMs,
                    InvalidArgumentStatus("fence requires an asynchronous element"), ERROR,
                    "fence requires an asynchronous element");
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = std::find_if(
      elements_.begin(), elements_.end(),
      [&element](const std::weak_ptr<Element> &value) { return value.lock() == element; });
  if (iter == elements_.end()) {
    elements_.emplace_back(element);
  }
  return Status();
}

Status Fence::WaitElements(const std::vector<std::shared_ptr<Element>> &elements) {
  for (const auto &element : elements) {
    OMNI_RETURN_STATUS_IF_NOT_OK(WaitElement(element));
  }
  return Status();
}

bool Fence::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  elements_.clear();
  return true;
}

Status Fence::Run() {
  std::vector<std::shared_ptr<Element>> elements;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &element : elements_) {
      const auto value = element.lock();
      if (value != nullptr) {
        elements.emplace_back(value);
      }
    }
  }
  Status first_error;
  for (const auto &element : elements) {
    const Status status = element->WaitAsync();
    if (first_error.ok() && !status.ok()) {
      first_error = status;
    }
  }
  return first_error;
}

} // namespace graph
} // namespace omni_runtime
