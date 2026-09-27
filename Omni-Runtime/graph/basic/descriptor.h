#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace omni_runtime {
namespace graph {

class Descriptor {
public:
  explicit Descriptor(const std::string &type_name) : session_(MakeSession(type_name)) {
  }
  virtual ~Descriptor() = default;

  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;

  const std::string &name() const {
    return name_.empty() ? session_ : name_;
  }

  const std::string &session() const {
    return session_;
  }

  const std::string &description() const {
    return description_;
  }

  bool set_name(const std::string &name) {
    name_ = name;
    return true;
  }

  bool set_description(const std::string &description) {
    description_ = description;
    return true;
  }

private:
  static std::string MakeSession(const std::string &type_name) {
    static std::atomic<uint64_t> next_id = 0U;
    const uint64_t id = next_id.fetch_add(1U, std::memory_order_relaxed);
    return type_name + "_" + std::to_string(id);
  }

  std::string name_;
  std::string session_;
  std::string description_;
};

} // namespace graph
} // namespace omni_runtime
