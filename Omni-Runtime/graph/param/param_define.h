#pragma once

#include <memory>

namespace omni_runtime {
namespace graph {

class PassedParam;
using AspectParam = PassedParam;
using DaemonParam = PassedParam;
using ElementParam = PassedParam;
using EventParam = PassedParam;
using StageParam = PassedParam;
using PassedParamPtr = std::shared_ptr<PassedParam>;

} // namespace graph
} // namespace omni_runtime
