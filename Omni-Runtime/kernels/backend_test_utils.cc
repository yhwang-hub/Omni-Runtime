#include "kernels/backend_test_utils.h"

#include <algorithm>

namespace omni_runtime::kernels::test {
namespace {

struct BackendState final {
  std::array<std::shared_ptr<backend::Backend>, 2U> devices;
  std::array<bool, 2U> available = {};
  std::array<bool, 2U> initialized = {};

  BackendState() {
    const std::array<backend::BackendKind, 2U> kinds = {backend::BackendKind::kCuda,
                                                        backend::BackendKind::kOpenCL};
    for (size_t index = 0U; index < kinds.size(); ++index) {
      devices[index] = backend::CreateBackend(kinds[index]);
      available[index] = devices[index] != nullptr && devices[index]->IsAvailable();
      if (available[index]) {
        initialized[index] = devices[index]->InitializeCompute();
      }
    }
  }
};

const BackendState &SharedBackends() {
  static const BackendState state;
  return state;
}

} // namespace

void BackendParityFixture::SetUp() {
  const auto &state = SharedBackends();
  for (size_t index = 0U; index < state.devices.size(); ++index) {
    ASSERT_NE(state.devices[index], nullptr) << "Backend factory failed for index " << index;
    ASSERT_TRUE(!state.available[index] || state.initialized[index])
        << backend::BackendName(state.devices[index]->kind())
        << " is present but InitializeCompute failed";
  }
  if (!state.available[0] || !state.available[1]) {
    GTEST_SKIP() << "CUDA/OpenCL parity requires both devices; CUDA=" << state.available[0]
                 << ", OpenCL=" << state.available[1];
  }
}

const std::array<std::shared_ptr<backend::Backend>, 2U> &BackendParityFixture::devices() const {
  return SharedBackends().devices;
}

void ExpectBackendClose(const std::vector<float> &actual, const std::vector<float> &expected,
                        const BackendTolerance tolerance) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0U; index < actual.size(); ++index) {
    ASSERT_TRUE(std::isfinite(actual[index])) << "element " << index;
    ASSERT_TRUE(std::isfinite(expected[index])) << "reference element " << index;
    EXPECT_NEAR(actual[index], expected[index],
                tolerance.absolute + tolerance.relative * std::fabs(expected[index]))
        << "element " << index;
  }
}

} // namespace omni_runtime::kernels::test
