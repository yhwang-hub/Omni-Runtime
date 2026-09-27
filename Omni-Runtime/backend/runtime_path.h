#pragma once

#include <cstdlib>
#include <filesystem>

namespace omni_runtime::backend {

inline std::filesystem::path FindLibraryRoot() {
  const char *const root = std::getenv("OMNI_RUNTIME_ROOT");
  if (root != nullptr && root[0] != '\0') {
    return std::filesystem::path(root);
  }
  const char *const test_srcdir = std::getenv("TEST_SRCDIR");
  if (test_srcdir != nullptr && test_srcdir[0] != '\0') {
    return std::filesystem::path(test_srcdir) / "_main" / "Omni-Runtime";
  }
  std::error_code error;
  std::filesystem::path directory = std::filesystem::current_path(error);
  while (!error) {
    if (std::filesystem::is_directory(directory / "Omni-Runtime" / "kernels", error)) {
      return directory / "Omni-Runtime";
    }
    error.clear();
    const std::filesystem::path parent = directory.parent_path();
    if (parent == directory) {
      break;
    }
    directory = parent;
  }
  return std::filesystem::current_path();
}

} // namespace omni_runtime::backend
