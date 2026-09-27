#pragma once

#include <algorithm>
#include <filesystem>
#include <format>
#include <string_view>
#include <system_error>
#include <vector>

#include "Omni-Runtime/kernel/utils/env.h"
#include "Omni-Runtime/kernel/utils/filesystem.h"
#include "Omni-Runtime/kernel/utils/uuid.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace disk {

inline constexpr std::string_view kCommitFileName = ".committed";

class DiskCacheEntry final {
public:
  bool hit = false;
  bool committed = false;
  std::filesystem::path path;
  std::filesystem::path commit_path;

  DiskCacheEntry(const DiskCacheEntry &) = delete;
  DiskCacheEntry &operator=(const DiskCacheEntry &) = delete;
  DiskCacheEntry(const bool hit, std::filesystem::path path, std::filesystem::path commit_path)
      : hit(hit), path(std::move(path)), commit_path(std::move(commit_path)) {
  }

  std::filesystem::path commit() {
    if (hit or committed) {
      return path;
    }
    const std::filesystem::path commit_marker = path / kCommitFileName;
    ::omni_runtime::jit::utils::write_file_sync(commit_marker, "");
    ::omni_runtime::jit::utils::fsync_dir(path);
    ::omni_runtime::jit::utils::make_dirs(commit_path.parent_path());
    std::error_code error;
    std::filesystem::rename(path, commit_path, error);
    if (error) {
      ::omni_runtime::jit::utils::safe_remove_all(path);
    }
    path = commit_path;
    hit = true;
    committed = true;
    return path;
  }

  ~DiskCacheEntry() {
    if (not hit and not committed) {
      ::omni_runtime::jit::utils::safe_remove_all(path);
    }
  }
};

struct DiskCache final {
  std::vector<std::filesystem::path> paths;

  explicit DiskCache(std::vector<std::filesystem::path> cache_paths)
      : paths(std::move(cache_paths)) {
  }

  static DiskCache from_env(const ::omni_runtime::jit::utils::Env &env) {
    std::vector<std::filesystem::path> cache_paths;
    if (const auto value = env.get<std::string>("JIT_CACHE_DIR")) {
      std::size_t begin = 0;
      while (true) {
        const std::size_t end = value->find(':', begin);
        const std::string item = value->substr(begin, end - begin);
        cache_paths.emplace_back(item);
        if (end == std::string::npos) {
          break;
        }
        begin = end + 1;
      }
    } else {
      const std::string home = ::omni_runtime::jit::utils::get_env<std::string>("HOME", "");
      const std::filesystem::path cache_root = home.empty()
                                                   ? std::filesystem::path(".omni_jit")
                                                   : std::filesystem::path(home) / ".omni_jit";
      cache_paths.emplace_back(cache_root);
    }
    return DiskCache(cache_paths);
  }

  [[nodiscard]] DiskCacheEntry entry(const std::string &tag, const std::string &digest) const {
    const auto is_valid_char = [](const char item) {
      return (item >= 'a' and item <= 'z') or (item >= 'A' and item <= 'Z') or
             (item >= '0' and item <= '9') or item == '_';
    };
    if (tag.empty() or not std::ranges::all_of(tag, is_valid_char)) {
      return DiskCacheEntry(false, std::filesystem::path(), std::filesystem::path());
    }
    const std::string entry_name = std::format("{}.{}", tag, digest);
    for (const auto &dir : paths) {
      const std::filesystem::path path = dir / "cache" / entry_name;
      if (std::filesystem::exists(path / kCommitFileName)) {
        ::omni_runtime::jit::utils::try_update_mtime(path / kCommitFileName);
        return DiskCacheEntry(true, path, std::filesystem::path());
      }
    }
    const std::filesystem::path temporary_path =
        paths[0] / "tmp" / ::omni_runtime::jit::utils::get_uuid();
    std::filesystem::create_directories(temporary_path);
    return DiskCacheEntry(false, temporary_path, paths[0] / "cache" / entry_name);
  }
};

} // namespace disk
} // namespace backend
} // namespace jit
} // namespace omni_runtime
