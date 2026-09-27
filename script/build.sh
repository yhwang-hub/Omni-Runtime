# Build script for Omni-Runtime.
# Usage examples are listed in the usage function below.
#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

build_mode="opt"
job_count="$(nproc)"
target_list=()
is_clean_enabled=false
is_keep_going_enabled=false
is_verbose_enabled=false

function info() {
  printf '\033[1;34m[INFO]\033[0m %s\n' "$*"
}

function error() {
  printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"
}

function success() {
  printf '\033[1;32m[OK]\033[0m %s\n' "$*"
}

function usage() {
  # Keep this help text synchronized with the script options and examples.
  cat <<'USAGE'
Usage: script/build.sh [options]

Options:
  -m, --mode <mode>        Build mode: opt or dbg. Default: opt.
  -j, --jobs <count>       Parallel jobs. Default: number of CPU cores.
  -t, --target <label>     Bazel target to build. Can be repeated.
                            Default: //...
  -c, --clean              Remove the Bazel output base and workspace convenience symlinks
                            (bazel-bin, bazel-out, bazel-testlogs, bazel-Omni-Runtime)
                            before building.
  -k, --keep-going         Continue as much as possible after an error.
  -v, --verbose            Print the executed Bazel commands.
  -h, --help               Show this help message.

Examples:
  ./script/build.sh
  ./script/build.sh --mode dbg
  ./script/build.sh --clean
  ./script/build.sh --target //Omni-Runtime/kernel:kernel
  ./script/build.sh --target //Omni-Runtime/kernels:kernels_tests
USAGE
}

function require_bazel() {
  # Bazel is the only build system used by this repository.
  if ! command -v bazel >/dev/null 2>&1; then
    error "Bazel is not installed or not in PATH."
    exit 1
  fi
}

function ensure_cuda_environment() {
  # CUDA_HOME and CUDA_PATH are optional but are set when the default toolkit is present.
  if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
    export CUDA_HOME="/usr/local/cuda"
  fi

  if [[ -z "${CUDA_PATH:-}" && -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="${CUDA_HOME}"
  fi
}

function parse_arguments() {
  # Parse long and short options. Every option requires an explicit value.
  while (($# > 0)); do
    case "$1" in
      -m | --mode)
        if (($# < 2)); then
          error "Option $1 requires a value."
          exit 1
        fi
        build_mode="$2"
        shift 2
        ;;
      -j | --jobs)
        if (($# < 2)); then
          error "Option $1 requires a value."
          exit 1
        fi
        job_count="$2"
        shift 2
        ;;
      -t | --target)
        if (($# < 2)); then
          error "Option $1 requires a value."
          exit 1
        fi
        target_list+=("$2")
        shift 2
        ;;
      -c | --clean)
        is_clean_enabled=true
        shift
        ;;
      -k | --keep-going)
        is_keep_going_enabled=true
        shift
        ;;
      -v | --verbose)
        is_verbose_enabled=true
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        error "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
  done

  if [[ "${build_mode}" != "opt" && "${build_mode}" != "dbg" ]]; then
    error "Invalid build mode: ${build_mode}. Supported values are opt and dbg."
    exit 1
  fi

  if ! [[ "${job_count}" =~ ^[1-9][0-9]*$ ]]; then
    error "Invalid job count: ${job_count}. It must be a positive integer."
    exit 1
  fi

  if ((${#target_list[@]} == 0)); then
    target_list=("//...")
  fi
}

function remove_bazel_convenience_links() {
  # Bazel creates these symlinks in the workspace root. `bazel clean --expunge`
  # removes the real output base, but leaves the convenience symlinks behind.
  local bazel_link
  local bazel_links=(
    bazel-bin
    bazel-out
    bazel-testlogs
    bazel-Omni-Runtime
  )

  for bazel_link in "${bazel_links[@]}"; do
    if [[ -L "${bazel_link}" ]]; then
      info "Removing Bazel convenience symlink: ${bazel_link}"
      rm "${bazel_link}"
    fi
  done
}

function clean_bazel_outputs() {
  info "Cleaning Bazel output base..."
  bazel clean --expunge
  remove_bazel_convenience_links
}

function build() {
  # Build all targets by default. Use --target to build one or more Bazel labels.
  cd "${PROJECT_ROOT}"

  if [[ "${is_clean_enabled}" == true ]]; then
    clean_bazel_outputs
  fi

  local bazel_arguments=(
    "build"
    "--compilation_mode=${build_mode}"
    "--jobs=${job_count}"
  )

  if [[ "${is_keep_going_enabled}" == true ]]; then
    bazel_arguments+=("--keep_going")
  fi

  if [[ "${is_verbose_enabled}" == true ]]; then
    bazel_arguments+=("--subcommands")
  fi

  bazel_arguments+=("${target_list[@]}")

  info "Building Omni-Runtime in ${build_mode} mode..."
  bazel "${bazel_arguments[@]}"
}

function main() {
  require_bazel
  ensure_cuda_environment
  parse_arguments "$@"
  build
  success "Omni-Runtime build completed."
}

main "$@"
