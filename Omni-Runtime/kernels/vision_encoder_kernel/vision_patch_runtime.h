#pragma once

#include <tuple>

#include "kernels/vision_encoder_kernel/vision_encoder_cuda_runtime.h"

namespace omni_runtime::kernels::runtime {

template <typename T> struct CudaOperator<T, backend::PatchEmbedding> final {
  static bool Launch(const backend::PatchEmbedding &args) {
    OMNI_RETURN_VAL_IF_LOG(
        args.frames <= 0 || args.height <= 0 || args.width <= 0 || args.channels <= 0 ||
            args.output_channels <= 0 || args.temporal_patch <= 0 || args.spatial_patch <= 0 ||
            args.frames % args.temporal_patch != 0 || args.height % args.spatial_patch != 0 ||
            args.width % args.spatial_patch != 0,
        false, ERROR, "Invalid spatial temporal patch shape");
    const uint64_t count = uint64_t(args.frames / args.temporal_patch) *
                           (args.height / args.spatial_patch) * (args.width / args.spatial_patch) *
                           args.output_channels;
    return VisionPatchEmbedKernelRuntime<T>::CompileAndLaunch(
        "backend_PatchEmbedding",
        {.options = {.grid_dim = {static_cast<uint32_t>(count), 1U, 1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .input = backend::cuda_internal::CudaAddress<void>(args.input),
         .weight = backend::cuda_internal::CudaAddress<void>(args.weight),
         .bias = backend::cuda_internal::CudaAddress<void>(args.bias),
         .frames = args.frames,
         .height = args.height,
         .width = args.width,
         .channels = args.channels,
         .output_channels = args.output_channels,
         .temporal_patch = args.temporal_patch,
         .spatial_patch = args.spatial_patch,
         .output = backend::cuda_internal::CudaAddress<void>(args.output)});
  }
};

template <> struct OpenCLOperator<backend::PatchEmbedding> final {
  static bool Make(const backend::PatchEmbedding &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(
        args.frames <= 0 || args.height <= 0 || args.width <= 0 || args.channels <= 0 ||
            args.output_channels <= 0 || args.temporal_patch <= 0 || args.spatial_patch <= 0 ||
            args.frames % args.temporal_patch != 0 || args.height % args.spatial_patch != 0 ||
            args.width % args.spatial_patch != 0,
        false, ERROR, "Invalid spatial temporal patch shape");
    const uint64_t patches = uint64_t(args.frames / args.temporal_patch) *
                             (args.height / args.spatial_patch) * (args.width / args.spatial_patch);
    const uint64_t inner =
        uint64_t(args.channels) * args.temporal_patch * args.spatial_patch * args.spatial_patch;
    const size_t bytes = backend::ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(
        !backend::HasElements(args.input, patches * inner, bytes) ||
            !backend::HasElements(args.weight, uint64_t(args.output_channels) * inner, bytes) ||
            !backend::HasElements(args.bias, args.output_channels, bytes) ||
            !backend::HasElements(args.output, patches * args.output_channels, bytes),
        false, ERROR, "Invalid spatial temporal patch buffers");
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::ConfigureOpenCL(
                           "vision_encoder_kernel/vision_patch.cl", "PatchEmbedding",
                           patches * args.output_channels, 256U, call),
                       false);
    OMNI_RETURN_VAL_IF(
        !backend::opencl_internal::BindOpenCLArguments(
            std::forward_as_tuple(args.input, args.weight, args.bias, args.frames, args.height,
                                  args.width, args.channels, args.output_channels,
                                  args.temporal_patch, args.spatial_patch, args.output),
            call),
        false);
    return true;
  }
};

template <typename T> struct CudaOperator<T, backend::PatchMerge> final {
  static bool Launch(const backend::PatchMerge &args) {
    OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.grid_width <= 0 || args.hidden_size <= 0 ||
                               args.merge_size <= 0,
                           false, ERROR, "Invalid patch merge dimensions");
    const uint64_t count = uint64_t(args.token_count) * args.hidden_size;
    return VisionMergeKernelRuntime<T>::CompileAndLaunch(
        "backend_PatchMerge",
        {.options = {.grid_dim = {static_cast<uint32_t>((count + 255U) / 256U), 1U, 1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .input = backend::cuda_internal::CudaAddress<void>(args.input),
         .token_count = args.token_count,
         .grid_width = args.grid_width,
         .hidden_size = args.hidden_size,
         .merge_size = args.merge_size,
         .output = backend::cuda_internal::CudaAddress<void>(args.output)});
  }
};

template <> struct OpenCLOperator<backend::PatchMerge> final {
  static bool Make(const backend::PatchMerge &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const uint64_t count = uint64_t(args.token_count) * args.hidden_size;
    const size_t bytes = backend::ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.grid_width <= 0 || args.hidden_size <= 0 ||
                               args.merge_size <= 0 || args.token_count % args.grid_width != 0 ||
                               args.grid_width % args.merge_size != 0 ||
                               (args.token_count / args.grid_width) % args.merge_size != 0 ||
                               !backend::HasElements(args.input, count, bytes) ||
                               !backend::HasElements(args.output, count, bytes),
                           false, ERROR, "Invalid patch merge shape or buffers");
    OMNI_RETURN_VAL_IF(
        !backend::opencl_internal::ConfigureOpenCL("vision_encoder_kernel/vision_patch.cl",
                                                   "PatchMerge", (count + 255U) / 256U, 256U, call),
        false);
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::BindOpenCLArguments(
                           std::forward_as_tuple(args.input, args.token_count, args.grid_width,
                                                 args.hidden_size, args.merge_size, args.output),
                           call),
                       false);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
