#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace {
constexpr int kBaseStates = 4;
constexpr int kPatternDim = 256;

__device__ __forceinline__ unsigned char unpack_low(uint8_t byte) {
    return static_cast<unsigned char>(byte & 0x0F);
}

__device__ __forceinline__ unsigned char unpack_high(uint8_t byte) {
    return static_cast<unsigned char>((byte >> 4) & 0x0F);
}

template <typename index_t>
__device__ __forceinline__ int load_index(const index_t* quartet_indices, int quartet_idx, int offset) {
    return static_cast<int>(quartet_indices[quartet_idx * 4 + offset]);
}

template <typename index_t, int kThreadsPerBlock, int kUnroll>
__global__ __launch_bounds__(kThreadsPerBlock, 2) void pattern_freq_packed_kernel_v2(
    const uint8_t* __restrict__ sequences_packed,
    const index_t* __restrict__ quartet_indices,
    float* __restrict__ result,
    int n_quartets,
    int seq_length,
    int packed_seq_length,
    int n_species
) {
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

    const int quartet_idx = blockIdx.x;
    if (quartet_idx >= n_quartets) {
        return;
    }

    const int sp0 = load_index(quartet_indices, quartet_idx, 0);
    const int sp1 = load_index(quartet_indices, quartet_idx, 1);
    const int sp2 = load_index(quartet_indices, quartet_idx, 2);
    const int sp3 = load_index(quartet_indices, quartet_idx, 3);
    if (sp0 < 0 || sp0 >= n_species || sp1 < 0 || sp1 >= n_species ||
        sp2 < 0 || sp2 >= n_species || sp3 < 0 || sp3 >= n_species) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ int warp_hist[kWarpsPerBlock][kPatternDim];
    __shared__ int warp_valid[kWarpsPerBlock];
    __shared__ int total_valid_shared;

    for (int idx = tid; idx < kWarpsPerBlock * kPatternDim; idx += kThreadsPerBlock) {
        warp_hist[idx / kPatternDim][idx % kPatternDim] = 0;
    }
    if (tid < kWarpsPerBlock) {
        warp_valid[tid] = 0;
    }
    __syncthreads();

    int valid_count = 0;
    const uint8_t* row0 = sequences_packed + static_cast<int64_t>(sp0) * packed_seq_length;
    const uint8_t* row1 = sequences_packed + static_cast<int64_t>(sp1) * packed_seq_length;
    const uint8_t* row2 = sequences_packed + static_cast<int64_t>(sp2) * packed_seq_length;
    const uint8_t* row3 = sequences_packed + static_cast<int64_t>(sp3) * packed_seq_length;

    const int stride = kThreadsPerBlock * kUnroll;
    for (int packed_base = tid; packed_base < packed_seq_length; packed_base += stride) {
        #pragma unroll
        for (int unroll_idx = 0; unroll_idx < kUnroll; ++unroll_idx) {
            const int packed_pos = packed_base + unroll_idx * kThreadsPerBlock;
            if (packed_pos >= packed_seq_length) {
                continue;
            }

            const uint8_t b0 = row0[packed_pos];
            const uint8_t b1 = row1[packed_pos];
            const uint8_t b2 = row2[packed_pos];
            const uint8_t b3 = row3[packed_pos];

            const int site0 = packed_pos << 1;

            const unsigned char s0_lo = unpack_low(b0);
            const unsigned char s1_lo = unpack_low(b1);
            const unsigned char s2_lo = unpack_low(b2);
            const unsigned char s3_lo = unpack_low(b3);
            if (s0_lo < kBaseStates && s1_lo < kBaseStates && s2_lo < kBaseStates && s3_lo < kBaseStates) {
                const int pattern_idx =
                    static_cast<int>(s0_lo) * 64 +
                    static_cast<int>(s1_lo) * 16 +
                    static_cast<int>(s2_lo) * 4 +
                    static_cast<int>(s3_lo);
                atomicAdd(&warp_hist[warp_id][pattern_idx], 1);
                ++valid_count;
            }

            if (site0 + 1 < seq_length) {
                const unsigned char s0_hi = unpack_high(b0);
                const unsigned char s1_hi = unpack_high(b1);
                const unsigned char s2_hi = unpack_high(b2);
                const unsigned char s3_hi = unpack_high(b3);
                if (s0_hi < kBaseStates && s1_hi < kBaseStates && s2_hi < kBaseStates && s3_hi < kBaseStates) {
                    const int pattern_idx =
                        static_cast<int>(s0_hi) * 64 +
                        static_cast<int>(s1_hi) * 16 +
                        static_cast<int>(s2_hi) * 4 +
                        static_cast<int>(s3_hi);
                    atomicAdd(&warp_hist[warp_id][pattern_idx], 1);
                    ++valid_count;
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        valid_count += __shfl_down_sync(0xFFFFFFFF, valid_count, offset);
    }
    if (lane == 0) {
        warp_valid[warp_id] = valid_count;
    }
    __syncthreads();

    if (tid == 0) {
        int total_valid = 0;
        for (int warp = 0; warp < kWarpsPerBlock; ++warp) {
            total_valid += warp_valid[warp];
        }
        total_valid_shared = total_valid;
    }
    __syncthreads();

    if (tid < kPatternDim) {
        int count = 0;
        for (int warp = 0; warp < kWarpsPerBlock; ++warp) {
            count += warp_hist[warp][tid];
        }
        const int total_valid = total_valid_shared;
        float* result_base = result + static_cast<int64_t>(quartet_idx) * kPatternDim;
        result_base[tid] = total_valid > 0
            ? static_cast<float>(count) / static_cast<float>(total_valid)
            : 0.0f;
    }
}

template <typename index_t>
void launch_pattern_freq_kernel(
    const uint8_t* sequences_packed,
    const index_t* quartet_indices,
    float* result,
    int n_quartets,
    int seq_length,
    int packed_seq_length,
    int n_species,
    cudaStream_t stream
) {
    if (packed_seq_length >= (1 << 20)) {
        pattern_freq_packed_kernel_v2<index_t, 512, 4><<<n_quartets, 512, 0, stream>>>(
            sequences_packed, quartet_indices, result, n_quartets, seq_length, packed_seq_length, n_species
        );
    } else if (packed_seq_length >= (1 << 14)) {
        pattern_freq_packed_kernel_v2<index_t, 256, 4><<<n_quartets, 256, 0, stream>>>(
            sequences_packed, quartet_indices, result, n_quartets, seq_length, packed_seq_length, n_species
        );
    } else {
        pattern_freq_packed_kernel_v2<index_t, 128, 2><<<n_quartets, 128, 0, stream>>>(
            sequences_packed, quartet_indices, result, n_quartets, seq_length, packed_seq_length, n_species
        );
    }
}
}  // namespace

torch::Tensor compute_pattern_frequencies_cuda_backend_v2_packed_kernel(
    torch::Tensor sequences_packed,
    torch::Tensor quartet_indices,
    int64_t seq_length
) {
    c10::cuda::CUDAGuard device_guard(sequences_packed.device());
    const int n_species = static_cast<int>(sequences_packed.size(0));
    const int packed_seq_length = static_cast<int>(sequences_packed.size(1));
    const int n_quartets = static_cast<int>(quartet_indices.size(0));

    auto result = torch::zeros(
        {n_quartets, kPatternDim},
        torch::TensorOptions().dtype(torch::kFloat32).device(sequences_packed.device())
    );

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (quartet_indices.scalar_type() == torch::kInt32) {
        launch_pattern_freq_kernel<int32_t>(
            sequences_packed.data_ptr<uint8_t>(),
            quartet_indices.data_ptr<int32_t>(),
            result.data_ptr<float>(),
            n_quartets,
            static_cast<int>(seq_length),
            packed_seq_length,
            n_species,
            stream
        );
    } else if (quartet_indices.scalar_type() == torch::kInt64) {
        launch_pattern_freq_kernel<int64_t>(
            sequences_packed.data_ptr<uint8_t>(),
            quartet_indices.data_ptr<int64_t>(),
            result.data_ptr<float>(),
            n_quartets,
            static_cast<int>(seq_length),
            packed_seq_length,
            n_species,
            stream
        );
    } else {
        throw std::runtime_error("quartet_indices must be torch.int32 or torch.int64");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}

torch::Tensor compute_pattern_frequencies_cuda_backend_v2_grouped_kernel(
    torch::Tensor,
    torch::Tensor,
    int64_t
) {
    throw std::runtime_error("Grouped kernel path removed in V2; use compute_pattern_frequencies_cuda_packed instead");
}
