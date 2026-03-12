#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {
constexpr int kBaseStates = 4;
constexpr int kPatternDim = 256;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

__device__ __forceinline__ unsigned char unpack_low(uint8_t byte) {
    return static_cast<unsigned char>(byte & 0x0F);
}

__device__ __forceinline__ unsigned char unpack_high(uint8_t byte) {
    return static_cast<unsigned char>((byte >> 4) & 0x0F);
}
}  // namespace

__global__ void pattern_freq_packed_kernel(
    const uint8_t* __restrict__ sequences_packed,
    const int64_t* __restrict__ quartet_indices,
    float* __restrict__ result,
    int n_quartets,
    int seq_length,
    int packed_seq_length,
    int n_species
) {
    const int quartet_idx = blockIdx.x;
    if (quartet_idx >= n_quartets) return;

    const int sp0 = static_cast<int>(quartet_indices[quartet_idx * 4 + 0]);
    const int sp1 = static_cast<int>(quartet_indices[quartet_idx * 4 + 1]);
    const int sp2 = static_cast<int>(quartet_indices[quartet_idx * 4 + 2]);
    const int sp3 = static_cast<int>(quartet_indices[quartet_idx * 4 + 3]);

    if (sp0 < 0 || sp0 >= n_species || sp1 < 0 || sp1 >= n_species || sp2 < 0 || sp2 >= n_species || sp3 < 0 ||
        sp3 >= n_species) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ int warp_hist[kWarpsPerBlock][kPatternDim];
    __shared__ int valid_count_shared[kThreadsPerBlock];

    for (int i = lane; i < kPatternDim; i += 32) {
        warp_hist[warp_id][i] = 0;
    }
    __syncthreads();

    int valid_count = 0;
    const uint8_t* row0 = sequences_packed + static_cast<int64_t>(sp0) * packed_seq_length;
    const uint8_t* row1 = sequences_packed + static_cast<int64_t>(sp1) * packed_seq_length;
    const uint8_t* row2 = sequences_packed + static_cast<int64_t>(sp2) * packed_seq_length;
    const uint8_t* row3 = sequences_packed + static_cast<int64_t>(sp3) * packed_seq_length;

    for (int packed_pos = tid; packed_pos < packed_seq_length; packed_pos += blockDim.x) {
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
            const int pattern_idx = static_cast<int>(s0_lo) * 64 + static_cast<int>(s1_lo) * 16 +
                                    static_cast<int>(s2_lo) * 4 + static_cast<int>(s3_lo);
            atomicAdd(&warp_hist[warp_id][pattern_idx], 1);
            valid_count++;
        }

        if (site0 + 1 < seq_length) {
            const unsigned char s0_hi = unpack_high(b0);
            const unsigned char s1_hi = unpack_high(b1);
            const unsigned char s2_hi = unpack_high(b2);
            const unsigned char s3_hi = unpack_high(b3);
            if (s0_hi < kBaseStates && s1_hi < kBaseStates && s2_hi < kBaseStates && s3_hi < kBaseStates) {
                const int pattern_idx = static_cast<int>(s0_hi) * 64 + static_cast<int>(s1_hi) * 16 +
                                        static_cast<int>(s2_hi) * 4 + static_cast<int>(s3_hi);
                atomicAdd(&warp_hist[warp_id][pattern_idx], 1);
                valid_count++;
            }
        }
    }

    valid_count_shared[tid] = valid_count;
    __syncthreads();

    if (tid == 0) {
        int total_valid = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            total_valid += valid_count_shared[i];
        }

        float* result_base = result + quartet_idx * kPatternDim;
        if (total_valid > 0) {
            for (int i = 0; i < kPatternDim; ++i) {
                int count = 0;
                for (int warp = 0; warp < kWarpsPerBlock; ++warp) {
                    count += warp_hist[warp][i];
                }
                result_base[i] = static_cast<float>(count) / static_cast<float>(total_valid);
            }
        } else {
            for (int i = 0; i < kPatternDim; ++i) {
                result_base[i] = 0.0f;
            }
        }
    }
}

torch::Tensor compute_pattern_frequencies_cuda13_packed(
    torch::Tensor sequences_packed,
    torch::Tensor quartet_indices,
    int64_t seq_length
) {
    const int n_species = static_cast<int>(sequences_packed.size(0));
    const int packed_seq_length = static_cast<int>(sequences_packed.size(1));
    const int n_quartets = static_cast<int>(quartet_indices.size(0));

    auto result = torch::zeros(
        {n_quartets, kPatternDim},
        torch::TensorOptions().dtype(torch::kFloat32).device(sequences_packed.device())
    );

    pattern_freq_packed_kernel<<<n_quartets, kThreadsPerBlock>>>(
        sequences_packed.data_ptr<uint8_t>(),
        quartet_indices.data_ptr<int64_t>(),
        result.data_ptr<float>(),
        n_quartets,
        static_cast<int>(seq_length),
        packed_seq_length,
        n_species
    );

    return result;
}
