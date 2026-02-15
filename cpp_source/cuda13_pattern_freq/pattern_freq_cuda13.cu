#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void pattern_freq_kernel(
    const int8_t* __restrict__ sequences,
    const int64_t* __restrict__ quartet_indices,
    float* __restrict__ result,
    int n_quartets,
    int seq_length,
    int n_species
) {
    int quartet_idx = blockIdx.x;
    if (quartet_idx >= n_quartets) return;

    int sp0 = static_cast<int>(quartet_indices[quartet_idx * 4 + 0]);
    int sp1 = static_cast<int>(quartet_indices[quartet_idx * 4 + 1]);
    int sp2 = static_cast<int>(quartet_indices[quartet_idx * 4 + 2]);
    int sp3 = static_cast<int>(quartet_indices[quartet_idx * 4 + 3]);

    if (sp0 < 0 || sp0 >= n_species || sp1 < 0 || sp1 >= n_species || sp2 < 0 || sp2 >= n_species || sp3 < 0 ||
        sp3 >= n_species) {
        return;
    }

    __shared__ int histogram[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) histogram[i] = 0;
    __syncthreads();

    int valid_count = 0;
    for (int pos = threadIdx.x; pos < seq_length; pos += blockDim.x) {
        int8_t s0 = sequences[(int64_t)sp0 * seq_length + pos];
        int8_t s1 = sequences[(int64_t)sp1 * seq_length + pos];
        int8_t s2 = sequences[(int64_t)sp2 * seq_length + pos];
        int8_t s3 = sequences[(int64_t)sp3 * seq_length + pos];
        if (s0 < 4 && s1 < 4 && s2 < 4 && s3 < 4) {
            int pattern_idx = s0 * 64 + s1 * 16 + s2 * 4 + s3;
            atomicAdd(&histogram[pattern_idx], 1);
            valid_count++;
        }
    }

    __shared__ int valid_count_shared[256];
    valid_count_shared[threadIdx.x] = valid_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total_valid = 0;
        for (int i = 0; i < blockDim.x; i++) total_valid += valid_count_shared[i];

        float* result_base = result + quartet_idx * 256;
        if (total_valid > 0) {
            for (int i = 0; i < 256; i++) result_base[i] = (float)histogram[i] / total_valid;
        } else {
            for (int i = 0; i < 256; i++) result_base[i] = 0.0f;
        }
    }
}

torch::Tensor compute_pattern_frequencies_cuda13(torch::Tensor sequences, torch::Tensor quartet_indices) {
    const int n_species = static_cast<int>(sequences.size(0));
    const int seq_length = static_cast<int>(sequences.size(1));
    const int n_quartets = static_cast<int>(quartet_indices.size(0));

    auto result = torch::zeros({n_quartets, 256},
                               torch::TensorOptions().dtype(torch::kFloat32).device(sequences.device()));

    const int threads = 256;
    const int blocks = n_quartets;

    pattern_freq_kernel<<<blocks, threads>>>(
        sequences.data_ptr<int8_t>(),
        quartet_indices.data_ptr<int64_t>(),
        result.data_ptr<float>(),
        n_quartets,
        seq_length,
        n_species
    );

    return result;
}

