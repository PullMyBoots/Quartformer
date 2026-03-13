#include <torch/extension.h>

#include <stdexcept>

torch::Tensor compute_pattern_frequencies_cuda_backend_v2_packed_kernel(
    torch::Tensor sequences_packed,
    torch::Tensor quartet_indices,
    int64_t seq_length
);

torch::Tensor compute_pattern_frequencies_cuda_backend_v2_grouped_kernel(
    torch::Tensor sequences_packed,
    torch::Tensor block_species,
    int64_t seq_length
);

torch::Tensor compute_pattern_frequencies_cuda_packed(
    torch::Tensor sequences_packed,
    torch::Tensor quartet_indices,
    int64_t seq_length
) {
    if (!sequences_packed.is_cuda() || !quartet_indices.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (sequences_packed.dtype() != torch::kUInt8) {
        throw std::runtime_error("packed sequences must be torch.uint8");
    }
    if (quartet_indices.dtype() != torch::kInt32 && quartet_indices.dtype() != torch::kInt64) {
        throw std::runtime_error("quartet_indices must be torch.int32 or torch.int64");
    }
    if (sequences_packed.dim() != 2) {
        throw std::runtime_error("packed sequences must have shape (n_species, packed_seq_length)");
    }
    if (quartet_indices.dim() != 2 || quartet_indices.size(1) != 4) {
        throw std::runtime_error("quartet_indices must have shape (n_quartets, 4)");
    }
    if (seq_length <= 0) {
        throw std::runtime_error("seq_length must be > 0");
    }

    return compute_pattern_frequencies_cuda_backend_v2_packed_kernel(
        sequences_packed.contiguous(),
        quartet_indices.contiguous(),
        seq_length
    );
}

torch::Tensor compute_pattern_frequencies_cuda_grouped_uniform(
    torch::Tensor sequences_packed,
    torch::Tensor block_species,
    int64_t seq_length
) {
    if (!sequences_packed.is_cuda() || !block_species.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (sequences_packed.dtype() != torch::kUInt8) {
        throw std::runtime_error("packed sequences must be torch.uint8");
    }
    if (block_species.dtype() != torch::kInt32) {
        throw std::runtime_error("block_species must be torch.int32");
    }
    if (sequences_packed.dim() != 2) {
        throw std::runtime_error("packed sequences must have shape (n_species, packed_seq_length)");
    }
    if (block_species.dim() != 2 || block_species.size(1) != 24) {
        throw std::runtime_error("block_species must have shape (n_blocks, 24)");
    }
    if (seq_length <= 0) {
        throw std::runtime_error("seq_length must be > 0");
    }

    return compute_pattern_frequencies_cuda_backend_v2_grouped_kernel(
        sequences_packed.contiguous(),
        block_species.contiguous(),
        seq_length
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "compute_pattern_frequencies_cuda_packed",
        &compute_pattern_frequencies_cuda_packed,
        "Pattern frequency computation for packed sequences (CUDA)"
    );
    m.def(
        "compute_pattern_frequencies_cuda_grouped_uniform",
        &compute_pattern_frequencies_cuda_grouped_uniform,
        "Deprecated grouped path retained for compatibility"
    );
}
