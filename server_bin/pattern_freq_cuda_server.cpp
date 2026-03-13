#include <torch/extension.h>

#include <stdexcept>

torch::Tensor compute_pattern_frequencies_cuda13(
    torch::Tensor sequences,
    torch::Tensor quartet_indices
);

torch::Tensor compute_pattern_frequencies_cuda(
    torch::Tensor sequences,
    torch::Tensor quartet_indices
) {
    if (!sequences.is_cuda() || !quartet_indices.is_cuda()) {
        throw std::runtime_error("inputs must be CUDA tensors");
    }
    if (sequences.dtype() != torch::kInt8) {
        throw std::runtime_error("sequences must be torch.int8");
    }
    if (quartet_indices.dtype() != torch::kInt64) {
        throw std::runtime_error("quartet_indices must be torch.int64");
    }
    if (sequences.dim() != 2) {
        throw std::runtime_error("sequences must have shape (n_species, seq_length)");
    }
    if (quartet_indices.dim() != 2 || quartet_indices.size(1) != 4) {
        throw std::runtime_error("quartet_indices must have shape (n_quartets, 4)");
    }

    return compute_pattern_frequencies_cuda13(
        sequences.contiguous(),
        quartet_indices.contiguous()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "compute_pattern_frequencies_cuda",
        &compute_pattern_frequencies_cuda,
        "Pattern frequency computation for dense sequences (CUDA)"
    );
}
