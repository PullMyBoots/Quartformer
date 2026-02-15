#include <torch/extension.h>

#include <stdexcept>

torch::Tensor compute_pattern_frequencies_cuda13(torch::Tensor sequences, torch::Tensor quartet_indices);

torch::Tensor compute_pattern_frequencies_cuda(torch::Tensor sequences, torch::Tensor quartet_indices) {
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
    return compute_pattern_frequencies_cuda13(sequences.contiguous(), quartet_indices.contiguous());
}

torch::Tensor compute_pattern_frequencies_cuda_grouped(
    torch::Tensor sequences,
    torch::Tensor quartet_indices,
    torch::Tensor group_offsets,
    torch::Tensor group_lengths,
    int64_t target_len
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
    if (group_offsets.dtype() != torch::kInt64 || group_lengths.dtype() != torch::kInt64) {
        throw std::runtime_error("group_offsets/group_lengths must be torch.int64");
    }
    if (group_offsets.dim() != 1 || group_lengths.dim() != 1) {
        throw std::runtime_error("group_offsets/group_lengths must be 1-D tensors");
    }
    if (group_offsets.size(0) != group_lengths.size(0)) {
        throw std::runtime_error("group_offsets/group_lengths size mismatch");
    }
    if (target_len < 0) {
        throw std::runtime_error("target_len must be >= 0");
    }

    auto group_offsets_cpu = group_offsets.contiguous().to(torch::kCPU);
    auto group_lengths_cpu = group_lengths.contiguous().to(torch::kCPU);

    const int64_t num_groups = group_offsets_cpu.size(0);
    const int64_t total_quartets = quartet_indices.size(0);
    auto offs_ptr = group_offsets_cpu.data_ptr<int64_t>();
    auto lens_ptr = group_lengths_cpu.data_ptr<int64_t>();

    for (int64_t i = 0; i < num_groups; ++i) {
        const int64_t off = offs_ptr[i];
        const int64_t len = lens_ptr[i];
        if (off < 0 || len < 0 || off + len > total_quartets) {
            throw std::runtime_error("group offsets/lengths out of bounds");
        }
        if (len > target_len) {
            throw std::runtime_error("group length exceeds target_len");
        }
    }

    auto flat = compute_pattern_frequencies_cuda13(sequences.contiguous(), quartet_indices.contiguous());
    auto output = torch::zeros(
        {num_groups, target_len, 256},
        torch::TensorOptions().dtype(torch::kFloat32).device(sequences.device())
    );

    for (int64_t i = 0; i < num_groups; ++i) {
        const int64_t off = offs_ptr[i];
        const int64_t len = lens_ptr[i];
        if (len == 0) continue;
        output.select(0, i).narrow(0, 0, len).copy_(flat.narrow(0, off, len));
    }

    return output;
}

torch::Tensor compute_pattern_frequencies_cuda_grouped_uniform(
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
    if (quartet_indices.dim() != 3 || quartet_indices.size(2) != 4) {
        throw std::runtime_error("quartet_indices must have shape (n_blocks, n_quartets, 4)");
    }

    const int64_t n_blocks = quartet_indices.size(0);
    const int64_t n_quartets = quartet_indices.size(1);
    auto flat_idx = quartet_indices.contiguous().view({n_blocks * n_quartets, 4});
    auto flat_out = compute_pattern_frequencies_cuda13(sequences.contiguous(), flat_idx);
    return flat_out.view({n_blocks, n_quartets, 256});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_pattern_frequencies_cuda", &compute_pattern_frequencies_cuda, "Pattern frequency computation (CUDA)");
    m.def(
        "compute_pattern_frequencies_cuda_grouped",
        &compute_pattern_frequencies_cuda_grouped,
        "Pattern frequency computation for grouped quartets (CUDA)"
    );
    m.def(
        "compute_pattern_frequencies_cuda_grouped_uniform",
        &compute_pattern_frequencies_cuda_grouped_uniform,
        "Pattern frequency computation for uniform grouped quartets (CUDA)"
    );
}
