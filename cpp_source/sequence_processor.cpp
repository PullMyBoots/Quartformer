#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <sstream>
#include <memory>
#include <cctype>

namespace py = pybind11;

class SequenceProcessor {
private:
    std::unordered_map<std::string, std::string> sequences_;
    int sequence_length_;

    // 字符到数字的映射（用于模式编码，支持大小写）
    std::unordered_map<char, int> nucleotide_map_ = {
        {'A', 0}, {'a', 0}, {'C', 1}, {'c', 1}, {'G', 2}, {'g', 2}, {'T', 3}, {'t', 3}, {'-', 4}
    };

    // 模式到索引的映射（5^4 = 625种可能）
    std::vector<std::string> pattern_index_;

    void initialize_patterns() {
        pattern_index_.clear();
        std::string nucleotides = "ACGT-";
        for (char c1 : nucleotides) {
            for (char c2 : nucleotides) {
                for (char c3 : nucleotides) {
                    for (char c4 : nucleotides) {
                        pattern_index_.push_back(std::string({c1, c2, c3, c4}));
                    }
                }
            }
        }
    }

    int pattern_to_index(const std::string& pattern, bool include_gap) {
        if (pattern.length() != 4) return -1;

        int base = include_gap ? 5 : 4;  // 根据include_gap选择进制
        int index = 0;
        
        for (int i = 0; i < 4; ++i) {
            char c = pattern[i];
            int value;
            
            if (include_gap) {
                // 五进制：A=0, C=1, G=2, T=3, -=4
                auto it = nucleotide_map_.find(c);
                if (it == nucleotide_map_.end()) return -1;
                value = it->second;
            } else {
                // 四进制：A=0, C=1, G=2, T=3，不包含gap，支持大小写
                if (c == 'A' || c == 'a') value = 0;
                else if (c == 'C' || c == 'c') value = 1;
                else if (c == 'G' || c == 'g') value = 2;
                else if (c == 'T' || c == 't') value = 3;
                else return -1;  // 遇到gap或其他无效字符
            }
            
            index = index * base + value;
        }
        return index;
    }

public:
    SequenceProcessor() {
        initialize_patterns();
    }

    bool load_phy_file(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filepath << std::endl;
            return false;
        }

        sequences_.clear();
        std::string line;
        int line_num = 0;

        while (std::getline(file, line)) {
            line_num++;
            if (line.empty()) continue;

            if (line_num == 1) {
                // 解析第一行：物种数量和序列长度
                size_t space_pos = line.find(' ');
                if (space_pos == std::string::npos) {
                    std::cerr << "Invalid format in first line" << std::endl;
                    return false;
                }
                sequence_length_ = std::stoi(line.substr(space_pos + 1));
                continue;
            }

            // 解析序列行
            size_t space_pos = line.find(' ');
            if (space_pos == std::string::npos || space_pos == 0) {
                continue;
            }

            std::string species_name = line.substr(0, space_pos);
            std::string sequence = line.substr(space_pos + 1);

            // 移除空格
            sequence.erase(std::remove(sequence.begin(), sequence.end(), ' '), sequence.end());
            sequences_[species_name] = sequence;
        }

        file.close();
        return true;
    }

    py::array_t<float> compute_pattern_frequencies(
        const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& species_tuples,
        bool include_gap,
        bool exclude_constant_sites = false,
        bool drop_conserved_sites = false) {

        int num_tuples = species_tuples.size();
        int num_patterns = include_gap ? 625 : 256;  // 4^4 = 256, 5^4 = 625
        std::vector<int> kept_positions;
        kept_positions.reserve(static_cast<size_t>(sequence_length_));

        if (drop_conserved_sites) {
            std::vector<const std::string*> all_sequences;
            all_sequences.reserve(sequences_.size());
            for (const auto& pair : sequences_) {
                all_sequences.push_back(&pair.second);
            }

            for (int pos = 0; pos < sequence_length_; ++pos) {
                char first_char = (*(all_sequences[0]))[pos];
                bool all_same = true;
                for (size_t seq_idx = 1; seq_idx < all_sequences.size(); ++seq_idx) {
                    if ((*(all_sequences[seq_idx]))[pos] != first_char) {
                        all_same = false;
                        break;
                    }
                }
                if (!all_same) {
                    kept_positions.push_back(pos);
                }
            }
        } else {
            for (int pos = 0; pos < sequence_length_; ++pos) {
                kept_positions.push_back(pos);
            }
        }

        // 创建numpy数组
        auto result = py::array_t<float>({num_tuples, num_patterns});
        auto buffer = result.mutable_unchecked<2>();

        for (int tuple_idx = 0; tuple_idx < num_tuples; ++tuple_idx) {
            auto [sp1, sp2, sp3, sp4] = species_tuples[tuple_idx];

            // 检查物种是否存在
            if (sequences_.find(sp1) == sequences_.end() ||
                sequences_.find(sp2) == sequences_.end() ||
                sequences_.find(sp3) == sequences_.end() ||
                sequences_.find(sp4) == sequences_.end()) {
                std::cerr << "Species not found: " << sp1 << ", " << sp2 << ", " << sp3 << ", " << sp4 << std::endl;
                continue;
            }

            const auto& seq1 = sequences_[sp1];
            const auto& seq2 = sequences_[sp2];
            const auto& seq3 = sequences_[sp3];
            const auto& seq4 = sequences_[sp4];

            // 计算模式频率
            std::vector<int> pattern_counts(num_patterns, 0);
            int valid_sites = 0;

            for (int pos : kept_positions) {
                char chars[4] = {seq1[pos], seq2[pos], seq3[pos], seq4[pos]};
                std::string pattern(chars, chars + 4);

                // 当include_gap=False时，如果任何位置包含gap或模糊字符，则跳过该列
                if (!include_gap) {
                    bool has_invalid_char = false;
                    for (char c : pattern) {
                        if (c == '-' || c == 'N' || c == 'n' || c == '?' || c == 'X' || c == 'x') {
                            has_invalid_char = true;
                            break;
                        }
                    }
                    if (has_invalid_char) {
                        continue;  // 跳过包含gap或模糊字符的列
                    }
                }

                if (exclude_constant_sites &&
                    chars[0] == chars[1] &&
                    chars[1] == chars[2] &&
                    chars[2] == chars[3]) {
                    continue;
                }

                int index = pattern_to_index(pattern, include_gap);
                // std::cout << "Pos " << pos << ": " << pattern << " -> Index " << index << std::endl;
                if (index >= 0 && index < num_patterns) {
                    pattern_counts[index]++;
                    valid_sites++;
                    // std::cout << "  Valid, count=" << pattern_counts[index] << std::endl;
                }
            }

            // 归一化频率并写入结果数组
            if (valid_sites > 0) {
                for (int i = 0; i < num_patterns; ++i) {
                    buffer(tuple_idx, i) = static_cast<float>(pattern_counts[i]) / valid_sites;
                }
            } else {
                // 如果没有有效位点，填充0
                for (int i = 0; i < num_patterns; ++i) {
                    buffer(tuple_idx, i) = 0.0f;
                }
            }
        }

        return result;
    }

    int get_sequence_length() const {
        return sequence_length_;
    }

    std::vector<std::string> get_species_names() const {
        std::vector<std::string> names;
        for (const auto& pair : sequences_) {
            names.push_back(pair.first);
        }
        return names;
    }
};

namespace {
int8_t map_base_to_int8(char c) {
    static int8_t lookup[256];
    static bool lookup_initialized = false;
    if (!lookup_initialized) {
        std::fill(std::begin(lookup), std::end(lookup), static_cast<int8_t>(4));
        lookup[static_cast<unsigned char>('A')] = lookup[static_cast<unsigned char>('a')] = 0;
        lookup[static_cast<unsigned char>('C')] = lookup[static_cast<unsigned char>('c')] = 1;
        lookup[static_cast<unsigned char>('G')] = lookup[static_cast<unsigned char>('g')] = 2;
        lookup[static_cast<unsigned char>('T')] = lookup[static_cast<unsigned char>('t')] = 3;
        lookup[static_cast<unsigned char>('-')] = 4;
        lookup_initialized = true;
    }
    return lookup[static_cast<unsigned char>(c)];
}

long long parse_species_index(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && !std::isdigit(static_cast<unsigned char>(s[i]))) {
        i++;
    }
    if (i == s.size()) {
        throw std::runtime_error("Failed to parse species index from: " + s);
    }
    size_t j = i;
    while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) {
        j++;
    }
    return std::stoll(s.substr(i, j - i));
}
}  // namespace

// 新增：直接把 phy 转成 int8 张量（A,C,G,T,- => 0,1,2,3,4），返回 (tensor, species_names)
py::tuple load_phy_to_tensor(const std::string& phy_path, bool drop_conserved_sites = false) {
    std::ifstream file(phy_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + phy_path);
    }

    std::string header;
    if (!std::getline(file, header)) {
        throw std::runtime_error("Failed to read header");
    }
    std::istringstream iss(header);
    int n_species = 0;
    int sequence_length = 0;
    if (!(iss >> n_species >> sequence_length) || n_species <= 0 || sequence_length <= 0) {
        throw std::runtime_error("Invalid header format");
    }

    size_t total_size =
        static_cast<size_t>(n_species) * static_cast<size_t>(sequence_length);
    std::vector<int8_t> buffer;
    try {
        buffer.resize(total_size);
    } catch (const std::bad_alloc& e) {
        throw std::runtime_error("Failed to allocate memory for sequences: " + std::string(e.what()));
    }

    std::vector<std::string> species_names;
    species_names.reserve(static_cast<size_t>(n_species));

    const size_t CHUNK_SIZE = 1024 * 1024;
    std::vector<char> chunk_buffer(CHUNK_SIZE);

    int species_idx = 0;
    std::string species_name;
    int8_t* current_row = nullptr;
    int pos = 0;
    bool reading_name = true;

    while (species_idx < n_species && !file.eof()) {
        file.read(chunk_buffer.data(), static_cast<std::streamsize>(CHUNK_SIZE));
        std::streamsize bytes_read = file.gcount();

        for (std::streamsize i = 0; i < bytes_read; ++i) {
            char c = chunk_buffer[static_cast<size_t>(i)];

            if (c == '\n') {
                if (!species_name.empty() && !reading_name) {
                    species_names.push_back(species_name);
                    species_idx++;
                }
                species_name.clear();
                reading_name = true;
                current_row = nullptr;
                pos = 0;
                continue;
            }
            if (c == '\r') {
                continue;
            }

            if (reading_name) {
                if (c == ' ' || c == '\t') {
                    if (!species_name.empty()) {
                        reading_name = false;
                        current_row = buffer.data() + static_cast<size_t>(species_idx) * sequence_length;
                        pos = 0;
                    }
                } else {
                    species_name.push_back(c);
                }
            } else {
                if (c == ' ' || c == '\t') continue;
                if (!current_row) continue;
                if (pos >= sequence_length) continue;
                current_row[pos++] = map_base_to_int8(c);
            }
        }
    }

    if (species_idx < n_species && !species_name.empty() && !reading_name) {
        species_names.push_back(species_name);
        species_idx++;
    }

    if (species_idx != n_species) {
        throw std::runtime_error(
            "PHY file ended early: expected " + std::to_string(n_species) +
            " species, got " + std::to_string(species_idx));
    }

    // If species names look like numbered labels (e.g. Sp0..SpN-1), reorder rows so that
    // row i corresponds to the numeric suffix i. This makes it compatible with quartet
    // indices computed via int(name[2:]) style logic.
    {
        bool can_reorder = true;
        std::vector<uint8_t> used(static_cast<size_t>(n_species), 0);
        std::vector<std::string> ordered_names(static_cast<size_t>(n_species));
        std::vector<int8_t> ordered_buffer;
        ordered_buffer.resize(total_size);

        for (int r = 0; r < n_species; ++r) {
            long long idx_ll = -1;
            try {
                idx_ll = parse_species_index(species_names[static_cast<size_t>(r)]);
            } catch (...) {
                can_reorder = false;
                break;
            }
            if (idx_ll < 0 || idx_ll >= n_species) {
                can_reorder = false;
                break;
            }
            size_t idx = static_cast<size_t>(idx_ll);
            if (used[idx]) {
                can_reorder = false;
                break;
            }
            used[idx] = 1;
            ordered_names[idx] = species_names[static_cast<size_t>(r)];

            auto* dst = ordered_buffer.data() + idx * static_cast<size_t>(sequence_length);
            auto* src = buffer.data() + static_cast<size_t>(r) * static_cast<size_t>(sequence_length);
            std::copy(src, src + static_cast<size_t>(sequence_length), dst);
        }

        if (can_reorder) {
            species_names = std::move(ordered_names);
            buffer = std::move(ordered_buffer);
        }
    }

    if (drop_conserved_sites) {
        std::vector<int> keep_positions;
        keep_positions.reserve(static_cast<size_t>(sequence_length));

        for (int pos = 0; pos < sequence_length; ++pos) {
            int8_t first_val = buffer[static_cast<size_t>(pos)];
            bool all_same = true;
            for (int row = 1; row < n_species; ++row) {
                if (buffer[static_cast<size_t>(row) * static_cast<size_t>(sequence_length) + static_cast<size_t>(pos)] != first_val) {
                    all_same = false;
                    break;
                }
            }
            if (!all_same) {
                keep_positions.push_back(pos);
            }
        }

        const int filtered_sequence_length = static_cast<int>(keep_positions.size());
        std::vector<int8_t> filtered_buffer(
            static_cast<size_t>(n_species) * static_cast<size_t>(filtered_sequence_length)
        );

        for (int row = 0; row < n_species; ++row) {
            for (int new_pos = 0; new_pos < filtered_sequence_length; ++new_pos) {
                filtered_buffer[
                    static_cast<size_t>(row) * static_cast<size_t>(filtered_sequence_length) + static_cast<size_t>(new_pos)
                ] = buffer[
                    static_cast<size_t>(row) * static_cast<size_t>(sequence_length) +
                    static_cast<size_t>(keep_positions[static_cast<size_t>(new_pos)])
                ];
            }
        }

        buffer = std::move(filtered_buffer);
        sequence_length = filtered_sequence_length;
    }

    auto* raw_buffer = new std::vector<int8_t>(std::move(buffer));
    py::capsule capsule(raw_buffer, [](void* v) {
        delete reinterpret_cast<std::vector<int8_t>*>(v);
    });

    auto tensor = py::array_t<int8_t>(
        {n_species, sequence_length},
        {static_cast<py::ssize_t>(std::max(sequence_length, 1)), static_cast<py::ssize_t>(1)},
        raw_buffer->data(),
        capsule
    );

    return py::make_tuple(tensor, species_names);
}

// 新增：从 pkl（路径/bytes/对象）计算 quartet 的数字索引，返回 torch.int64 Tensor（或 numpy int64 array）
py::object compute_quartet_indices(const py::object& pkl_or_obj, bool as_torch = true) {
    py::object sample_quartets;

    if (py::isinstance<py::str>(pkl_or_obj)) {
        auto pickle = py::module_::import("pickle");
        auto builtins = py::module_::import("builtins");
        py::object f = builtins.attr("open")(pkl_or_obj, "rb");
        try {
            sample_quartets = pickle.attr("load")(f);
        } catch (...) {
            try {
                f.attr("close")();
            } catch (...) {
            }
            throw;
        }
        f.attr("close")();
    } else if (py::isinstance<py::bytes>(pkl_or_obj)) {
        auto pickle = py::module_::import("pickle");
        sample_quartets = pickle.attr("loads")(pkl_or_obj);
    } else {
        sample_quartets = pkl_or_obj;
    }

    py::sequence seq = sample_quartets.cast<py::sequence>();
    py::ssize_t n = seq.size();

    py::array_t<long long> out({n, static_cast<py::ssize_t>(4)});
    auto out_buf = out.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < n; ++i) {
        py::sequence quartet = seq[i].cast<py::sequence>();
        if (quartet.size() != 4) {
            throw std::runtime_error("Each quartet must have 4 species strings");
        }
        for (py::ssize_t j = 0; j < 4; ++j) {
            std::string name = py::str(quartet[j]);
            out_buf(i, j) = parse_species_index(name);
        }
    }

    if (!as_torch) {
        return out;
    }

    py::object torch = py::module_::import("torch");
    py::object tensor = torch.attr("from_numpy")(out).attr("to")(torch.attr("int64")).attr("clone")();
    return tensor;
}

// 新增：直接暴露的全局函数
py::array_t<float> process_phy_file(
    const std::string& phy_path,
    const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& species_tuples,
    bool include_gap = false,
    bool exclude_constant_sites = false,
    bool drop_conserved_sites = false
) {
    SequenceProcessor processor;
    if (!processor.load_phy_file(phy_path)) {
        throw std::runtime_error("Failed to load phy file: " + phy_path);
    }
    return processor.compute_pattern_frequencies(
        species_tuples, include_gap, exclude_constant_sites, drop_conserved_sites
    );
}

PYBIND11_MODULE(sequence_processor, m) {
    py::class_<SequenceProcessor>(m, "SequenceProcessor")
        .def(py::init<>())
        .def("load_phy_file", &SequenceProcessor::load_phy_file)
        .def(
            "compute_pattern_frequencies",
            &SequenceProcessor::compute_pattern_frequencies,
            py::arg("species_tuples"),
            py::arg("include_gap"),
            py::arg("exclude_constant_sites") = false,
            py::arg("drop_conserved_sites") = false,
            py::return_value_policy::copy
        )
        .def("get_sequence_length", &SequenceProcessor::get_sequence_length)
        .def("get_species_names", &SequenceProcessor::get_species_names);

    // 新增：直接暴露的全局函数
    m.def(
        "process_phy_file",
        &process_phy_file,
        py::arg("phy_path"),
        py::arg("species_tuples"),
        py::arg("include_gap") = false,
        py::arg("exclude_constant_sites") = false,
        py::arg("drop_conserved_sites") = false,
        "直接处理phy文件并返回模式频率张量（无需类封装）");
    m.def("load_phy_to_tensor", &load_phy_to_tensor, py::arg("phy_path"), py::arg("drop_conserved_sites") = false,
        "Load PHY file and return (sequences_tensor, species_names) tuple");
    m.def("compute_quartet_indices", &compute_quartet_indices, py::arg("pkl_or_obj"), py::arg("as_torch") = true,
        "Compute quartet indices from a pickle path/bytes/object; returns torch.int64 Tensor by default");
    m.doc() = "Sequence pattern frequency analysis module";
}
