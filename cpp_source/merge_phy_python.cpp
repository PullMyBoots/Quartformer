#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <filesystem>

namespace py = pybind11;
namespace fs = std::filesystem;

class PhyMerger {
private:
    std::map<std::string, std::string> species_sequences;
    std::vector<std::string> species_order;
    
public:
    
    // 读取PHY文件
    std::pair<std::map<std::string, std::string>, int> read_phy_file(const std::string& phy_file) {
        std::map<std::string, std::string> sequences;
        int length = 0;
        
        std::ifstream file(phy_file);
        if (!file.is_open()) {
            std::cerr << "无法打开PHY文件: " << phy_file << std::endl;
            return {sequences, length};
        }
        
        std::string line;
        std::getline(file, line); // 读取第一行（序列数量和长度）
        
        std::istringstream iss(line);
        int num_sequences;
        iss >> num_sequences >> length;
        
        std::string current_seq_name;
        std::string current_seq;
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            // 检查是否是序列名称行（包含空格且不以空格开头）
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos && space_pos > 0) {
                // 保存前一个序列
                if (!current_seq_name.empty()) {
                    sequences[current_seq_name] = current_seq;
                }
                
                // 开始新序列
                current_seq_name = line.substr(0, space_pos);
                current_seq = line.substr(space_pos + 1);
                // 移除序列名后的空格
                while (!current_seq.empty() && current_seq[0] == ' ') {
                    current_seq.erase(0, 1);
                }
                // 移除可能的换行符和回车符
                current_seq.erase(std::remove(current_seq.begin(), current_seq.end(), '\r'), current_seq.end());
                current_seq.erase(std::remove(current_seq.begin(), current_seq.end(), '\n'), current_seq.end());
            } else {
                // 继续当前序列（多行格式）
                current_seq += line;
            }
        }
        
        // 保存最后一个序列
        if (!current_seq_name.empty()) {
            sequences[current_seq_name] = current_seq;
        }
        
        file.close();
        return {sequences, length};
    }
    
    // 写入PHY文件
    void write_phy_file(const std::string& output_file, const std::map<std::string, std::string>& sequences) {
        if (sequences.empty()) return;
        
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "无法创建输出文件: " << output_file << std::endl;
            return;
        }
        
        // 计算序列长度
        int seq_length = sequences.begin()->second.length();
        int num_sequences = sequences.size();
        
        // 写入头部信息
        file << num_sequences << " " << seq_length << std::endl;
        
        // 写入序列
        for (const auto& pair : sequences) {
            file << pair.first << " " << pair.second << std::endl;
        }
        
        file.close();
    }
    
    // 合并PHY文件
    void merge_phy_files(const std::vector<std::string>& phy_files, const std::string& output_file) {
        // 初始化物种容器
        for (const auto& species : species_order) {
            species_sequences[species] = "";
        }
        
        // 遍历所有PHY文件
        for (const auto& phy_file : phy_files) {
            auto [sequences, length] = read_phy_file(phy_file);
            
            // 为每个物种添加序列
            for (const auto& species : species_order) {
                std::string species_seq = "";
                
                // 查找该物种在当前PHY文件中的序列
                for (const auto& seq_pair : sequences) {
                    const std::string& seq_name = seq_pair.first;
                    const std::string& sequence = seq_pair.second;
                    
                    // 解析序列名称：species_locus_individual
                    // 例如：0_0_0 -> Sp0, 14_0_1 -> Sp14
                    size_t first_underscore = seq_name.find('_');
                    if (first_underscore != std::string::npos) {
                        std::string species_id = seq_name.substr(0, first_underscore);
                        std::string expected_name = "Sp" + species_id;
                        
                        if (expected_name == species) {
                            species_seq = sequence;
                            break;
                        }
                    }
                }
                
                // 如果找到序列就添加，否则用gap填充
                if (!species_seq.empty()) {
                    species_sequences[species] += species_seq;
                } else {
                    species_sequences[species] += std::string(length, '-');
                }
            }
        }
        
        // 写入合并后的文件
        write_phy_file(output_file, species_sequences);
    }
    
    // 主处理函数
    int process_folder(const std::string& folder_path, const std::vector<std::string>& species_list, bool keep_individual_files = false) {
        // 1. 设置物种顺序
        species_order = species_list;
        
        if (species_order.empty()) {
            std::cerr << "物种列表为空" << std::endl;
            return 0;
        }
        
        // 2. 查找所有基因树的MSA文件
        std::vector<std::string> msa_files;
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("g_trees") == 0 && filename.find(".phy") != std::string::npos) {
                    msa_files.push_back(entry.path().string());
                }
            }
        }
        
        if (msa_files.empty()) {
            std::cerr << "文件夹中未找到MSA文件: " << folder_path << std::endl;
            return 0;
        }
        
        // 排序确保一致性
        std::sort(msa_files.begin(), msa_files.end());
        
        // 3. 合并所有MSA
        std::string supermatrix_file = folder_path + "/supermatrix.phy";
        merge_phy_files(msa_files, supermatrix_file);
        
        // 4. 根据用户选择决定是否清理基因树MSA文件
        if (!keep_individual_files) {
            for (const auto& msa_file : msa_files) {
                try {
                    fs::remove(msa_file);
                } catch (const std::exception& e) {
                    std::cerr << "删除文件失败: " << msa_file << " - " << e.what() << std::endl;
                }
            }
        }
        
        return msa_files.size();
    }
};

// Python绑定
PYBIND11_MODULE(merge_phy_cpp, m) {
    m.doc() = "高性能PHY文件合并模块";
    
    py::class_<PhyMerger>(m, "PhyMerger")
        .def(py::init<>())
        .def("process_folder", &PhyMerger::process_folder, "处理文件夹，合并PHY文件", 
             py::arg("folder_path"), py::arg("species_list"), py::arg("keep_individual_files") = false)
        .def("read_phy_file", &PhyMerger::read_phy_file, "读取PHY文件")
        .def("write_phy_file", &PhyMerger::write_phy_file, "写入PHY文件")
        .def("merge_phy_files", &PhyMerger::merge_phy_files, "合并PHY文件");
}
