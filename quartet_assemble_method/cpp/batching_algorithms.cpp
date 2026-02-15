#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_set>
#include <array>
#include <cstdint>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Utility Functions
// ============================================================================

inline int64_t binomial(int n, int k) {
    if (k > n || k < 0) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;

    int64_t result = 1;
    for (int i = 0; i < k; i++) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

// ============================================================================
// Algorithm V4: Pair-Balanced Block-Design (Original)
// ============================================================================

std::vector<std::vector<std::array<int, 4>>>
pair_balanced_block_design_v4(int num_species, double k_param, int batch_size,
                               int seed, double species_weight) {
    std::mt19937 rng(seed);
    int n = num_species;
    int64_t target_total = std::max(1L, static_cast<int64_t>(std::pow(num_species, k_param)));

    int block_size = 4;
    for (int s = 4; s <= n; s++) {
        if (binomial(s, 4) <= batch_size) {
            block_size = s;
        } else {
            break;
        }
    }

    int64_t quartets_per_block = binomial(block_size, 4);
    int num_batches = std::max(1, (int)((target_total + quartets_per_block - 1) / quartets_per_block));

    std::vector<std::vector<int>> pair_counts(n, std::vector<int>(n, 0));
    std::vector<int> species_counts(n, 0);
    std::vector<std::vector<std::array<int, 4>>> batches;
    batches.reserve(num_batches);

    for (int b_idx = 0; b_idx < num_batches; b_idx++) {
        int min_val = INT32_MAX;
        std::vector<std::pair<int, int>> best_pairs;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int val = pair_counts[i][j];
                if (val < min_val) {
                    min_val = val;
                    best_pairs.clear();
                    best_pairs.push_back({i, j});
                } else if (val == min_val) {
                    best_pairs.push_back({i, j});
                }
            }
        }

        std::uniform_int_distribution<int> dist(0, best_pairs.size() - 1);
        auto [p1, p2] = best_pairs[dist(rng)];

        std::vector<int> block = {p1, p2};
        std::unordered_set<int> block_set = {p1, p2};

        while (static_cast<int>(block.size()) < block_size) {
            std::vector<int> best_cands;
            double best_score = 1e100;

            for (int cand = 0; cand < n; cand++) {
                if (block_set.count(cand)) continue;

                double pair_score = 0;
                for (int u : block) {
                    pair_score += pair_counts[cand][u];
                }

                double freq_score = species_weight * species_counts[cand];
                double total_score = pair_score + freq_score;

                if (total_score < best_score - 1e-9) {
                    best_score = total_score;
                    best_cands.clear();
                    best_cands.push_back(cand);
                } else if (std::abs(total_score - best_score) < 1e-9) {
                    best_cands.push_back(cand);
                }
            }

            if (best_cands.empty()) break;

            std::uniform_int_distribution<int> cand_dist(0, best_cands.size() - 1);
            int chosen = best_cands[cand_dist(rng)];
            block.push_back(chosen);
            block_set.insert(chosen);
        }

        std::sort(block.begin(), block.end());
        std::vector<std::array<int, 4>> new_batch;
        int bs = block.size();

        for (int i = 0; i < bs - 3; i++) {
            for (int j = i + 1; j < bs - 2; j++) {
                for (int k = j + 1; k < bs - 1; k++) {
                    for (int l = k + 1; l < bs; l++) {
                        new_batch.push_back({block[i], block[j], block[k], block[l]});
                    }
                }
            }
        }

        batches.push_back(new_batch);

        for (int u : block) species_counts[u]++;
        for (size_t i = 0; i < block.size(); i++) {
            for (size_t j = i + 1; j < block.size(); j++) {
                int u = block[i], v = block[j];
                pair_counts[u][v]++;
                pair_counts[v][u]++;
            }
        }
    }

    return batches;
}

// ============================================================================
// Algorithm V6: Connectivity-First Hybrid Batching
// ============================================================================

std::vector<std::vector<std::array<int, 4>>>
connectivity_first_block_design_v6(int num_species, double k_param, int batch_size,
                                    int seed, double species_weight) {
    std::mt19937 rng(seed);
    int n = num_species;
    int64_t target_total = std::max(1L, static_cast<int64_t>(std::pow(num_species, k_param)));

    // 1. Determine Block Size
    int block_size = 4;
    for (int s = 4; s <= n; s++) {
        if (binomial(s, 4) <= batch_size) {
            block_size = s;
        } else {
            break;
        }
    }

    // Data Structures
    std::vector<std::vector<bool>> is_covered(n, std::vector<bool>(n, false));
    std::vector<std::vector<int>> pair_counts(n, std::vector<int>(n, 0));
    std::vector<int> species_counts(n, 0);
    
    std::vector<std::vector<std::array<int, 4>>> batches;
    
    int64_t current_quartet_count = 0;
    int covered_pairs_count = 0;
    int total_pairs = n * (n - 1) / 2;

    // =========================================================
    // PHASE 1: CONNECTIVITY (The Patching Logic)
    // =========================================================
    
    while (covered_pairs_count < total_pairs) {
        // A. Find a seed pair (u, v) that is NOT covered
        int u = -1, v = -1;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (!is_covered[i][j]) {
                    u = i; v = j;
                    goto found_seed;
                }
            }
        }
        found_seed:;
        
        if (u == -1) break; // All pairs covered

        // B. Initialize Block with seed
        std::vector<int> block = {u, v};
        std::vector<bool> in_block(n, false);
        in_block[u] = true; in_block[v] = true;

        // C. Greedily Grow Block
        while (static_cast<int>(block.size()) < block_size) {
            int best_gain = -1;
            
            // Collect candidates
            std::vector<int> candidates;
            
            for (int c = 0; c < n; c++) {
                if (in_block[c]) continue;
                
                int gain = 0;
                for (int member : block) {
                    int p1 = std::min(c, member);
                    int p2 = std::max(c, member);
                    if (!is_covered[p1][p2]) {
                        gain++;
                    }
                }
                
                if (gain > best_gain) {
                    best_gain = gain;
                    candidates.clear();
                    candidates.push_back(c);
                } else if (gain == best_gain) {
                    candidates.push_back(c);
                }
            }
            
            if (candidates.empty()) break;
            
            std::uniform_int_distribution<int> c_dist(0, candidates.size() - 1);
            int chosen = candidates[c_dist(rng)];
            
            block.push_back(chosen);
            in_block[chosen] = true;
        }

        // D. Commit Block
        std::sort(block.begin(), block.end());
        
        // Generate Quartets
        std::vector<std::array<int, 4>> new_batch;
        int bs = block.size();
        for (int i = 0; i < bs - 3; i++) {
            for (int j = i + 1; j < bs - 2; j++) {
                for (int k = j + 1; k < bs - 1; k++) {
                    for (int l = k + 1; l < bs; l++) {
                        new_batch.push_back({block[i], block[j], block[k], block[l]});
                    }
                }
            }
        }
        
        if (!new_batch.empty()) {
            batches.push_back(new_batch);
            current_quartet_count += new_batch.size();
        }

        // Update Counts and Covered Matrix
        for (int node : block) species_counts[node]++;
        for (size_t i = 0; i < block.size(); i++) {
            for (size_t j = i + 1; j < block.size(); j++) {
                int p1 = block[i];
                int p2 = block[j];
                pair_counts[p1][p2]++;
                pair_counts[p2][p1]++;
                
                if (p1 > p2) std::swap(p1, p2);
                if (!is_covered[p1][p2]) {
                    is_covered[p1][p2] = true;
                    covered_pairs_count++;
                }
            }
        }
    }

    // =========================================================
    // PHASE 2: BALANCING (Fill up to k_param if needed)
    // =========================================================
    
    while (current_quartet_count < target_total) {
        // V4 Logic: Find least covered pair
        int min_val = INT32_MAX;
        std::vector<std::pair<int, int>> best_pairs;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int val = pair_counts[i][j];
                if (val < min_val) {
                    min_val = val;
                    best_pairs.clear();
                    best_pairs.push_back({i, j});
                } else if (val == min_val) {
                    best_pairs.push_back({i, j});
                }
            }
        }

        std::uniform_int_distribution<int> dist(0, best_pairs.size() - 1);
        auto [p1, p2] = best_pairs[dist(rng)];

        std::vector<int> block = {p1, p2};
        std::unordered_set<int> block_set = {p1, p2};

        while (static_cast<int>(block.size()) < block_size) {
            std::vector<int> best_cands;
            double best_score = 1e100;

            for (int cand = 0; cand < n; cand++) {
                if (block_set.count(cand)) continue;

                double pair_score = 0;
                for (int u : block) {
                    pair_score += pair_counts[cand][u];
                }

                double freq_score = species_weight * species_counts[cand];
                double total_score = pair_score + freq_score;

                if (total_score < best_score - 1e-9) {
                    best_score = total_score;
                    best_cands.clear();
                    best_cands.push_back(cand);
                } else if (std::abs(total_score - best_score) < 1e-9) {
                    best_cands.push_back(cand);
                }
            }

            if (best_cands.empty()) break;

            std::uniform_int_distribution<int> cand_dist(0, best_cands.size() - 1);
            int chosen = best_cands[cand_dist(rng)];
            block.push_back(chosen);
            block_set.insert(chosen);
        }

        // Commit Block (V4 style)
        std::sort(block.begin(), block.end());
        std::vector<std::array<int, 4>> new_batch;
        int bs = block.size();
        for (int i = 0; i < bs - 3; i++) {
            for (int j = i + 1; j < bs - 2; j++) {
                for (int k = j + 1; k < bs - 1; k++) {
                    for (int l = k + 1; l < bs; l++) {
                        new_batch.push_back({block[i], block[j], block[k], block[l]});
                    }
                }
            }
        }

        if (!new_batch.empty()) {
            batches.push_back(new_batch);
            current_quartet_count += new_batch.size();
        }

        for (int u : block) species_counts[u]++;
        for (size_t i = 0; i < block.size(); i++) {
            for (size_t j = i + 1; j < block.size(); j++) {
                int u = block[i], v = block[j];
                pair_counts[u][v]++;
                pair_counts[v][u]++;
            }
        }
    }

    return batches;
}

// ============================================================================
// Algorithm V5: Species-sized Batches (Existing)
// ============================================================================
std::vector<std::vector<int>>
pair_balanced_block_design_v5(int num_species, double k_param, int batch_size,
                              int seed, double species_weight) {
    std::mt19937 rng(seed);
    int n = num_species;
    int block_size = std::max(4, batch_size);
    block_size = std::min(block_size, n);
    int64_t target_total = std::max(1L, static_cast<int64_t>(std::pow(num_species, k_param)));
    int64_t quartets_per_block = binomial(block_size, 4);

    if (quartets_per_block <= 0) return {};

    int num_batches = std::max(1, (int)((target_total + quartets_per_block - 1) / quartets_per_block));

    std::vector<std::vector<int>> pair_counts(n, std::vector<int>(n, 0));
    std::vector<int> species_counts(n, 0);
    std::vector<std::vector<int>> batches;
    batches.reserve(num_batches);

    for (int b_idx = 0; b_idx < num_batches; b_idx++) {
        int min_val = INT32_MAX;
        std::vector<std::pair<int, int>> best_pairs;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int val = pair_counts[i][j];
                if (val < min_val) {
                    min_val = val;
                    best_pairs.clear();
                    best_pairs.push_back({i, j});
                } else if (val == min_val) {
                    best_pairs.push_back({i, j});
                }
            }
        }

        if (best_pairs.empty()) break;

        std::uniform_int_distribution<int> dist(0, best_pairs.size() - 1);
        auto [p1, p2] = best_pairs[dist(rng)];

        std::vector<int> block = {p1, p2};
        std::unordered_set<int> block_set = {p1, p2};

        while (static_cast<int>(block.size()) < block_size) {
            std::vector<int> best_cands;
            double best_score = 1e100;

            for (int cand = 0; cand < n; cand++) {
                if (block_set.count(cand)) continue;

                double pair_score = 0;
                for (int u : block) {
                    pair_score += pair_counts[cand][u];
                }

                double freq_score = species_weight * species_counts[cand];
                double total_score = pair_score + freq_score;

                if (total_score < best_score - 1e-9) {
                    best_score = total_score;
                    best_cands.clear();
                    best_cands.push_back(cand);
                } else if (std::abs(total_score - best_score) < 1e-9) {
                    best_cands.push_back(cand);
                }
            }

            if (best_cands.empty()) break;

            std::uniform_int_distribution<int> cand_dist(0, best_cands.size() - 1);
            int chosen = best_cands[cand_dist(rng)];
            block.push_back(chosen);
            block_set.insert(chosen);
        }

        std::sort(block.begin(), block.end());
        batches.push_back(block);

        for (int u : block) species_counts[u]++;
        for (size_t i = 0; i < block.size(); i++) {
            for (size_t j = i + 1; j < block.size(); j++) {
                int u = block[i], v = block[j];
                pair_counts[u][v]++;
                pair_counts[v][u]++;
            }
        }
    }

    return batches;
}

// ============================================================================
// Python Interface
// ============================================================================

static PyObject* method_pair_balanced_block_design_v4(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_species;
    double k_param;
    int batch_size;
    int seed = 42;
    double species_weight = 2.0;
    int threads = 1;

    static char* kwlist[] = {
        (char*)"num_species",
        (char*)"k_param",
        (char*)"batch_size",
        (char*)"seed",
        (char*)"species_weight",
        (char*)"threads",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "idi|idi", kwlist,
                                     &num_species, &k_param, &batch_size,
                                     &seed, &species_weight, &threads)) {
        return NULL;
    }

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    auto batches = pair_balanced_block_design_v4(num_species, k_param, batch_size,
                                                  seed, species_weight);

    PyObject* py_batches = PyList_New(batches.size());
    for (size_t i = 0; i < batches.size(); i++) {
        PyObject* py_batch = PyList_New(batches[i].size());
        for (size_t j = 0; j < batches[i].size(); j++) {
            PyObject* py_quartet = PyTuple_Pack(4,
                PyLong_FromLong(batches[i][j][0]),
                PyLong_FromLong(batches[i][j][1]),
                PyLong_FromLong(batches[i][j][2]),
                PyLong_FromLong(batches[i][j][3])
            );
            PyList_SET_ITEM(py_batch, j, py_quartet);
        }
        PyList_SET_ITEM(py_batches, i, py_batch);
    }

    return py_batches;
}

static PyObject* method_connectivity_first_block_design_v6(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_species;
    double k_param;
    int batch_size;
    int seed = 42;
    double species_weight = 2.0;
    int threads = 1;

    static char* kwlist[] = {
        (char*)"num_species",
        (char*)"k_param",
        (char*)"batch_size",
        (char*)"seed",
        (char*)"species_weight",
        (char*)"threads",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "idi|idi", kwlist,
                                     &num_species, &k_param, &batch_size,
                                     &seed, &species_weight, &threads)) {
        return NULL;
    }

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    auto batches = connectivity_first_block_design_v6(num_species, k_param, batch_size,
                                                  seed, species_weight);

    PyObject* py_batches = PyList_New(batches.size());
    for (size_t i = 0; i < batches.size(); i++) {
        PyObject* py_batch = PyList_New(batches[i].size());
        for (size_t j = 0; j < batches[i].size(); j++) {
            PyObject* py_quartet = PyTuple_Pack(4,
                PyLong_FromLong(batches[i][j][0]),
                PyLong_FromLong(batches[i][j][1]),
                PyLong_FromLong(batches[i][j][2]),
                PyLong_FromLong(batches[i][j][3])
            );
            PyList_SET_ITEM(py_batch, j, py_quartet);
        }
        PyList_SET_ITEM(py_batches, i, py_batch);
    }

    return py_batches;
}

static PyObject* method_pair_balanced_block_design_v5(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_species;
    double k_param;
    int batch_size;
    int seed = 42;
    double species_weight = 2.0;
    int threads = 1;

    static char* kwlist[] = {
        (char*)"num_species",
        (char*)"k_param",
        (char*)"batch_size",
        (char*)"seed",
        (char*)"species_weight",
        (char*)"threads",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "idi|idi", kwlist,
                                     &num_species, &k_param, &batch_size,
                                     &seed, &species_weight, &threads)) {
        return NULL;
    }

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    auto batches = pair_balanced_block_design_v5(num_species, k_param, batch_size,
                                                  seed, species_weight);

    PyObject* py_batches = PyList_New(batches.size());
    for (size_t i = 0; i < batches.size(); i++) {
        PyObject* py_block = PyList_New(batches[i].size());
        for (size_t j = 0; j < batches[i].size(); j++) {
            PyList_SET_ITEM(py_block, j, PyLong_FromLong(batches[i][j]));
        }
        PyList_SET_ITEM(py_batches, i, py_block);
    }

    return py_batches;
}

static PyMethodDef BatchingMethods[] = {
    {"pair_balanced_block_design_v4",
     (PyCFunction)method_pair_balanced_block_design_v4,
     METH_VARARGS | METH_KEYWORDS,
     "Pair-Balanced Block-Design Batching V4 algorithm"},
    {"connectivity_first_block_design_v6",
     (PyCFunction)method_connectivity_first_block_design_v6,
     METH_VARARGS | METH_KEYWORDS,
     "V6 Algorithm: Guarantees full pairwise connectivity before meeting count target"},
    {"pair_balanced_block_design_v5",
     (PyCFunction)method_pair_balanced_block_design_v5,
     METH_VARARGS | METH_KEYWORDS,
     "Pair-Balanced Block-Design Batching V5 algorithm (returns species blocks, not quartets)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef batchingmodule = {
    PyModuleDef_HEAD_INIT,
    "batching_algorithms",
    "High-performance quartet batching algorithms",
    -1,
    BatchingMethods
};

PyMODINIT_FUNC PyInit_batching_algorithms(void) {
    return PyModule_Create(&batchingmodule);
}
