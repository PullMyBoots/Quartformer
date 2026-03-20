#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_set>
#include <array>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Utility Functions
// ============================================================================

// Compute binomial coefficient C(n, k)
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

// Generate all C(n, 4) quartets from n species
std::vector<std::array<int, 4>> generate_all_quartets(int n) {
    std::vector<std::array<int, 4>> quartets;
    int64_t total = binomial(n, 4);
    quartets.reserve(total);

    for (int a = 0; a < n - 3; a++) {
        for (int b = a + 1; b < n - 2; b++) {
            for (int c = b + 1; c < n - 1; c++) {
                for (int d = c + 1; d < n; d++) {
                    quartets.push_back({a, b, c, d});
                }
            }
        }
    }
    return quartets;
}

// ============================================================================
// Algorithm 1: Pair-Balanced Block-Design Batching V4
// ============================================================================

std::vector<std::vector<std::array<int, 4>>>
pair_balanced_block_design_v4(int num_species, double k_param, int batch_size,
                               int seed, double species_weight) {
    std::mt19937 rng(seed);
    int n = num_species;
    int64_t target_total = std::max(1L, static_cast<int64_t>(std::pow(num_species, k_param)));

    // 1. Choose Block Size
    int block_size = 4;
    for (int s = 4; s <= n; s++) {
        if (binomial(s, 4) <= batch_size) {
            block_size = s;
        } else {
            break;
        }
    }

    int64_t quartets_per_block = binomial(block_size, 4);
    int num_batches = std::max(1L, (target_total + quartets_per_block - 1) / quartets_per_block);

    // Data structures
    std::vector<std::vector<int>> pair_counts(n, std::vector<int>(n, 0));
    std::vector<int> species_counts(n, 0);
    std::vector<std::vector<std::array<int, 4>>> batches;
    batches.reserve(num_batches);

    for (int b_idx = 0; b_idx < num_batches; b_idx++) {
        // A. Pick Seed Pair: Strictly the pair with minimal coverage
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

        // B. Grow Block
        while (static_cast<int>(block.size()) < block_size) {
            std::vector<int> best_cands;
            double best_score = 1e100;

            for (int cand = 0; cand < n; cand++) {
                if (block_set.count(cand)) continue;

                // Score: Minimize the increase in global variance
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

        // C. Generate all quartets from this block
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

        // D. Update Counts
        for (int u : block) {
            species_counts[u]++;
        }
        for (size_t i = 0; i < block.size(); i++) {
            int u = block[i];
            for (size_t j = i + 1; j < block.size(); j++) {
                int v = block[j];
                pair_counts[u][v]++;
                pair_counts[v][u]++;
            }
        }
    }

    return batches;
}

// ============================================================================
// Algorithm 1: Pair-Balanced Block-Design Batching V5 (species-sized batches)
// ============================================================================

std::vector<std::vector<int>>
pair_balanced_block_design_v5(int num_species, double k_param, int batch_size,
                              int seed, double species_weight) {
    std::mt19937 rng(seed);
    int n = num_species;
    // In V5, batch_size is treated as the desired species set size per batch.
    int block_size = std::max(4, batch_size);
    block_size = std::min(block_size, n);
    int64_t target_total = std::max(1L, static_cast<int64_t>(std::pow(num_species, k_param)));
    int64_t quartets_per_block = binomial(block_size, 4);

    if (quartets_per_block <= 0) {
        return {};
    }

    int num_batches = std::max(1L, (target_total + quartets_per_block - 1) / quartets_per_block);

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

        if (best_pairs.empty()) {
            break;
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
        batches.push_back(block);

        for (int u : block) {
            species_counts[u]++;
        }
        for (size_t i = 0; i < block.size(); i++) {
            int u = block[i];
            for (size_t j = i + 1; j < block.size(); j++) {
                int v = block[j];
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

    // Convert to Python list of lists of tuples
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
