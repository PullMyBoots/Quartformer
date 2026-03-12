#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct AggregateEntry {
    uint64_t key;
    double w0;
    double w1;
    double w2;
    int64_t count;
};

inline int normalize_shard_count(int num_shards) {
    if (num_shards <= 1) {
        return 1;
    }
    int out = 1;
    while (out < num_shards) {
        out <<= 1;
    }
    return out;
}

inline uint64_t encode_quartet_key(const int32_t* row) {
    return (static_cast<uint64_t>(static_cast<uint16_t>(row[0])) << 48) |
           (static_cast<uint64_t>(static_cast<uint16_t>(row[1])) << 32) |
           (static_cast<uint64_t>(static_cast<uint16_t>(row[2])) << 16) |
           static_cast<uint64_t>(static_cast<uint16_t>(row[3]));
}

std::vector<AggregateEntry> reduce_entries(std::vector<AggregateEntry>& entries, int num_shards) {
    if (entries.empty()) {
        return {};
    }

    num_shards = normalize_shard_count(num_shards);
    const uint64_t shard_mask = static_cast<uint64_t>(num_shards - 1);

    std::vector<size_t> shard_counts(num_shards, 0);
    for (const auto& entry : entries) {
        shard_counts[static_cast<size_t>(entry.key & shard_mask)]++;
    }

    std::vector<size_t> shard_offsets(num_shards + 1, 0);
    for (int shard = 0; shard < num_shards; ++shard) {
        shard_offsets[shard + 1] = shard_offsets[shard] + shard_counts[shard];
    }

    std::vector<AggregateEntry> partitioned(entries.size());
    std::vector<size_t> write_offsets = shard_offsets;
    for (auto& entry : entries) {
        const size_t shard = static_cast<size_t>(entry.key & shard_mask);
        partitioned[write_offsets[shard]++] = entry;
    }

    std::vector<std::vector<AggregateEntry>> shard_results(num_shards);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int shard = 0; shard < num_shards; ++shard) {
        const size_t begin = shard_offsets[shard];
        const size_t end = shard_offsets[shard + 1];
        if (begin == end) {
            continue;
        }

        auto shard_begin = partitioned.begin() + static_cast<std::ptrdiff_t>(begin);
        auto shard_end = partitioned.begin() + static_cast<std::ptrdiff_t>(end);
        std::sort(shard_begin, shard_end, [](const AggregateEntry& lhs, const AggregateEntry& rhs) {
            return lhs.key < rhs.key;
        });

        std::vector<AggregateEntry> reduced;
        reduced.reserve(end - begin);

        AggregateEntry acc = *shard_begin;
        for (auto it = shard_begin + 1; it != shard_end; ++it) {
            if (it->key == acc.key) {
                acc.w0 += it->w0;
                acc.w1 += it->w1;
                acc.w2 += it->w2;
                acc.count += it->count;
            } else {
                reduced.push_back(acc);
                acc = *it;
            }
        }
        reduced.push_back(acc);
        shard_results[shard] = std::move(reduced);
    }

    size_t total_reduced = 0;
    for (const auto& shard : shard_results) {
        total_reduced += shard.size();
    }

    std::vector<AggregateEntry> output;
    output.reserve(total_reduced);
    for (auto& shard : shard_results) {
        output.insert(output.end(), shard.begin(), shard.end());
    }
    if (output.size() > 1) {
        std::sort(output.begin(), output.end(), [](const AggregateEntry& lhs, const AggregateEntry& rhs) {
            return lhs.key < rhs.key;
        });
    }
    return output;
}

PyObject* build_result_tuple(const std::vector<AggregateEntry>& reduced) {
    npy_intp key_dims[1] = {static_cast<npy_intp>(reduced.size())};
    npy_intp sum_dims[2] = {static_cast<npy_intp>(reduced.size()), 3};

    PyObject* key_arr = PyArray_SimpleNew(1, key_dims, NPY_UINT64);
    PyObject* sum_arr = PyArray_SimpleNew(2, sum_dims, NPY_FLOAT64);
    PyObject* count_arr = PyArray_SimpleNew(1, key_dims, NPY_INT64);
    if (key_arr == nullptr || sum_arr == nullptr || count_arr == nullptr) {
        Py_XDECREF(key_arr);
        Py_XDECREF(sum_arr);
        Py_XDECREF(count_arr);
        return PyErr_NoMemory();
    }

    auto* key_ptr = static_cast<uint64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(key_arr)));
    auto* sum_ptr = static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(sum_arr)));
    auto* count_ptr = static_cast<int64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(count_arr)));

    for (size_t i = 0; i < reduced.size(); ++i) {
        key_ptr[i] = reduced[i].key;
        sum_ptr[i * 3 + 0] = reduced[i].w0;
        sum_ptr[i * 3 + 1] = reduced[i].w1;
        sum_ptr[i * 3 + 2] = reduced[i].w2;
        count_ptr[i] = reduced[i].count;
    }

    PyObject* out = PyTuple_Pack(3, key_arr, sum_arr, count_arr);
    Py_DECREF(key_arr);
    Py_DECREF(sum_arr);
    Py_DECREF(count_arr);
    return out;
}

PyObject* method_aggregate_quartets(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    PyObject* quartets_obj = nullptr;
    PyObject* weights_obj = nullptr;
    int num_shards = 64;
    static const char* kwlist[] = {"quartets", "weights", "num_shards", nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO|i", const_cast<char**>(kwlist), &quartets_obj, &weights_obj, &num_shards)) {
        return nullptr;
    }

    PyArrayObject* quartets_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(quartets_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* weights_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(weights_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY));
    if (quartets_arr == nullptr || weights_arr == nullptr) {
        Py_XDECREF(quartets_arr);
        Py_XDECREF(weights_arr);
        return nullptr;
    }

    if (PyArray_NDIM(quartets_arr) != 2 || PyArray_DIM(quartets_arr, 1) != 4) {
        Py_DECREF(quartets_arr);
        Py_DECREF(weights_arr);
        PyErr_SetString(PyExc_ValueError, "quartets must have shape (n, 4)");
        return nullptr;
    }
    if (PyArray_NDIM(weights_arr) != 2 || PyArray_DIM(weights_arr, 1) != 3) {
        Py_DECREF(quartets_arr);
        Py_DECREF(weights_arr);
        PyErr_SetString(PyExc_ValueError, "weights must have shape (n, 3)");
        return nullptr;
    }
    if (PyArray_DIM(quartets_arr, 0) != PyArray_DIM(weights_arr, 0)) {
        Py_DECREF(quartets_arr);
        Py_DECREF(weights_arr);
        PyErr_SetString(PyExc_ValueError, "quartets and weights must have the same first dimension");
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(quartets_arr, 0);
    const auto* quartets_ptr = static_cast<const int32_t*>(PyArray_DATA(quartets_arr));
    const auto* weights_ptr = static_cast<const double*>(PyArray_DATA(weights_arr));

    std::vector<AggregateEntry> entries(static_cast<size_t>(n));
    std::vector<AggregateEntry> reduced;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < n; ++i) {
        const int32_t* quartet_row = quartets_ptr + i * 4;
        const double* weight_row = weights_ptr + i * 3;
        entries[static_cast<size_t>(i)] = AggregateEntry{
            encode_quartet_key(quartet_row),
            weight_row[0],
            weight_row[1],
            weight_row[2],
            1,
        };
    }
    reduced = reduce_entries(entries, num_shards);
    Py_END_ALLOW_THREADS

    Py_DECREF(quartets_arr);
    Py_DECREF(weights_arr);
    return build_result_tuple(reduced);
}

PyObject* method_reduce_aggregates(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    PyObject* keys_obj = nullptr;
    PyObject* sums_obj = nullptr;
    PyObject* counts_obj = nullptr;
    int num_shards = 1;
    static const char* kwlist[] = {"keys", "weight_sums", "counts", "num_shards", nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOO|i", const_cast<char**>(kwlist),
            &keys_obj, &sums_obj, &counts_obj, &num_shards)) {
        return nullptr;
    }

    PyArrayObject* keys_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(keys_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* sums_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(sums_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* counts_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(counts_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    if (keys_arr == nullptr || sums_arr == nullptr || counts_arr == nullptr) {
        Py_XDECREF(keys_arr);
        Py_XDECREF(sums_arr);
        Py_XDECREF(counts_arr);
        return nullptr;
    }

    if (PyArray_NDIM(keys_arr) != 1) {
        Py_DECREF(keys_arr);
        Py_DECREF(sums_arr);
        Py_DECREF(counts_arr);
        PyErr_SetString(PyExc_ValueError, "keys must have shape (n,)");
        return nullptr;
    }
    if (PyArray_NDIM(sums_arr) != 2 || PyArray_DIM(sums_arr, 1) != 3) {
        Py_DECREF(keys_arr);
        Py_DECREF(sums_arr);
        Py_DECREF(counts_arr);
        PyErr_SetString(PyExc_ValueError, "weight_sums must have shape (n, 3)");
        return nullptr;
    }
    if (PyArray_NDIM(counts_arr) != 1) {
        Py_DECREF(keys_arr);
        Py_DECREF(sums_arr);
        Py_DECREF(counts_arr);
        PyErr_SetString(PyExc_ValueError, "counts must have shape (n,)");
        return nullptr;
    }
    if (PyArray_DIM(keys_arr, 0) != PyArray_DIM(sums_arr, 0) ||
        PyArray_DIM(keys_arr, 0) != PyArray_DIM(counts_arr, 0)) {
        Py_DECREF(keys_arr);
        Py_DECREF(sums_arr);
        Py_DECREF(counts_arr);
        PyErr_SetString(PyExc_ValueError, "keys, weight_sums, and counts must have the same first dimension");
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(keys_arr, 0);
    const auto* keys_ptr = static_cast<const uint64_t*>(PyArray_DATA(keys_arr));
    const auto* sums_ptr = static_cast<const double*>(PyArray_DATA(sums_arr));
    const auto* counts_ptr = static_cast<const int64_t*>(PyArray_DATA(counts_arr));

    std::vector<AggregateEntry> entries(static_cast<size_t>(n));
    std::vector<AggregateEntry> reduced;

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < n; ++i) {
        const double* sum_row = sums_ptr + i * 3;
        entries[static_cast<size_t>(i)] = AggregateEntry{
            keys_ptr[i],
            sum_row[0],
            sum_row[1],
            sum_row[2],
            counts_ptr[i],
        };
    }
    reduced = reduce_entries(entries, num_shards);
    Py_END_ALLOW_THREADS

    Py_DECREF(keys_arr);
    Py_DECREF(sums_arr);
    Py_DECREF(counts_arr);
    return build_result_tuple(reduced);
}

PyMethodDef module_methods[] = {
    {
        "aggregate_quartets",
        reinterpret_cast<PyCFunction>(method_aggregate_quartets),
        METH_VARARGS | METH_KEYWORDS,
        "Aggregate duplicate quartets into unique keys, summed weights, and counts.",
    },
    {
        "reduce_aggregates",
        reinterpret_cast<PyCFunction>(method_reduce_aggregates),
        METH_VARARGS | METH_KEYWORDS,
        "Reduce pre-aggregated quartet keys, summed weights, and counts.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "quartet_aggregate",
    "CPU quartet aggregation helpers.",
    -1,
    module_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit_quartet_aggregate(void) {
    import_array();
    return PyModule_Create(&module_def);
}
