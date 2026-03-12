#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct RawEntry {
    uint64_t key;
    float w0;
    float w1;
    float w2;
};

struct AggregateEntry {
    uint64_t key;
    double w0;
    double w1;
    double w2;
    int64_t count;
};

typedef struct {
    PyObject_HEAD
    std::vector<RawEntry>* entries;
    int num_shards;
} QuartetSinkObject;

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

std::vector<AggregateEntry> reduce_entries(
    const std::vector<RawEntry>& entries,
    int num_shards
) {
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

    std::vector<RawEntry> partitioned(entries.size());
    std::vector<size_t> write_offsets = shard_offsets;
    for (const auto& entry : entries) {
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
        std::sort(shard_begin, shard_end, [](const RawEntry& lhs, const RawEntry& rhs) {
            return lhs.key < rhs.key;
        });

        std::vector<AggregateEntry> reduced;
        reduced.reserve(end - begin);

        AggregateEntry acc{
            shard_begin->key,
            static_cast<double>(shard_begin->w0),
            static_cast<double>(shard_begin->w1),
            static_cast<double>(shard_begin->w2),
            1,
        };
        for (auto it = shard_begin + 1; it != shard_end; ++it) {
            if (it->key == acc.key) {
                acc.w0 += static_cast<double>(it->w0);
                acc.w1 += static_cast<double>(it->w1);
                acc.w2 += static_cast<double>(it->w2);
                acc.count += 1;
            } else {
                reduced.push_back(acc);
                acc = AggregateEntry{
                    it->key,
                    static_cast<double>(it->w0),
                    static_cast<double>(it->w1),
                    static_cast<double>(it->w2),
                    1,
                };
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

int QuartetSink_init(QuartetSinkObject* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {"reserve_hint", "num_shards", nullptr};
    Py_ssize_t reserve_hint = 0;
    int num_shards = 64;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "|ni", const_cast<char**>(kwlist), &reserve_hint, &num_shards)) {
        return -1;
    }

    self->num_shards = normalize_shard_count(num_shards);
    self->entries = new (std::nothrow) std::vector<RawEntry>();
    if (self->entries == nullptr) {
        PyErr_NoMemory();
        return -1;
    }
    if (reserve_hint > 0) {
        self->entries->reserve(static_cast<size_t>(reserve_hint));
    }
    return 0;
}

void QuartetSink_dealloc(QuartetSinkObject* self) {
    delete self->entries;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* QuartetSink_append_quartets(QuartetSinkObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* quartets_obj = nullptr;
    PyObject* weights_obj = nullptr;
    static const char* kwlist[] = {"quartets", "weights", nullptr};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO", const_cast<char**>(kwlist), &quartets_obj, &weights_obj)) {
        return nullptr;
    }

    PyArrayObject* quartets_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(quartets_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* weights_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(weights_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
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
    const auto* weights_ptr = static_cast<const float*>(PyArray_DATA(weights_arr));

    try {
        self->entries->reserve(self->entries->size() + static_cast<size_t>(n));
    } catch (const std::bad_alloc&) {
        Py_DECREF(quartets_arr);
        Py_DECREF(weights_arr);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < n; ++i) {
        const int32_t* quartet_row = quartets_ptr + i * 4;
        const float* weight_row = weights_ptr + i * 3;
        self->entries->push_back(RawEntry{
            encode_quartet_key(quartet_row),
            weight_row[0],
            weight_row[1],
            weight_row[2],
        });
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(quartets_arr);
    Py_DECREF(weights_arr);
    Py_RETURN_NONE;
}

PyObject* QuartetSink_finalize(QuartetSinkObject* self, PyObject* /*args*/) {
    std::vector<AggregateEntry> reduced;
    Py_BEGIN_ALLOW_THREADS
    reduced = reduce_entries(*self->entries, self->num_shards);
    self->entries->clear();
    self->entries->shrink_to_fit();
    Py_END_ALLOW_THREADS
    return build_result_tuple(reduced);
}

PyObject* QuartetSink_num_entries(QuartetSinkObject* self, void* /*closure*/) {
    return PyLong_FromSize_t(self->entries->size());
}

PyMethodDef QuartetSink_methods[] = {
    {
        "append_quartets",
        reinterpret_cast<PyCFunction>(QuartetSink_append_quartets),
        METH_VARARGS | METH_KEYWORDS,
        "Append raw quartet rows and weights to the sink.",
    },
    {
        "finalize",
        reinterpret_cast<PyCFunction>(QuartetSink_finalize),
        METH_NOARGS,
        "Aggregate all appended rows and return (keys, sums, counts).",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef QuartetSink_getset[] = {
    {
        const_cast<char*>("num_entries"),
        reinterpret_cast<getter>(QuartetSink_num_entries),
        nullptr,
        const_cast<char*>("Number of raw quartet rows stored in the sink."),
        nullptr,
    },
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyTypeObject QuartetSinkType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

PyMethodDef module_methods[] = {
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "quartet_sink",
    "Append-only quartet sink with C++ final aggregation.",
    -1,
    module_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit_quartet_sink(void) {
    import_array();

    QuartetSinkType.tp_name = "quartet_sink.QuartetSink";
    QuartetSinkType.tp_basicsize = sizeof(QuartetSinkObject);
    QuartetSinkType.tp_flags = Py_TPFLAGS_DEFAULT;
    QuartetSinkType.tp_doc = "Append-only quartet sink.";
    QuartetSinkType.tp_new = PyType_GenericNew;
    QuartetSinkType.tp_init = reinterpret_cast<initproc>(QuartetSink_init);
    QuartetSinkType.tp_dealloc = reinterpret_cast<destructor>(QuartetSink_dealloc);
    QuartetSinkType.tp_methods = QuartetSink_methods;
    QuartetSinkType.tp_getset = QuartetSink_getset;

    if (PyType_Ready(&QuartetSinkType) < 0) {
        return nullptr;
    }

    PyObject* module = PyModule_Create(&module_def);
    if (module == nullptr) {
        return nullptr;
    }

    Py_INCREF(&QuartetSinkType);
    if (PyModule_AddObject(module, "QuartetSink", reinterpret_cast<PyObject*>(&QuartetSinkType)) < 0) {
        Py_DECREF(&QuartetSinkType);
        Py_DECREF(module);
        return nullptr;
    }

    return module;
}
