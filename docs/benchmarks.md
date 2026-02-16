# Performance Benchmarks

This document reports runtime benchmarks for QuartFormer compared with ASTER and FastTree under a fixed local environment.

## Test Environment

### Hardware Configuration
- **CPU**: Intel Core i9-14900K (16 cores, 32 threads)
- **RAM**: 128 GB DDR5
- **GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **OS**: Linux (WSL2)

### Software Configuration
- **Python**: 3.10.18
- **CUDA**: 12.6
- **PyTorch**: Enabled with CUDA support
- **Compiler**: GCC with C++17 support

---

## Methodology

### Tools and Commands

**QuartFormer (QF)**
```bash
python run_qf.py --phy input.phy --out output.nwk --task-type heterogeneous --run-mode regular
```

**ASTER (caster-site)**
```bash
caster-site -i input.phy -o output.nwk -t 32
```

**FastTree / FastTreeMP**
```bash
# Requires PHYLIP to FASTA conversion first
# For FastTreeMP (multi-threaded), set OMP_NUM_THREADS:
OMP_NUM_THREADS=32 FastTreeMP -nt -gtr input.fasta > output.nwk
# or single-threaded:
FastTree -nt -gtr input.fasta > output.nwk
```

### Dataset Specifications
- **Model**: GTR (General Time Reversible)
- **Sequence lengths**: 100k, 1M, 10M sites
- **Taxon counts**: 24, 48, 96, 192 species
- **Replicates**: 1 per configuration (single-run measurement)
- **Thread count**: 32 (for multi-threaded tools)

### Performance Metrics
- **Metric**: Wall-clock time (seconds)
- **Measurement**: Unix `/usr/bin/time -v` for precise timing
- **Speedup calculation**: (competitor_time / qf_time)

## Scope and Interpretation

- Results are specific to this hardware/software stack and command settings.
- No variance estimates are provided because each configuration was measured once.
- `N/A` indicates the corresponding method did not finish within the practical run window used in our experiments.
- These benchmarks focus on runtime only; accuracy is reported separately in `docs/accuracy.md`.

---

## Benchmark Results

### 100k Sequence Length

| Taxa | QF (s) | ASTER (s) | FastTree (s) | QF vs ASTER | QF vs FastTree |
|------|---------|-----------|--------------|--------------|----------------|
| 24   | 0.969   | 2.950     | 27.970       | **3.04x**    | **28.87x**     |
| 48   | 1.451   | 8.660     | 51.620       | **5.97x**    | **35.58x**     |
| 96   | 5.380   | 60.080    | 130.550      | **11.17x**   | **24.27x**     |
| 192  | 49.569  | 91.850    | 218.480      | **1.85x**    | **4.41x**      |

**Observation (100k):**
- In this setup, QuartFormer is faster than ASTER and FastTree for all listed taxon counts.

---

### 1M Sequence Length

| Taxa | QF (s) | ASTER (s) | FastTree (s) | QF vs ASTER | QF vs FastTree |
|------|---------|-----------|--------------|--------------|----------------|
| 24   | 0.537   | 5.120     | 332.850      | **9.54x**    | **619.81x**    |
| 48   | 1.762   | 66.160    | 896.780      | **37.55x**   | **509.02x**    |
| 96   | 6.793   | 206.170   | 1946.760     | **30.35x**   | **286.52x**    |
| 192  | 53.284  | 707.800   | 3840.000     | **13.29x**   | **72.09x**     |

**Observation (1M):**
- In this setup, QuartFormer remains faster than ASTER and FastTree for all listed taxon counts.

---

### 10M Sequence Length

| Taxa | QF (s) | ASTER (s) | FastTree (s) | QF vs ASTER | QF vs FastTree |
|------|---------|-----------|--------------|--------------|----------------|
| 24   | 1.454   | 137.320   | 8946.000     | **94.41x**   | **6151.99x**   |
| 48   | 6.394   | 159.520   | 17223.000    | **24.95x**   | **2693.90x**   |
| 96   | 23.012  | 6387.000  | N/A          | **277.61x**  | N/A            |
| 192  | 165.497 | N/A       | N/A          | N/A          | N/A            |

**Observation (10M):**
- Runtime gaps increase in this dataset group, and some baseline runs are reported as `N/A`.


---

## Summary

Under this benchmark configuration, QuartFormer shows lower wall-clock runtime than ASTER and FastTree across all completed comparisons in the tables above.  
Observed speedup ranges are:

- vs ASTER: **1.85x to 277.61x**
- vs FastTree: **4.41x to 6151.99x**

These values should be interpreted as environment-specific benchmark results rather than universal performance guarantees.
