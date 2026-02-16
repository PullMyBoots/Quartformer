# Real Dataset Analysis

This document presents QuartFormer inference results on real biological datasets, comparing against published phylogenetic trees from peer-reviewed studies.

## Test Configuration

### Inference Parameters
- **Task Type**: heterogeneous / homogeneous
- **Run Mode**: regular / fast
- **K-Param**: [default value used]
- **Inference Batch Size**: [default value used]
- **Quartet Assembler**: TREE-QMC / QFM-FI

### Evaluation Metrics
- **Robinson-Foulds (RF) Distance**: Normalized RF distance to reference tree
- **Quartet Distance**: Quartet concordance with reference tree
- **Topological Comparison**: Major clade agreement

---

## Datasets

### Dataset 1: [Name]

**Description:**
- **Taxa Count**: X species
- **Sequence Length**: Y bp
- **Gene / Locus**: [Gene name or multi-gene]
- **Publication**: [Citation]
- **Reference Tree**: [Source of published tree]
- **Download**: [URL or source]

**Results:**

| Metric | QuartFormer | Reference Tree |
|--------|-------------|----------------|
| RF Distance | TBD | - |
| Quartet Distance | TBD | - |
| Runtime | TBD s | - |

**Topological Comparison:**
[Describe major clades, agreement/disagreement with published tree]

**Visualization:**
[Placeholder for tree figure]

---

### Dataset 2: [Name]

**Description:**
- **Taxa Count**: X species
- **Sequence Length**: Y bp
- **Gene / Locus**: [Gene name or multi-gene]
- **Publication**: [Citation]
- **Reference Tree**: [Source of published tree]
- **Download**: [URL or source]

**Results:**

| Metric | QuartFormer | Reference Tree |
|--------|-------------|----------------|
| RF Distance | TBD | - |
| Quartet Distance | TBD | - |
| Runtime | TBD s | - |

**Topological Comparison:**
[Describe major clades, agreement/disagreement with published tree]

---

### Dataset 3: [Name]

**Description:**
- **Taxa Count**: X species
- **Sequence Length**: Y bp
- **Gene / Locus**: [Gene name or multi-gene]
- **Publication**: [Citation]
- **Reference Tree**: [Source of published tree]

**Results:**

| Metric | QuartFormer | Reference Tree |
|--------|-------------|----------------|
| RF Distance | TBD | - |
| Quartet Distance | TBD | - |
| Runtime | TBD s | - |

**Topological Comparison:**
[Describe major clades, agreement/disagreement with published tree]

---

### Dataset 4: [Name]

**Description:**
- **Taxa Count**: X species
- **Sequence Length**: Y bp
- **Gene / Locus**: [Gene name or multi-gene]
- **Publication**: [Citation]
- **Reference Tree**: [Source of published tree]

**Results:**

| Metric | QuartFormer | Reference Tree |
|--------|-------------|----------------|
| RF Distance | TBD | - |
| Quartet Distance | TBD | - |
| Runtime | TBD s | - |

**Topological Comparison:**
[Describe major clades, agreement/disagreement with published tree]

---

### Dataset 5: [Name]

**Description:**
- **Taxa Count**: X species
- **Sequence Length**: Y bp
- **Gene / Locus**: [Gene name or multi-gene]
- **Publication**: [Citation]
- **Reference Tree**: [Source of published tree]

**Results:**

| Metric | QuartFormer | Reference Tree |
|--------|-------------|----------------|
| RF Distance | TBD | - |
| Quartet Distance | TBD | - |
| Runtime | TBD s | - |

**Topological Comparison:**
[Describe major clades, agreement/disagreement with published tree]

---

## Summary Table

| Dataset | Taxa | Seq Length (bp) | RF Distance | Quartet Distance | Runtime |
|---------|------|-----------------|-------------|------------------|---------|
| [Name 1] | X | Y | TBD | TBD | TBD s |
| [Name 2] | X | Y | TBD | TBD | TBD s |
| [Name 3] | X | Y | TBD | TBD | TBD s |
| [Name 4] | X | Y | TBD | TBD | TBD s |
| [Name 5] | X | Y | TBD | TBD | TBD s |

---

## Key Observations

### Overall Performance
[Summary of how QuartFormer performs across real datasets]

### Topological Accuracy
[Patterns in clade recovery, common disagreements]

### Taxon Count vs Accuracy
[How accuracy varies with dataset size]

### Sequence Length vs Accuracy
[How accuracy varies with amount of data]

---

## Case Studies

### High Agreement Example
[Dataset where QuartFormer closely matches published tree]

### Interesting Disagreement Example
[Dataset where QuartFormer differs from published tree - explore potential reasons]

---

## Comparison with Other Methods on Real Data

[If available, include comparison with ASTER, FastTree, IQ-TREE on same datasets]

---

## Discussion

### Strengths on Real Data
[Where QuartFormer excels]

### Limitations on Real Data
[Challenges or areas for improvement]

### Real vs Simulated Data Performance
[Compare accuracy on real vs simulated datasets]

---

## Conclusion

[Summary of real dataset performance and practical implications]

---

## Appendix: Dataset Details

### Data Preprocessing
[How raw sequences were processed for QuartFormer input]

### Reference Tree Sources
[Where published trees were obtained]

### Reproducibility
[Commands / parameters to reproduce these results]
