# Example Datasets

This directory contains example phylogenetic inference datasets for testing QuartFormer.

## Datasets

### Small (24 species)
- **MSA.phy**: 2.3 MB - Multiple sequence alignment
- **tree.nwk**: 594 B - Reference tree for accuracy evaluation
- **Use case**: Quick testing and validation

### Medium (48 species)
- **MSA.phy**: 4.6 MB - Multiple sequence alignment
- **tree.nwk**: 1.2 KB - Reference tree for accuracy evaluation
- **Use case**: Standard usage demonstration

### Large (96 species)
- **MSA.phy**: 9.2 MB - Multiple sequence alignment
- **tree.nwk**: 2.4 KB - Reference tree for accuracy evaluation
- **Use case**: Large-scale performance demonstration

### Real dataset (Wolbachia)
- **MSA.phy**: Concatenated alignment from the Wolbachia dataset
- **ref_tree.newick**: Reference tree used for comparison
- **Use case**: Real-world validation example
- **Source paper**: *Phylogenomic Analysis of Wolbachia Strains Reveals Patterns of Genome Evolution and Recombination*

## Quick Start

```bash
# Move to examples directory
cd examples

# Run all examples
./run_examples.sh
```

You can also run cases manually:

```bash
# 1) Basic inference (24 species)
python ../infer_tree.py \
  --phy 24/MSA.phy \
  --out output/output_24.nwk \
  --task-type homogeneous

# 2) Inference + RF evaluation (48 species)
python ../infer_tree.py \
  --phy 48/MSA.phy \
  --out output/output_48.nwk \
  --ref-tree 48/tree.nwk \
  --metric rf \
  --task-type homogeneous

# 3) Inference + quartet concordance (96 species)
python ../infer_tree.py \
  --phy 96/MSA.phy \
  --out output/output_96.nwk \
  --ref-tree 96/tree.nwk \
  --metric quartet \
  --task-type homogeneous

# 4) Real dataset (Wolbachia) + support + plot
python ../infer_tree.py \
  --phy Wolbachia/MSA.phy \
  --out output/output_wolbachia.nwk \
  --ref-tree Wolbachia/ref_tree.newick \
  --metric rf \
  --task-type heterogeneous \
  --compute-branch-support \
  --plot-tree
```

## Notes

- All datasets were simulated using GTR model with 100,000 site alignments
- Reference trees are provided for accuracy evaluation
- The `Wolbachia` example is a real dataset and is included for qualitative validation
- Basic parameters can be passed via CLI; advanced parameters are configured in `infer_config.jsonc`
- Parameter precedence in current implementation:
  - `basic`: `CLI > config > default`
  - `advanced`: `config > CLI > default`
