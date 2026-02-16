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
# Basic usage (24 species)
python ../run_qf.py --phy 24/MSA.phy --out output_24.nwk

# With reference tree evaluation (48 species)
python ../run_qf.py --phy 48/MSA.phy --out output_48.nwk --ref-tree 48/tree.nwk --metric rf

# Large dataset with quartet accuracy (96 species)
python ../run_qf.py --phy 96/MSA.phy --out output_96.nwk --ref-tree 96/tree.nwk --metric quartet

# Real dataset (Wolbachia) with RF evaluation
python ../run_qf.py --phy Wolbachia/MSA.phy --out output_wolbachia.nwk --ref-tree Wolbachia/ref_tree.newick --metric rf
```

## Notes

- All datasets were simulated using GTR model with 100,000 site alignments
- Reference trees are provided for accuracy evaluation
- The `Wolbachia` example is a real dataset and is included for qualitative validation
- Adjust `--k-param`, `--run-mode`, and `--task-type` for different scenarios
