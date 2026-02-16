#!/bin/bash
# Example usage scripts for QuartFormer

echo "=========================================="
echo "QuartFormer Example Scripts"
echo "=========================================="
echo ""

# Small dataset (24 species) - Quick test
echo "1. Running small dataset (24 species)..."
python ../run_qf.py --phy 24/MSA.phy --out output_small.nwk

# Medium dataset (48 species) - With RF evaluation
echo ""
echo "2. Running medium dataset (48 species) with RF distance evaluation..."
python ../run_qf.py --phy 48/MSA.phy --out output_medium.nwk --ref-tree 48/tree.nwk --metric rf

# Large dataset (96 species) - With quartet accuracy
echo ""
echo "3. Running large dataset (96 species) with quartet accuracy evaluation..."
python ../run_qf.py --phy 96/MSA.phy --out output_large.nwk --ref-tree 96/tree.nwk --metric quartet

echo ""
echo "=========================================="
echo "All examples completed!"
echo "Output trees: output_*.nwk"
echo "=========================================="
