#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER_SCRIPT="${ROOT_DIR}/infer_tree.py"
OUT_DIR="${SCRIPT_DIR}/output"

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "QuartFormer Example Runner"
echo "=========================================="
echo "Root: ${ROOT_DIR}"
echo "Examples: ${SCRIPT_DIR}"
echo "Output: ${OUT_DIR}"
echo

echo "1) Small dataset (24 taxa): basic inference"
python "${INFER_SCRIPT}" \
  --phy "${SCRIPT_DIR}/24/MSA.phy" \
  --out "${OUT_DIR}/output_24.nwk" \
  --task-type homogeneous

echo
echo "2) Medium dataset (48 taxa): inference + RF evaluation"
python "${INFER_SCRIPT}" \
  --phy "${SCRIPT_DIR}/48/MSA.phy" \
  --out "${OUT_DIR}/output_48.nwk" \
  --ref-tree "${SCRIPT_DIR}/48/tree.nwk" \
  --metric rf \
  --task-type homogeneous

echo
echo "3) Large dataset (96 taxa): inference + quartet concordance"
python "${INFER_SCRIPT}" \
  --phy "${SCRIPT_DIR}/96/MSA.phy" \
  --out "${OUT_DIR}/output_96.nwk" \
  --ref-tree "${SCRIPT_DIR}/96/tree.nwk" \
  --metric quartet \
  --task-type homogeneous

echo
echo "4) Wolbachia: heterogeneous mode + support + plot + RF evaluation"
python "${INFER_SCRIPT}" \
  --phy "${SCRIPT_DIR}/Wolbachia/MSA.phy" \
  --out "${OUT_DIR}/output_wolbachia.nwk" \
  --ref-tree "${SCRIPT_DIR}/Wolbachia/ref_tree.newick" \
  --metric rf \
  --task-type heterogeneous \
  --compute-branch-support \
  --plot-tree

echo
echo "=========================================="
echo "All examples completed."
echo "Generated files are in: ${OUT_DIR}"
echo "=========================================="
