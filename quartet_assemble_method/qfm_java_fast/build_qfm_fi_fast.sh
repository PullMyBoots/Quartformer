#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p build_fast/classes

# Thin adapter build:
# - Compile only FastMain
# - Reuse official qfm_java QFM-FI.jar classes for algorithm core
CORE_JAR="$ROOT_DIR/../qfm_java/QFM-FI_unzipped/QFM-FI.jar"
if [[ ! -f "$CORE_JAR" ]]; then
  echo "[ERROR] Missing core jar: $CORE_JAR" >&2
  exit 1
fi

rm -rf build_fast/classes/*
javac -cp "$CORE_JAR" -d build_fast/classes qfm_fi/src/qfm_fi/FastMain.java

cat > build_fast/manifest.mf <<'EOF'
Main-Class: qfm_fi.FastMain
Class-Path: ../../qfm_java/QFM-FI_unzipped/QFM-FI.jar
EOF

jar cfm QFM-FI_unzipped/QFM-FI-fast.jar build_fast/manifest.mf -C build_fast/classes .

echo "[SUCCESS] Built: $ROOT_DIR/QFM-FI_unzipped/QFM-FI-fast.jar"
