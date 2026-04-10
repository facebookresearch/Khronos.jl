#!/bin/bash
# Sync Julia source files into the Python package for pip distribution.
#
# Run this before `pip install .` or `python -m build` from the python/ directory.
# Not needed for editable installs (`pip install -e .`) since the bridge
# auto-detects the repo root in development mode.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/python/khronos/julia_src"

echo "Syncing Julia source into $DEST ..."

# Create directory structure
mkdir -p "$DEST/src"
mkdir -p "$DEST/precompile"

# Ensure Python package markers exist
touch "$DEST/__init__.py"
touch "$DEST/src/__init__.py"
touch "$DEST/precompile/__init__.py"

# Write the __init__.py content if empty
if [ ! -s "$DEST/__init__.py" ]; then
    echo "# This directory contains the bundled Khronos.jl Julia source code." > "$DEST/__init__.py"
    echo "# It exists as a Python package so importlib.resources can locate it." >> "$DEST/__init__.py"
fi

# Copy Project.toml
cp "$REPO_ROOT/Project.toml" "$DEST/Project.toml"

# Copy all Julia source files preserving directory structure
rsync -a --include='*/' --include='*.jl' --exclude='*' \
    "$REPO_ROOT/src/" "$DEST/src/"

# Copy precompile workload (lives in the Python package tree)
if [ -f "$DEST/precompile/workload.jl" ]; then
    echo "  precompile/workload.jl already exists, keeping."
else
    echo "  WARNING: precompile/workload.jl not found in $DEST/precompile/"
    echo "  Create it manually or copy from the template."
fi

# Count files
N_JL=$(find "$DEST/src" -name '*.jl' | wc -l)
echo "Done. Synced $N_JL Julia source files + Project.toml."
