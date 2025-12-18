#!/bin/bash
# Build script for the Rust grammar extension

set -e

echo "Building Rust grammar extension..."

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed. Install from https://rustup.rs/"
    exit 1
fi

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Build in release mode for maximum performance
echo "Building with maturin (release mode)..."
maturin develop --release

echo "Build complete! The Rust extension is now available."
echo ""
echo "To test, run:"
echo "  python benchmark_rust.py"
