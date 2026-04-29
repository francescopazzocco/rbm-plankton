#!/bin/bash

# Sweep L (hidden units) for refactored RBM plankton code
# Config is in src/config.py, entry point is src/main.py

for L in 3 5 7 10; do
    echo "=========================================="
    echo "Running with L = $L"
    echo "=========================================="

    # Replace N_HIDDEN in config.py
    sed -i "s/^N_HIDDEN   = [0-9]*/N_HIDDEN   = $L/" src/config.py

    # Run with PYTHONPATH set to src
    PYTHONPATH=src python src/main.py

    echo ""
done
