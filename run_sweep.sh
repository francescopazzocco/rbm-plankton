#!/bin/bash

# Sweep L (hidden units) for refactored RBM plankton code
# OUTPUT_DIR in config.py auto-generates based on VISIBLE_MODEL and N_HIDDEN

for L in 3 5 7 10; do
    echo "=========================================="
    echo "Running with L = $L"
    echo "=========================================="

    # Replace N_HIDDEN in config.py
    sed -i "s/^N_HIDDEN     = [0-9]*/N_HIDDEN     = $L/" src/config.py

    # Run with PYTHONPATH set to src
    PYTHONPATH=src python src/main.py

    echo ""
done
