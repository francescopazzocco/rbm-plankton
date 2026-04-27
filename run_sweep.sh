#!/bin/bash

# Sweep L (hidden units) and activation functions

for ACT in sigmoid relu; do
    for L in 3 5 7 10; do
        echo "=========================================="
        echo "Running with L = $L, HIDDEN_ACT = $ACT"
        echo "=========================================="

        # Replace N_HIDDEN and HIDDEN_ACT in script
        sed -i "s/^N_HIDDEN    = [0-9]*/N_HIDDEN    = $L/" src/rbm_plankton.py
        sed -i "s/^HIDDEN_ACT  = \"[^\"]*\"/HIDDEN_ACT  = \"$ACT\"/" src/rbm_plankton.py

        # Run
        python src/rbm_plankton.py

        echo ""
    done
done