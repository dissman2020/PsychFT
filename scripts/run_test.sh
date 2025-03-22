#!/bin/bash
export PYTHONPATH="/data/kankan.lan/repos/psy101/src/:$PYTHONPATH"

echo "Current directory: $(pwd)"

python src/test.py --config params/test_params.json