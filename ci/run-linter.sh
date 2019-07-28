#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, flake8, isort)"

source activate esm-analysis-dev

echo "[flake8]"
flake8 esm_analysis --max-line-length=88 --exclude=__init__.py --ignore=W605,W503

echo "[black]"
black --check --line-length=88 -S esm_analysis 