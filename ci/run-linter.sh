#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, flake8, isort)"

source activate esmtools-dev

echo "[flake8]"
# NOTE: F401 should be removed if we can figure out how to import an accessor
# created within the package.
flake8 esmtools --max-line-length=88 --exclude=__init__.py --ignore=W605,W503,F722,C901,F401

echo "[black]"
black --check --line-length=88 -S esmtools

echo "[isort]"
isort --recursive -w 88 --check-only esmtools
