# esmtools

A toolbox for functions related to Earth System Model analysis, with a focus on biogeochemical oceanography.

**Note**: This is more of a personal package with functions I use commonly. It doesn't have any testing (as of now) or documentation, and isn't supposed to function as a full analysis package.

Please check out [`climpred`](https://github.com/bradyrx/climpred) for a package in development made for analysis of climate prediction ensembles.

## Installation
```shell
pip install git+https://github.com/bradyrx/esmtools
```

**Note**: Little development in the future will go towards visualization. We suggest [proplot](https://github.com/lukelbd/proplot) to users for a phenomenal `matplotlib` wrapper.

## Contribution Guide

1. Install `pre-commit` and its hook on the `esmtools` repository. Make sure you're in the main repository when you run `pre-commit install`.

```bash
$ pip install --user pre-commit
$ pre-commit install
```

Afterwards, `pre-commit` will be run with every commit so that formatting is handled at commit-time.

## Contact
Developed and maintained by Riley Brady.

email: riley.brady@colorado.edu

web: https://www.rileyxbrady.com
