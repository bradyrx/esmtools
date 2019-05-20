# esmtools 

A toolbox for functions related to Earth System Model analysis, with a focus on biogeochemical oceanography.

**Note**: This is more of a personal package with functions I use commonly. It doesn't have any testing (as of now) or documentation, and isn't supposed to function as a full analysis package.

Please check out [`climpred`](https://github.com/bradyrx/climpred) for a package in development made for analysis of climate prediction ensembles.

## Installation
```shell
pip install git+https://github.com/bradyrx/esmtools
```

## Modules

`carbon`

Functions for the carbon cycle and carbonate chemistry.

`loadutils`

Load in sample datasets or masks.

`filtering`

Functions for filtering output over space and time.

`physics`

Functions related to physical conversions.

`stats`

Functions for time series and spatial statistics

`vis`

Functions for colorbars, coloarmaps, and projecting data globally or regionally.

**Note**: Little development in the future will go towards visualization. We suggest [proplot](https://github.com/lukelbd/proplot) to users for a phenomenal `matplotlib` wrapper.

## Contact
Developed and maintained by Riley Brady.

email: riley.brady@colorado.edu

web: https://www.rileyxbrady.com
