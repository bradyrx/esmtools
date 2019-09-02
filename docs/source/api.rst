API Reference
=============

This page provides an auto-generated summary of esm_analysis's API.
For more details and examples, refer to the relevant chapters in the main part of the documentation.

Carbon
~~~~~~

Functions related to analyzing ocean biogeochemistry.

.. automodsumm:: esm_analysis.carbon
    :functions-only:
    :toctree: api
    :skip: check_xarray, linear_regression, nanmean, rm_poly

Composite Analysis
~~~~~~~~~~~~~~~~~~

Functions pertaining to composite analysis. Composite analysis takes the mean view of some field (e.g., sea surface temperature) when some climate index (e.g., El Nino Southern Oscillation) is in its negative or positive mode.

.. automodsumm:: esm_analysis.composite
    :functions-only:
    :toctree: api
    :skip: check_xarray, ttest_ind_from_stats, standardize

Grid Tools
~~~~~~~~~~

Functions related to climate model grids.

.. automodsumm:: esm_analysis.grid
    :functions-only:
    :toctree: api
    :skip: check_xarray

MPAS Model Tools
~~~~~~~~~~~~~~~~

Functions related to analyzing output from the Model for Prediction Across Scales (MPAS) ocean model. Since the grid is comprised of unstructured hexagons, analysis and visualization is not as straight-forward as working with regularly gridded data. The best visualizations on the unstructured grid can be made in ParaView.

.. automodsumm:: esm_analysis.mpas
    :functions-only:
    :toctree: api
    :skip: make_cartopy

Physics
~~~~~~~

Functions related to physics/dynamics.

.. automodsumm:: esm_analysis.physics
    :functions-only:
    :toctree: api

Spatial
~~~~~~~

Functions related to spatial analysis.

.. automodsumm:: esm_analysis.spatial
    :functions-only:
    :toctree: api
    :skip: check_xarray

Statistics
~~~~~~~~~~

Statistical functions. A portion directly wrap functions from ``climpred``.

.. automodsumm:: esm_analysis.stats
    :functions-only:
    :toctree: api
    :skip: check_xarray, get_dims, has_dims, lreg, tti_from_stats


Temporal
~~~~~~~~

Functions related to time.

.. automodsumm:: esm_analysis.temporal
    :functions-only:
    :toctree: api


Unit Conversions
~~~~~~~~~~~~~~~~

Functions related to converting units.

.. automodsumm:: esm_analysis.conversions
    :functions-only:
    :toctree: api
    :skip: check_xarray

Visualization
~~~~~~~~~~~~~

Functions related to visualization.

.. automodsumm:: esm_analysis.vis
    :functions-only:
    :toctree: api
    :skip: add_cyclic_point
