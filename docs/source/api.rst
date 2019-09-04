API Reference
=============

This page provides an auto-generated summary of esmtools's API.
For more details and examples, refer to the relevant chapters in the main part of the documentation.

Carbon
~~~~~~

Functions related to analyzing ocean biogeochemistry.

.. automodsumm:: esmtools.carbon
    :functions-only:
    :toctree: api
    :skip: check_xarray, linear_regression, nanmean, rm_poly

Composite Analysis
~~~~~~~~~~~~~~~~~~

Functions pertaining to composite analysis. Composite analysis takes the mean view of some field (e.g., sea surface temperature) when some climate index (e.g., El Nino Southern Oscillation) is in its negative or positive mode.

.. automodsumm:: esmtools.composite
    :functions-only:
    :toctree: api
    :skip: check_xarray, ttest_ind_from_stats, standardize

Grid Tools
~~~~~~~~~~

Functions related to climate model grids.

.. automodsumm:: esmtools.grid
    :functions-only:
    :toctree: api
    :skip: check_xarray

MPAS Model Tools
~~~~~~~~~~~~~~~~

Functions related to analyzing output from the Model for Prediction Across Scales (MPAS) ocean model. Since the grid is comprised of unstructured hexagons, analysis and visualization is not as straight-forward as working with regularly gridded data. The best visualizations on the unstructured grid can be made in ParaView.

.. automodsumm:: esmtools.mpas
    :functions-only:
    :toctree: api
    :skip: make_cartopy

Physics
~~~~~~~

Functions related to physics/dynamics.

.. automodsumm:: esmtools.physics
    :functions-only:
    :toctree: api

Spatial
~~~~~~~

Functions related to spatial analysis.

.. automodsumm:: esmtools.spatial
    :functions-only:
    :toctree: api
    :skip: check_xarray

Statistics
~~~~~~~~~~

Statistical functions. A portion directly wrap functions from ``climpred``.

.. automodsumm:: esmtools.stats
    :functions-only:
    :toctree: api
    :skip: check_xarray, get_dims, has_dims, lreg, tti_from_stats


Temporal
~~~~~~~~

Functions related to time.

.. automodsumm:: esmtools.temporal
    :functions-only:
    :toctree: api


Unit Conversions
~~~~~~~~~~~~~~~~

Functions related to converting units.

.. automodsumm:: esmtools.conversions
    :functions-only:
    :toctree: api
    :skip: check_xarray

Visualization
~~~~~~~~~~~~~

Functions related to visualization.

.. automodsumm:: esmtools.vis
    :functions-only:
    :toctree: api
    :skip: add_cyclic_point
