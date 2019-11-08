API Reference
=============

This page provides an auto-generated summary of esmtools's API.
For more details and examples, refer to the relevant chapters in the main part of the documentation.

Carbon
------

``from esmtools.carbon import ...``

.. currentmodule:: esmtools.carbon

Functions related to analyzing ocean (and perhaps terrestrial) biogeochemistry.

.. autosummary::
    :toctree: api/

    co2_sol
    schmidt
    temp_decomp_takahashi
    potential_pco2
    spco2_sensitivity
    spco2_decomposition_index
    spco2_decomposition
    calculate_compatible_emissions
    get_iam_emissions
    plot_compatible_emissions

Composite Analysis
------------------

``from esmtools.composite import ...``

.. currentmodule:: esmtools.composite

Functions pertaining to composite analysis. Composite analysis takes the mean view of some field (e.g., sea surface temperature) when some climate index (e.g., El Nino Southern Oscillation) is in its negative or positive mode.

.. autosummary::
    :toctree: api/

    composite_analysis

Grid Tools
----------

``from esmtools.grid import ...``

.. currentmodule:: esmtools.grid

Functions related to climate model grids.

.. autosummary::
    :toctree: api/

    convert_lon


MPAS Model Tools
----------------

``from esmtools.mpas import ...``

.. currentmodule:: esmtools.mpas

Functions related to analyzing output from the Model for Prediction Across Scales (MPAS) ocean model. Since the grid is comprised of unstructured hexagons, analysis and visualization is not as straight-forward as working with regularly gridded data. The best visualizations on the unstructured grid can be made in ParaView.

.. autosummary::
    :toctree: api/

    scatter
    xyz_to_lat_lon
    convert_rad_to_deg
    convert_deg_to_rad

Physics
-------

``from esmtools.physics import ...``

.. currentmodule:: esmtools.physics

Functions related to physics/dynamics.

.. autosummary::
    :toctree: api/

    stress_to_speed

Spatial
-------

``from esmtools.spatial import ...``

.. currentmodule:: esmtools.spatial

Functions related to spatial analysis.

.. autosummary::
    :toctree: api/

    find_indices
    extract_region

Stats
-----

``from esmtools.stats import ...``

.. currentmodule:: esmtools.stats

Statistical functions. A portion directly wrap functions from ``climpred``.

.. autosummary::
    :toctree: api/

    standardize
    nanmean
    cos_weight
    area_weight
    smooth_series
    fit_poly
    linear_regression
    corr
    rm_poly
    rm_trend
    autocorr
    ACF
    ttest_ind_from_stats

Temporal
--------

``from esmtools.temporal import ...``

.. currentmodule:: esmtools.temporal

Functions related to time.

.. autosummary::
    :toctree: api/

    to_annual

Visualization
-------------

``from esmtools.vis import ...``

.. currentmodule:: esmtools.vis

Functions related to visualization. Most of functions these are useless with the advent of ``proplot``.

.. autosummary::
    :toctree: api/

    deseam
    discrete_cmap
    make_cartopy
    add_box
    savefig
    meshgrid
    outer_legend
    quick_pcolor
    global_subplot_colorbar
