API Reference
=============

This page provides an auto-generated summary of esmtools's API.
For more details and examples, refer to the relevant chapters in the main part of the
documentation.

Carbon
------

``from esmtools.carbon import ...``

.. currentmodule:: esmtools.carbon

Functions related to analyzing ocean (and perhaps terrestrial) biogeochemistry.

.. autosummary::
    :toctree: api/

    calculate_compatible_emissions
    co2_sol
    get_iam_emissions
    plot_compatible_emissions
    potential_pco2
    schmidt
    spco2_sensitivity
    spco2_decomposition_index
    spco2_decomposition
    temp_decomp_takahashi

Composite Analysis
------------------

``from esmtools.composite import ...``

.. currentmodule:: esmtools.composite

Functions pertaining to composite analysis. Composite analysis takes the mean view of
some field (e.g., sea surface temperature) when some climate index
(e.g., El Nino Southern Oscillation) is in its negative or positive mode.

.. autosummary::
    :toctree: api/

    composite_analysis

Conversions
-----------

``from esmtools.conversions import ...``

.. currentmodule:: esmtools.conversions

Functions related to unit conversions.

.. autosummary::
    :toctree: api/

    convert_mpas_fgco2

Grid Tools
----------

``from esmtools.grid import ...``

.. currentmodule:: esmtools.grid

Functions related to climate model grids.

.. autosummary::
    :toctree: api/

    convert_lon


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

    extract_region
    find_indices

Statistics
----------

``from esmtools.stats import ...``

.. currentmodule:: esmtools.stats

Functions dealing with statistics.

.. autosummary::
    :toctree: api/

    ACF
    autocorr
    corr
    linear_slope
    linregress
    polyfit
    nanmean
    rm_poly
    rm_trend
    standardize

Temporal
--------

``from esmtools.temporal import ...``

.. currentmodule:: esmtools.temporal

Functions related to time.

.. autosummary::
    :toctree: api/

    to_annual


Testing
-------

``from esmtools.testing import ...``

.. currentmodule:: esmtools.testing

Functions specifically focused on statistical testing.

.. autosummary::
    :toctree: api/

    multipletests
    ttest_ind_from_stats
