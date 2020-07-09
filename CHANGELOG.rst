=================
Changelog History
=================

esmtools v1.1.2 (2020-07-09)
============================

Internals/Minor Fixes
---------------------
- Fix ``flake8`` F401 error by using ``TimeUtilAccessor`` directly in first instance
  in code. (:pr:`86`) `Riley X. Brady`_.
- Add ``conda`` badge and ``conda`` installation instructions. (:pr:`87`) `Riley X. Brady`_.
- Migrate ``corr`` and ``autocorr`` from ``climpred`` to ``esmtools`` with some light edits to
  the code. (:pr:`88`) `Riley X. Brady`_.

Deprecated
----------
- ``climpred`` removed as a dependency for ``esmtools``. (:pr:`88`) `Riley X. Brady`_.
- ``autocorr`` deprecated, since it can be run via ``corr(x, x)``. ``ACF`` renamed to
  ``autocorr``, which reflects ``pandas``-style naming. (:pr:`88`) `Riley X. Brady`_.

esmtools v1.1.1 (2020-07-08)
============================

Features
--------
- ``xarray`` implementation of ``statsmodels.stats.multitest.multipletests``.
  (:pr:`71`) `Aaron Spring`_
- Implements ``nan_policy=...`` keyword for :py:func:`~esmtools.stats.linear_slope`,
  :py:func:`~esmtools.stats.linregress`, :py:func:`~esmtools.stats.polyfit`,
  :py:func:`~esmtools.stats.rm_poly`, :py:func:`~esmtools.stats.rm_trend`.
  (:pr:`70`) `Riley X. Brady`_.

  * ``'none', 'propagate'``: Propagate nans through function. I.e., return a nan for
    a given grid cell if a nan exists in it.
  * ``'raise'``: Raise an error if there are any nans in the datasets.
  * ``'drop', 'omit'``: Like ``skipna``, compute statistical function after removing
    nans.

- Adds support for datetime axes in :py:func:`~esmtools.stats.linear_slope`,
  :py:func:`~esmtools.stats.linregress`, :py:func:`~esmtools.stats.polyfit`,
  :py:func:`~esmtools.stats.rm_poly`, :py:func:`~esmtools.stats.rm_trend`. Converts
  datetimes to numeric time, computes function, and then converts back to datetime.
  (:pr:`70`)`Riley X. Brady`_.
- :py:func:`~esmtools.stats.linear_slope`,
  :py:func:`~esmtools.stats.linregress`, :py:func:`~esmtools.stats.polyfit`,
  :py:func:`~esmtools.stats.rm_poly`, :py:func:`~esmtools.stats.rm_trend` are now
  dask-compatible and vectorized better.
  (:pr:`70`) `Riley X. Brady`_.

Bug Fixes
---------
- Does not eagerly evaluate ``dask`` arrays anymore. (:pr:`70`) `Riley X. Brady`_.

Internals/Minor Fixes
---------------------
- Adds ``isort`` and ``nbstripout`` to CI for development. Blacken and isort code.
  (:pr:`73`) `Riley X. Brady`_

Documentation
-------------
- Add more robust API docs page, information on how to contribute, CHANGELOG, etc. to
  ``sphinx``. (:pr:`67`) `Riley X. Brady`_.

Deprecations
------------
- Removes ``mpas`` and ``vis`` modules. The former is better for a project-dependent
  package. The latter essentially poorly replicates some of ``proplot`` functionality.
  (:pr:`69`) `Riley X. Brady`_.
- Removes ``stats.smooth_series``, since there is an easy ``xarray`` function for it.
  (:pr:`70`) `Riley X. Brady`_.
- Changes ``stats.linear_regression`` to ``stats.linregress``.
  (:pr:`70`) `Riley X. Brady`_.
- Changes ``stats.compute_slope`` to ``stats.linear_slope``.
  (:pr:`70`) `Riley X. Brady`_.
- Removes ``stats.area_weight`` and ``stats.cos_weight`` since they are available
  through ``xarray``. (:pr:`83`) `Riley X. Brady`_.

esmtools v1.1 (2019-09-04)
==========================

Features
--------
- ``co2_sol`` and ``schmidt`` now can be computed on grids and do not do time-averaging
  (:pr:`45`) `Riley X. Brady`_.
- ``temp_decomp_takahashi`` now returns a dataset with thermal/non-thermal components
  (:pr:`45`) `Riley X. Brady`_.
- ``temporal`` module that includes a ``to_annual()`` function for weighted temporal
  resampling (:pr:`50`) `Riley X. Brady`_.
- ``filtering`` module renamed to ``spatial`` and ``find_indices`` made public.
  (:pr:`52`) `Riley X. Brady`_.
- ``standardize`` function moved to stats. (:pr:`52`) `Riley X. Brady`_.
- ``loadutils`` removed (:pr:`52`) `Riley X. Brady`_.
- ``calculate_compatible_emissions`` following Jones et al. 2013
  (:pr:`54`) `Aaron Spring`_
- Update ``corr`` to broadcast ``x`` and ``y`` such that a single time series can be
  correlated across a grid. (:pr:`58`) `Riley X. Brady`_.
- ``convert_lon_to_180to180`` and ``convert_lon_to_0to360`` now wrapped with
  ``convert_lon`` and now supports 2D lat/lon grids. ``convert_lon()`` is also
  available as an accessor.  (:pr:`60`) `Riley X. Brady`_.

Internals/Minor Fixes
---------------------
- Changed name back to ``esmtools`` now that the readthedocs domain was cleared up.
  Thanks Andrew Walter! (:pr:`61`) `Riley X. Brady`_.
- ``esmtools`` documentation created with docstring updates for all functions.

esm_analysis v1.0.2 (2019-07-27)
================================

Internals/Minor Fixes
---------------------
- Changed name from ``esmtools`` to ``esm_analysis`` since the former was registered
  on readthedocs.

esmtools v1.0.1 (2019-07-25)
============================

Internals/Minor Fixes
---------------------
- Add versioning and clean up setup file.
- Add travis continuous integration and coveralls for testing.

esmtools v1.0.0 (2019-07-25)
============================
Formally releases ``esmtools`` on pip for easy installing by other packages.

.. _`Riley X. Brady`: https://github.com/bradyrx
.. _`Aaron Spring`: https://github.com/aaronspring
