=================
Changelog History
=================

esmtools v1.1 (2019-09-04)
================================

Features
--------
- ``co2_sol`` and ``schmidt`` now can be computed on grids and do not do time-averaging (:pr:`45`) `Riley X. Brady`_.
- ``temp_decomp_takahashi`` now returns a dataset with thermal/non-thermal components (:pr:`45`) `Riley X. Brady`_.
- ``temporal`` module that includes a ``to_annual()`` function for weighted temporal resampling (:pr:`50`) `Riley X. Brady`_.
- ``filtering`` module renamed to ``spatial`` and ``find_indices`` made public. (:pr:`52`) `Riley X. Brady`_.
- ``standardize`` function moved to stats. (:pr:`52`) `Riley X. Brady`_.
- ``loadutils`` removed (:pr:`52`) `Riley X. Brady`_.
- ``calculate_compatible_emissions`` following Jones et al. 2013  (:pr:`54`) `Aaron Spring`_
- Update ``corr`` to broadcast ``x`` and ``y`` such that a single time series can be correlated across a grid. (:pr:`58`) `Riley X. Brady`_.
- ``convert_lon_to_180to180`` and ``convert_lon_to_0to360`` now wrapped with ``convert_lon`` and now supports 2D lat/lon grids. ``convert_lon()`` is also available as an accessor.  (:pr:`60`) `Riley X. Brady`_.

Internals/Minor Fixes
---------------------
- Changed name back to ``esmtools`` now that the readthedocs domain was cleared up. Thanks Andrew Walter! (:pr:`61`) `Riley X. Brady`_.
- ``esmtools`` documentation created with docstring updates for all functions.

esm_analysis v1.0.2 (2019-07-27)
================================

Internals/Minor Fixes
---------------------
- Changed name from ``esmtools`` to ``esm_analysis`` since the former was registered on readthedocs.

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
