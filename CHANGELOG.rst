=================
Changelog History
=================

esm_analysis v1.0.3 (2019-07-##)
================================

Features
--------
- ``co2_sol`` and ``schmidt`` now can be computed on grids and does not do time-averaging (:pr:`45`) `Riley X. Brady`_.
- ``temp_decomp_takahashi`` now returns a dataset with thermal/non-thermal components (:pr:`45`) `Riley X. Brady`_.
- ``temporal`` module that includes a ``to_annual()`` function for weighted temporal resampling (:pr:`50`) `Riley X. Brady`_.
- ``filtering`` module renamed to ``spatial`` and ``find_indices`` made public. (:pr:`52`) `Riley X. Brady`_.
- ``standardize`` function moved to stats. (:pr:`52`) `Riley X. Brady`_.

Internals/Minor Fixes
---------------------
- ``esm_analysis`` documentation created with docstring updates for all functions.

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
