# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] — 2026-09-30

### Changed — BREAKING

- **Time series are refined globally by default.** The series is now a single
  tensor and the windows are views of it, so one backward pass gives each time
  step the *sum* of the contributions of every window containing it (the chain
  rule), and the data is optimised with Adam against

      criterion(f(windows), targets) + lam * mean((z - z_observed)^2)

  The proximity term is what makes the problem well posed: with stride 1 there
  is one constraint per time step and one unknown per channel and step.

  The previous rule — a Gauss-Newton step per window, averaged over the windows
  covering each time step — scaled every step by that window's residual, so a
  backbone that fits well left the data almost where it was. Dividing by the
  window count is also not the derivative of anything, and it shrank the
  interior of the series relative to its edges. On the benchmarks of the
  accompanying study the global rule recovered 60–91 % of the injected noise on
  four of five series where the window rule recovered under 2 %, at about a
  fifth of its cost.

  `is_ts=True` code written against 1.2.0 **runs without error under the new
  rule**. Pass `ts_strategy="window"` to reproduce 1.2.0 results; `nrr`,
  `nr_threshold`, `eta_x` and `eta_y` have no effect under the global rule, and
  setting any of them there raises a warning. Tabular refinement is unchanged.

- `max_epochs` now defaults to `None`, which resolves to 100 for the
  Gauss-Newton step (unchanged) and 300 for global refinement.

### Added

- `ts_strategy` (`"global"` | `"window"`), and `lam`, `lr` and `tol` for the
  global rule.
- `fit(target_cols=...)`: declares that the target *is* one or more columns of
  `X`, the usual autoregressive setup. Global refinement then refines each
  target value once, as the single variable it is, instead of an independent
  copy passed as `y`. Column names passed as `y` with a DataFrame `X` are
  treated the same way. With `denoise_y=False` these columns are held fixed —
  under the window rule that flag cannot protect a target that is also an input
  channel.

### Fixed

- `transform()` refined the fitted data in place, so a second call started from
  the first call's output, overwrote the array that call had returned, and left
  `x_noisy` holding refined data. Every call now starts from the data passed to
  `fit()`, which is what fitting once and sweeping settings requires.

## [1.2.0] — 2026-09-23

First stable release of the Gauss-Newton step rule. It supersedes the `1.1.0b1`
and `1.1.0b2` pre-releases, which `pip install denograd` never selected: under
PEP 440 a pre-release sorts *below* its final version, so `1.1.0` (March 2026)
remained the installed default while the work that superseded it sat unreachable.

### Changed — BREAKING

- **`nrr` means something different.** It was an absolute L2 step length, where
  useful values were around `0.01`. It is now the **fraction of the distance to
  the consistency surface `{f(x) = y}` that a sample closes per epoch**: a
  dimensionless quantity where `nrr=1.0` lands on the surface in a single step
  and the useful range is roughly `0.3`–`0.9`. The default moved from `0.05` to
  `0.5` to match.

  The parameter kept its name, so **code written against `1.1.0` or earlier runs
  without error and produces meaningless results**: an `nrr=0.01` tuned for the
  old rule now closes 1 % of the gap per epoch instead of taking a step of length
  0.01, and an `nrr` tuned for the new rule is ~50x too large under the old one.
  Results obtained with the two rules are not comparable at equal `nrr`.

  A sample can now never travel further than its own residual, which is what
  makes the step self-limiting and removes the need to tune a step length per
  dataset.

- `requires-python` raised from `>=3.6` to `>=3.9`. The module itself is still
  3.6-compatible, but current `torch` and `numpy` are not, so the old bound
  described an installation that cannot actually resolve.

### Added

- **`eta_x` and `eta_y`**: per-block multipliers applied to the two adjustments
  *after* the joint normalisation, so they redistribute nothing — lowering
  `eta_y` shrinks the target correction without enlarging the feature one.
  Useful because with `eta_y=1` the target races to meet `f(x)` (the cheap
  direction: a scalar with gradient −1 absorbs most of the step in the windowed
  case), the residual collapses, the loop converges, and the features barely
  move. Holding the target back keeps the residual alive so the features have to
  do the work. Both default to `1.0`, which reproduces the previous behaviour.

- `denograd.__version__`, read from the installed package metadata. Previously
  there was no way for a caller to record which build produced a result.

### Fixed

- Nine bugs in the correction loop, fixed alongside the step-rule change
  (`e6d9fdf`).

### Removed

- `matplotlib` and `ipython` from the install requirements. Neither appears
  anywhere in `denograd.py`; they had been declared since `0.1.x` and forced
  every user to pull IPython in order to import the library.

- `setup.py`. All metadata now lives in `pyproject.toml` under PEP 621.

## [1.1.0] — 2026-03-26

Published to PyPI without a corresponding commit in this repository. Its
provenance could not be reconstructed, and `1.2.0` supersedes it.

## [1.0.2] — 2026-02-26

## [1.0.1] — 2026-02-23

## [1.0.0] — 2026-01-30

Unified dataset handling, future-step prediction for sliding windows, and
improved gradient stability.

## [0.1.3] — 2026-01-27

## [0.1.0] — 2026-01-19

Initial release.
