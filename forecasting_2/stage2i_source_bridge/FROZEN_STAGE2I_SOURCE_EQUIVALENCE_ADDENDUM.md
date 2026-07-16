# Frozen Stage 2I Source-Equivalence Addendum

Frozen after the exact primary M3 URL returned HTTP 403 in both the local runtime and a clean GitHub-hosted Ubuntu runner, and before any M3/M4 value was supplied to the forecasting policy or any canonical outcome metric was computed.

## Purpose

The original canonical holdout required `https://forecasters.org/data/m3comp/M3C.xls` and prohibited unrecorded mirror substitution. That exact transport is unavailable. This addendum does not reinterpret the failed transport as a pass. It creates a new, explicit source namespace and permits M3 values only after a dual-source semantic-equivalence proof.

All model, policy, threshold, interval, candidate, selection, metric, bootstrap, and holdout gates remain unchanged. Only the M3 transport clause is replaced by the gate below.

## Immutable source bridge

Two independently maintained, version-pinned M3 representations must be acquired:

1. CRAN's `Mcomp` 2.8 data object, pinned to CRAN mirror commit `28f4e9babfe26607c9274c2c8d0e900a6aa61eec`, with `data/M3.rda` MD5 `f420fb522d3467b7fd2f96477350202b`.
2. The Monash/Zenodo M3 TSF archives identified by records 4656222, 4656262, 4656298, and 4656335 for yearly, quarterly, monthly, and other series.

M4 remains on the exact URLs in the original pre-value manifest.

## Mandatory equivalence gates

All gates must pass before a task suite may be emitted.

1. The original pre-value manifest hash is exactly `977f3db34bdd9e45940190858889494a1bfa19d6af2e4f3d7ab3984f30cb64a9`.
2. Every bridge URL, redirect target, byte count, SHA-256, and relevant upstream commit/record identifier is saved.
3. The CRAN `M3.rda` MD5 equals `f420fb522d3467b7fd2f96477350202b`.
4. Both representations contain exactly 3,003 series: 645 yearly, 756 quarterly, 1,428 monthly, and 174 other.
5. Canonical IDs are exactly `Y1..Y645`, `Q1..Q756`, `M1..M1428`, and `O1..O174` in both representations.
6. Every paired series has the same length and horizon metadata.
7. Concatenated CRAN history and future values match the corresponding Zenodo TSF series elementwise with `rtol <= 1e-12` and `atol <= 1e-10`; no mismatch is tolerated.
8. No series value, error, route, source label, or outcome may influence the frozen SHA-256 series-selection rule.
9. The resulting 700-task inventory and all source/equivalence files are hashed before forecasting.
10. Any failed or unverified gate stops the run; no third source is substituted automatically.

## Holdout status

A passing bridge authorizes the unchanged Stage 2I forecasting policy to run on a new source-equivalent canonical panel. Results must be labeled `Stage2I-SE`, not as a literal pass of the original exact-URL source gate. The remaining 14 canonical holdout gates and all forecasting thresholds are unchanged.
