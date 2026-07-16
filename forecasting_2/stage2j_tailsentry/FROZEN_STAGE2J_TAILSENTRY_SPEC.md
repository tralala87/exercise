# Frozen Stage 2J TAILSENTRY Confirmation Specification

Frozen on 2026-07-16 before M1 source values are downloaded into the forecasting runtime and before any Stage 2J M1 or fresh-synthetic outcome is computed.

## Scientific question

Does a parameter-free, single-pass robust forecast envelope repair the catastrophic tail-risk failure discovered in the Stage 2I archival-official M3/M4 holdout while preserving the validated compiler/router strengths?

The Stage 2I-AO holdout is closed. It may motivate the failure category, but no Stage 2J threshold, gate, or decision may be selected from its outcomes.

## Development-only rule selection

The guard family was evaluated only on the already-open Stage 2H 14-source development set. The allowed family was the no-guard baseline plus symmetric pointwise empirical candidate intervals with lower quantiles in `{0.05, 0.075, 0.10, 0.125, 0.15, 0.20}`.

Selection rule:

1. reject a candidate if any Stage 2H source is more than 2% worse than the unguarded forecast;
2. among the remaining candidates, minimize equal-source macro MASE;
3. break ties by macro WQL and then by the less restrictive interval.

This selected lower quantile `q=0.125`. Leave-one-source-out selection chose `q=0.125` for 13 of 14 held-out sources and `q=0.10` for one.

## Frozen TAILSENTRY transform

For ten frozen candidate means `m_1,...,m_10`, routed mean `r`, and routed scale `s`, independently at every horizon step:

```
L = empirical_quantile({m_j}, 0.125)
U = empirical_quantile({m_j}, 0.875)
r_guard = clip(r, L, U)
s_guard = sqrt(s^2 + (r - r_guard)^2)
```

No model feature, source label, future value, route probability, calibration outcome, or trainable parameter enters the transform. The candidate set and quantile interpolation are NumPy defaults, frozen by the source hash.

## Fresh public confirmation: M1

Source:

- CRAN `Mcomp` 2.8, repository commit `28f4e9babfe26607c9274c2c8d0e900a6aa61eec`;
- `data/M1.rda` expected MD5 `78a8fd28b46f657836f577b636ba70e4`.

Panel construction:

- use every M1 series with finite-span internal missingness <=15%;
- concatenate official history `x` and official future `xx`;
- require `len(full) >= 32 + 2*h`;
- create origins at `len(full)-2*h` and `len(full)-h`;
- keep at most the latest 512 context values;
- linearly interpolate internal gaps only after trimming leading/trailing non-finite values;
- retain the original M1 series ID, period, type, and horizon;
- use period mapping `YEARLY->1`, `QUARTERLY->4`, `MONTHLY->12`, `OTHER->1`;
- use all eligible series; there is no outcome-dependent series selection.

Primary macro weights the seven M1 economic/domain types equally. Period and period-by-type summaries are mandatory. Bootstrap resamples type, then series, then origin with seed `202607162101` for 10,000 draws.

## Fresh synthetic confirmation

Using untouched seed namespace `202607162201` through `202607162208`, evaluate 64 tasks in each of eight cells:

- supported ID at 64x16 and 128x32;
- compositional OOD at 64x16 and 128x48;
- irregular/missing OOD at 64x16;
- long horizon at 128x48;
- misspecified OOD at 64x16 and 32x32.

The frozen Stage 2I experts/router remain unchanged. Only TAILSENTRY is added after the routed forecast.

## Promotion gates

All gates are required:

1. Artifact/source integrity: all frozen source, model, policy, guard, and task hashes match.
2. Freshness: no M1 series or Stage 2J synthetic seed appeared in Stage 2H development or Stage 2I-AO.
3. Exact future-information firewall and bitwise prediction replay on a frozen 5% sample.
4. Public macro safety: guarded MASE is no more than 0.5% worse than unguarded Stage 2I.
5. Public tail repair: 95th-percentile per-series MASE is at least 10% lower than unguarded.
6. Annual repair: M1 yearly MASE is at least 5% lower than unguarded.
7. Catastrophic-risk repair: the fraction of series with MASE >5 is at least 20% lower than unguarded.
8. Domain safety: no M1 type is more than 2% worse than unguarded.
9. Empirical competitiveness: guarded macro MASE is no more than 3% worse than the best fixed candidate and no worse than seasonal naive.
10. Probabilistic safety: guarded WQL is no more than 1% worse than unguarded and central-80% coverage lies in [0.70, 0.90].
11. Synthetic support safety: guarded MASE regression is <=2% on both supported ID and compositional OOD.
12. Synthetic tail value: pooled 95th-percentile task MASE is at least 5% lower than unguarded on long/misspecified/irregular tasks.
13. Action non-triviality: at least 1% and at most 25% of public horizon steps are changed.
14. Deployment cost: added latency <=5%, memory <=1%, and no additional candidate/model fit.
15. Ablation completeness: unguarded Stage 2I, TAILSENTRY, all ten fixed candidates, and candidate median are reported.

A pass authorizes a source-disjoint public benchmark against accessible frontier forecasters. It does not itself establish frontier superiority. A failure authorizes only failure diagnosis in a new namespace; M1 outcomes may not be used for threshold tuning.
