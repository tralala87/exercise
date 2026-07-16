# Frozen Stage 2M-R source-feasibility addendum

Frozen after the Stage 2M source suite was acquired and the panel builder stopped before any model prediction or outcome score.

## Observed feasibility blocker

The predeclared `CarParts` cohort contains 2,674 monthly series, all of length 51. The unchanged Stage 2M eligibility rule requires at least `32 + 2*12 = 56` observations. Therefore exactly zero CarParts series are eligible; the planned 30-series cohort is mathematically impossible.

No forecast, route, interval, metric, prediction lock, task outcome, or model comparison was produced before this addendum.

## Sole repair

Replace `CarParts` with `Electricity_Weekly`, using the pinned Monash TSF source:

`https://zenodo.org/record/4656141/files/electricity_weekly_dataset.zip`

The primary GluonTS Monash registry identifies this exact file and record. The replacement is selected from source metadata only, before its bytes are downloaded.

## Unchanged items

- CAPE-R1S-R weights, candidates, risk forest, route, and guard;
- BRACE point mean: `0.5*CAPE + 0.5*seasonal_naive`;
- ARC q70 replay statistic, pseudocount 64, and scale bounds [1.0, 1.5];
- 30 series per cohort, two origins, context 32–512;
- deterministic SHA-256 series selection;
- all 17 Stage 2M promotion gates;
- no source identity feature and no tuning on this panel.

The repaired experiment uses a new `Stage2M-R` namespace. The failed Stage 2M source-feasibility attempt remains immutable evidence and is not scored.
