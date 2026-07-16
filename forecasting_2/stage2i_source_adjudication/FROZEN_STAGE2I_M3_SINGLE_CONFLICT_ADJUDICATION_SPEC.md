# Frozen Stage 2I M3 Single-Conflict Adjudication Specification

Frozen after the exact official M3 transport returned HTTP 403 and after the preregistered CRAN-versus-Zenodo semantic-equivalence gate identified exactly one conflicting cell, but before any third representation is accessed and before any canonical forecast outcome is computed.

## Known evidence at freeze time

- Exact official URL: `https://forecasters.org/data/m3comp/M3C.xls`.
- CRAN `Mcomp` 2.8 and the four Monash/Zenodo TSF archives agree on all series IDs, group counts, lengths, horizons, and all but one numeric cell.
- The only conflict is monthly canonical series `M1385`, zero-based position `83`:
  - CRAN `Mcomp`: `+1200.0`
  - Monash/Zenodo TSF: `-1200.0`
- No model prediction, route, loss, or canonical metric has been computed from either alternative.

## Adjudication source hierarchy

1. Query the Internet Archive CDX index for snapshots of the exact official URL and its HTTP/HTTPS variants.
2. Download each distinct archived digest through an `id_` replay URL.
3. Accept a candidate only if it is an OLE Excel workbook, contains the four expected sheets, has exactly 3,003 series with the expected group counts, and passes the original M3 horizon metadata checks.
4. If at least one valid archived official workbook exists, its cell is authoritative. Multiple distinct workbook digests must agree on the conflict; otherwise the gate fails.
5. A valid official workbook must match one existing representation on every cell and differ from the other only at the already declared conflict. Any additional disagreement fails the gate.
6. If no valid archived official workbook is retrievable, stop. Do not substitute another third source automatically.

## Promotion rule

The gate passes only when all accepted archived official workbooks unanimously select either `+1200.0` or `-1200.0`, and the selected workbook has no additional semantic mismatch against the corresponding full M3 representation.

A pass authorizes a new `Stage2I-SA` source-adjudicated data namespace. The unchanged Stage 2I forecasting policy, thresholds, metrics, and remaining holdout gates may then run on a 700-task panel built from the archived official workbook plus the exact M4 sources.

The original exact-live-URL gate remains recorded as transport-blocked; it is not retroactively reclassified.
