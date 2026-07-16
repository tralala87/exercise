# Frozen Stage 2I Archival-Official M3 Adjudication

Frozen on 2026-07-16 after the exact live M3 URL returned HTTP 403 in both the local runtime and a clean GitHub-hosted Ubuntu runner, and after CRAN Mcomp 2.8 and the Monash/Zenodo TSF representation were found to disagree at exactly one value. It is frozen before an archived official M3 spreadsheet is downloaded, before any M3 representation is supplied to the forecasting policy, and before any canonical holdout metric is computed.

## Scientific purpose

Determine the canonical value source without tuning the model or choosing the representation that yields a favorable forecast. The adjudicator is an archived byte-for-byte capture of the official IIF M3C spreadsheet URL, not another transformed dataset.

## Immutable primary URL

`https://forecasters.org/data/m3comp/M3C.xls`

The current official M3 page still links this exact target, but the target is protected by a JavaScript security challenge. The recorded live-transport failure remains a failure; this addendum creates a separately labeled archival-official namespace.

## Frozen retrieval rule

1. Query the Internet Archive CDX index for all four exact URL variants: HTTP/HTTPS and with/without `www`.
2. Retain only HTTP-200 captures whose replay bytes begin with the OLE Compound File signature `d0cf11e0a1b11ae1` and whose workbook contains the expected M3 sheets.
3. Deduplicate valid captures by SHA-256 of the downloaded workbook bytes.
4. Use the earliest and latest capture of every distinct valid workbook digest. No timestamp may be selected based on data values.
5. Record the CDX response, requested and resolved replay URLs, timestamps, original URLs, byte counts, SHA-256 digests, and workbook validation results.
6. If no valid archived official workbook is available, stop. No third-party replacement is selected automatically.

## Frozen full-dataset adjudication rule

For each valid archived official workbook:

- read all four sheets (`M3Year`, `M3Quart`, `M3Month`, `M3Other`);
- require counts 645, 756, 1,428, and 174;
- require horizons 6, 8, 18, and 8;
- trim only trailing spreadsheet blanks exactly as in the original M3 loader;
- construct IDs `Y1..Y645`, `Q1..Q756`, `M1..M1428`, and `O1..O174` by sheet order;
- compare all 3,003 series elementwise against the pinned CRAN Mcomp 2.8 representation and the pinned Monash/Zenodo TSF representation.

A representation is authorized only if every valid archived official workbook agrees with it on every series length and every numeric value under `rtol=1e-12`, `atol=1e-10`. If all official captures do not select one representation unambiguously, stop. A one-cell patch or majority vote is forbidden.

## Downstream rule

After adjudication, construct the unchanged 700-task, 350-series, ten-cohort panel using the original SHA-256 selection rule and exact M4 URLs. Hash source bytes, comparison tables, selected representation, task inventory, and task values before forecasting.

The forecasting architecture, checkpoints, policies, thresholds, intervals, candidates, bootstrap seed, metrics, and the remaining canonical holdout gates are unchanged. Results must be labeled `Stage2I-AO` (archival official). They are not a literal pass of the original live-URL transport gate.
