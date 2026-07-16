from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_MD5 = {
    'M1.rda': '78a8fd28b46f657836f577b636ba70e4',
    'tourism.rda': '431f57effb90c94f9029ab50c8471646',
}
SOURCE_URLS = {
    'M1.rda': 'https://raw.githubusercontent.com/cran/Mcomp/28f4e9babfe26607c9274c2c8d0e900a6aa61eec/data/M1.rda',
    'tourism.rda': 'https://raw.githubusercontent.com/cran/Tcomp/8420cd2fcc2797d3b4ae7cc2c71dddbaf35010b7/data/tourism.rda',
}


def digest(path: Path, algorithm: str = 'sha256') -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: finalize_source_suite.py output_dir')
    root = Path(sys.argv[1])
    metadata_path = root / 'm1_tourism_metadata.csv'
    values_path = root / 'm1_tourism_values.csv.gz'
    checks = {}
    for name, expected in EXPECTED_MD5.items():
        actual = digest(root / name, 'md5')
        checks[f'{name}_md5'] = actual == expected
        if actual != expected:
            raise RuntimeError(f'{name} MD5 mismatch: {actual} != {expected}')
    metadata = pd.read_csv(metadata_path)
    checks['m1_series_count'] = int((metadata.benchmark == 'M1').sum()) == 1001
    checks['tourism_series_count'] = int((metadata.benchmark == 'Tourism').sum()) == 1311
    checks['six_required_cohorts'] = set(
        metadata.loc[metadata.period.isin(['YEARLY', 'QUARTERLY', 'MONTHLY']), ['benchmark', 'period']]
        .itertuples(index=False, name=None)
    ) == {
        ('M1', 'YEARLY'), ('M1', 'QUARTERLY'), ('M1', 'MONTHLY'),
        ('Tourism', 'YEARLY'), ('Tourism', 'QUARTERLY'), ('Tourism', 'MONTHLY'),
    }

    expected_rows = int(metadata.total_length.sum())
    observed_rows = 0
    all_values_finite = True
    observed_counts: Counter[tuple[str, str]] = Counter()
    required_columns = {'benchmark', 'series_id', 'position', 'value'}
    for chunk in pd.read_csv(values_path, chunksize=200_000):
        if not required_columns.issubset(chunk.columns):
            raise RuntimeError(f'value export columns missing: {required_columns - set(chunk.columns)}')
        observed_rows += len(chunk)
        all_values_finite &= bool(np.isfinite(pd.to_numeric(chunk.value, errors='coerce')).all())
        counts = chunk.groupby(['benchmark', 'series_id'], sort=False).size()
        for key, value in counts.items():
            observed_counts[(str(key[0]), str(key[1]))] += int(value)
    expected_counts = {
        (str(row.benchmark), str(row.series_id)): int(row.total_length)
        for row in metadata.itertuples(index=False)
    }
    checks['gzip_complete_and_row_count_exact'] = observed_rows == expected_rows
    checks['all_values_finite'] = all_values_finite
    checks['per_series_lengths_exact'] = observed_counts == expected_counts

    if not all(checks.values()):
        raise RuntimeError(f'validation failed: {checks}')
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file():
            files[path.name] = {
                'sha256': digest(path),
                'size_bytes': path.stat().st_size,
            }
    manifest = {
        'status': 'STAGE2K_SOURCE_SUITE_FROZEN_BEFORE_FORECASTING',
        'sources': SOURCE_URLS,
        'expected_md5': EXPECTED_MD5,
        'series': {
            'M1': 1001,
            'Tourism': 1311,
            'total': 2312,
        },
        'value_rows': observed_rows,
        'required_periods': ['YEARLY', 'QUARTERLY', 'MONTHLY'],
        'checks': checks,
        'files': files,
    }
    (root / 'SOURCE_SUITE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    decision = {
        'decision': 'SOURCE_SUITE_READY',
        'checks_passed': sum(checks.values()),
        'checks_total': len(checks),
        'metadata_rows': len(metadata),
        'value_rows': observed_rows,
        'values_sha256': digest(values_path),
        'metadata_sha256': digest(metadata_path),
        'manifest_sha256': digest(root / 'SOURCE_SUITE_MANIFEST.json'),
    }
    (root / 'RUN_DECISION.json').write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
