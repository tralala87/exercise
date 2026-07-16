from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests

PRE_VALUE_SHA256 = "0965409e0f72e601f8a98dcf8dc76f4686005272915c2cd5f5a1adf7af87a7f6"
ADDENDUM_SHA256 = "3e1f8afcdf39bc9311dc3c56af002fe9edce4b4078dc21e6a54fedbeddbb78f4"
URL = "https://zenodo.org/record/4656141/files/electricity_weekly_dataset.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path) -> dict:
    headers = {"User-Agent": "stage2mr-source-repair/1.0"}
    error = None
    for attempt in range(1, 5):
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for block in response.iter_content(1 << 20):
                        if block:
                            handle.write(block)
                return {
                    "requested_url": url,
                    "resolved_url": response.url,
                    "status": response.status_code,
                    "content_type": response.headers.get("content-type"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "attempt": attempt,
                }
        except Exception as exc:
            error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed: {error}")


def parse_tsf(path: Path) -> dict:
    attrs = []
    frequency = None
    global_horizon = None
    in_data = False
    lengths = []
    missing_fractions = []
    with path.open("r", encoding="cp1252") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@attribute"):
                parts = line.split()
                attrs.append((parts[1], parts[2]))
            elif line.startswith("@frequency"):
                frequency = line.split(maxsplit=1)[1]
            elif line.startswith("@horizon"):
                global_horizon = int(line.split(maxsplit=1)[1])
            elif line.startswith("@data"):
                in_data = True
            elif in_data and not line.startswith("@"):
                parts = line.split(":")
                values = parts[-1].split(",")
                finite = np.asarray([v != "?" and v != "" for v in values], dtype=bool)
                if finite.any():
                    first, last = np.flatnonzero(finite)[[0, -1]]
                    span = values[int(first): int(last) + 1]
                    lengths.append(len(span))
                    missing_fractions.append(sum(v == "?" or v == "" for v in span) / len(span))
    horizon = global_horizon if global_horizon is not None else 8
    eligible = sum(length >= 32 + 2 * horizon and missing <= 0.15 for length, missing in zip(lengths, missing_fractions))
    if eligible < 30:
        raise RuntimeError(f"Electricity_Weekly has only {eligible} eligible series")
    return {
        "frequency": frequency,
        "global_horizon": global_horizon,
        "frozen_fallback_horizon": 8,
        "series": len(lengths),
        "eligible_series": eligible,
        "minimum_length": min(lengths),
        "median_length": float(np.median(lengths)),
        "maximum_length": max(lengths),
        "attributes": attrs,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: build_replacement.py PRE_VALUE ADDENDUM OUTPUT")
    pre_value, addendum, root = map(Path, sys.argv[1:])
    root.mkdir(parents=True, exist_ok=True)
    if sha256(pre_value) != PRE_VALUE_SHA256:
        raise RuntimeError("pre-value manifest hash mismatch")
    if sha256(addendum) != ADDENDUM_SHA256:
        raise RuntimeError("source-feasibility addendum hash mismatch")
    (root / pre_value.name).write_bytes(pre_value.read_bytes())
    (root / addendum.name).write_bytes(addendum.read_bytes())
    archive = root / "Electricity_Weekly.zip"
    transport = download(URL, archive)
    extract = root / "Electricity_Weekly"
    extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".tsf")]
        if len(members) != 1:
            raise RuntimeError(f"expected one TSF, found {members}")
        zf.extract(members[0], extract)
        tsf = extract / members[0]
    inspection = parse_tsf(tsf)
    manifest = {
        "status": "STAGE2MR_REPLACEMENT_SOURCE_READY_BEFORE_PREDICTIONS",
        "pre_value_manifest_sha256": PRE_VALUE_SHA256,
        "source_feasibility_addendum_sha256": ADDENDUM_SHA256,
        "replacement": "Electricity_Weekly",
        "transport": transport,
        "dataset": {"relative_path": str(tsf.relative_to(root)), **inspection},
        "model_policy_metric_gate_changes": "none",
    }
    (root / "REPLACEMENT_SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    decision = {
        "decision": "REPLACEMENT_SOURCE_READY",
        "eligible_series": inspection["eligible_series"],
        "series": inspection["series"],
        "manifest_sha256": sha256(root / "REPLACEMENT_SOURCE_MANIFEST.json"),
    }
    (root / "RUN_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
