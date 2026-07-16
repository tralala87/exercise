from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

SOURCES = {
    "CIF2016": "https://zenodo.org/record/4656042/files/cif_2016_dataset.zip",
    "NN5_Daily": "https://zenodo.org/record/4656117/files/nn5_daily_dataset_without_missing_values.zip",
    "NN5_Weekly": "https://zenodo.org/record/4656125/files/nn5_weekly_dataset.zip",
    "Hospital": "https://zenodo.org/record/4656014/files/hospital_dataset.zip",
    "CarParts": "https://zenodo.org/record/4656021/files/car_parts_dataset_without_missing_values.zip",
    "Solar_Weekly": "https://zenodo.org/record/4656151/files/solar_weekly_dataset.zip",
}
PRE_VALUE_SHA256 = "fb4989c9c0a782a771a61f32daaea1129bb5c9aeb76647c84a1e3808a78767ce"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path) -> dict:
    headers = {"User-Agent": "stage2m-source-lock/1.0"}
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
    raise RuntimeError(f"download failed for {url}: {error}")


def inspect_tsf(path: Path) -> dict:
    frequency = None
    horizon = None
    attributes = []
    series = 0
    found_data = False
    with path.open("r", encoding="cp1252") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@frequency"):
                frequency = line.split(maxsplit=1)[1]
            elif line.startswith("@horizon"):
                horizon = int(line.split(maxsplit=1)[1])
            elif line.startswith("@attribute"):
                parts = line.split()
                attributes.append({"name": parts[1], "type": parts[2]})
            elif line.startswith("@data"):
                found_data = True
            elif found_data and not line.startswith("@"):
                series += 1
    if not found_data or series <= 0:
        raise RuntimeError(f"invalid TSF file: {path}")
    return {
        "frequency": frequency,
        "global_horizon": horizon,
        "attributes": attributes,
        "series": series,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_source_suite.py PRE_VALUE_MANIFEST.json OUTPUT_DIR")
    pre_value = Path(sys.argv[1])
    root = Path(sys.argv[2])
    root.mkdir(parents=True, exist_ok=True)
    if sha256(pre_value) != PRE_VALUE_SHA256:
        raise RuntimeError("pre-value manifest hash mismatch")
    (root / "STAGE2M_PRE_VALUE_MANIFEST.json").write_bytes(pre_value.read_bytes())

    downloads = []
    inspections = {}
    for name, url in SOURCES.items():
        archive = root / f"{name}.zip"
        record = download(url, archive)
        downloads.append({"name": name, **record})
        extract = root / name
        extract.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".tsf")]
            if len(members) != 1:
                raise RuntimeError(f"{name}: expected one TSF, found {members}")
            zf.extract(members[0], extract)
            tsf = extract / members[0]
        inspections[name] = {"relative_path": str(tsf.relative_to(root)), **inspect_tsf(tsf)}

    manifest = {
        "status": "STAGE2M_SOURCE_SUITE_READY",
        "pre_value_manifest_sha256": PRE_VALUE_SHA256,
        "downloads": downloads,
        "datasets": inspections,
    }
    (root / "SOURCE_SUITE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    decision = {
        "decision": "SOURCE_SUITE_READY",
        "datasets": len(inspections),
        "total_series": sum(v["series"] for v in inspections.values()),
        "manifest_sha256": sha256(root / "SOURCE_SUITE_MANIFEST.json"),
        "pre_value_manifest_sha256": PRE_VALUE_SHA256,
    }
    (root / "RUN_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
