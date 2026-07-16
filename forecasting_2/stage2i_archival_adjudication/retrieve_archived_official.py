from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

EXPECTED_MANIFEST_SHA256 = "da98603e38d1cfc1326ddab94a2a8a6aa30c6116c746f3eb5dbc6cd68c08719c"
EXPECTED_ADDENDUM_SHA256 = "3edfedf2e0c0cc73484fb4ca36b4dcf3ab4c70250be8508fb96e0f1d52310443"
OLE = bytes.fromhex("d0cf11e0a1b11ae1")
EXPECTED_COUNTS = {"M3Year": 645, "M3Quart": 756, "M3Month": 1428, "M3Other": 174}
EXPECTED_HORIZONS = {"M3Year": 6, "M3Quart": 8, "M3Month": 18, "M3Other": 8}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def request_with_retries(session: requests.Session, url: str, *, stream: bool = False) -> requests.Response:
    last: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = session.get(url, stream=stream, timeout=(30, 300), allow_redirects=True)
            response.raise_for_status()
            return response
        except Exception as exc:
            last = exc
            if attempt < 4:
                time.sleep(float(2**attempt))
    raise RuntimeError(f"request failed: {url}") from last


def query_cdx(session: requests.Session, endpoint: str, variants: list[str], output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    raw_queries: list[dict[str, Any]] = []
    fields = ["timestamp", "original", "statuscode", "mimetype", "digest", "length"]
    for index, variant in enumerate(variants):
        params = (
            f"url={quote(variant, safe='')}&output=json&filter=statuscode%3A200"
            f"&fl={','.join(fields)}&from=1990&to=2026"
        )
        url = f"{endpoint}?{params}"
        try:
            response = request_with_retries(session, url)
            text = response.text
            (output_dir / f"cdx_query_{index}.json").write_text(text)
            payload = response.json()
            if payload and isinstance(payload[0], list):
                header = payload[0]
                for record in payload[1:]:
                    row = {str(key): str(value) for key, value in zip(header, record)}
                    row["query_variant"] = variant
                    rows.append(row)
            raw_queries.append({"url": url, "status": response.status_code, "resolved_url": response.url, "rows": max(0, len(payload)-1)})
        except Exception as exc:
            raw_queries.append({"url": url, "error_type": type(exc).__name__, "error": str(exc)})
    (output_dir / "cdx_query_summary.json").write_text(json.dumps(raw_queries, indent=2, sort_keys=True) + "\n")
    return rows


def query_availability(session: requests.Session, variants: list[str], output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    probes = ["20000101000000", "20050101000000", "20100101000000", "20150101000000", "20200101000000", "20260101000000"]
    raw: list[dict[str, Any]] = []
    for variant in variants:
        for timestamp in probes:
            url = f"https://archive.org/wayback/available?url={quote(variant, safe='')}&timestamp={timestamp}"
            try:
                response = request_with_retries(session, url)
                payload = response.json()
                raw.append({"url": url, "payload": payload})
                closest = payload.get("archived_snapshots", {}).get("closest", {})
                if closest.get("available") and str(closest.get("status")) == "200":
                    rows.append({
                        "timestamp": str(closest.get("timestamp")),
                        "original": variant,
                        "statuscode": "200",
                        "mimetype": "unknown",
                        "digest": "availability-api",
                        "length": "unknown",
                        "query_variant": variant,
                    })
            except Exception as exc:
                raw.append({"url": url, "error_type": type(exc).__name__, "error": str(exc)})
    (output_dir / "availability_queries.json").write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return rows


def select_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unique = {(r.get("timestamp", ""), r.get("original", ""), r.get("digest", "")): r for r in rows if r.get("timestamp") and r.get("original")}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in unique.values():
        key = row.get("digest") or f"no-digest::{row.get('original')}"
        grouped[key].append(row)
    selected: list[dict[str, str]] = []
    for group in grouped.values():
        group.sort(key=lambda r: (r["timestamp"], r["original"]))
        selected.append(group[0])
        if group[-1] is not group[0]:
            selected.append(group[-1])
    selected.sort(key=lambda r: (r["timestamp"], r["original"]))
    return selected


def download_snapshot(session: requests.Session, row: dict[str, str], destination: Path) -> dict[str, Any]:
    timestamp, original = row["timestamp"], row["original"]
    urls = [
        f"https://web.archive.org/web/{timestamp}id_/{original}",
        f"https://web.archive.org/web/{timestamp}if_/{original}",
    ]
    errors: list[dict[str, str]] = []
    for url in urls:
        try:
            response = request_with_retries(session, url, stream=True)
            partial = destination.with_suffix(".part")
            partial.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("wb") as handle:
                for chunk in response.iter_content(1 << 20):
                    if chunk:
                        handle.write(chunk)
            prefix = partial.read_bytes()[: max(160, len(OLE))]
            if prefix[:8] != OLE:
                raise RuntimeError(f"non-OLE replay prefix={prefix[:120]!r}")
            partial.replace(destination)
            return {
                "requested_url": url,
                "resolved_url": response.url,
                "status": response.status_code,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "timestamp": timestamp,
                "original": original,
                "cdx_digest": row.get("digest"),
                "cdx_length": row.get("length"),
                "path": str(destination),
            }
        except Exception as exc:
            errors.append({"url": url, "error_type": type(exc).__name__, "error": str(exc)})
            destination.with_suffix(".part").unlink(missing_ok=True)
    raise RuntimeError(json.dumps(errors))


def validate_workbook(path: Path) -> dict[str, Any]:
    workbook = pd.ExcelFile(path, engine="xlrd")
    result: dict[str, Any] = {"sheets": workbook.sheet_names, "sheet_checks": {}}
    for sheet, expected_count in EXPECTED_COUNTS.items():
        if sheet not in workbook.sheet_names:
            raise RuntimeError(f"missing sheet {sheet}")
        frame = pd.read_excel(path, sheet_name=sheet, engine="xlrd")
        horizons = pd.to_numeric(frame["NF"], errors="coerce")
        values = frame.iloc[:, 6:].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        finite_counts = np.isfinite(values).sum(axis=1)
        check = {
            "rows": len(frame),
            "expected_rows": expected_count,
            "horizon_unique": sorted(int(v) for v in horizons.dropna().unique()),
            "expected_horizon": EXPECTED_HORIZONS[sheet],
            "all_lengths_match_N": bool(np.array_equal(finite_counts, pd.to_numeric(frame["N"], errors="coerce").to_numpy(np.int64))),
        }
        result["sheet_checks"][sheet] = check
        if len(frame) != expected_count or check["horizon_unique"] != [EXPECTED_HORIZONS[sheet]] or not check["all_lengths_match_N"]:
            raise RuntimeError(f"workbook validation failed: {sheet}: {check}")
    result["valid"] = True
    return result


def run(output_root: Path, manifest_path: Path, addendum_path: Path) -> dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    diagnostic = output_root / "diagnostic"
    snapshots = output_root / "snapshots"
    diagnostic.mkdir(); snapshots.mkdir()
    started = time.time()
    try:
        if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
            raise RuntimeError("archival manifest hash mismatch")
        if sha256(addendum_path) != EXPECTED_ADDENDUM_SHA256:
            raise RuntimeError("archival addendum hash mismatch")
        manifest = json.loads(manifest_path.read_text())
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Stage2I archival source audit)", "Accept": "*/*"})
        rows = query_cdx(session, manifest["wayback_cdx"], manifest["url_variants"], diagnostic)
        if not rows:
            rows = query_availability(session, manifest["url_variants"], diagnostic)
        (diagnostic / "all_capture_rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        selected = select_rows(rows)
        (diagnostic / "selected_capture_rows.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
        if not selected:
            raise RuntimeError("no archived captures discovered")
        attempts: list[dict[str, Any]] = []
        valid: list[dict[str, Any]] = []
        for index, row in enumerate(selected):
            destination = snapshots / f"capture_{index:03d}_{row['timestamp']}.xls"
            try:
                record = download_snapshot(session, row, destination)
                record["workbook"] = validate_workbook(destination)
                attempts.append({"status": "valid", **record})
                valid.append(record)
            except Exception as exc:
                attempts.append({"status": "invalid", "row": row, "error_type": type(exc).__name__, "error": str(exc)})
        (diagnostic / "capture_attempts.json").write_text(json.dumps(attempts, indent=2, sort_keys=True) + "\n")
        if not valid:
            raise RuntimeError("no valid archived official workbook downloaded")
        by_hash: dict[str, dict[str, Any]] = {}
        for record in valid:
            by_hash.setdefault(record["sha256"], record)
        decision = {
            "status": "ARCHIVED_OFFICIAL_WORKBOOKS_ACQUIRED",
            "valid_capture_count": len(valid),
            "distinct_workbook_sha256_count": len(by_hash),
            "distinct_workbooks": list(by_hash.values()),
            "manifest_sha256": sha256(manifest_path),
            "addendum_sha256": sha256(addendum_path),
            "elapsed_seconds": time.time() - started,
        }
        (output_root / "ARCHIVAL_ACQUISITION_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        decision = {
            "status": "BLOCKED_ARCHIVAL_OFFICIAL_ACQUISITION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
            "policy": "No third-party source was substituted.",
        }
        (output_root / "ARCHIVAL_ACQUISITION_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root.resolve(), args.manifest.resolve(), args.addendum.resolve())
