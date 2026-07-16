from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

MANIFEST_SHA256 = "dc5a6264fabdf60827e0e6ff26b4f357bcadfc04af3daf3334548e223d8d4207"
SPEC_SHA256 = "37d6ebe38c285928a0ad265442753acc423aa2d082023c15869a16642de1a8fd"
CRAN_URL = "https://raw.githubusercontent.com/cran/Mcomp/28f4e9babfe26607c9274c2c8d0e900a6aa61eec/data/M3.rda"
CRAN_MD5 = "f420fb522d3467b7fd2f96477350202b"
ZENODO = {
    "Yearly": "https://zenodo.org/api/records/4656222/files/m3_yearly_dataset.zip/content",
    "Quarterly": "https://zenodo.org/api/records/4656262/files/m3_quarterly_dataset.zip/content",
    "Monthly": "https://zenodo.org/api/records/4656298/files/m3_monthly_dataset.zip/content",
    "Other": "https://zenodo.org/api/records/4656335/files/m3_other_dataset.zip/content",
}
SHEET = {"Yearly": "M3Year", "Quarterly": "M3Quart", "Monthly": "M3Month", "Other": "M3Other"}
PREFIX = {"Yearly": "Y", "Quarterly": "Q", "Monthly": "M", "Other": "O"}
COUNT = {"Yearly": 645, "Quarterly": 756, "Monthly": 1428, "Other": 174}
HORIZON = {"Yearly": 6, "Quarterly": 8, "Monthly": 18, "Other": 8}
OLE = bytes.fromhex("d0cf11e0a1b11ae1")


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "requests": requests.__version__,
        "github_sha": os.getenv("GITHUB_SHA"),
        "github_ref": os.getenv("GITHUB_REF"),
    }


def download(session: requests.Session, url: str, path: Path, minimum: int = 1) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        partial = path.with_suffix(path.suffix + ".part")
        partial.unlink(missing_ok=True)
        try:
            with session.get(url, stream=True, timeout=(30, 300), allow_redirects=True) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        if chunk:
                            handle.write(chunk)
                if partial.stat().st_size < minimum:
                    raise RuntimeError(f"download too small: {partial.stat().st_size}")
                partial.replace(path)
                return {
                    "requested_url": url,
                    "resolved_url": response.url,
                    "status": int(response.status_code),
                    "bytes": int(path.stat().st_size),
                    "sha256": hash_file(path),
                    "content_type": response.headers.get("content-type"),
                    "attempt": attempt,
                    "path": str(path),
                }
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(float(2**attempt))
    raise RuntimeError(f"download failed: {url}") from last_error


def cdx_rows(session: requests.Session, endpoint: str, original: str) -> list[dict[str, str]]:
    params = [
        ("url", original),
        ("output", "json"),
        ("fl", "timestamp,original,statuscode,mimetype,digest,length"),
        ("filter", "statuscode:200"),
        ("collapse", "digest"),
    ]
    response = session.get(endpoint, params=params, timeout=(30, 120))
    response.raise_for_status()
    payload = response.json()
    if not payload or len(payload) < 2:
        return []
    header = payload[0]
    return [dict(zip(header, row)) for row in payload[1:]]


def parse_tsf(path: Path) -> list[np.ndarray]:
    attributes: list[tuple[str, str]] = []
    rows: list[np.ndarray] = []
    in_data = False
    with path.open("r", encoding="cp1252") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                parts = line.split(" ")
                if parts[0].lower() == "@attribute":
                    attributes.append((parts[1], parts[2]))
                elif parts[0].lower() == "@data":
                    in_data = True
                continue
            if not in_data:
                raise RuntimeError("TSF row before @data")
            fields = line.split(":")
            if len(fields) != len(attributes) + 1:
                raise RuntimeError("TSF field mismatch")
            rows.append(np.asarray([np.nan if value == "?" else float(value) for value in fields[-1].split(",")], dtype=np.float64))
    return rows


def load_zenodo(raw: Path) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for group in ZENODO:
        archive = raw / f"zenodo/{group}.zip"
        folder = raw / f"zenodo/{group}"
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zip_handle:
            names = [name for name in zip_handle.namelist() if name.lower().endswith(".tsf")]
            if len(names) != 1:
                raise RuntimeError(f"unexpected TSF members for {group}: {names}")
            zip_handle.extract(names[0], folder)
        arrays = parse_tsf(folder / names[0])
        if len(arrays) != COUNT[group]:
            raise RuntimeError(f"Zenodo {group} count mismatch")
        output[group] = {f"{PREFIX[group]}{index + 1}": values for index, values in enumerate(arrays)}
    return output


def load_cran(values_csv: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(values_csv)
    return {
        str(series_id): group.sort_values("position").value.to_numpy(np.float64)
        for series_id, group in frame.groupby("series_id", sort=False)
    }


def load_xls(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    if path.read_bytes()[:8] != OLE:
        raise RuntimeError("not an OLE workbook")
    book = pd.ExcelFile(path, engine="xlrd")
    expected_sheets = set(SHEET.values())
    if not expected_sheets.issubset(set(book.sheet_names)):
        raise RuntimeError(f"missing sheets: {expected_sheets - set(book.sheet_names)}")
    output: dict[str, dict[str, np.ndarray]] = {}
    checks: dict[str, Any] = {}
    for group, sheet in SHEET.items():
        frame = pd.read_excel(book, sheet_name=sheet)
        if len(frame) != COUNT[group]:
            raise RuntimeError(f"{sheet}: expected {COUNT[group]} rows, observed {len(frame)}")
        if not (pd.to_numeric(frame["NF"], errors="coerce") == HORIZON[group]).all():
            raise RuntimeError(f"{sheet}: horizon metadata mismatch")
        values: dict[str, np.ndarray] = {}
        for index, (_, row) in enumerate(frame.iterrows(), start=1):
            array = pd.to_numeric(row.iloc[6:], errors="coerce").to_numpy(np.float64)
            finite = np.flatnonzero(np.isfinite(array))
            if len(finite) == 0:
                raise RuntimeError(f"{sheet} row {index}: no values")
            array = array[: finite[-1] + 1]
            values[f"{PREFIX[group]}{index}"] = array
        output[group] = values
        checks[group] = {"rows": len(frame), "sheet": sheet, "horizon": HORIZON[group]}
    return output, checks


def flatten(groups: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {series_id: values for group in groups.values() for series_id, values in group.items()}


def compare(left: dict[str, np.ndarray], right: dict[str, np.ndarray], atol: float = 1e-10, rtol: float = 1e-12) -> dict[str, Any]:
    if set(left) != set(right):
        return {"id_sets_match": False, "left_ids": len(left), "right_ids": len(right)}
    mismatches: list[dict[str, Any]] = []
    for series_id in sorted(left):
        left_values = left[series_id]
        right_values = right[series_id]
        if len(left_values) != len(right_values):
            mismatches.append({"series_id": series_id, "kind": "length", "left": len(left_values), "right": len(right_values)})
            continue
        mask = ~np.isclose(left_values, right_values, atol=atol, rtol=rtol, equal_nan=True)
        for position in np.flatnonzero(mask):
            mismatches.append(
                {
                    "series_id": series_id,
                    "position": int(position),
                    "left": float(left_values[position]),
                    "right": float(right_values[position]),
                    "absolute_delta": float(abs(left_values[position] - right_values[position])),
                }
            )
    return {"id_sets_match": True, "series": len(left), "mismatch_count": len(mismatches), "mismatches": mismatches}


def write_manifest(directory: Path) -> str:
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256")
    manifest = directory / "MANIFEST.sha256"
    manifest.write_text("".join(f"{hash_file(path)}  {path.relative_to(directory)}\n" for path in files))
    return hash_file(manifest)


def run(output: Path, manifest_path: Path, spec_path: Path, extractor: Path) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    raw = output / "raw"
    evidence = output / "evidence"
    diagnostic = output / "diagnostic"
    for directory in (raw, evidence, diagnostic):
        directory.mkdir(parents=True, exist_ok=True)
    transport: list[dict[str, Any]] = []
    started = time.time()
    try:
        if hash_file(manifest_path) != MANIFEST_SHA256:
            raise RuntimeError("adjudication manifest hash mismatch")
        if hash_file(spec_path) != SPEC_SHA256:
            raise RuntimeError("adjudication spec hash mismatch")
        manifest = json.loads(manifest_path.read_text())
        shutil.copy2(manifest_path, evidence / manifest_path.name)
        shutil.copy2(spec_path, evidence / spec_path.name)

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Stage2I archival source audit)", "Accept": "*/*"})

        cran = raw / "cran/M3.rda"
        transport.append(download(session, CRAN_URL, cran, 100_000))
        if hash_file(cran, "md5") != CRAN_MD5:
            raise RuntimeError("CRAN M3.rda MD5 mismatch")
        cran_values = raw / "cran/m3_values.csv.gz"
        cran_metadata = raw / "cran/m3_metadata.csv"
        subprocess.run(["Rscript", str(extractor), str(cran), str(cran_values), str(cran_metadata)], check=True)
        cran_map = load_cran(cran_values)

        for group, url in ZENODO.items():
            transport.append(download(session, url, raw / f"zenodo/{group}.zip", 1_000))
        zenodo_map = flatten(load_zenodo(raw))

        cdx_records: list[dict[str, str]] = []
        cdx_errors: list[dict[str, str]] = []
        for original in manifest["official_url_variants"]:
            try:
                rows = cdx_rows(session, manifest["archive_index"], original)
                for row in rows:
                    row["query_original"] = original
                    cdx_records.append(row)
            except Exception as exc:
                cdx_errors.append({"original": original, "error_type": type(exc).__name__, "error": str(exc)})
        (evidence / "cdx_records.json").write_text(json.dumps(cdx_records, indent=2, sort_keys=True) + "\n")
        (evidence / "cdx_errors.json").write_text(json.dumps(cdx_errors, indent=2, sort_keys=True) + "\n")
        if not cdx_records:
            raise RuntimeError(f"no archived official snapshots returned; errors={cdx_errors}")

        by_digest: dict[str, dict[str, str]] = {}
        for row in cdx_records:
            by_digest.setdefault(row.get("digest") or f"timestamp:{row['timestamp']}", row)
        accepted: list[dict[str, Any]] = []
        workbook_maps: list[dict[str, np.ndarray]] = []
        rejected: list[dict[str, Any]] = []
        for index, (archive_digest, row) in enumerate(sorted(by_digest.items())):
            replay = manifest["archive_replay_template"].format(timestamp=row["timestamp"], original=row["original"])
            path = raw / f"archive/M3C_{index:02d}_{row['timestamp']}.xls"
            try:
                record = download(session, replay, path, 100_000)
                transport.append(record)
                groups, sheet_checks = load_xls(path)
                mapping = flatten(groups)
                workbook_maps.append(mapping)
                accepted.append(
                    {
                        "archive_digest": archive_digest,
                        "timestamp": row["timestamp"],
                        "original": row["original"],
                        "replay_url": replay,
                        "path": str(path),
                        "sha256": hash_file(path),
                        "bytes": path.stat().st_size,
                        "sheet_checks": sheet_checks,
                    }
                )
            except Exception as exc:
                rejected.append(
                    {
                        "archive_digest": archive_digest,
                        "timestamp": row.get("timestamp"),
                        "original": row.get("original"),
                        "replay_url": replay,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        (evidence / "archive_candidates_accepted.json").write_text(json.dumps(accepted, indent=2, sort_keys=True) + "\n")
        (evidence / "archive_candidates_rejected.json").write_text(json.dumps(rejected, indent=2, sort_keys=True) + "\n")
        if not workbook_maps:
            raise RuntimeError("no valid archived official workbook")

        archive_vs_cran: list[dict[str, Any]] = []
        archive_vs_zenodo: list[dict[str, Any]] = []
        conflict_values: list[float] = []
        for candidate, mapping in zip(accepted, workbook_maps):
            against_cran = compare(mapping, cran_map)
            against_zenodo = compare(mapping, zenodo_map)
            archive_vs_cran.append({"sha256": candidate["sha256"], **against_cran})
            archive_vs_zenodo.append({"sha256": candidate["sha256"], **against_zenodo})
            conflict_values.append(float(mapping["M1385"][83]))
        (evidence / "archive_vs_cran.json").write_text(json.dumps(archive_vs_cran, indent=2, sort_keys=True) + "\n")
        (evidence / "archive_vs_zenodo.json").write_text(json.dumps(archive_vs_zenodo, indent=2, sort_keys=True) + "\n")

        unanimous = len(set(conflict_values)) == 1
        selected_value = conflict_values[0] if unanimous else None
        expected_conflict = manifest["known_conflict"]
        declared_pair = {float(expected_conflict["cran_mcomp_value"]), float(expected_conflict["zenodo_tsf_value"])}
        selected_declared = selected_value in declared_pair if selected_value is not None else False
        source_match_flags: list[bool] = []
        for left, right in zip(archive_vs_cran, archive_vs_zenodo):
            cran_exact = left.get("mismatch_count") == 0
            zenodo_exact = right.get("mismatch_count") == 0
            cran_one = left.get("mismatch_count") == 1 and left.get("mismatches") == [
                {"series_id": "M1385", "position": 83, "left": -1200.0, "right": 1200.0, "absolute_delta": 2400.0}
            ]
            zenodo_one = right.get("mismatch_count") == 1 and right.get("mismatches") == [
                {"series_id": "M1385", "position": 83, "left": 1200.0, "right": -1200.0, "absolute_delta": 2400.0}
            ]
            source_match_flags.append(bool((cran_exact and zenodo_one) or (zenodo_exact and cran_one)))

        checks = {
            "manifest_exact": hash_file(evidence / manifest_path.name) == MANIFEST_SHA256,
            "spec_exact": hash_file(evidence / spec_path.name) == SPEC_SHA256,
            "cran_md5_exact": hash_file(cran, "md5") == CRAN_MD5,
            "at_least_one_valid_archived_workbook": len(workbook_maps) >= 1,
            "all_archived_conflict_values_unanimous": unanimous,
            "selected_value_is_predeclared_alternative": selected_declared,
            "no_undeclared_semantic_mismatch": bool(all(source_match_flags)),
        }
        if not all(checks.values()):
            raise RuntimeError(f"adjudication gates failed: {checks}")
        selected_source = "CRAN_Mcomp" if selected_value == 1200.0 else "Monash_Zenodo_TSF"
        decision = {
            "status": "STAGE2I_SA_M3_CONFLICT_ADJUDICATED",
            "selected_value": selected_value,
            "selected_existing_representation": selected_source,
            "accepted_archived_workbooks": accepted,
            "rejected_archived_candidates": rejected,
            "checks": checks,
            "environment": environment(),
            "caveat": "The live exact URL remains transport-blocked; authority comes from archived official workbook bytes.",
        }
        (evidence / "ADJUDICATION_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
        decision["evidence_manifest_sha256"] = write_manifest(evidence)
        (output / "RUN_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        decision = {
            "status": "BLOCKED_M3_CONFLICT_ADJUDICATION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "transport": transport,
            "environment": environment(),
            "elapsed_seconds": time.time() - started,
            "policy": "No unrecorded third source was substituted.",
        }
        (diagnostic / "ADJUDICATION_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
        (output / "RUN_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output.resolve(), arguments.manifest.resolve(), arguments.spec.resolve(), arguments.extractor.resolve())


if __name__ == "__main__":
    main()
