from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
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

ORIGINAL_MANIFEST_SHA256 = "977f3db34bdd9e45940190858889494a1bfa19d6af2e4f3d7ab3984f30cb64a9"
BRIDGE_MANIFEST_SHA256 = "1da4e0b028e75c9c4d01d2ea29d3092e40ea696c021f043fe9027db84694aae7"
ADDENDUM_SHA256 = "ca7905c2683f341c300b7b1df693f3254446730d6ea621011c422271879fb57b"
EXPECTED_M3_MD5 = "f420fb522d3467b7fd2f96477350202b"
EXPECTED_COUNTS = {"Yearly": 645, "Quarterly": 756, "Monthly": 1428, "Other": 174}
PREFIX = {"Yearly": "Y", "Quarterly": "Q", "Monthly": "M", "Other": "O"}
PERIOD = {"Yearly": "YEARLY", "Quarterly": "QUARTERLY", "Monthly": "MONTHLY", "Other": "OTHER"}
HORIZON = {"Yearly": 6, "Quarterly": 8, "Monthly": 18, "Other": 8}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def download(session: requests.Session, url: str, destination: Path, minimum: int) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    error: Exception | None = None
    for attempt in range(1, 5):
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.unlink(missing_ok=True)
        try:
            with session.get(url, stream=True, timeout=(30, 600), allow_redirects=True) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        if chunk:
                            handle.write(chunk)
                if partial.stat().st_size < minimum:
                    raise RuntimeError(f"download too small: {partial.stat().st_size} bytes")
                partial.replace(destination)
                return {
                    "requested_url": url,
                    "resolved_url": response.url,
                    "status": int(response.status_code),
                    "content_type": response.headers.get("content-type"),
                    "etag": response.headers.get("etag"),
                    "last_modified": response.headers.get("last-modified"),
                    "bytes": int(destination.stat().st_size),
                    "sha256": digest(destination),
                    "attempt": attempt,
                    "path": str(destination),
                }
        except Exception as exc:
            error = exc
            partial.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(float(2**attempt))
    raise RuntimeError(f"source download failed: {url}") from error


def unzip_single_tsf(archive: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".tsf")]
        if len(members) != 1:
            raise RuntimeError(f"expected one TSF in {archive.name}, found {members}")
        zf.extract(members[0], output_dir)
    return output_dir / members[0]


def parse_tsf(path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    attributes: list[tuple[str, str]] = []
    rows: list[np.ndarray] = []
    metadata: dict[str, Any] = {}
    in_data = False
    with path.open("r", encoding="cp1252") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                parts = line.split(" ")
                key = parts[0].lower()
                if key == "@attribute":
                    if len(parts) != 3:
                        raise ValueError(f"invalid attribute line: {line}")
                    attributes.append((parts[1], parts[2]))
                elif key == "@data":
                    in_data = True
                elif len(parts) == 2:
                    metadata[key[1:]] = parts[1]
                continue
            if not in_data:
                raise ValueError("data row before @data")
            fields = line.split(":")
            if len(fields) != len(attributes) + 1:
                raise ValueError(f"TSF field mismatch: {len(fields)} vs {len(attributes)+1}")
            values = fields[-1].split(",")
            array = np.asarray([np.nan if value == "?" else float(value) for value in values], dtype=np.float64)
            rows.append(array)
    if not rows:
        raise RuntimeError(f"no TSF rows parsed from {path}")
    return rows, metadata


def clean(values: np.ndarray) -> tuple[np.ndarray, float]:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    finite = np.isfinite(series.to_numpy())
    if not finite.any():
        return np.array([], dtype=np.float64), 1.0
    first, last = np.flatnonzero(finite)[[0, -1]]
    series = series.iloc[int(first) : int(last) + 1].reset_index(drop=True)
    missing = float(series.isna().mean())
    series = series.interpolate(method="linear", limit_direction="both")
    return series.to_numpy(np.float64), missing


def load_cran_export(values_path: Path, metadata_path: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    values = pd.read_csv(values_path)
    metadata = pd.read_csv(metadata_path, dtype={"series_id": str})
    mapping = {
        str(series_id): group.sort_values("position").value.to_numpy(np.float64)
        for series_id, group in values.groupby("series_id", sort=False)
    }
    return mapping, metadata


def compare_m3(
    cran: dict[str, np.ndarray],
    cran_meta: pd.DataFrame,
    zenodo: dict[str, dict[str, np.ndarray]],
    rtol: float,
    atol: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_ids: list[str] = []
    for group, count in EXPECTED_COUNTS.items():
        expected_ids.extend(f"{PREFIX[group]}{index}" for index in range(1, count + 1))
        if len(zenodo[group]) != count:
            raise RuntimeError(f"Zenodo {group}: expected {count}, observed {len(zenodo[group])}")
    if len(cran) != 3003 or set(cran) != set(expected_ids):
        raise RuntimeError(f"CRAN ID set mismatch: count={len(cran)}")
    meta_index = cran_meta.set_index("series_id")
    for group, count in EXPECTED_COUNTS.items():
        for index in range(1, count + 1):
            series_id = f"{PREFIX[group]}{index}"
            left = np.asarray(cran[series_id], np.float64)
            right = np.asarray(zenodo[group][series_id], np.float64)
            same_length = len(left) == len(right)
            if same_length:
                absolute = np.abs(left - right)
                max_abs = float(np.nanmax(absolute)) if len(absolute) else 0.0
                scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
                max_rel = float(np.nanmax(absolute / scale)) if len(absolute) else 0.0
                matched = bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
            else:
                max_abs = float("inf")
                max_rel = float("inf")
                matched = False
            meta = meta_index.loc[series_id]
            expected_horizon = HORIZON[group]
            metadata_ok = (
                str(meta.period).upper() == PERIOD[group]
                and int(meta.horizon) == expected_horizon
                and int(meta.total_length) == len(left)
            )
            rows.append(
                {
                    "group": group,
                    "series_id": series_id,
                    "length_cran": len(left),
                    "length_zenodo": len(right),
                    "horizon_cran": int(meta.horizon),
                    "max_abs_delta": max_abs,
                    "max_relative_delta": max_rel,
                    "value_match": matched,
                    "metadata_match": bool(metadata_ok),
                }
            )
    frame = pd.DataFrame(rows)
    checks = {
        "series_exactly_3003": len(frame) == 3003,
        "group_counts_exact": frame.groupby("group").size().to_dict() == EXPECTED_COUNTS,
        "all_lengths_match": bool((frame.length_cran == frame.length_zenodo).all()),
        "all_horizons_match": bool(frame.metadata_match.all()),
        "all_values_match": bool(frame.value_match.all()),
        "max_abs_delta": float(frame.max_abs_delta.max()),
        "max_relative_delta": float(frame.max_relative_delta.max()),
        "rtol": rtol,
        "atol": atol,
    }
    if not all(checks[key] for key in ("series_exactly_3003", "group_counts_exact", "all_lengths_match", "all_horizons_match", "all_values_match")):
        raise RuntimeError(f"M3 source equivalence failed: {checks}")
    return frame, checks


def parse_numeric_cells(cells: list[str]) -> np.ndarray:
    output = np.empty(len(cells), dtype=np.float64)
    for index, cell in enumerate(cells):
        text = cell.strip()
        output[index] = np.nan if text == "" else float(text)
    return output


def m4_test_map(path: Path) -> dict[str, np.ndarray]:
    mapping: dict[str, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            mapping[str(row[0])] = parse_numeric_cells(row[1:])
    return mapping


def select_m4(dataset: str, spec: dict[str, Any], train_path: Path, test_path: Path) -> tuple[list[tuple], int]:
    test = m4_test_map(test_path)
    heap: list[tuple[int, str, np.ndarray, float, str]] = []
    eligible = 0
    target = int(spec["target_series"])
    with train_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            series_id = str(row[0])
            if series_id not in test:
                raise RuntimeError(f"M4 test row missing for {series_id}")
            raw = np.concatenate([parse_numeric_cells(row[1:]), test[series_id]])
            values, missing = clean(raw)
            if missing <= 0.15 and len(values) >= 32 + 2 * int(spec["horizon"]) and np.isfinite(values).all():
                eligible += 1
                rank = hashlib.sha256(f"{dataset}::{series_id}".encode()).hexdigest()
                key = int(rank, 16)
                item = (-key, series_id, values, missing, rank)
                if len(heap) < target:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
    chosen = [(rank, series_id, values, missing) for _, series_id, values, missing, rank in heap]
    chosen.sort(key=lambda item: item[0])
    if len(chosen) != target:
        raise RuntimeError(f"{dataset}: only {len(chosen)} eligible series")
    return chosen, eligible


def build_tasks(
    original: dict[str, Any],
    m3: dict[str, dict[str, np.ndarray]],
    raw: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected: list[tuple] = []
    source_rows: list[dict[str, Any]] = []
    for dataset, spec in original["groups"].items():
        if spec["benchmark"] == "M3":
            group = spec["group"]
            eligible: list[tuple[str, str, np.ndarray, float]] = []
            for series_id, raw_values in m3[group].items():
                values, missing = clean(raw_values)
                if missing <= 0.15 and len(values) >= 32 + 2 * int(spec["horizon"]) and np.isfinite(values).all():
                    rank = hashlib.sha256(f"{dataset}::{series_id}".encode()).hexdigest()
                    eligible.append((rank, series_id, values, missing))
            eligible.sort(key=lambda item: item[0])
            chosen = eligible[: int(spec["target_series"])]
            if len(chosen) != int(spec["target_series"]):
                raise RuntimeError(f"{dataset}: only {len(chosen)} eligible series")
            raw_series = len(m3[group])
            eligible_count = len(eligible)
        else:
            group = spec["group"]
            chosen, eligible_count = select_m4(
                dataset,
                spec,
                raw / f"m4/{group}-train.csv",
                raw / f"m4/{group}-test.csv",
            )
            with (raw / f"m4/{group}-train.csv").open(newline="", encoding="utf-8-sig") as handle:
                raw_series = sum(1 for _ in handle) - 1
        for rank, series_id, values, missing in chosen:
            selected.append((dataset, series_id, spec, values, missing, rank))
        source_rows.append(
            {
                "dataset": dataset,
                "raw_series": raw_series,
                "eligible_series": eligible_count,
                "selected_series": len(chosen),
            }
        )

    tasks: list[dict[str, Any]] = []
    series_rows: list[dict[str, Any]] = []
    value_frames: list[pd.DataFrame] = []
    task_id = 0
    for dataset, series_id, spec, values, missing, rank in selected:
        horizon = int(spec["horizon"])
        series_rows.append(
            {
                "dataset": dataset,
                "series_id": series_id,
                "length": len(values),
                "frequency": int(spec["frequency"]),
                "horizon": horizon,
                "imputed_fraction": missing,
                "selection_hash": rank,
            }
        )
        for origin_number, origin in enumerate((len(values) - 2 * horizon, len(values) - horizon), start=1):
            context = values[max(0, origin - 512) : origin].astype(np.float32)
            future = values[origin : origin + horizon].astype(np.float32)
            if len(context) < 32 or len(future) != horizon:
                raise RuntimeError((dataset, series_id, origin_number, len(context), len(future)))
            tasks.append(
                {
                    "task_id": task_id,
                    "dataset": dataset,
                    "series_id": series_id,
                    "origin_number": origin_number,
                    "context": len(context),
                    "horizon": horizon,
                    "frequency": int(spec["frequency"]),
                    "imputed_fraction": missing,
                }
            )
            value_frames.append(
                pd.DataFrame(
                    {
                        "task_id": task_id,
                        "part": "context",
                        "position": np.arange(len(context), dtype=np.int32),
                        "value": context.astype(np.float64),
                    }
                )
            )
            value_frames.append(
                pd.DataFrame(
                    {
                        "task_id": task_id,
                        "part": "future",
                        "position": np.arange(len(future), dtype=np.int32),
                        "value": future.astype(np.float64),
                    }
                )
            )
            task_id += 1
    return pd.DataFrame(tasks), pd.DataFrame(series_rows), pd.concat(value_frames, ignore_index=True), pd.DataFrame(source_rows)


def write_manifest(directory: Path) -> str:
    excluded = {"MANIFEST.sha256", "SOURCE_SUITE_DECISION.json"}
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name not in excluded)
    manifest = directory / "MANIFEST.sha256"
    manifest.write_text("".join(f"{digest(path)}  {path.relative_to(directory)}\n" for path in files))
    return digest(manifest)


def create_archive(directory: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory))


def run(output_root: Path, original_manifest: Path, bridge_manifest: Path, addendum: Path, extractor: Path) -> dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    raw = output_root / "raw"
    suite = output_root / "suite"
    diagnostic = output_root / "diagnostic"
    for directory in (raw, suite, diagnostic):
        directory.mkdir(parents=True, exist_ok=True)
    transport: list[dict[str, Any]] = []
    started = time.time()
    try:
        if digest(original_manifest) != ORIGINAL_MANIFEST_SHA256:
            raise RuntimeError("original manifest hash mismatch")
        if digest(bridge_manifest) != BRIDGE_MANIFEST_SHA256:
            raise RuntimeError("bridge manifest hash mismatch")
        if digest(addendum) != ADDENDUM_SHA256:
            raise RuntimeError("addendum hash mismatch")
        original = json.loads(original_manifest.read_text())
        bridge = json.loads(bridge_manifest.read_text())
        shutil.copy2(original_manifest, suite / "PRE_VALUE_MANIFEST.json")
        shutil.copy2(bridge_manifest, suite / "SOURCE_EQUIVALENCE_PRE_VALUE_MANIFEST.json")
        shutil.copy2(addendum, suite / "FROZEN_STAGE2I_SOURCE_EQUIVALENCE_ADDENDUM.md")

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Stage2I source-equivalence audit)", "Accept": "*/*"})
        cran_path = raw / "m3/cran/M3.rda"
        transport.append(download(session, bridge["m3_cran"]["url"], cran_path, 100_000))
        observed_md5 = digest(cran_path, "md5")
        if observed_md5 != bridge["m3_cran"]["expected_md5"] or observed_md5 != EXPECTED_M3_MD5:
            raise RuntimeError(f"CRAN M3 MD5 mismatch: {observed_md5}")

        zenodo_files: dict[str, Path] = {}
        for group, url in bridge["m3_zenodo"].items():
            if group == "parser_reference_commit":
                continue
            archive = raw / f"m3/zenodo/{group}.zip"
            transport.append(download(session, url, archive, 1_000))
            zenodo_files[group] = unzip_single_tsf(archive, raw / f"m3/zenodo/{group}")

        m4_groups = sorted({spec["group"] for spec in original["groups"].values() if spec["benchmark"] == "M4"})
        base = original["source_urls"]["M4_base"].rstrip("/")
        for group in m4_groups:
            for split in ("Train", "Test"):
                filename = f"{group}-{split.lower()}.csv"
                transport.append(download(session, f"{base}/{split}/{filename}", raw / f"m4/{filename}", 256))
        transport.append(download(session, original["source_urls"]["M4_info"], raw / "m4/M4-info.csv", 10_000))

        cran_values = raw / "m3/cran/m3_values.csv.gz"
        cran_metadata = raw / "m3/cran/m3_metadata.csv"
        subprocess.run(["Rscript", str(extractor), str(cran_path), str(cran_values), str(cran_metadata)], check=True)
        cran_map, cran_meta = load_cran_export(cran_values, cran_metadata)

        zenodo_map: dict[str, dict[str, np.ndarray]] = {}
        zenodo_metadata: dict[str, Any] = {}
        for group, path in zenodo_files.items():
            arrays, metadata = parse_tsf(path)
            prefix = PREFIX[group]
            zenodo_map[group] = {f"{prefix}{index + 1}": values for index, values in enumerate(arrays)}
            zenodo_metadata[group] = metadata
            if len(arrays) != EXPECTED_COUNTS[group]:
                raise RuntimeError(f"Zenodo {group} count mismatch")
            if int(metadata.get("horizon", -1)) != HORIZON[group]:
                raise RuntimeError(f"Zenodo {group} horizon mismatch: {metadata}")

        tolerance = bridge["equivalence_tolerance"]
        equivalence, equivalence_checks = compare_m3(
            cran_map,
            cran_meta,
            zenodo_map,
            float(tolerance["rtol"]),
            float(tolerance["atol"]),
        )
        equivalence.to_csv(suite / "m3_source_equivalence_by_series.csv", index=False)
        (suite / "m3_source_equivalence_summary.json").write_text(
            json.dumps({"checks": equivalence_checks, "zenodo_metadata": zenodo_metadata}, indent=2, sort_keys=True) + "\n"
        )

        tasks, series, values, source_inventory = build_tasks(original, zenodo_map, raw)
        tasks.to_csv(suite / "task_inventory.csv", index=False)
        series.to_csv(suite / "series_inventory.csv", index=False)
        values.to_csv(suite / "task_values.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        source_inventory.to_csv(suite / "source_inventory.csv", index=False)
        pd.DataFrame(transport).to_csv(suite / "source_transport.csv", index=False)
        source_hashes = [
            {"path": str(path.relative_to(raw)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(raw.rglob("*"))
            if path.is_file()
        ]
        pd.DataFrame(source_hashes).to_csv(suite / "downloaded_source_hashes.csv", index=False)

        checks = {
            "original_manifest_exact": digest(suite / "PRE_VALUE_MANIFEST.json") == ORIGINAL_MANIFEST_SHA256,
            "bridge_manifest_exact": digest(suite / "SOURCE_EQUIVALENCE_PRE_VALUE_MANIFEST.json") == BRIDGE_MANIFEST_SHA256,
            "addendum_exact": digest(suite / "FROZEN_STAGE2I_SOURCE_EQUIVALENCE_ADDENDUM.md") == ADDENDUM_SHA256,
            "cran_md5_exact": observed_md5 == EXPECTED_M3_MD5,
            "m3_equivalence_passed": bool(equivalence_checks["all_values_match"] and equivalence_checks["all_lengths_match"]),
            "tasks_exactly_700": len(tasks) == 700,
            "series_exactly_350": len(series) == 350,
            "cohorts_exactly_10": tasks.dataset.nunique() == 10,
            "two_origins_each": bool((tasks.groupby(["dataset", "series_id"]).origin_number.nunique() == 2).all()),
            "contexts_32_to_512": bool(tasks.context.between(32, 512).all()),
            "horizons_6_to_48": bool(tasks.horizon.between(6, 48).all()),
            "all_values_finite": bool(np.isfinite(values.value.to_numpy()).all()),
        }
        if not all(checks.values()):
            raise RuntimeError(f"source-equivalent suite validation failed: {checks}")
        validation = {
            "status": "SOURCE_EQUIVALENCE_AND_PANEL_PASS",
            "checks": checks,
            "equivalence": equivalence_checks,
            "environment": environment(),
            "tasks": len(tasks),
            "series": len(series),
            "cohorts": tasks.dataset.nunique(),
        }
        (suite / "VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
        manifest_hash = write_manifest(suite)
        decision = {
            "status": "STAGE2I_SE_SOURCE_SUITE_AUTHORIZED",
            "label": "Stage2I-SE",
            "suite_manifest_sha256": manifest_hash,
            "task_inventory_sha256": digest(suite / "task_inventory.csv"),
            "task_values_sha256": digest(suite / "task_values.csv.gz"),
            "source_hash_inventory_sha256": digest(suite / "downloaded_source_hashes.csv"),
            "equivalence_summary_sha256": digest(suite / "m3_source_equivalence_summary.json"),
            "validation": validation,
            "caveat": "The original exact M3 URL remains transport-blocked; this is an explicitly source-equivalent namespace.",
        }
        (suite / "SOURCE_SUITE_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
        archive = output_root / "Stage2I_SE_Canonical_Source_Suite.zip"
        create_archive(suite, archive)
        (output_root / f"{archive.name}.sha256").write_text(f"{digest(archive)}  {archive.name}\n")
        decision.update({"archive": archive.name, "archive_sha256": digest(archive)})
    except Exception as exc:
        decision = {
            "status": "BLOCKED_SOURCE_EQUIVALENCE_GATE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "transport_completed": transport,
            "environment": environment(),
            "elapsed_seconds": time.time() - started,
            "policy": "No unrecorded source or third representation was substituted.",
        }
        (diagnostic / "SOURCE_EQUIVALENCE_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    (output_root / "RUN_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--bridge-manifest", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.output_root.resolve(),
        args.original_manifest.resolve(),
        args.bridge_manifest.resolve(),
        args.addendum.resolve(),
        args.extractor.resolve(),
    )


if __name__ == "__main__":
    main()
