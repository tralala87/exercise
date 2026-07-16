from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adjudicate_m3_archive as core

SNAPSHOT_TIMESTAMP = "20210225020840"
SNAPSHOT_ORIGINAL = "http://forecasters.org/data/m3comp/M3C.xls"
AVAILABILITY_URL = "https://archive.org/wayback/available"
REPLAY_URL = f"https://web.archive.org/web/{SNAPSHOT_TIMESTAMP}id_/{SNAPSHOT_ORIGINAL}"


def quick_download(session: requests.Session, url: str, path: Path, minimum: int = 1) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, 3):
        partial = path.with_suffix(path.suffix + ".part")
        partial.unlink(missing_ok=True)
        try:
            with session.get(url, stream=True, timeout=(20, 75), allow_redirects=True) as response:
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
                    "sha256": core.hash_file(path),
                    "content_type": response.headers.get("content-type"),
                    "attempt": attempt,
                    "path": str(path),
                }
        except Exception as exc:
            last = exc
            partial.unlink(missing_ok=True)
    raise RuntimeError(f"download failed: {url}") from last


def run(output: Path, manifest_path: Path, spec_path: Path, extractor: Path) -> dict:
    if output.exists():
        shutil.rmtree(output)
    raw = output / "raw"
    evidence = output / "evidence"
    diagnostic = output / "diagnostic"
    for directory in (raw, evidence, diagnostic):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        if core.hash_file(manifest_path) != core.MANIFEST_SHA256:
            raise RuntimeError("adjudication manifest hash mismatch")
        if core.hash_file(spec_path) != core.SPEC_SHA256:
            raise RuntimeError("adjudication spec hash mismatch")
        manifest = json.loads(manifest_path.read_text())
        shutil.copy2(manifest_path, evidence / manifest_path.name)
        shutil.copy2(spec_path, evidence / spec_path.name)

        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Stage2I archived-source audit)", "Accept": "*/*"})
        availability = session.get(
            AVAILABILITY_URL,
            params={"url": SNAPSHOT_ORIGINAL, "timestamp": "20180101"},
            timeout=(20, 45),
        )
        availability.raise_for_status()
        availability_payload = availability.json()
        (evidence / "wayback_availability.json").write_text(json.dumps(availability_payload, indent=2, sort_keys=True) + "\n")
        closest = availability_payload.get("archived_snapshots", {}).get("closest", {})
        if closest.get("available") is not True or str(closest.get("status")) != "200":
            raise RuntimeError(f"availability API did not return a valid snapshot: {closest}")
        if closest.get("timestamp") != SNAPSHOT_TIMESTAMP:
            raise RuntimeError(f"unexpected closest snapshot timestamp: {closest}")

        transport = []
        workbook = raw / f"archive/M3C_{SNAPSHOT_TIMESTAMP}.xls"
        transport.append(quick_download(session, REPLAY_URL, workbook, 100_000))
        archived_groups, sheet_checks = core.load_xls(workbook)
        archived_map = core.flatten(archived_groups)

        cran = raw / "cran/M3.rda"
        transport.append(quick_download(session, core.CRAN_URL, cran, 100_000))
        if core.hash_file(cran, "md5") != core.CRAN_MD5:
            raise RuntimeError("CRAN M3.rda MD5 mismatch")
        cran_values = raw / "cran/m3_values.csv.gz"
        cran_metadata = raw / "cran/m3_metadata.csv"
        subprocess.run(["Rscript", str(extractor), str(cran), str(cran_values), str(cran_metadata)], check=True)
        cran_map = core.load_cran(cran_values)

        for group, url in core.ZENODO.items():
            transport.append(quick_download(session, url, raw / f"zenodo/{group}.zip", 1_000))
        zenodo_map = core.flatten(core.load_zenodo(raw))

        archived_vs_cran = core.compare(archived_map, cran_map)
        archived_vs_zenodo = core.compare(archived_map, zenodo_map)
        (evidence / "archive_vs_cran.json").write_text(json.dumps(archived_vs_cran, indent=2, sort_keys=True) + "\n")
        (evidence / "archive_vs_zenodo.json").write_text(json.dumps(archived_vs_zenodo, indent=2, sort_keys=True) + "\n")
        (evidence / "transport.json").write_text(json.dumps(transport, indent=2, sort_keys=True) + "\n")

        selected_value = float(archived_map["M1385"][83])
        expected = manifest["known_conflict"]
        declared = {float(expected["cran_mcomp_value"]), float(expected["zenodo_tsf_value"])}
        cran_exact = archived_vs_cran.get("mismatch_count") == 0
        zenodo_exact = archived_vs_zenodo.get("mismatch_count") == 0
        cran_one = archived_vs_cran.get("mismatch_count") == 1 and archived_vs_cran.get("mismatches") == [
            {"series_id": "M1385", "position": 83, "left": -1200.0, "right": 1200.0, "absolute_delta": 2400.0}
        ]
        zenodo_one = archived_vs_zenodo.get("mismatch_count") == 1 and archived_vs_zenodo.get("mismatches") == [
            {"series_id": "M1385", "position": 83, "left": 1200.0, "right": -1200.0, "absolute_delta": 2400.0}
        ]
        checks = {
            "manifest_exact": core.hash_file(evidence / manifest_path.name) == core.MANIFEST_SHA256,
            "spec_exact": core.hash_file(evidence / spec_path.name) == core.SPEC_SHA256,
            "availability_timestamp_exact": closest.get("timestamp") == SNAPSHOT_TIMESTAMP,
            "workbook_ole_magic": workbook.read_bytes()[:8] == core.OLE,
            "sheet_and_count_checks_passed": set(sheet_checks) == set(core.SHEET),
            "selected_value_is_predeclared": selected_value in declared,
            "no_undeclared_semantic_mismatch": bool((cran_exact and zenodo_one) or (zenodo_exact and cran_one)),
        }
        if not all(checks.values()):
            raise RuntimeError(f"known-snapshot adjudication gates failed: {checks}")
        selected_source = "CRAN_Mcomp" if selected_value == 1200.0 else "Monash_Zenodo_TSF"
        decision = {
            "status": "STAGE2I_SA_M3_CONFLICT_ADJUDICATED",
            "snapshot_timestamp": SNAPSHOT_TIMESTAMP,
            "snapshot_original": SNAPSHOT_ORIGINAL,
            "snapshot_replay_url": REPLAY_URL,
            "snapshot_sha256": core.hash_file(workbook),
            "snapshot_bytes": workbook.stat().st_size,
            "selected_value": selected_value,
            "selected_existing_representation": selected_source,
            "sheet_checks": sheet_checks,
            "checks": checks,
            "environment": core.environment(),
            "caveat": "The live exact URL remains transport-blocked; authority comes from an archived official workbook.",
        }
        (evidence / "ADJUDICATION_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
        decision["evidence_manifest_sha256"] = core.write_manifest(evidence)
    except Exception as exc:
        import traceback
        decision = {
            "status": "BLOCKED_M3_KNOWN_SNAPSHOT_ADJUDICATION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "environment": core.environment(),
            "policy": "No source outside the frozen archival hierarchy was substituted.",
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
    args = parser.parse_args()
    run(args.output.resolve(), args.manifest.resolve(), args.spec.resolve(), args.extractor.resolve())


if __name__ == "__main__":
    main()
