from __future__ import annotations

import argparse, hashlib, json, os, platform, shutil, sys, time, traceback, zipfile
from pathlib import Path

import datasetsforecast, numpy as np, pandas as pd, requests

EXPECTED = "977f3db34bdd9e45940190858889494a1bfa19d6af2e4f3d7ab3984f30cb64a9"
OLE = bytes.fromhex("d0cf11e0a1b11ae1")
SHEETS = {"Yearly":"M3Year", "Quarterly":"M3Quart", "Monthly":"M3Month", "Other":"M3Other"}
M3_COUNTS = {"Yearly":645, "Quarterly":756, "Monthly":1428, "Other":174}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def env() -> dict:
    return {"python":sys.version, "platform":platform.platform(), "pandas":pd.__version__,
            "numpy":np.__version__, "datasetsforecast":datasetsforecast.__version__,
            "github_sha":os.getenv("GITHUB_SHA"), "github_ref":os.getenv("GITHUB_REF")}


def get(session, url: str, path: Path, minimum: int, magic: bytes | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, 5):
        part = path.with_suffix(path.suffix + ".part"); part.unlink(missing_ok=True)
        try:
            with session.get(url, stream=True, timeout=(30, 240), allow_redirects=True) as r:
                r.raise_for_status()
                with part.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk: f.write(chunk)
                if part.stat().st_size < minimum: raise RuntimeError(f"too small: {part.stat().st_size}")
                prefix = part.read_bytes()[:max(160, len(magic or b""))]
                if magic and prefix[:len(magic)] != magic:
                    raise RuntimeError(f"bad magic={prefix[:len(magic)].hex()} prefix={prefix[:120]!r}")
                part.replace(path)
                return {"requested_url":url, "resolved_url":r.url, "status":r.status_code,
                        "content_type":r.headers.get("content-type"), "bytes":path.stat().st_size,
                        "sha256":sha(path), "attempt":attempt, "path":str(path)}
        except Exception as e:
            last = e; part.unlink(missing_ok=True)
            if attempt < 4: time.sleep(2 ** attempt)
    raise RuntimeError(f"authorized source download failed: {url}") from last


def clean(values) -> tuple[np.ndarray, float]:
    s = pd.Series(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    finite = np.isfinite(s.to_numpy())
    if not finite.any(): return np.array([], dtype=np.float64), 1.0
    first, last = np.flatnonzero(finite)[[0,-1]]
    s = s.iloc[int(first):int(last)+1].reset_index(drop=True)
    missing = float(s.isna().mean())
    return s.interpolate(method="linear", limit_direction="both").to_numpy(np.float64), missing


def load_m3(path: Path, group: str) -> dict[str, np.ndarray]:
    df = pd.read_excel(path, sheet_name=SHEETS[group])
    ids = [f"{group[0]}{i+1}" for i in range(len(df))]
    cols = list(df.columns[6:])
    out = {uid:pd.to_numeric(row[cols], errors="coerce").to_numpy(np.float64)
           for uid, (_, row) in zip(ids, df.iterrows())}
    if len(out) != M3_COUNTS[group]: raise RuntimeError(f"M3 {group} count={len(out)}")
    return out


def load_m4(train: Path, test: Path) -> dict[str, np.ndarray]:
    tr, te = pd.read_csv(train), pd.read_csv(test)
    id1, id2 = tr.columns[0], te.columns[0]
    if id1 != id2: raise RuntimeError("M4 ID-column mismatch")
    te = te.set_index(id2); out = {}
    for _, row in tr.iterrows():
        uid = str(row.iloc[0]); tail = te.loc[uid]
        if isinstance(tail, pd.DataFrame): raise RuntimeError(f"duplicate M4 ID {uid}")
        out[uid] = np.r_[pd.to_numeric(row.iloc[1:], errors="coerce").to_numpy(np.float64),
                         pd.to_numeric(tail, errors="coerce").to_numpy(np.float64)]
    return out


def lock_manifest(suite: Path) -> str:
    excluded = {"MANIFEST.sha256", "SOURCE_SUITE_DECISION.json"}
    files = sorted(p for p in suite.rglob("*") if p.is_file() and p.name not in excluded)
    (suite/"MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(suite)}\n" for p in files))
    return sha(suite/"MANIFEST.sha256")


def build(out: Path, manifest_source: Path) -> dict:
    if out.exists(): shutil.rmtree(out)
    raw, suite, diag = out/"raw", out/"suite", out/"diagnostic"
    raw.mkdir(parents=True); suite.mkdir(); diag.mkdir()
    transport = []; started = time.time()
    try:
        if sha(manifest_source) != EXPECTED: raise RuntimeError("pre-value manifest hash mismatch")
        if datasetsforecast.__version__ != "1.0.0": raise RuntimeError("datasetsforecast version mismatch")
        manifest_path = suite/"PRE_VALUE_MANIFEST.json"; shutil.copy2(manifest_source, manifest_path)
        manifest = json.loads(manifest_path.read_text())
        s = requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0", "Accept":"*/*"})
        transport.append(get(s, manifest["source_urls"]["M3"], raw/"m3/M3C.xls", 100000, OLE))
        base = manifest["source_urls"]["M4_base"].rstrip("/")
        m4groups = sorted({v["group"] for v in manifest["groups"].values() if v["benchmark"]=="M4"})
        for group in m4groups:
            for split in ("Train", "Test"):
                fn = f"{group}-{split.lower()}.csv"
                transport.append(get(s, f"{base}/{split}/{fn}", raw/f"m4/{fn}", 256))
        transport.append(get(s, manifest["source_urls"]["M4_info"], raw/"m4/M4-info.csv", 10000))

        m3, m4, selected, sources = {}, {}, [], []
        for dataset, spec in manifest["groups"].items():
            group = spec["group"]
            if spec["benchmark"] == "M3":
                m3.setdefault(group, load_m3(raw/"m3/M3C.xls", group)); mapping = m3[group]
            else:
                m4.setdefault(group, load_m4(raw/f"m4/{group}-train.csv", raw/f"m4/{group}-test.csv")); mapping = m4[group]
            eligible = []
            for uid, raw_values in mapping.items():
                values, missing = clean(raw_values)
                if missing <= .15 and len(values) >= 32 + 2*int(spec["horizon"]) and np.isfinite(values).all():
                    eligible.append((hashlib.sha256(f"{dataset}::{uid}".encode()).hexdigest(), uid, values, missing))
            eligible.sort(key=lambda x:x[0]); chosen = eligible[:int(spec["target_series"])]
            if len(chosen) != int(spec["target_series"]): raise RuntimeError(f"{dataset}: eligible={len(chosen)}")
            selected += [(dataset, uid, spec, values, missing, rank) for rank,uid,values,missing in chosen]
            sources.append({"dataset":dataset,"raw_series":len(mapping),"eligible_series":len(eligible),"selected_series":len(chosen)})

        tasks, series, value_parts, task_id = [], [], [], 0
        for dataset, uid, spec, values, missing, rank in selected:
            h = int(spec["horizon"])
            series.append({"dataset":dataset,"series_id":uid,"length":len(values),"frequency":spec["frequency"],
                           "horizon":h,"imputed_fraction":missing,"selection_hash":rank})
            for origin_no, origin in enumerate((len(values)-2*h, len(values)-h), 1):
                context = values[max(0,origin-512):origin].astype(np.float32)
                future = values[origin:origin+h].astype(np.float32)
                if len(context)<32 or len(future)!=h: raise RuntimeError("origin geometry failure")
                tasks.append({"task_id":task_id,"dataset":dataset,"series_id":uid,"origin_number":origin_no,
                              "context":len(context),"horizon":h,"frequency":spec["frequency"],"imputed_fraction":missing})
                value_parts += [pd.DataFrame({"task_id":task_id,"part":"context","position":np.arange(len(context)),"value":context}),
                                pd.DataFrame({"task_id":task_id,"part":"future","position":np.arange(h),"value":future})]
                task_id += 1
        tasks, series, values = pd.DataFrame(tasks), pd.DataFrame(series), pd.concat(value_parts, ignore_index=True)
        tasks.to_csv(suite/"task_inventory.csv", index=False); series.to_csv(suite/"series_inventory.csv", index=False)
        values.to_csv(suite/"task_values.csv.gz", index=False, compression={"method":"gzip","mtime":0})
        pd.DataFrame(sources).to_csv(suite/"source_inventory.csv", index=False)
        pd.DataFrame(transport).to_csv(suite/"source_transport.csv", index=False)
        pd.DataFrame([{"path":str(p.relative_to(raw)),"bytes":p.stat().st_size,"sha256":sha(p)}
                      for p in sorted(raw.rglob("*")) if p.is_file()]).to_csv(suite/"downloaded_source_hashes.csv",index=False)
        checks = {"tasks_exactly_700":len(tasks)==700,"series_exactly_350":len(series)==350,
                  "ten_datasets":tasks.dataset.nunique()==10,
                  "two_origins_each":bool((tasks.groupby(["dataset","series_id"]).origin_number.nunique()==2).all()),
                  "contexts_32_to_512":bool(tasks.context.between(32,512).all()),
                  "horizons_1_to_96":bool(tasks.horizon.between(1,96).all()),
                  "all_values_finite":bool(np.isfinite(values.value.to_numpy()).all()),
                  "manifest_unchanged":sha(manifest_path)==EXPECTED,"m3_official_xls_magic":(raw/"m3/M3C.xls").read_bytes()[:8]==OLE}
        if not all(checks.values()): raise RuntimeError(f"suite validation failed: {checks}")
        validation = {"checks":checks,"tasks":len(tasks),"series":len(series),"datasets":tasks.dataset.nunique(),"environment":env()}
        (suite/"VALIDATION.json").write_text(json.dumps(validation,indent=2,sort_keys=True)+"\n")
        decision = {"status":"CANONICAL_SOURCE_SUITE_BUILT","suite_manifest_sha256":lock_manifest(suite),
                    "task_inventory_sha256":sha(suite/"task_inventory.csv"),"task_values_sha256":sha(suite/"task_values.csv.gz"),
                    "source_hash_inventory_sha256":sha(suite/"downloaded_source_hashes.csv"),"validation":validation}
        (suite/"SOURCE_SUITE_DECISION.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n")
        archive = out/"Stage2I_Canonical_Source_Suite.zip"
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in sorted(suite.rglob("*")):
                if p.is_file(): z.write(p,p.relative_to(suite))
        (out/(archive.name+".sha256")).write_text(f"{sha(archive)}  {archive.name}\n")
        decision |= {"archive":archive.name,"archive_sha256":sha(archive)}
    except Exception as exc:
        decision = {"status":"BLOCKED_AUTHORIZED_SOURCE_ACQUISITION","error_type":type(exc).__name__,"error":str(exc),
                    "traceback":traceback.format_exc(),"environment":env(),"transport_completed":transport,
                    "elapsed_seconds":time.time()-started,"policy":"No unrecorded mirror substitution was attempted."}
        (diag/"SOURCE_ACQUISITION_DECISION.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n")
    (out/"RUN_DECISION.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n")
    print(json.dumps(decision,indent=2,sort_keys=True)); return decision


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True)
    a=p.parse_args(); build(a.output_root.resolve(),a.manifest.resolve())
