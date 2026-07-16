from __future__ import annotations

import argparse, json, shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from build_m1_source_lock import (
    EXPECTED_PRE_VALUE_SHA256, EXPECTED_SPEC_SHA256, EXPECTED_GUARD_SHA256,
    EXPECTED_MD5, digest, download, write_manifest,
)


def run(output_root: Path, pre_value: Path, spec: Path, guard: Path, extractor: Path) -> dict:
    if output_root.exists(): shutil.rmtree(output_root)
    raw, lock, diagnostic = output_root/'raw', output_root/'source_lock', output_root/'diagnostic'
    for directory in (raw, lock, diagnostic): directory.mkdir(parents=True, exist_ok=True)
    started=time.time(); transport=None
    try:
        observed={'pre_value':digest(pre_value),'spec':digest(spec),'guard':digest(guard)}
        expected={'pre_value':EXPECTED_PRE_VALUE_SHA256,'spec':EXPECTED_SPEC_SHA256,'guard':EXPECTED_GUARD_SHA256}
        if observed != expected: raise RuntimeError(f'frozen artifact hash mismatch: observed={observed}')
        manifest=json.loads(pre_value.read_text()); source=manifest['m1_source']; rda=raw/'M1.rda'
        transport=download(source['url'],rda)
        if transport['md5'] != source['expected_md5'] or transport['md5'] != EXPECTED_MD5:
            raise RuntimeError(f"M1.rda MD5 mismatch: {transport['md5']}")
        values,metadata=lock/'m1_values.csv.gz',lock/'m1_metadata.csv'
        subprocess.run(['Rscript',str(extractor),str(rda),str(values),str(metadata)],check=True)
        meta=pd.read_csv(metadata,dtype={'series_id':str}); val=pd.read_csv(values,dtype={'series_id':str})
        observed_lengths=val.groupby('series_id').size().reindex(meta.series_id).to_numpy()
        expected_lengths=meta.total_length.to_numpy()
        checks={
            'series_exactly_1001':len(meta)==1001,
            'unique_ids':meta.series_id.nunique()==1001,
            'values_cover_all_ids':set(val.series_id)==set(meta.series_id),
            'total_lengths_match':bool(np.array_equal(observed_lengths,expected_lengths)),
            'horizons_positive':bool((meta.horizon>0).all()),
            'recognized_periods':set(meta.period).issubset({'YEARLY','QUARTERLY','MONTHLY','OTHER'}),
            'source_md5_exact':transport['md5']==EXPECTED_MD5,
            'all_exported_values_numeric_or_missing':bool(pd.to_numeric(val.value,errors='coerce').notna().sum()>0),
        }
        if not all(checks.values()): raise RuntimeError(f'M1 source validation failed: {checks}')
        for source_path,name in ((pre_value,'STAGE2J_PRE_VALUE_MANIFEST.json'),(spec,'FROZEN_STAGE2J_TAILSENTRY_SPEC.md'),(guard,'tailsentry.py')):
            shutil.copy2(source_path,lock/name)
        (lock/'source_transport.json').write_text(json.dumps(transport,indent=2,sort_keys=True)+'\n')
        validation={'status':'M1_SOURCE_LOCK_PASS','checks':checks,'period_counts':meta.groupby('period').size().sort_index().to_dict(),'type_counts':meta.groupby('type').size().sort_index().to_dict(),'metadata_rows':len(meta),'value_rows':len(val),'source_rda_sha256':digest(rda),'source_rda_md5':digest(rda,'md5'),'python':sys.version}
        (lock/'VALIDATION.json').write_text(json.dumps(validation,indent=2,sort_keys=True)+'\n')
        manifest_sha=write_manifest(lock);decision={'status':'STAGE2J_M1_SOURCE_LOCK_AUTHORIZED','source_lock_manifest_sha256':manifest_sha,'m1_values_sha256':digest(values),'m1_metadata_sha256':digest(metadata),'validation':validation}
        (lock/'SOURCE_LOCK_DECISION.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
        archive=output_root/'Stage2J_M1_Source_Lock.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
            for path in sorted(lock.rglob('*')):
                if path.is_file(): zf.write(path,path.relative_to(lock))
        (output_root/f'{archive.name}.sha256').write_text(f'{digest(archive)}  {archive.name}\n');decision.update({'archive':archive.name,'archive_sha256':digest(archive)})
    except Exception as exc:
        decision={'status':'BLOCKED_STAGE2J_M1_SOURCE_LOCK','error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'transport':transport,'elapsed_seconds':time.time()-started}
        (diagnostic/'SOURCE_LOCK_DECISION.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
    (output_root/'RUN_DECISION.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n');print(json.dumps(decision,indent=2,sort_keys=True),flush=True);return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--pre-value',type=Path,required=True);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--guard',type=Path,required=True);parser.add_argument('--extractor',type=Path,required=True);args=parser.parse_args();run(args.output_root.resolve(),args.pre_value.resolve(),args.spec.resolve(),args.guard.resolve(),args.extractor.resolve())
