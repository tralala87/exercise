from __future__ import annotations
import argparse, hashlib, json, shutil, time, traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
import numpy as np, pandas as pd, requests

MANIFEST='da98603e38d1cfc1326ddab94a2a8a6aa30c6116c746f3eb5dbc6cd68c08719c'
ADDENDUM='3edfedf2e0c0cc73484fb4ca36b4dcf3ab4c70250be8508fb96e0f1d52310443'
OLE=bytes.fromhex('d0cf11e0a1b11ae1')
COUNTS={'M3Year':645,'M3Quart':756,'M3Month':1428,'M3Other':174}
HORIZONS={'M3Year':6,'M3Quart':8,'M3Month':18,'M3Other':8}

def sha(p):
 h=hashlib.sha256(); f=open(p,'rb')
 for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 f.close(); return h.hexdigest()

def fetch_json(url):
 r=requests.get(url,headers={'User-Agent':'Mozilla/5.0 (Stage2I archival audit)'},timeout=(15,40)); r.raise_for_status(); return url,r.url,r.json()

def discover(m,out):
 fields='timestamp,original,statuscode,mimetype,digest,length'; rows=[]; diag=[]
 urls=[f"{m['wayback_cdx']}?url={quote(v,safe='')}&output=json&filter=statuscode%3A200&fl={fields}&from=1990&to=2026" for v in m['url_variants']]
 with ThreadPoolExecutor(max_workers=4) as ex:
  futs={ex.submit(fetch_json,u):u for u in urls}
  for fut in as_completed(futs):
   u=futs[fut]
   try:
    _,resolved,p=fut.result(); h=p[0] if p and isinstance(p[0],list) else []
    for rec in p[1:]: rows.append(dict(zip(h,map(str,rec))))
    diag.append({'url':u,'resolved_url':resolved,'rows':max(0,len(p)-1)})
   except Exception as e: diag.append({'url':u,'error_type':type(e).__name__,'error':str(e)})
 if not rows:
  probes=['20000101000000','20050101000000','20100101000000','20150101000000','20200101000000','20260101000000']
  aus=[f"https://archive.org/wayback/available?url={quote(v,safe='')}&timestamp={t}" for v in m['url_variants'] for t in probes]
  with ThreadPoolExecutor(max_workers=12) as ex:
   futs={ex.submit(fetch_json,u):u for u in aus}
   for fut in as_completed(futs):
    u=futs[fut]
    try:
     _,resolved,p=fut.result(); c=p.get('archived_snapshots',{}).get('closest',{})
     diag.append({'url':u,'resolved_url':resolved,'closest':c})
     if c.get('available') and str(c.get('status'))=='200':
      rows.append({'timestamp':str(c['timestamp']),'original':p.get('url') or u.split('url=')[1].split('&')[0], 'digest':'availability-api','length':'unknown'})
    except Exception as e: diag.append({'url':u,'error_type':type(e).__name__,'error':str(e)})
 (out/'discovery.json').write_text(json.dumps({'diagnostic':diag,'rows':rows},indent=2,sort_keys=True)+'\n'); return rows

def choose(rows):
 unique={(r.get('timestamp',''),r.get('original',''),r.get('digest','')):r for r in rows if r.get('timestamp') and r.get('original')}
 groups=defaultdict(list)
 for r in unique.values(): groups[r.get('digest') or ('none:'+r['original'])].append(r)
 selected=[]
 for g in groups.values():
  g.sort(key=lambda r:(r['timestamp'],r['original'])); selected.append(g[0]);
  if len(g)>1: selected.append(g[-1])
 return sorted(selected,key=lambda r:(r['timestamp'],r['original']))

def validate(path):
 x=pd.ExcelFile(path,engine='xlrd'); result={}
 for s,n in COUNTS.items():
  d=pd.read_excel(path,sheet_name=s,engine='xlrd'); vals=d.iloc[:,6:].apply(pd.to_numeric,errors='coerce').to_numpy(float)
  check={'rows':len(d),'horizons':sorted(pd.to_numeric(d['NF'],errors='coerce').dropna().astype(int).unique().tolist()),'lengths_ok':bool(np.array_equal(np.isfinite(vals).sum(1),pd.to_numeric(d['N'],errors='coerce').astype(int).to_numpy()))}
  if len(d)!=n or check['horizons']!=[HORIZONS[s]] or not check['lengths_ok']: raise RuntimeError(f'{s}:{check}')
  result[s]=check
 return result

def get_snapshot(row,path):
 errors=[]
 for marker in ('id_','if_'):
  u=f"https://web.archive.org/web/{row['timestamp']}{marker}/{row['original']}"
  try:
   r=requests.get(u,headers={'User-Agent':'Mozilla/5.0 (Stage2I archival audit)'},stream=True,timeout=(15,90),allow_redirects=True); r.raise_for_status()
   part=path.with_suffix('.part'); part.parent.mkdir(parents=True,exist_ok=True)
   with open(part,'wb') as f:
    for c in r.iter_content(1<<20):
     if c:f.write(c)
   prefix=open(part,'rb').read(120)
   if prefix[:8]!=OLE: raise RuntimeError(f'non-OLE {prefix!r}')
   part.replace(path); return {'requested_url':u,'resolved_url':r.url,'bytes':path.stat().st_size,'sha256':sha(path),'timestamp':row['timestamp'],'original':row['original'],'workbook':validate(path)}
  except Exception as e: errors.append({'url':u,'error_type':type(e).__name__,'error':str(e)}); path.with_suffix('.part').unlink(missing_ok=True)
 raise RuntimeError(json.dumps(errors))

def run(root,mp,ap):
 if root.exists():shutil.rmtree(root)
 (root/'diagnostic').mkdir(parents=True); (root/'snapshots').mkdir(); start=time.time()
 try:
  if sha(mp)!=MANIFEST or sha(ap)!=ADDENDUM: raise RuntimeError('frozen hash mismatch')
  m=json.loads(mp.read_text()); rows=discover(m,root/'diagnostic'); selected=choose(rows)
  (root/'diagnostic/selected.json').write_text(json.dumps(selected,indent=2,sort_keys=True)+'\n')
  if not selected: raise RuntimeError('no archived captures discovered')
  valid=[]; attempts=[]
  for i,row in enumerate(selected):
   try: rec=get_snapshot(row,root/f"snapshots/capture_{i:03d}_{row['timestamp']}.xls");valid.append(rec);attempts.append({'status':'valid',**rec})
   except Exception as e:attempts.append({'status':'invalid','row':row,'error_type':type(e).__name__,'error':str(e)})
  (root/'diagnostic/attempts.json').write_text(json.dumps(attempts,indent=2,sort_keys=True)+'\n')
  if not valid: raise RuntimeError('no valid archived official workbook downloaded')
  distinct={r['sha256']:r for r in valid}
  decision={'status':'ARCHIVED_OFFICIAL_WORKBOOKS_ACQUIRED','valid_capture_count':len(valid),'distinct_workbook_sha256_count':len(distinct),'distinct_workbooks':list(distinct.values()),'manifest_sha256':sha(mp),'addendum_sha256':sha(ap),'elapsed_seconds':time.time()-start}
 except Exception as e:
  decision={'status':'BLOCKED_ARCHIVAL_OFFICIAL_ACQUISITION','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start,'policy':'No third-party source was substituted.'}
 (root/'ARCHIVAL_ACQUISITION_DECISION.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n');print(json.dumps(decision,indent=2,sort_keys=True));return decision

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--addendum',type=Path,required=True);a=p.parse_args();run(a.output_root.resolve(),a.manifest.resolve(),a.addendum.resolve())
