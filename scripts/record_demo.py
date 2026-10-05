"""Record an immutable genuine live session from a development example."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import time
import httpx

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--policy',default='information')
    parser.add_argument('--budget',type=float,default=6)
    args=parser.parse_args()
    with httpx.Client(base_url=args.url,timeout=120) as client:
        health=client.get('/api/health').json()
        if not health['model_ready']: raise RuntimeError(health['blockers'])
        response=client.post('/api/sessions',json={'budget':args.budget,'policy':args.policy})
        response.raise_for_status(); state=response.json()
        response=client.post(f"/api/sessions/{state['id']}/run",json={'version':state['version']})
        response.raise_for_status(); jid=response.json()['job_id']
        deadline=time.monotonic()+300
        while True:
            job=client.get('/api/jobs/'+jid).json()
            if job['status']=='error': raise RuntimeError(job['error'])
            if job['status']=='completed': break
            if time.monotonic()>deadline: raise TimeoutError('Recorded demo job exceeded five minutes')
            time.sleep(.5)
        state=client.get('/api/sessions/'+state['id']).json()
        response=client.get('/api/sessions/'+state['id']+'/report'); response.raise_for_status()
        provenance=client.get('/api/model-info').json()
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    folder=ROOT/'artifacts'/'replays'; folder.mkdir(parents=True,exist_ok=True)
    record=dict(id='development-demo',title='Illustrative development inspection',timestamp=datetime.now(timezone.utc).isoformat(),
        model_provenance=provenance,input_hashes={'features_sha256':digest(ROOT/'artifacts/data/features.npz')},
        split_hash=digest(ROOT/'artifacts/data/splits.json'),policy_configuration={'budget':args.budget,'policy':args.policy},session=state,report=response.json())
    from nextcheck.replay.public import public_replay
    manifest=json.loads((ROOT/'demo_manifest.json').read_text(encoding='utf-8'))
    target=folder/'development-demo.json'
    target.write_text(json.dumps(public_replay(record,manifest),indent=2,allow_nan=False),encoding='utf-8')
    print(target)

if __name__=='__main__': main()
