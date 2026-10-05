"""Live API acceptance checks against actual TabPFN. The backend must be running."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import uuid
import httpx

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--url',default='http://127.0.0.1:8000'); args=parser.parse_args()
    checks=[]
    with httpx.Client(base_url=args.url,timeout=120) as client:
        def get(path):
            response=client.get(path); response.raise_for_status(); return response.json()
        def post(path,payload):
            response=client.post(path,json=payload); response.raise_for_status(); return response.json()
        def job(path,version):
            jid=post(path,{'version':version})['job_id']; deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                result=get('/api/jobs/'+jid)
                if result['status']=='error': raise RuntimeError(result['error'])
                if result['status']=='completed': return result['result']
                time.sleep(.1)
            raise TimeoutError('Live job exceeded deadline')
        health=get('/api/health'); assert health['model_ready'] and health['data_ready']
        model=get('/api/model-info'); assert model['ready'] and model['model_version']=='v3.5'
        checks.append('real_model_ready')
        manifest=get('/api/demo-manifest'); panels={p['id']:p for p in manifest['panels']}
        state=post('/api/sessions',{'budget':6,'policy':'information','lambda_cost':0})
        sid=state['id']; url='/api/sessions/'+sid
        assert len(state['visible_values'])==5 and 'true_label' not in json.dumps(state)
        assert client.get(url+'/report').status_code==409
        checks.append('active_hidden_values_and_label_omitted')
        state=job(url+'/recommend',state['version']); group=state['recommendation']['selected_action']; assert group
        purchase={'version':state['version'],'group_id':group,'idempotency_key':uuid.uuid4().hex}
        state=post(url+'/acquire',purchase)
        assert state['spent']==panels[group]['cost']
        assert len(state['visible_values'])==5+len(panels[group]['features'])
        assert post(url+'/acquire',purchase)['spent']==state['spent']
        assert client.post(url+'/recommend',json={'version':0}).status_code==409
        checks.append('recommend_atomic_reveal_single_charge_idempotent_and_stale_rejection')
        state=post(url+'/stop',{'version':state['version']}); report=get(url+'/report')
        assert report['true_label'] in [0,1,2] and report['log_loss']>=0
        assert client.post(url+'/run',json={'version':state['version']}).status_code==409
        checks.append('completed_report_cannot_resume')
        zero=post('/api/sessions',{'budget':0})
        zero=job('/api/sessions/'+zero['id']+'/run',zero['version'])
        assert zero['spent']==0 and zero['stop_reason']=='budget_exhausted'
        checks.append('zero_budget_stops')
        state=post('/api/sessions',{'budget':6,'policy':'value'})
        state=job('/api/sessions/'+state['id']+'/run',state['version'])
        assert state['status']=='completed' and state['spent']<=6 and len(state['observed_groups'])<=6
        checks.append('learned_value_autopilot_bounded')
        benchmark=get('/api/benchmarks'); assert any(run['status']=='complete' and run['results'] for run in benchmark['runs'])
        checks.append('actual_benchmark_results_available')
        replays=get('/api/replays')['replays']; assert replays
        record=get('/api/replays/'+replays[0]['id']); assert record['session']['mode']=='recorded'
        assert 'row_index' not in json.dumps(record)
        checks.append('sanitized_recorded_replay_available')
    result={'status':'passed','timestamp':datetime.now(timezone.utc).isoformat(),'checks':checks,'model_provenance':model}
    target=ROOT/'artifacts/verification/live_api.json'; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({'status':'passed','checks':len(checks)},indent=2))

if __name__=='__main__': main()
