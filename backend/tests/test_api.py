"""Explicit mathematical fixture backend. Never imported by the application."""
import json
import time
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from nextcheck.api import create_app
from nextcheck.runtime import Runtime
from nextcheck.settings import Settings

class MathematicalPredictor:
    provenance={'checkpoint_sha256':'mathematical-fixture-only','backend':'fixture'}
    def predict_proba(self,X):
        z=np.nan_to_num(X,nan=0.).mean(axis=1)
        logits=np.array([z,-z,np.zeros(len(z))]).T
        q=np.exp(logits-logits.max(axis=1,keepdims=True))
        return q/q.sum(axis=1,keepdims=True)

@pytest.fixture
def app_client(tmp_path):
    root=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'demo_manifest.json').read_text())
    n=sum(len(p['features']) for p in manifest['panels'])
    data={'X':np.random.default_rng(9).normal(size=(40,n)), 'y':np.arange(40)%3}
    runtime=Runtime(Settings(root=root,artifacts=tmp_path),manifest=manifest,predictor=MathematicalPredictor(),data=data,
                    splits={'context':list(range(30)),'selection':list(range(30,36)),'test':list(range(36,40))})
    with TestClient(create_app(runtime)) as client:
        yield client,runtime

def queued(client,url,version):
    response=client.post(url,json={'version':version})
    assert response.status_code==200,response.text
    jid=response.json()['job_id']
    for _ in range(200):
        job=client.get('/api/jobs/'+jid).json()
        if job['status'] in ['completed','error']:
            assert job['status']=='completed',job
            return job['result']
        time.sleep(.01)
    pytest.fail('Fixture job did not finish')

def test_visible_boundary_purchase_idempotency_stale_and_terminal(app_client):
    client,runtime=app_client
    state=client.post('/api/sessions',json={'budget':6,'lambda_cost':0}).json()
    sid=state['id']; url='/api/sessions/'+sid
    assert state['observed_groups']==['initial']
    assert len(state['visible_values'])==5
    assert 'true_label' not in json.dumps(state)
    assert client.get(url+'/report').status_code==409
    state=queued(client,url+'/recommend',state['version'])
    group=state['recommendation']['selected_action']
    assert group is not None
    payload={'version':state['version'],'group_id':group,'idempotency_key':'purchase-first'}
    acquired=client.post(url+'/acquire',json=payload)
    assert acquired.status_code==200,acquired.text
    state=acquired.json()
    assert state['spent']==runtime.panels[group]['cost']
    assert len(state['visible_values'])==5+len(runtime.panels[group]['features'])
    assert client.post(url+'/acquire',json=payload).json()['spent']==state['spent']
    assert client.post(url+'/recommend',json={'version':0}).status_code==409
    ended=client.post(url+'/stop',json={'version':state['version']}).json()
    assert ended['status']=='completed'
    assert client.get(url+'/report').json()['true_label'] in [0,1,2]
    assert client.post(url+'/run',json={'version':ended['version']}).status_code==409

def test_hidden_changes_cannot_change_recommendation_or_leak(app_client):
    client,runtime=app_client
    state=client.post('/api/sessions',json={}).json()
    url='/api/sessions/'+state['id']
    before=queued(client,url+'/recommend',state['version'])['recommendation']
    _,index=runtime.store.get(state['id'])
    runtime.data['X'][index,5:]=987654321.25
    runtime.data['y'][index]=2
    after=queued(client,url+'/recommend',state['version'])['recommendation']
    for key in ['probabilities','candidates','selected_action','state_hash']:
        assert before[key]==after[key]
    for endpoint in [url,url+'/trace']:
        text=client.get(endpoint).text
        assert '987654321' not in text
        assert 'true_label' not in text
        assert 'row_index' not in text

def test_zero_budget_and_bounded_run(app_client):
    client,_=app_client
    zero=client.post('/api/sessions',json={'budget':0}).json()
    ended=queued(client,'/api/sessions/'+zero['id']+'/run',zero['version'])
    assert ended['spent']==0
    assert ended['stop_reason']=='budget_exhausted'
    state=client.post('/api/sessions',json={'budget':11,'policy':'random'}).json()
    ended=queued(client,'/api/sessions/'+state['id']+'/run',state['version'])
    assert len(ended['observed_groups'])<=6
    assert ended['spent']<=11
    assert ended['status']=='completed'

def test_invalid_inputs_and_unavailable_no_fallback(app_client):
    client,runtime=app_client
    assert client.post('/api/sessions',json={'budget':-1}).status_code==422
    assert client.post('/api/sessions',json={'costs':{'pressure':-1}}).status_code==409
    assert client.post('/api/sessions',json={'policy':'value'}).status_code==409
    assert client.get('/api/benchmarks').json()['runs']==[]
    runtime.engine=None
    runtime.blockers=['Model access disabled for this test.']
    assert client.get('/api/health').json()['model_ready'] is False
    assert client.post('/api/sessions',json={}).status_code==503

def test_model_failure_is_error_not_completed(app_client):
    client,runtime=app_client
    state=client.post('/api/sessions',json={}).json()
    def fail(X): raise RuntimeError('Deliberate fixture failure')
    runtime.predictor.predict_proba=fail
    response=client.post('/api/sessions/'+state['id']+'/run',json={'version':state['version']})
    jid=response.json()['job_id']
    for _ in range(200):
        job=client.get('/api/jobs/'+jid).json()
        if job['status']=='error': break
        time.sleep(.01)
    assert job['status']=='error'
    assert client.get('/api/sessions/'+state['id']).json()['status']=='error'
    assert client.get('/api/sessions/'+state['id']+'/report').status_code==409
