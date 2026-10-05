import json
from nextcheck.replay.public import public_replay
from test_api import app_client

def test_changed_artifacts_cannot_resume_or_report(app_client):
    client,runtime=app_client
    session=client.post('/api/sessions',json={}).json()
    runtime.artifact_ref='changed-data-or-model'
    assert client.get('/api/sessions/'+session['id']).status_code==409
    assert client.post('/api/sessions/'+session['id']+'/stop',json={'version':0}).status_code==409
    assert client.get('/api/sessions/'+session['id']+'/report').status_code==409

def test_interrupted_charged_reveal_is_error_on_recovery(app_client):
    client,runtime=app_client
    session=client.post('/api/sessions',json={}).json()
    with runtime.store.transaction():
        state,index=runtime.store.get(session['id'])
        state['probabilities']=None
        runtime.store.save(state,index)
    runtime._recover_interrupted()
    state=client.get('/api/sessions/'+session['id']).json()
    assert state['status']=='error'
    assert client.get('/api/sessions/'+session['id']+'/report').status_code==409

def test_replay_allowlist_drops_unpurchased_fields_and_label(app_client):
    _,runtime=app_client
    record={'session':{'id':'record','status':'active','observed_groups':['initial'],'visible_values':{'TS1__mean':1.,'PS1__mean':987654.},
                       'true_label':2,'row_index':99,'trace':[{'kind':'started','visible_values':{'TS1__mean':1,'PS1__mean':987654.},'true_label':2}]},
            'report':{'true_label':2},'hidden_row':[987654.]}
    public=public_replay(record,runtime.manifest)
    serialized=json.dumps(public)
    for private in ('987654','true_label','row_index','hidden_row'): assert private not in serialized

def test_default_session_and_initial_cost_contract(app_client):
    client,_=app_client
    response=client.post('/api/sessions',json={'policy':'default','costs':{'initial':0,'power':2}})
    assert response.status_code==200,response.text
    assert response.json()['policy']=='information'
    assert response.json()['trace'][0]['requested_policy']=='default'
    assert response.json()['trace'][0]['policy']=='information'
    assert client.post('/api/sessions',json={'policy':'all','budget':6}).status_code==409
    assert client.post('/api/sessions',json={'policy':'all','budget':11}).status_code==200

def test_persisted_benchmark_blockers_are_visible(app_client):
    client,runtime=app_client
    folder=runtime.settings.artifacts/'run_failed'
    folder.mkdir()
    (folder/'summary.json').write_text(json.dumps({'run_id':'failed','status':'blocked','results':[],'blockers':['Checkpoint unavailable']}))
    result=client.get('/api/benchmarks').json()
    assert result['blockers']==['failed: Checkpoint unavailable']

def test_corrupt_optional_utility_does_not_disable_model(app_client):
    client,runtime=app_client
    folder=runtime.settings.artifacts/'utility'; folder.mkdir()
    (folder/'execution_status.json').write_text('{"status":"complete"}')
    (folder/'value.json').write_text('{"corrupt":true}')
    runtime._load_utility()
    health=client.get('/api/health').json()
    assert health['model_ready'] is True
    assert health['utility_ready'] is False
    assert health['utility_blockers']

def test_frozen_default_requires_exact_utility_files(app_client):
    from nextcheck.evaluation.provenance import sha256
    client,runtime=app_client
    root=runtime.settings.artifacts
    data=root/'data'; data.mkdir()
    (data/'features.npz').write_bytes(b'data')
    (data/'splits.json').write_text('{}')
    utility=root/'utility'; utility.mkdir()
    for name in ('value','value_no_residual'): (utility/(name+'.json')).write_text('{}')
    run=root/'run_test'; run.mkdir()
    frozen={'input_hashes':{'features_sha256':sha256(data/'features.npz'),'splits_sha256':sha256(data/'splits.json'),
        'manifest_sha256':sha256(runtime.settings.root/'demo_manifest.json')},
        'utility_model_hashes':{name:sha256(utility/(name+'.json')) for name in ('value','value_no_residual')},
        'selection_rows':[30,31], 'choices':{'budget=6;lambda=0.02':{'default_policy':'prior','policies':{'prior':{'configuration':{}}}}}}
    (run/'frozen_choices.json').write_text(json.dumps(frozen))
    (run/'summary.json').write_text(json.dumps({'status':'complete','scope':'smoke','model_provenance':runtime.provenance(),
        'frozen_choices_sha256':sha256(run/'frozen_choices.json')}))
    selected=client.post('/api/sessions',json={'policy':'default'}).json()
    assert selected['policy']=='prior'
    (utility/'value.json').write_text('{"different":true}')
    selected=client.post('/api/sessions',json={'policy':'default'}).json()
    assert selected['policy']=='information'
