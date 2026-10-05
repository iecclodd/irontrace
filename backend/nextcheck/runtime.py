from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from time import perf_counter
from uuid import uuid4
import hashlib
import json
import math
import numpy as np

from .settings import Settings
from .storage import Storage
from .jobs import Jobs

class StateError(ValueError):
    pass

def now():
    return datetime.now(timezone.utc).isoformat()

class Runtime:
    def __init__(self, settings=None, *, manifest=None, predictor=None, data=None, splits=None):
        self.settings = settings or Settings()
        self.manifest = manifest or json.loads((self.settings.root / 'demo_manifest.json').read_text(encoding='utf-8'))
        self.panels = {p['id']:p for p in self.manifest['panels']}
        self.features = [f for p in self.manifest['panels'] for f in p['features']]
        self.columns = {f:i for i,f in enumerate(self.features)}
        self.store = Storage(self.settings.artifacts / 'nextcheck.sqlite3')
        self.jobs = Jobs()
        self.lock = RLock()
        self.model_lock = RLock()
        self.predictor = predictor
        self.data = data
        self.splits = splits
        self.utility = {}
        self.utility_hashes = {}
        self.blockers = []
        self.loading = False
        self.engine = None
        self.artifact_ref = None
        self.utility_blockers = []
        if predictor is not None and data is not None:
            self._make_engine()
            self._recover_interrupted()

    def _make_engine(self):
        from .agent.loop import AcquisitionEngine
        self.engine = AcquisitionEngine(self.predictor,self.data['X'][self.splits['context']],self.manifest)
        digest=hashlib.sha256()
        for array in (self.data['X'],self.data['y']):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(json.dumps(self.splits,sort_keys=True).encode())
        digest.update(json.dumps(self.manifest,sort_keys=True).encode())
        digest.update(str(self.provenance().get('checkpoint_sha256','unverified')).encode())
        digest.update(str(self.provenance().get('context_sha256','unverified')).encode())
        for field in ('tabpfn_version','torch_version','n_estimators_resolved','fit_mode','random_state'):
            digest.update(str(self.provenance().get(field)).encode())
        self.artifact_ref=digest.hexdigest()

    def initialize(self):
        self.loading = True
        try:
            import os
            if os.environ.get('NEXTCHECK_DISABLE_MODEL')=='1':
                raise RuntimeError('Real model access is disabled by NEXTCHECK_DISABLE_MODEL=1. No fallback is used.')
            folder = self.settings.artifacts / 'data'
            if not (folder / 'features.npz').exists():
                raise RuntimeError('Dataset is not prepared. Run python scripts/prepare_data.py.')
            self.data = dict(np.load(folder / 'features.npz', allow_pickle=False))
            self.splits = json.loads((folder / 'splits.json').read_text(encoding='utf-8'))
            from .evaluation.benchmark import validate_splits
            validate_splits(self.splits,len(self.data['y']))
            if list(self.data['feature_names']) != self.features:
                raise RuntimeError('Prepared feature order does not match the panel manifest. Prepare the data again.')
            from .model.tabpfn35 import TabPFN35Predictor
            self.predictor = TabPFN35Predictor(self.data['X'][self.splits['context']],self.data['y'][self.splits['context']])
            # Readiness is established by real inference, not construction alone.
            row = self._masked(self.data['X'][self.splits['selection'][0]],['initial'])
            self._predict(row)
            self._make_engine()
            self._load_utility()
            self._recover_interrupted()
            self.blockers = []
        except Exception as exc:
            self.engine = None
            self.blockers = [str(exc)[:2000]]
        finally:
            self.loading = False

    def health(self):
        return dict(status='loading' if self.loading else ('ready' if self.engine else 'blocked'),model_ready=self.engine is not None,
            data_ready=self.data is not None,utility_ready=len(self.utility)==2,blockers=self.blockers,utility_blockers=self.utility_blockers,mode='live')

    def _load_utility(self):
        from .acquisition.utility import UtilityModel
        from .evaluation.runner import context_hash
        directory=self.settings.artifacts/'utility'
        if not directory.exists():
            return
        try:
            status=json.loads((directory/'execution_status.json').read_text(encoding='utf-8'))
            if status.get('status') not in ('completed','complete'):
                raise ValueError('Utility preparation is incomplete.')
            pending={}
            for name in ('value','value_no_residual'):
                model=UtilityModel.load(directory/(name+'.json'))
                metadata=model.training_metadata or {}
                if metadata.get('context_hash')!=context_hash(self.data['X'],self.data['y'],self.splits['context']):
                    raise ValueError('Utility context differs from the active prediction context.')
                if metadata.get('model_provenance',{}).get('checkpoint_sha256')!=self.provenance().get('checkpoint_sha256'):
                    raise ValueError('Utility checkpoint differs from the active predictor.')
                if not set(model.fitting_rows).issubset(set(self.splits['utility'])):
                    raise ValueError('Utility fitting rows are outside the utility partition.')
                from .evaluation.provenance import sha256
                hashes=metadata.get('input_hashes',{})
                expected_paths={'features_sha256':self.settings.artifacts/'data'/'features.npz',
                    'splits_sha256':self.settings.artifacts/'data'/'splits.json','manifest_sha256':self.settings.root/'demo_manifest.json'}
                for key,path in expected_paths.items():
                    if hashes.get(key)!=sha256(path):
                        raise ValueError('Utility input artifact differs from the active data/schema.')
                pending[name]=model
            self.utility=pending
            self.utility_hashes={name:sha256(directory/(name+'.json')) for name in pending}
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.utility={}
            self.utility_hashes={}
            self.utility_blockers=[str(exc)]

    def _assert_identity(self,payload):
        if payload.get('artifact_ref')!=self.artifact_ref:
            raise StateError('This session belongs to different data or model artifacts. Start a new session.')
        if payload.get('utility_model_ref') and payload['utility_model_ref']!=self.utility_hashes.get(payload['policy']):
            raise StateError('This session belongs to a different empirical value model. Start a new session.')

    def _recover_interrupted(self):
        with self.store.transaction():
            rows=self.store.connection.execute('SELECT payload,row_index FROM sessions').fetchall()
            for text,index in rows:
                state=json.loads(text)
                if state['status']=='active' and state.get('probabilities') is None:
                    state.update(status='error',stop_reason='error',version=state['version']+1)
                    self.store.save(state,index)
                    self._event(state['id'],'error',message='Prediction was interrupted after a charged reveal. Start a new session.')

    def provenance(self):
        return deepcopy(getattr(self.predictor,'provenance',{})) if self.predictor else {'status':'unavailable','blockers':self.blockers}

    def _ready(self):
        if self.engine is None:
            raise RuntimeError('; '.join(self.blockers) or 'Model is loading. Please wait.')

    def _masked(self,row,groups):
        masked = np.full(len(self.features),np.nan)
        for group in groups:
            for name in self.panels[group]['features']:
                masked[self.columns[name]] = row[self.columns[name]]
        return masked

    def _predict(self,row):
        with self.model_lock:
            start = perf_counter()
            q = np.asarray(self.predictor.predict_proba(np.asarray(row).reshape(1,-1)),dtype=float)[0]
            duration = (perf_counter()-start)*1000
        if q.shape != (3,) or not np.isfinite(q).all() or (q<0).any() or q.sum()<=0:
            raise RuntimeError('Model returned invalid class probabilities.')
        return (q/q.sum()).tolist(), duration

    def _public(self,payload):
        output = deepcopy(payload)
        output['trace'] = self.store.events(payload['id'])
        return output

    def get(self,sid):
        payload,_ = self.store.get(sid)
        self._assert_identity(payload)
        return self._public(payload)

    def _check(self,payload,version):
        self._assert_identity(payload)
        if payload['version'] != version:
            raise StateError('Stale session version. Refresh the session before continuing.')
        if payload['status'] != 'active':
            raise StateError('This session has ended. Create a new session to acquire more panels.')
        if payload.get('probabilities') is None:
            raise StateError('Session has no completed prediction. Start a new session.')

    def _event(self,sid,kind,**fields):
        self.store.append_event(sid,dict(kind=kind,timestamp=now(),**fields))

    def create(self,request):
        self._ready()
        policies = ['default','static','all','information','raw_kl','entropy_drop','random','initial','prior','value','value_no_residual']
        if request.policy not in policies:
            raise StateError('Unknown or unavailable session policy.')
        if request.policy.startswith('value') and request.policy not in self.utility:
            raise StateError('Fit the empirical value models before choosing this policy.')
        costs = {p['id']:float(p['cost']) for p in self.manifest['panels']}
        for group,cost in (request.costs or {}).items():
            if group=='initial' and cost==0:
                continue
            if group not in self.panels or group=='initial' or not math.isfinite(cost) or cost<0 or cost>1000:
                raise StateError('Costs must be finite nonnegative assumptions for purchasable panels.')
            costs[group] = cost
        policy=request.policy
        configuration={}
        selection_note='Provisional information heuristic; no matching development-selected configuration.'
        default=self._selected_configuration(request.budget,request.lambda_cost,costs)
        if policy=='default':
            policy=default['default_policy'] if default else 'information'
        if default and policy in default['policies']:
            configuration=default['policies'][policy].get('configuration',{})
            selection_note=default['selection_note']
        if policy=='static' and not default:
            raise StateError('No development-selected static subset matches this budget, cost, and lambda configuration.')
        if policy=='all' and sum(costs[g] for g in costs if g!='initial')>request.budget:
            raise StateError('All panels exceed this session budget.')
        if policy.startswith('value') and policy not in self.utility:
            raise StateError('Matching empirical value models are unavailable.')
        sid = uuid4().hex
        # Sessions draw from development examples; final test rows stay evaluator-only.
        pool = self.splits['selection']
        index = int(pool[int(sid[:8],16)%len(pool)])
        masked = self._masked(self.data['X'][index],['initial'])
        if policy=='prior':
            q=(np.bincount(self.data['y'][self.splits['context']].astype(int),minlength=3)/len(self.splits['context'])).tolist()
            duration=0.
        else:
            q,duration = self._predict(masked)
        values = {f:float(masked[i]) for f,i in self.columns.items() if np.isfinite(masked[i])}
        payload = dict(id=sid,version=0,status='active',mode='live',budget=request.budget,spent=0.,remaining_budget=request.budget,
            policy=policy,lambda_cost=request.lambda_cost,costs=costs,observed_groups=['initial'],visible_values=values,
            policy_configuration=configuration,selection_note=selection_note,
            utility_model_ref=self.utility_hashes.get(policy),
            probabilities=q,recommendation=None,stop_reason=None,inference_ms=duration,
            model_ref=str(self.provenance().get('checkpoint_sha256','unverified')),artifact_ref=self.artifact_ref,created_at=now())
        with self.store.transaction():
            self.store.save(payload,index)
            self._event(sid,'started',probabilities=q,visible_values=values,budget=request.budget,
                requested_policy=request.policy,policy=policy,policy_configuration=configuration,selection_note=selection_note)
        return self.get(sid)

    def _selected_configuration(self,budget,lambda_cost,costs):
        if any(costs[p['id']]!=p['cost'] for p in self.manifest['panels']):
            return None
        from .evaluation.provenance import sha256
        key=f'budget={budget:g};lambda={lambda_cost:g}'
        for summary_path in sorted(self.settings.artifacts.glob('run_*/summary.json'),reverse=True):
            try:
                summary=json.loads(summary_path.read_text(encoding='utf-8'))
                if summary.get('status') not in ('complete','partial'): continue
                if summary['model_provenance'].get('context_sha256')!=self.provenance().get('context_sha256'): continue
                if summary['model_provenance'].get('checkpoint_sha256')!=self.provenance().get('checkpoint_sha256'): continue
                path=summary_path.parent/'frozen_choices.json'
                if sha256(path)!=summary['frozen_choices_sha256']: continue
                frozen=json.loads(path.read_text(encoding='utf-8'))
                hashes=frozen['input_hashes']
                if hashes['features_sha256']!=sha256(self.settings.artifacts/'data'/'features.npz'): continue
                if hashes['splits_sha256']!=sha256(self.settings.artifacts/'data'/'splits.json'): continue
                if hashes['manifest_sha256']!=sha256(self.settings.root/'demo_manifest.json'): continue
                if any(frozen.get('utility_model_hashes',{}).get(name)!=sha256(self.settings.artifacts/'utility'/(name+'.json')) for name in ('value','value_no_residual')): continue
                if self.utility_hashes and frozen['utility_model_hashes']!=self.utility_hashes: continue
                selected=frozen['choices'][key]
                selected['selection_note']=f"Selected on {len(frozen['selection_rows'])} development rows ({summary['scope']} evaluation)."
                return selected
            except (OSError,ValueError,KeyError,TypeError):
                continue
        return None

    def recommend(self,sid,version):
        self._ready()
        with self.lock:
            payload,_=self.store.get(sid)
            self._check(payload,version)
            snapshot=deepcopy(payload)
        from .agent.state import ObservedCase
        from .agent.loop import AcquisitionEngine
        manifest=deepcopy(self.manifest)
        for panel in manifest['panels']:
            panel['cost']=snapshot['costs'][panel['id']]
        engine=AcquisitionEngine(self.predictor,self.data['X'][self.splits['context']],manifest,utility_model=self.utility.get(snapshot['policy']))
        observed=ObservedCase(tuple(snapshot['observed_groups']),snapshot['visible_values'])
        if snapshot['policy'] in ('prior','static','all'):
            groups=snapshot.get('policy_configuration',{}).get('groups',[]) if snapshot['policy']=='static' else [p for p in self.panels if p!='initial']
            if snapshot['policy']=='prior': groups=[]
            remaining=[g for g in groups if g not in snapshot['observed_groups'] and snapshot['costs'][g]<=snapshot['remaining_budget']]
            selected=remaining[0] if remaining else None
            analysis=dict(probabilities=snapshot['probabilities'],candidates=[],selected_action=None,
                stop_reason='no_predicted_net_value',inference_ms=0.,prediction_calls=0,hypothetical_query_rows=0,
                state_hash=hashlib.sha256(json.dumps(snapshot['visible_values'],sort_keys=True).encode()).hexdigest())
            analysis['selected_action']=selected
            analysis['stop_reason']=None if selected else ('no_groups_remaining' if len(snapshot['observed_groups'])==6 else 'no_predicted_net_value')
            analysis['candidates']=[dict(group_id=g,cost=snapshot['costs'][g],affordable=True,raw_kl=None,information=None,residual=None,entropy_drop=None,predicted_delta=None,net_value=None,sampler=None) for g in remaining]
        else:
            with self.model_lock:
                threshold=snapshot.get('policy_configuration',{}).get('threshold',snapshot['lambda_cost']) if snapshot['policy'] in ('information','raw_kl','entropy_drop') else snapshot['lambda_cost']
                analysis=engine.analyze(observed,snapshot['remaining_budget'],policy=snapshot['policy'],lambda_cost=threshold,seed=self.settings.seed)
        # Convert any NumPy scalar outputs before persisting.
        analysis=json.loads(json.dumps(analysis,default=lambda v:v.tolist() if hasattr(v,'tolist') else float(v),allow_nan=False))
        analysis['version']=version
        analysis['model_ref']=snapshot['model_ref']
        with self.lock,self.store.transaction():
            current,index=self.store.get(sid)
            self._check(current,version)
            current['recommendation']=analysis
            current['inference_ms']=analysis.get('inference_ms',0)
            self.store.save(current,index)
            self._event(sid,'recommended',**analysis)
        return self.get(sid)

    def acquire(self,sid,request):
        self._ready()
        with self.lock:
            with self.store.transaction():
                current,index=self.store.get(sid)
                self._assert_identity(current)
                previous=self.store.purchase(sid,request.idempotency_key)
                if previous:
                    if previous!=request.group_id:
                        raise StateError('Idempotency key was already used for another group.')
                    return self._public(current)
                self._check(current,request.version)
                if request.group_id not in self.panels or request.group_id=='initial':
                    raise StateError('Unknown purchasable panel.')
                if request.group_id in current['observed_groups']:
                    return self._public(current)
                recommendation=current['recommendation']
                if not recommendation or recommendation['version']!=current['version']:
                    raise StateError('Obtain a current recommendation before acquiring a panel.')
                if recommendation.get('selected_action')!=request.group_id:
                    raise StateError('Panel does not match the current recommended action.')
                cost=current['costs'][request.group_id]
                if cost>current['remaining_budget']+1e-12:
                    raise StateError('Panel exceeds remaining budget.')
                revealed={f:float(self.data['X'][index,self.columns[f]]) for f in self.panels[request.group_id]['features']}
                current['visible_values'].update(revealed)
                current['observed_groups'].append(request.group_id)
                current['spent']+=cost
                current['remaining_budget']=max(0.,current['budget']-current['spent'])
                current['version']+=1
                current['recommendation']=None
                current['probabilities']=None
                self.store.record_purchase(sid,request.idempotency_key,request.group_id)
                self.store.save(current,index)
                self._event(sid,'acquired',group_id=request.group_id,cost=cost,spent=current['spent'],visible_values=revealed)
            try:
                q,duration=self._predict(self._masked(self.data['X'][index],current['observed_groups']))
                with self.store.transaction():
                    current['probabilities']=q
                    current['inference_ms']=duration
                    self.store.save(current,index)
                    self._event(sid,'predicted',probabilities=q,inference_ms=duration)
            except Exception as exc:
                with self.store.transaction():
                    current.update(status='error',stop_reason='error')
                    self.store.save(current,index)
                    self._event(sid,'error',message='Model prediction failed after acquisition; no completed prediction is available.')
                raise RuntimeError('Model prediction failed after acquisition. Session ended with error.') from exc
            return self.get(sid)

    def stop(self,sid,version,reason='user_stopped'):
        with self.lock,self.store.transaction():
            payload,index=self.store.get(sid)
            self._check(payload,version)
            payload.update(status='completed',stop_reason=reason,version=version+1,recommendation=None)
            self.store.save(payload,index)
            self._event(sid,'stopped',reason=reason)
        return self.get(sid)

    def queue_recommend(self,sid,version):
        self._check(self.store.get(sid)[0],version)
        def work(progress):
            try:
                return self.recommend(sid,version)
            except StateError:
                raise
            except Exception as exc:
                self._fail(sid,version)
                raise RuntimeError('Numerical recommendation failed; inspect local model readiness.') from exc
        return self.jobs.submit(work)

    def _fail(self,sid,version):
        with self.lock,self.store.transaction():
            payload,index=self.store.get(sid)
            if payload['status']=='active' and payload['version']==version:
                payload.update(status='error',stop_reason='error',version=version+1)
                self.store.save(payload,index)
                self._event(sid,'error',message='Numerical inference failed.')

    def queue_run(self,sid,version):
        self._check(self.store.get(sid)[0],version)
        def work(progress):
            current_version=version
            try:
                for step in range(6):
                    state=self.get(sid)
                    if state['status']!='active':
                        return state
                    self._check(state,current_version)
                    state=self.recommend(sid,current_version)
                    rec=state['recommendation']
                    if not rec.get('selected_action'):
                        return self.stop(sid,state['version'],rec.get('stop_reason') or 'no_predicted_net_value')
                    from .schemas import Acquire
                    state=self.acquire(sid,Acquire(version=state['version'],group_id=rec['selected_action'],idempotency_key=uuid4().hex))
                    current_version=state['version']
                    progress((step+1)/6)
                raise RuntimeError('Purchase bound exceeded.')
            except StateError:
                if self.get(sid)['status']!='active':
                    return self.get(sid)
                raise
            except Exception:
                self._fail(sid,current_version)
                raise RuntimeError('Automated acquisition stopped because numerical inference failed.')
        return self.jobs.submit(work)

    def report(self,sid):
        state,index=self.store.get(sid)
        self._assert_identity(state)
        if state['status']!='completed':
            raise StateError('Reports and true labels are available only after successful session termination.')
        label=int(self.data['y'][index])
        return dict(session=self.get(sid),true_label=label,log_loss=-math.log(max(1e-8,state['probabilities'][label])),
            interpretation='Retrospective pump leakage classification; not a safety or repair assessment.')

    def benchmarks(self):
        runs=[]
        for path in sorted(self.settings.artifacts.glob('run_*/summary.json'),reverse=True):
            try:
                summary=json.loads(path.read_text(encoding='utf-8'))
                scatter=path.parent/'value_scatter.json'
                if scatter.is_file(): summary['value_scatter']=json.loads(scatter.read_text(encoding='utf-8'))
                runs.append(summary)
            except (OSError,ValueError):
                continue
        blockers=[f"{run.get('run_id','Evaluation')}: {message}" for run in runs for message in run.get('blockers',[])]
        if not runs: blockers=['No persisted evaluation results. Run python scripts/prepare_utility.py then python scripts/evaluate.py.']
        return {'runs':runs,'blockers':blockers}

    def replays(self):
        found=[]
        for path in sorted((self.settings.artifacts/'replays').glob('*.json')):
            try:
                content=json.loads(path.read_text(encoding='utf-8'))
                if content.get('model_provenance',{}).get('checkpoint_sha256'):
                    found.append(dict(id=path.stem,title=content.get('title',path.stem),timestamp=content.get('timestamp')))
            except (OSError,ValueError):
                continue
        return {'replays':found}
