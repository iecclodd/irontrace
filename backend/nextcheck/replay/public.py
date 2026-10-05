"""Allowlisted recorded-replay output; arbitrary artifact keys never reach a browser."""
from copy import deepcopy

SESSION_KEYS=('id','version','status','mode','budget','spent','remaining_budget','policy','lambda_cost','costs','observed_groups',
              'probabilities','stop_reason','inference_ms','model_ref','created_at','selection_note')
METRICS=('group_id','cost','affordable','raw_kl','information','residual','entropy_drop','predicted_delta','net_value','current_entropy','margin')
SAMPLER_KEYS=('neighborhood_size','unique_donor_count','mean_distance','min_distance','max_distance','seed','rng_seed','distances','draws')
EVENT_KEYS=('kind','timestamp','group_id','cost','spent','probabilities','inference_ms','reason','message','budget','policy',
            'requested_policy','policy_configuration','selection_note',
            'selected_action','stop_reason','state_hash','model_ref','version','prediction_calls','hypothetical_query_rows')

def public_replay(record,manifest):
    source=record.get('session',{})
    panels={p['id']:p['features'] for p in manifest['panels']}
    groups=source.get('observed_groups',[])
    if any(group not in panels for group in groups):
        raise ValueError('Unknown replay panel.')
    def values(raw,allowed_groups):
        permitted={f for g in allowed_groups for f in panels[g]}
        return {k:v for k,v in raw.items() if k in permitted}
    session={k:deepcopy(source[k]) for k in SESSION_KEYS if k in source}
    session['mode']='recorded'
    session['visible_values']=values(source.get('visible_values',{}),groups)
    session['recommendation']=None
    events=[]
    visible=['initial']
    for raw in source.get('trace',record.get('trace',[])):
        if raw.get('kind')=='acquired':
            group=raw.get('group_id')
            if group not in panels: continue
            if group not in visible: visible.append(group)
        event={k:deepcopy(raw[k]) for k in EVENT_KEYS if k in raw}
        if 'visible_values' in raw:
            event['visible_values']=values(raw['visible_values'],visible)
        if 'candidates' in raw:
            event['candidates']=[]
            for candidate in raw['candidates']:
                if candidate.get('group_id') not in panels: continue
                safe={k:deepcopy(candidate[k]) for k in METRICS if k in candidate}
                sampler=candidate.get('sampler') or {}
                safe['sampler']={k:deepcopy(sampler[k]) for k in SAMPLER_KEYS if k in sampler}
                event['candidates'].append(safe)
        events.append(event)
    session['trace']=events
    output={k:deepcopy(record[k]) for k in ('id','title','timestamp','model_provenance','input_hashes','split_hash','policy_configuration') if k in record}
    output.update(session=session,trace=events)
    if source.get('status')=='completed':
        report=record.get('report',{})
        output['report']={k:deepcopy(report[k]) for k in ('true_label','log_loss','interpretation') if k in report}
    return output
