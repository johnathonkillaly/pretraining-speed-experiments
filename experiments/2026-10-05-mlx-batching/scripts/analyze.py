"""Descriptive aggregation and unchanged preregistered two-round decision."""
import json
import statistics
from pathlib import Path

STUDY=Path(__file__).resolve().parents[1]


def analyze():
    records=[]
    for path in sorted((STUDY/"metrics").glob('B*_R*.json')):
        row=json.loads(path.read_text())
        if row.get('completed'): records.append(row)
    by_batch={}
    for record in records:
        by_batch.setdefault(record['batch_size'],[]).append(record)
    arms={}
    for batch,runs in sorted(by_batch.items()):
        measured=[row for run in runs for row in run['records'] if not row['warmup']]
        arms[str(batch)]={
            'runs':len(runs),'targets_per_second':32760/statistics.mean(r['total_seconds'] for r in measured),
            'model_seconds':statistics.mean(r['model_backward_accumulation_seconds'] for r in measured),
            'optimizer_seconds':statistics.mean(r['optimizer_seconds'] for r in measured),
            'peak_active_bytes':max(r['peak_active_bytes'] for r in runs),
            'numerical_gates_passed':all(r['numerical_gates_passed'] for r in runs),
            'finite_state':all(r['finite_final_state'] and r['finite_first_state'] for r in runs),
            'per_run':{r['tag']:r['steady_targets_per_second'] for r in runs}}
    decisions={}
    for batch,runs in sorted(by_batch.items()):
        if batch==1:continue
        paired=[]
        for i in (1,2):
            a=next((r for r in by_batch.get(1,[]) if r['tag']==f'B1_R{i}'),None)
            b=next((r for r in runs if r['tag']==f'B{batch}_R{i}'),None)
            if a and b:paired.append({'round':i,'speed_gain_pct':100*(b['steady_targets_per_second']/a['steady_targets_per_second']-1)})
        valid=arms[str(batch)]['numerical_gates_passed'] and arms[str(batch)]['finite_state'] and len(paired)==2
        decisions[str(batch)]={'pairs':paired,'useful':valid and all(p['speed_gain_pct']>=5 for p in paired),
                               'strong':valid and all(p['speed_gain_pct']>=10 for p in paired)}
    events=[json.loads(line) for line in (STUDY/'metrics/events.jsonl').read_text().splitlines()]
    monitors=[r for r in events if r['event']=='monitor']
    endings=[r for r in events if r['event'] in ('campaign_complete','campaign_stopped')]
    result={'arms':arms,'decisions':decisions,'full_context_validation':json.loads((STUDY/'metrics/validation_4095.json').read_text()) if (STUDY/'metrics/validation_4095.json').exists() else None,
            'campaign_end':endings[-1] if endings else None,
            'safety':{'maximum_thermal_state':max((r['thermal_state'] for r in monitors),default=None),
                      'minimum_recorded_memory_free_pct':min((r['memory_free_pct'] for r in monitors),default=None),
                      'all_monitored_on_ac':all(r['ac'] for r in monitors),
                      'maximum_worker_rss_bytes':max((r['worker_rss_bytes'] for r in monitors),default=None)}}
    (STUDY/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='full_context_validation'},indent=2))
    return result


if __name__=='__main__':analyze()
