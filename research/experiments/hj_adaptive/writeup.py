"""Compile the explicitly versioned v7 research manuscript."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np


NAMES={'mechanical':'Mechanical','reaction':'Reaction','building':'Building',
       'adaptive':'Teacher transport','fixed':'Fixed transport','fresh':'Prefix-Fresh',
       'dpc':'DPC','warm':'Warm DPC','short':'Short','dpc_full':'Full DPC',
       'warm_full':'Warm Full DPC','dpc_initial':'Initial DPC','warm_initial':'Initial Warm DPC'}


def number(x):
    if x is None:return '--'
    return f'{x:.5g}'


def table(filename,columns,headers,rows,caption,label):
    lines=[r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}',
           r'\begin{tabular}{@{}'+columns+r'@{}}',r'\toprule',
           ' & '.join(headers)+r'\\',r'\midrule']
    lines += [' & '.join(map(str,row))+r'\\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{'+caption+'}',
              r'\label{'+label+'}',r'\end{table}']
    Path('paper/sections',filename).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def result_tables(root):
    from .study import verify,digest
    cfg=verify(root)
    summary=json.loads((root/'report/summary.json').read_text())
    audit=json.loads((root/'completion_audit.json').read_text())
    assert audit['evidence_hashes'][str(root/'report/summary.json')]==digest(root/'report/summary.json')
    primary=[];costmap={};transfers=[];work=[];diagnostics=[];selection=[]
    evidence={'primary':{},'transfer':[],'work':[],'gradients':[],
        'actual_batch_gradient_checks':audit['actual_batch_gradient_checks'],
        'actual_batch_gradient_failures':audit['actual_batch_gradient_failures'],
        'prefix_equality_gradient_checks':audit['prefix_equality_gradient_checks']}
    for task in cfg['tasks']:
        family=task['family'];label=NAMES[family];ref=task['principal_reference']
        decision=summary['primary_decisions'][family]
        evidence['primary'][family]=decision
        for name in task['primary_references']:
            row=decision['comparisons'][name]
            primary.append([label,NAMES[name],f"{row['pooled_ratio']:.5f}",
                number(row['noninferiority_upper_mean_difference']),
                'Pass' if row['noninferior'] else 'Fail'])
        case=f"{family}_h{task['horizon']}_d32_nominal"
        c={row['method']:row['mean'] for row in summary['costs'] if row['case']==case and row['budget']==60}
        costmap[family]=c
        for row in summary['comparisons']:
            if row['family']!=family or row['dimension']!=256 or row['budget']!=60 or row['reference']!=ref:continue
            lo,hi=row['descriptive_ratio_interval']
            transfers.append([label,row['condition'].capitalize(),NAMES[ref],
                f"{row['pooled_ratio']:.5f}",f'[{lo:.5f}, {hi:.5f}]'])
            evidence['transfer'].append(row)
        for name in ['fresh','fixed','adaptive','dpc','warm','dpc_full','warm_full']:
            rows=[r for r in summary['training'] if r['family']==family and r['method']==name]
            attempts=sum(r['attempts'] for r in rows)
            accepted=sum(r['accepted_refreshes'] for r in rows);refreshes=sum(r['refreshes'] for r in rows)
            retained=sum(r['retains_parent'] is True for r in rows)
            q=dict(family=family,method=name,mean_updates=float(np.mean([r['attempts'] for r in rows])),
                mean_query_seconds=float(np.mean([r['query_seconds'] for r in rows])),
                mean_refreshes=refreshes/len(rows),accepted_fraction=accepted/refreshes if refreshes else None,
                retained_parent=retained,clipped_fraction=sum(r['clipped_updates'] for r in rows)/max(1,attempts),
                mean_setup_seconds=float(np.mean([r['setup_seconds'] for r in rows])))
            spec=next(s for s in task['methods'] if s['name']==name)
            if spec['mode']=='transport':
                reasons={'action_only':0,'age_only':0,'action_and_age':0,'no_trigger':0}
                teacher_changes=[]
                for seed in cfg['seeds']:
                    path=root/'training'/f'{family}_h{task["horizon"]}_s{seed}'/(name+'.json')
                    record=json.loads(path.read_text());last_refresh=0;accepted_costs=[]
                    for event in record['events']:
                        if event['reason']=='transport_refresh':
                            last_refresh=event['step']
                            if event['accepted']:accepted_costs.append(event['incumbent_training_mean'])
                        elif event['reason']=='action_probe':
                            action=event['action_drift'] is None or event['action_drift']>spec['action_tolerance']
                            age=event['step']-last_refresh>=spec['max_age']
                            assert event['trigger']==(action or age)
                            key='action_and_age' if action and age else 'action_only' if action else 'age_only' if age else 'no_trigger'
                            reasons[key]+=1
                    teacher_changes.append(100*(1-accepted_costs[-1]/accepted_costs[0]))
                q.update(action_probe_reasons=reasons,
                    mean_accepted_teacher_training_improvement_percent=float(np.mean(teacher_changes)),
                    teacher_improvement_scope='Float32 mean on the fixed512traininginitials, final accepted versus initial teacher. This does not measure fresh-state improvement or guarantee that the final teacher was available before the deadline.')
            evidence['work'].append(q)
            if name in ['fixed','adaptive']:
                work.append([label,NAMES[name],f"{q['mean_updates']:.0f}",f"{q['mean_query_seconds']:.2f}",
                    f"{q['mean_refreshes']:.1f}",f"{100*q['accepted_fraction']:.1f}\\%",f'{retained}/10'])
        rows=[r for r in audit['prefix_equality_gradient_checks'] if r['family']==family and r['method']=='adaptive']
        finite=[r for r in rows if r['finite_gradients']]
        transport=[r for r in summary['transport_gradient_diagnostics'] if r['family']==family]
        errors=[r['relative_gradient_error'] for r in transport if r['finite']]
        d=dict(family=family,prefix_finite=len(finite),prefix_checks=len(rows),
            prefix_median_cosine=float(np.median([r['gradient_cosine'] for r in finite])) if finite else None,
            prefix_non_descent=sum(not r['prefix_negative_gradient_is_full_descent'] for r in finite),
            transport_checks=len(transport),transport_passes=sum(r['agreement_within_audit_tolerance'] for r in transport),
            transport_max_relative_error=max(errors) if errors else None)
        evidence['gradients'].append(d)
        diagnostics.append([label,number(d['prefix_median_cosine']),f"{d['prefix_non_descent']}/{d['prefix_finite']}",
            f"{d['transport_passes']}/{d['transport_checks']}",number(d['transport_max_relative_error'])])
        spec=next(s for s in task['methods'] if s['name']=='adaptive')
        selection.append([label,'Anchor CV' if spec.get('variance_reduction') else 'Action drift',
            number(spec['lr']),str(spec['refresh_every']) if 'refresh_every' in spec else str(spec['action_tolerance']),
            NAMES[ref]])
    table('hj_v7_selection_table.tex','llrrl',['Family','Selected candidate','LR','Period / threshold','Principal reference'],selection,
        'Configurations fixed using development only. All selected candidates start from the same charged Short parent. The principal reference is selected before primary testing and does not replace the full reference set.', 'tab:v7-selection')
    table('hj_v7_primary_table.tex','llrrl',['Family','Reference','$\overline J_A/\overline J_B$','$U_{A-1.01B}$','Component'],primary,
        'Primary noninferiority components at dimension 32 and 60 seconds. Lower cost is better. The paired-seed arithmetic-difference upper bound uses one-sided $\\alpha=0.05/3$. A family passes only when every component passes; individual component labels are not separate multiplicity-adjusted discoveries.', 'tab:v7-primary')
    cost_rows=[[NAMES[name]]+[f"{costmap[t['family']][name]:.7g}" if name in costmap[t['family']] else '--' for t in cfg['tasks']]
        for name in ['adaptive','fixed','fresh','dpc','warm','short','dpc_full','warm_full','dpc_initial','warm_initial']]
    table('hj_v7_cost_table.tex','lrrr',['Method','Mechanical','Reaction','Building'],cost_rows,
        'Mean fresh cost at dimension 32 and the 60-second allocation. Each entry pools ten trained policies over the same 512 initial states. Full denotes full-initial-set DPC. Additional initial-objective counterparts are present when mixed-objective DPC or Warm DPC won development selection; otherwise the selected DPC and Warm DPC already use that objective.', 'tab:v7-costs')
    table('hj_v7_transfer_table.tex','lllrl',['Family','Condition','Reference','Cost ratio','Descriptive 95\\% interval'],transfers,
        'Transfer to dimension 256 with unchanged neural weights, relative to the principal reference selected during development. Crossed seed/state bootstrap intervals are descriptive; these cases do not replace the nominal primary decisions.', 'tab:v7-transfer')
    table('hj_v7_work_table.tex','llrrrrr',['Family','Method','Updates','Query s','Sweeps','Accepted','Parent'],work,
        'Actual training work for the two transport methods, averaged over ten seeds, including any final block that overruns its deadline. Query seconds include teacher evaluation and action probes. Acceptance is the pooled fraction of attempted teacher sweeps, including initial evaluation and possible unchanged-policy acceptance. Parent counts selected 60-second policies whose weights exactly retain the parent. Overrunning work cannot supply a selected policy.', 'tab:v7-work')
    table('hj_v7_gradient_table.tex','lrrrr',['Family','Prefix cosine','Non-descent','All-time pass','Max. error'],diagnostics,
        'Training-state diagnostics. Prefix columns assign identical selected candidate weights to teacher and student, comparing the exact prefix-plus-teacher gradient with the full student gradient over all 512 original initial states (ten policies per family). Non-descent counts nonpositive inner products among finite checks. All-time columns reevaluate both selected transport methods as their own teachers on the first 128 training states (20 checks per family), using a relative-gradient tolerance of $2\\times10^{-4}$. These descriptive checks do not select or change a policy.', 'tab:v7-gradients')
    (root/'report/writeup_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print('Generated six tables and supporting writeup_evidence.json from the completed audit')


def compile_paper():
    root=Path.cwd();target=root/'build/hj_teacher_transport_v7_compile.log'
    job='hj_teacher_transport_v7'
    commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=../build',job+'.tex'],
        ['bibtex','../build/'+job],
        ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=../build',job+'.tex'],
        ['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=../build',job+'.tex']]
    with target.open('w',encoding='utf-8') as log:
        for command in commands:
            code=subprocess.run(command,cwd=root/'paper',stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError('Compile failed; see '+str(target))
    print('Compiled '+job+'.pdf; visual review and final result audit still required')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--compile',action='store_true')
    parser.add_argument('--tables',action='store_true');parser.add_argument('--root',default='experiments/results/hj_transport_primary_v7');args=parser.parse_args()
    if args.tables:result_tables(Path(args.root))
    if args.compile:compile_paper()
