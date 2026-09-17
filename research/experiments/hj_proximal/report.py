"""Raw-evidence development report; no significance or classical-solver claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import problem,teacher,actor,sample,rollout
from .targets import expose, reconstruct
from .study import sha, utc


def run(root):
    cfg=json.loads((root/'config.json').read_text())
    report=json.loads((root/'report.json').read_text())
    counts=dict(phases=0,nonfinite_updates=0,budget_selections=0,proximal_accepted=0,proximal_rejected=0)
    work=[];gauge=[]
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            folder=root/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}'
            for spec in task['methods']:
                result=json.loads((folder/(spec['name']+'.json')).read_text())
                counts['phases']+=1;counts['nonfinite_updates']+=result['nonfinite_updates_skipped']
                for budget,item in result['budgets'].items():
                    assert item['available_seconds']<=float(budget)
                    counts['budget_selections']+=1
                if spec['mode']=='proximal':
                    counts['proximal_accepted']+=result['accepted_cycles'];counts['proximal_rejected']+=result['rejected_cycles']
                    previous=None
                    for event in result['events']:
                        current=event['incumbent_training_mean']
                        if previous is not None:assert current<=previous+1e-7*(1+abs(previous))
                        if not event['accepted']:assert current==previous
                        previous=current
                work.append(dict(family=task['family'],seed=seed,method=spec['name'],
                    updates=result['attempted_updates'],query_seconds=result.get('query_seconds',0.),
                    accepted=result.get('accepted_cycles'),rejected=result.get('rejected_cycles'),
                    setup_seconds=result['setup_seconds'],overrun_seconds=result['overrun_seconds'],
                    from_parent=result['budgets'].get('30.0',{}).get('from_parent'),
                    cost15=result['budgets'].get('15.0',{}).get('validation_mean'),
                    cost30=result['budgets'].get('30.0',{}).get('validation_mean')))
            # Explicitly post-hoc diagnostic: a zero physical action change can
            # still generate a nonzero absolute latent-regression target error.
            p=problem(task['family'],cfg['training_nodes'][task['family']],task['horizon'])
            a=actor(p,teacher(p),cfg['width']).cuda()
            saved=torch.load(folder/'parent.pt',map_location='cuda',weights_only=False)['budgets']['7.5']
            a.load_state_dict(saved['state_dict'])
            with torch.no_grad():
                _,xs,us=rollout(sample(p,16,110000+seed),a,p,keep=True)
                x=xs.flatten(0,1).contiguous();t=torch.arange(p.horizon,device='cuda').repeat(16)
                f,base,u=expose(a,x,t);z=a.net(f).squeeze(-1)
                scale=p.bound if p.family=='building' else 2*p.bound
                canonical=((u-base)/scale).clamp(-.995,.995).atanh()
                gauge.append(dict(family=p.family,seed=seed,latent_loss_for_unchanged_actions=float((z-canonical).square().mean()),
                    physical_action_reconstruction_error=float((reconstruct(a.net,f,base,p)-u).abs().max()),
                    scope='Post-hoc failure diagnosis on 16 original training initials, no new policy training.'))
            del a
    assert counts['phases']==sum(len(t['methods'])*len(cfg['seeds']) for t in cfg['tasks'])
    assert report['max_validation_replay_error']<1e-10
    preservation={
        'build/hj_teacher_transport_v7.pdf':'db2f8ed51bc2f2f1968b8f089ea00612c36e91d8a4bc71ec6555d0b805bd19ca',
        'experiments/results/hj_transport_primary_v7/completion_audit.json':'7a150aaad2da5094285cc978e91428fceeb9b63229c82bf24b290e0387173de2'}
    for path,expected in preservation.items():assert sha(path)==expected,path
    evidence=dict(created_utc=utc(),counts=counts,work=work,posthoc_latent_gauge_diagnostic=gauge,
        preserved_v7=preservation,evaluation_report_sha256=sha(root/'report.json'),
        max_validation_replay_error=report['max_validation_replay_error'])
    write(root/'development_audit.json',evidence)
    names={'mechanical':'기계계','reaction':'반응계','building':'건물'}
    methods={'fixed':'v7 고정 갱신','prox_action':'행동 목표 학습','prox_latent':'포화 전 출력 학습'}
    lines=['# v8 개발 실험 결과','',
        'HJB의 행동 최소화를 명시적인 teacher 목표로 만들고 student가 이를 학습하도록 구현했다. '
        '새 방법 두 개와 기존 비교군을 동일한 계산 예산으로 시험했다. 이 문서는 2개 seed의 개발 연구이며 확증 결과가 아니다.','',
        '## 바꾼 방법','',
        'Teacher가 자기 정책의 전체 미래 비용 미분을 계산한다. 이 정보로 제약을 만족하는 proximal Hamiltonian 행동 목표를 직접 구한다. '
        'Student는 행동 목표 또는 포화되기 전 신경망 출력 목표를 학습한다. 실제 training 비용 감소와 teacher가 예측한 감소를 비교해 개선 폭을 조절하고, 나빠진 시도는 되돌린다.','',
        '[방법과 실험 계획](hj_v8_improvement_plan.md) · [HJB 연결 및 한계](hj_v8_theory.md)','',
        '## 동일 예산에서의 새 상태 비용','',
        '총 30초에 부모 정책 7.5초, 준비·teacher 계산·거절한 시도·검증 시간이 포함된다. '
        '아래 비용은 두 seed가 공통의 새로운 256개 초기 상태에서 얻은 평균이다. 낮을수록 좋다. '
        '각 문제의 비교군은 Short, DPC, Warm DPC 중 가장 낮은 관측 평균이며, iLQR·QP·MPC를 직접 비교한 표는 아니다.','',
        '| 문제 | 가장 강한 학습 비교군 | 비교군 비용 | v7 고정 갱신 | 행동 목표 학습 | 포화 전 출력 학습 | 행동 목표의 비용 차이 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for case in report['cases']:
        if case['dimension']!=32 or case['condition']!='nominal':continue
        m=case['methods'];ref=case['best_learning_reference'];ratio=m['prox_action']['mean']/m[ref]['mean']
        lines.append(f"| {names[case['family']]} | {ref} | {m[ref]['mean']:.6f} | {m['fixed']['mean']:.6f} | {m['prox_action']['mean']:.6f} | {m['prox_latent']['mean']:.6f} | {(ratio-1)*100:+.3f}% |")
    lines+=['','## 고차원 가중치 전이와 분포 변화','','학습은 32차원에서 했다. 다음은 가중치를 256차원에 적용한 비용 차이다. 각 조건은 새로운 128개 초기 상태를 사용했다. 256차원에서 직접 학습한 결과가 아니다.','',
        '| 문제 | 조건 | 가장 강한 학습 비교군 | 행동 목표 비용 차이 | 포화 전 출력 비용 차이 |','|---|---|---|---:|---:|']
    for case in report['cases']:
        if case['dimension']!=256:continue
        m=case['methods'];ref=case['best_learning_reference']
        lines.append(f"| {names[case['family']]} | {case['condition']} | {ref} | {(m['prox_action']['mean']/m[ref]['mean']-1)*100:+.3f}% | {(m['prox_latent']['mean']/m[ref]['mean']-1)*100:+.3f}% |")
    lines+=['','## 검증과 실패 기록','',
        f"- 학습 {counts['phases']}개 완료, 저장 예산 선택 {counts['budget_selections']}개 확인.",
        f"- 저장된 모든 정책의 validation 비용 재실행 최대 오차 {report['max_validation_replay_error']:.3g}.",
        f"- 비정상 수치에 따른 업데이트 생략 {counts['nonfinite_updates']}회.",
        f"- 두 새 변형을 합쳐 실제 training 비용 검사에서 수용 {counts['proximal_accepted']}회, 거절 {counts['proximal_rejected']}회. 최초 teacher 생성은 수용 횟수에서 제외.",
        '- 해석적 행동 목표의 KKT 조건·목적 감소, 자동미분과 식의 일치, actor 재구성, CUDA/일반 Adam 업데이트 일치를 학습 전에 검사했다.',
        '- 새 상태 궤적의 동역학·비용을 별도 식으로 확인하고, 동일 배치 크기를 유지한 NumPy 동역학으로 feedback 비용을 검사했다.',
        '- v7 최종 PDF와 완료 검증 기록의 hash가 이전과 동일함을 확인했다.','',
        '포화 전 출력에 절대 목표를 주는 변형은 제어 입력을 바꾸지 않는 목표에서도 신경망 출력의 다른 표현을 요구할 수 있다. '
        '이는 projection이 여러 원래 출력을 같은 행동으로 보내기 때문이다. 이 문제를 확인하기 위한 사후 진단을 추가했으며, '
        '새 방법의 성능을 재조정하는 데 사용하지 않았다. 실패 결과도 그대로 보존했다.','',
        '| 문제 | seed | 행동 목표가 현재 행동과 같을 때의 latent 손실 |','|---|---:|---:|']
    for row in gauge:lines.append(f"| {names[row['family']]} | {row['seed']} | {row['latent_loss_for_unchanged_actions']:.6g} |")
    lines+=['','## 주장할 수 있는 범위','',
        '각 변형·문제·seed의 결과를 모두 남겼다. 두 seed의 개발 비교와 이들 정책의 새 상태 평가만으로 통계적으로 확증된 우위를 주장하지 않는다. '
        '강한 학습 비교군을 제외하거나 cold DPC 대비 큰 차이만 골라 고전 알고리즘을 압도했다고 해석해서는 안 된다. '
        '건물 문제는 이미 최적 비용에 가까워 큰 총비용 절감 자체의 여지가 작다.','',
        'MPC-Net의 Hamiltonian 학습과 guided policy search의 정책 지도라는 선행 아이디어가 있다. '
        '이 조합이 최초라고 주장하지 않으며, 실제 효용과 선행 방법 대비 차별성은 따로 입증해야 한다.','',
        '## 재현','',
        '```powershell',
        r'.\.venv-hj-cuda\Scripts\python.exe -m experiments.hj_proximal.audit',
        r'.\.venv-hj-cuda\Scripts\python.exe -m experiments.hj_proximal.study',
        r'.\.venv-hj-cuda\Scripts\python.exe -m experiments.hj_proximal.evaluate',
        r'.\.venv-hj-cuda\Scripts\python.exe -m experiments.hj_proximal.report',
        '```','',
        '기존 폴더에서는 완료된 학습을 건너뛴다. 새 전체 학습에는 별도 `--root`와 `--prepare`를 사용한다. '
        '설정과 소스 hash는 `development_lock.json`, 평가 전 가중치 hash는 `evaluation_lock.json`, '
        '원시 비용은 `evaluation/*.pt`, 최종 재검증은 `development_audit.json`에 있다.','']
    Path('docs/hj_v8_results_ko.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=[r for r in report['cases'] if r['dimension']==32 and r['condition']=='nominal']
    fig,axes=plt.subplots(1,3,figsize=(10,3.4),constrained_layout=True)
    colors=['#7c8799','#078377','#bc704d']
    for ax,case in zip(axes,cases):
        values=[(case['methods'][m]['ratio_to_best_learning_reference']-1)*100 for m in methods]
        ax.bar(['v7 fixed','Action target','Latent target'],values,color=colors)
        ax.axhline(0,color='black',lw=.7);ax.set_title(case['family'].capitalize())
        ax.tick_params(axis='x',labelrotation=20);ax.set_ylabel('Cost difference vs best learning reference (%)')
    fig.suptitle('Two-seed development study, matched 30 s total budget\nNew nominal d32 states; negative values are better',fontsize=12)
    fig.savefig(root/'cost_comparison.png',dpi=170);fig.savefig(root/'cost_comparison.pdf');plt.close(fig)
    write(root/'final_manifest.json',dict(completed_utc=utc(),status='complete',
        hashes={str(p):sha(p) for p in [root/'report.json',root/'development_audit.json',Path('docs/hj_v8_results_ko.md'),root/'cost_comparison.png',Path(__file__)]}))
    print(json.dumps(counts),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='experiments/results/hj_proximal_dev_v8a')
    args=parser.parse_args();torch.set_num_threads(1);run(Path(args.root))


if __name__=='__main__':main()
