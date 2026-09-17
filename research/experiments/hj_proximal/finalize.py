"""Combined, descriptive v8 assessment; preserves all previously locked sources."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.hj_gridfree.engine import write
from experiments.hj_cotangent.systems import problem,teacher,actor,sample
from .curvature import CurvatureGraph,perturbed_trajectory,weighted_target
from .targets import feasible
from .study import sha,utc


def supplemental_curvature_audit():
    """Check captured random-probe estimates against directional differences."""
    rows=[];torch.manual_seed(851901)
    for family in ['mechanical','reaction','building']:
        p=problem(family,2,4);a=actor(p,teacher(p),8).cuda()
        graph=CurvatureGraph(a,p,2,4)
        x=sample(p,2,851902);gen=torch.Generator(device='cuda').manual_seed(851903)
        got=graph(x,gen);torch.cuda.synchronize();expected=torch.zeros_like(got[-1]);eps=1e-3
        for sign in graph.signs:
            values=[]
            for direction in [-1.,1.]:
                delta=(direction*eps*sign).detach().requires_grad_(True)
                value=perturbed_trajectory(x,delta,a,p)[0]
                values.append(torch.autograd.grad(value.sum(),delta)[0])
            expected.add_(sign*(values[1]-values[0])/(2*eps*graph.probes))
        error=float(((expected-got[-1]).abs()/(1+got[-1].abs())).max())
        assert error<2e-3,(family,error)
        # Exercise anisotropic active constraints independently of trained data.
        old=feasible(10*torch.randn(64,p.n,device='cuda',dtype=torch.float64),p)
        g=torch.randn_like(old);diagonal=torch.exp(5*torch.randn_like(old))
        target,h=weighted_target(old,g,diagonal,p,16.)
        r=p.dt*({'mechanical':.04,'reaction':.1,'building':.025}[family])/p.n
        metric=h+16*2*r;stationarity=g+metric*(target-old)
        if family=='building':
            # Projected gradient fixed point is necessary and sufficient.
            residual=(feasible(target-stationarity/metric,p)-target).abs().max()
        else:
            multiplier=(-(stationarity*target).sum(-1,keepdim=True)/target.square().sum(-1,keepdim=True).clamp_min(1e-20)).clamp_min(0.)
            residual=((stationarity+multiplier*target).abs()/(1+g.abs()+metric*old.abs())).max()
        residual=float(residual);assert residual<2e-6,(family,residual)
        rows.append(dict(family=family,captured_diagonal_finite_difference_error=error,weighted_projection_kkt_error=residual))
        del graph,a
    return rows


def main():
    torch.set_num_threads(1)
    roots=[Path('experiments/results/hj_proximal_dev_v8a'),Path('experiments/results/hj_curvature_dev_v8b')]
    reports=[];work=[];verified_hashes=0;unique_phases=0;unique_selections=0;raw_costs=0
    for index,root in enumerate(roots):
        cfg=json.loads((root/'config.json').read_text())
        for file in ['development_lock.json','evaluation_lock.json']:
            lock=json.loads((root/file).read_text())
            for path,expected in lock['hashes'].items():assert sha(path)==expected,path;verified_hashes+=1
            for path,item in lock.get('copied_artifacts',{}).items():assert sha(path)==item['sha256'],path
        report=json.loads((root/'report.json').read_text());reports.append(report)
        assert report['max_validation_replay_error']<1e-10
        for path in (root/'evaluation').glob('*.pt'):
            raw=torch.load(path,map_location='cpu',weights_only=False)
            meta=json.loads(path.with_suffix('.json').read_text())
            for name,values in raw['methods'].items():
                assert abs(float(values.mean())-meta['methods'][name]['mean'])<1e-12
                assert np.max(np.abs(values.mean(1).numpy()-np.array(meta['methods'][name]['seed_means'])))<1e-12
                raw_costs+=values.numel()
        for task in cfg['tasks']:
            for seed in cfg['seeds']:
                folder=root/'training'/f'{task["family"]}_h{task["horizon"]}_s{seed}'
                for spec in task['methods']:
                    if index==1 and spec['name']!='curvature_action':continue
                    result=json.loads((folder/(spec['name']+'.json')).read_text());unique_phases+=1
                    assert result['nonfinite_updates_skipped']==0
                    for budget,selected in result['budgets'].items():
                        assert selected['available_seconds']<=float(budget);unique_selections+=1
                    if spec['mode']=='proximal':
                        previous=None
                        for event in result['events']:
                            value=event['incumbent_training_mean']
                            if previous is not None:assert value<=previous+1e-7*(1+abs(previous))
                            if not event['accepted']:assert value==previous
                            previous=value
                    if spec['name']=='curvature_action':
                        work.append(dict(family=task['family'],seed=seed,setup_seconds=result['setup_seconds'],
                            query_seconds=result['query_seconds'],updates=result['attempted_updates'],
                            accepted=result['accepted_cycles'],rejected=result['rejected_cycles'],
                            retained_parent=result['budgets']['30.0']['from_parent'],overrun_seconds=result['overrun_seconds']))
    supplement=supplemental_curvature_audit()
    dest=Path('build/hj_v8_final');dest.mkdir(exist_ok=True)
    audit=dict(completed_utc=utc(),passed=True,new_training_phases=unique_phases,
        unique_budget_selections=unique_selections,raw_costs_recomputed=raw_costs,verified_source_artifact_hashes=verified_hashes,
        additional_curvature_work=work,supplemental_curvature_audit=supplement,
        caveat='42 initial development phases plus 4 new curvature phases. The 28 baseline phase artifacts copied to v8b are not independent new training runs.')
    assert unique_phases==46 and unique_selections==126
    write(dest/'completion_audit.json',audit)
    names={'mechanical':'기계계','reaction':'반응계','building':'건물'}
    lines=['# v8 방법론 개선 평가','',
        '**방법론을 바꿔 실제 개선은 얻었다. 그러나 고전 알고리즘을 크게 압도했다는 근거는 아직 없다.** '
        '새 학습 46개와 새 상태 평가를 완료했고, 지금 추가 학습이 계속 돌아가는 상태는 아니다.','',
        '## 이번에 구현한 세 가지 수정','',
        '1. **행동 목표를 직접 가르치기.** Teacher의 미래 비용 미분으로 제약이 있는 proximal Hamiltonian 최소화 문제를 풀어 행동 목표를 만든다. '
        'Student는 그 목표를 학습한다. 실제 비용과 예측 비용의 차이로 다음 개선 폭을 조절한다.',
        '2. **포화 전 신경망 출력에 목표 주기.** 포화의 gradient 문제를 겨냥했지만, 현재 구현은 잘 작동하지 않았다. '
        '같은 행동에 여러 원래 출력이 대응해 불필요하게 다른 표현을 학습하도록 요구하는 문제를 진단했다.',
        '3. **미래 비용 곡률도 전달하기.** 전체 Hessian 없이 네 번의 Hessian–vector product로 대각 곡률을 추정했다. '
        '수치 검사는 통과했으나, 30초 예산에서는 준비·teacher 계산 비용에 비해 충분한 개선을 얻지 못했다.','',
        '## 가장 유망한 수정의 결과','',
        '다음은 첫 개발 연구의 새 32차원 초기 상태 256개에서 얻은 평균 비용 차이다. '
        '같은 총 30초 예산에 부모 정책, 준비, 실패한 시도, teacher 계산과 검증을 포함했다. '
        '각 행은 2개 training seed를 평균한 **개발 결과**다. 음수는 새 방법의 비용이 더 낮다는 뜻이다.','',
        '| 문제 | v7 고정 갱신 대비 | 가장 강한 학습 비교군 대비 | 비교군 |','|---|---:|---:|---|']
    for case in reports[0]['cases']:
        if case['dimension']!=32:continue
        m=case['methods'];ref=case['best_learning_reference']
        lines.append(f"| {names[case['family']]} | {(m['prox_action']['mean']/m['fixed']['mean']-1)*100:+.2f}% | {(m['prox_action']['mean']/m[ref]['mean']-1)*100:+.2f}% | {ref} |")
    lines+=['','반응계의 개선 방향은 두 seed에서 일치했다. 별도의 두 번째 새 상태 묶음에서도 같은 저장 정책을 평가했다. '
        '이것은 새로운 training seed로 반복한 실험이 아니므로 독립 재학습 확증으로 세지 않는다.','',
        '| 반응계 조건 | 첫 상태 묶음: 행동 목표 vs Warm DPC | 둘째 상태 묶음: 같은 정책 vs Warm DPC |','|---|---:|---:|']
    for dimension,condition in [(32,'nominal'),(256,'nominal'),(256,'changed')]:
        matches=[next(c for c in report['cases'] if c['family']=='reaction' and c['dimension']==dimension and c['condition']==condition) for report in reports]
        values=[(c['methods']['prox_action']['mean']/c['methods']['warm']['mean']-1)*100 for c in matches]
        lines.append(f'| d{dimension}, {condition} | {values[0]:+.2f}% | {values[1]:+.2f}% |')
    lines+=['','256차원은 32차원에서 학습한 공유 가중치의 전이 결과다. 고차원에서 직접 학습한 성능이나 일반적인 차원 독립성을 입증한 것은 아니다.','',
        '## 곡률을 추가했을 때','',
        '곡률 변형은 같은 부모·데이터를 사용하는 새 학습 네 개를 실행했다. 기존 비교군 학습 기록 28개를 복사해 재사용한 사실과 hash를 보존했다. '
        '둘째 새 상태 묶음에서 곡률 변형까지 모두 평가했다.','',
        '| 문제 | 곡률 변형의 강한 비교군 대비 비용 차이 (d32) | 평균 준비 시간 | 수용된 teacher 갱신 합계 |','|---|---:|---:|---:|']
    for case in reports[1]['cases']:
        if case['dimension']!=32:continue
        rows=[r for r in work if r['family']==case['family']];m=case['methods'];ref=case['best_learning_reference']
        lines.append(f"| {names[case['family']]} | {(m['curvature_action']['mean']/m[ref]['mean']-1)*100:+.2f}% | {np.mean([r['setup_seconds'] for r in rows]):.2f}초 | {sum(r['accepted'] for r in rows)}회 |")
    lines+=['','이 실험은 해당 구현과 짧은 예산에서 곡률의 이점을 보여주지 못했다. '
        'Hessian–vector product 네 번의 대각 추정에는 잡음이 남고, clipping된 대각 모형은 완전한 Bellman Hessian도 오차 보증도 아니다. '
        '장시간 학습에서의 효과까지 부정한 결과는 아니다.','',
        '## HJB 메시지와 다음 연구 판단','',
        '> Teacher는 자기 정책의 미래 비용을 평가한다. Student는 그 정보를 받아 Hamiltonian을 줄이는 행동을 학습한다. '
        '실제로 개선된 student가 다음 teacher가 된다.','',
        '이번에는 이 설명이 명시적인 행동 목표 생성과 teacher 승격 코드에 직접 대응한다. '
        '다만 Hamiltonian 학습은 [MPC-Net](https://arxiv.org/html/1909.05197v2), '
        '최적화가 정책 학습을 지도하는 접근은 [guided policy search](https://proceedings.mlr.press/v28/levine13.html)에도 있다. '
        '그 조합 자체를 최초 기여로 쓰지는 않는다.','',
        '**현 단계에서는 행동 목표 방식이 다음 연구의 후보이고, 압도적 성능은 미해결 목표다.** '
        '강한 비교군을 포함한 독립 training seed 반복이 먼저 필요하다. 그다음 큰 성능 차이를 노릴 축은 '
        '장시간 문제에서 같은 품질까지 드는 총 계산량, 실제 고차원 학습, 모델·외란 변화 대응이다. '
        '이는 새로 검증할 가설이며 이번 결과로 달성했다고 주장하지 않는다.','',
        '건물처럼 비교군도 거의 최적인 문제에서 큰 총비용 절감은 목표가 될 수 없다. '
        'iLQR·DDP·QP·MPC 같은 고전 제어 최적화기를 압도한다는 주장에는 같은 품질의 실행 시간 비교와 '
        'offline 학습비까지 포함한 손익분기점 평가가 별도로 필요하다. 이번 학습 비교를 그 주장으로 대체하지 않는다.','',
        '## 검증 자료','',
        f'- 새 학습 {unique_phases}개, 고유 예산 선택 {unique_selections}개, 원시 비용 {raw_costs:,}개 재계산.',
        '- 모든 저장 정책의 원래 validation 비용 재실행 오차 0. 예산 이후 정책 선택 없음.',
        '- 두 개발 연구의 설정·코드·평가 가중치 lock과 재사용한 자료 hash를 재검증.',
        '- KKT 조건, teacher gradient 일치, finite difference 곡률, CUDA 곡률 추정, 별도 동역학·비용 재계산 확인.',
        '- 실패한 변형과 부모 정책을 유지한 경우를 포함한 모든 결과 보존. v7 최종 원고는 변경하지 않음.','',
        '[첫 연구 상세 결과](hj_v8_results_ko.md) · [HJB 이론](hj_v8_theory.md) · [곡률 추가 실험 계획](hj_v8_curvature_plan.md)',
        '', '[비교 그림](../build/hj_v8_final/cost_comparison.png) · [최종 검사](../build/hj_v8_final/completion_audit.json)','']
    doc=Path('docs/hj_v8_assessment_ko.md');doc.write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cases=[c for c in reports[0]['cases'] if c['dimension']==32]
    fig,axes=plt.subplots(1,3,figsize=(10.5,4.1))
    for ax,case in zip(axes,cases):
        methods=['fixed','prox_action','prox_latent']
        values=[(case['methods'][name]['ratio_to_best_learning_reference']-1)*100 for name in methods]
        bars=ax.bar(['v7 fixed','Action target','Latent target'],values,color=['#7c8799','#078377','#bc704d'],width=.65)
        ax.axhline(0,color='#27333d',lw=.8);ax.set_title(case['family'].capitalize(),pad=10)
        ax.tick_params(axis='x',labelsize=9,rotation=15);ax.set_ylabel('Cost difference (%)')
        ax.bar_label(bars,labels=[f'{value:+.2f}%' for value in values],padding=4,fontsize=9)
        ax.margins(y=.2)
    fig.suptitle('Development comparison: same 30 s total budget, two training seeds',fontsize=13,y=.98)
    fig.text(.5,.90,'New nominal d32 states; relative to the best learning reference in each task',ha='center',fontsize=10)
    fig.text(.5,.025,'Negative is better. Axis scales differ. These are descriptive development results.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,bottom=.22,top=.77,wspace=.38)
    fig.savefig(dest/'cost_comparison.png',dpi=180);fig.savefig(dest/'cost_comparison.pdf');plt.close(fig)
    write(dest/'manifest.json',dict(completed_utc=utc(),status='numerical_work_complete',
        hashes={str(p):sha(p) for p in [doc,dest/'completion_audit.json',dest/'cost_comparison.png',dest/'cost_comparison.pdf',Path(__file__)]},
        visual_review='pending',training_status='All 46 new phases finished; no continuing training scheduled.'))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
