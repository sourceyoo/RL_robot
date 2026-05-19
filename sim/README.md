# `sim/` 구조

## Entry points (sim/ 루트)

| 파일 | 역할 |
|---|---|
| `rl_fish.xml` | 활성 MuJoCo 모델 (3DOF planar + V-cut fin + R5 fluidcoef). 모든 학습·진단·실험이 참조. |
| `fish_env.py` | Gymnasium `FishSwimEnv` (obs/reward/action). 다른 모든 학습/분석 코드의 import 의존성. |
| `train.py` | SAC 학습 + callback (`PolicySnapshotCallback`, `CurriculumStopCallback`, best-save). curriculum이 import. |
| `curriculum.py` | **entry: 학습**. 6-stage 정의 + `--start-stage`·`--end-stage`로 단일 stage 실행. |
| `eval_stages.py` | **entry: eval**. 모델 1개를 6-stage deterministic 평가 (catastrophic forgetting 점검). |
| `view_policy.py` | **entry: viewer**. 저장된 정책 rollout 시각화. |
| `analyze_policy.py` | 정책 ctrl pattern 분석 (F/D/B mode 식별). |
| `test_env.py` | FishSwimEnv smoke test. |

## 디렉토리

```
sim/
├── diagnostics/      # 추진 진단 스크립트 (rl_fish.xml만 의존)
│   ├── freq_sweep.py            # 주파수 스윕 (추진 방향)
│   ├── yaw_test.py              # 회전 능력 (v4 핵심)
│   ├── f_sweep.py               # F duty 비대칭 sweep
│   ├── freq_sweep_locked.py     # body fix 대칭성 검증
│   ├── freq_sweep_norollpitch.py
│   ├── multiseg_test.py         # 다단 passive fin
│   └── waveform_test.py
├── experiments/      # 사이드 실험 (학습 환경 분리)
│   └── box_fin/                 # 박스 fin 실패 기록 (xml + viz + sweep)
├── logs/             # 옛 학습 로그 archive
│   ├── m1_m3/   (63개)          # v2~v33 구 환경
│   └── m4_v8/   ( 6개)          # m4 v8·v8b (v10 이전)
├── runs/             # 현재 활성 학습 산출물 (m4_v10: V-cut + R5 + backward penalty)
│   └── r5_back05_seed0/         # stage별 model.zip + *_train.log + *_eval.log
├── runs_archive/     # 옛 학습 산출물 보존 (총 81개, README.md 참조)
│   ├── v4/ v8/                  # 매우 옛 백업 (m1~m3 초기)
│   ├── m1_m3_stages/            # stage 이름만 (s1, s2, s3*)
│   ├── m1_m3_multi_seed/        # v24~v33 multi-seed
│   ├── m4_v1_v8b/               # m4 v1~v8b 시리즈
│   ├── mode3_planar/            # 3DOF planar 도입기
│   └── smoke/                   # smoke 테스트
├── tb_logs/          # TensorBoard 로그
└── plots/            # 학습 곡선 plot 산출물
```

## 사용 빈도

빠른 시작은 `../CLAUDE.md` 참조. 진단 스크립트 호출은 `python3 diagnostics/<name>.py` 형식.
