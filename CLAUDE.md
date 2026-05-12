# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 목적

KUFIsh_III (사용자 자체 CAD) 기반 단일 관절 물고기 로봇의 강화학습. MuJoCo + SAC + Gymnasium. **수면 영법(BCF surface swimming) 가정**으로 단순화하여 학습.

---

## 빠른 시작

```bash
cd sim
python3 curriculum.py --no-viewer                       # 6단계 자동 학습
python3 view_policy.py runs/s2_anchor/model.zip         # 정책 보기
python3 -m mujoco.viewer --mjcf=rl_fish.xml             # 모델만 검수
tensorboard --logdir tb_logs/                           # 학습 곡선

# SSH/원격 백그라운드: tmux new -s train + python3 ... | tee log
```

### 학습 시간 (RTX 3090, fps ~305)

s1: ~100초 / s3a~c: 11~19분 / s3d 500k: ~28분 / **s3d 1M: ~55분** / 전체: 2~3시간.

병목은 SAC update + Python overhead 98% (env step만 1.4%). 1M step ≈ 1.8M SAC update.

---

## 활성 모델 — `sim/rl_fish.xml`

RL이 학습하는 단 하나의 모델. **이 파일이 모든 시뮬·학습의 진입점.**

### 핵심 사실 (한눈에 안 보이는 것들)

- **3DOF planar 운동**. base_link는 freejoint가 아니라 `slide_x` + `slide_y` + `hinge_yaw` 3개 명시 joint. roll/pitch/z는 *수학적으로 잠김*. **freejoint로 바꾸지 마세요** — pitch wobble로 추진 방향이 뒤집힙니다 (함정 #1).
- **qpos layout = 5**: `[root_x, root_y, root_yaw, tail_joint, fin_joint]`. env 코드의 인덱스가 이 순서에 의존.
- **수중 환경**. `<option density="1000" viscosity="0.001">` (실제 물). `<flag gravity="disable"/>`로 중력 끔. added mass + drag는 `fluidshape="ellipsoid"`로 자동. **중력 켜지 마세요** — 가라앉음.
- **euler="3.14159 0 0" + axis 부호 보정**. base_link가 x축 180° 회전된 상태로 시작. 이 회전 보상하기 위해 planar joint 축이 `(1,0,0)`, `(0,-1,0)`, `(0,0,-1)`. qpos는 *world* 좌표 (x, y, yaw)와 1:1.
- **추진 방향 규약: world −x = 머리 방향(전진).** freq_sweep에서 x_disp < 0이면 전진. 보상 함수 작성 시 *목표 위치를 머리 방향(−x)에 두기*.

### 관절 구조 + 액추에이터

```
world ─[slide_x][slide_y][hinge_yaw]─ base_link (PLA 강체)
                                       ├ tail_joint (active hinge, ±20°)
                                       │   └ tail_link (PLA 강체)
                                       │       └ fin_joint (passive hinge, ±30°)
                                       │           └ fin_1 (Ecoflex 00-30, density 1070)
```

- **액션 공간 = 1차원** (단일 motor): `position` actuator on `tail_joint`. ctrlrange `-1..1`, gear=0.349 → ±20°. kp=100, kv=5, forcerange=±3 Nm (BL4260).
- **fin_joint는 passive**. stiffness=1e-2, damping=5e-5. RL이 명령하지 않음.

### Inertial 값의 출처

| body | mass | CoM | 출처 |
|---|---|---|---|
| base_link | 2.4349 kg | (0, 0, 0.004535) | **MATLAB CG.m 검증값**. CAD URDF off-diag inertia는 임의 재질 인공물이라 0으로 정리. |
| tail_link | 0.0699 kg | (0.0511, 0, -0.005349) | MODE_3 URDF, y CoM=0으로 대칭화 |
| fin_1 | 0.01052 kg | (0.0773, -0.0004, 0.006) | MODE_3 URDF, density=1070 (Ecoflex) |

**임의로 수정 마세요**. base는 사용자 실물 측정값, 나머지는 사용자 합의 후 대칭화한 것.

### 추진 검증 (freq_sweep.py, ctrl=±1 sine, 12초)

1~6 Hz 모든 주파수에서 −x 머리 방향 전진 (1Hz −0.49m, 3Hz −2.17m, 6Hz −3.36m). 0.5Hz만 후진 (저주파 wave-thrust 약함).

---

## CAD 소스

`fish_urdf/RL_SIM_MODE_3_description/` mesh 사용 (`base_link.stl`, `tail_link_1_1.stl`, `fin_1.stl`). URDF는 형상만, inertial은 위 표대로. URDF가 fin을 fixed joint로 부착하지만 MJCF는 passive hinge로 교체 (실리콘 변형). `MODE_2`는 fin 미분리 구버전, 사용 안 함.

---

## Python 학습 인프라 — `sim/`

- `fish_env.py` — Gymnasium 환경. obs **(11 + action_history_n)D**: tail/fin qpos·qvel + world v + sin/cos yaw + target rel + 최근 N step ctrl 이력.
- `train.py` — SAC 학습 + 별도 thread viewer. `PolicySnapshotCallback` race-free, `CurriculumStopCallback` 자동 조기 종료.
- `curriculum.py` — 6단계 자동 진행 + `--seed` `--tb-tag` `--runs-subdir` (v29~ multi-seed 지원). **v30~ `model_best.zip` 자동 저장** (각 stage reach_rate peak 시점, `model.zip`와 공존).
- `view_policy.py` — 저장된 정책 viewer rollout.
- 진단: `freq_sweep.py` (추진 방향), `yaw_test.py` (회전 능력 — v4 핵심).

### Action history obs (v4~)

`action_history_n`. obs 끝에 최근 N step ctrl stack. v4 N=8, **v11 N=16** (1 wag cycle), v17 N=24, **v19~v29 N=20**. 비대칭 wagging 패턴(D2 등) 학습 enable. yaw_test.py: 대칭 sine 0.49°/s vs D2 4.6°/s.

### 보상 함수 (현재 코드, v22~v29 best)

```python
reward = progress·10 + (10 if reached else 0) - ctrl_cost
       + align_weight·align + YAW_W·|yaw_rate|
# align_weight = 0.02 (10s ep) / 0.012 (30s ep) — v14
# YAW_W = 0.005 — v22 best, sweet spot (v23~v28 변형 모두 reject 또는 noise 안)
```

- **progress**: delta dist × 10. align ∈ [-1, +1] (cos(머리, 목표)).
- **yaw_rate 보상 (v21)**: 회전 시도 자체에 +보상. v20 "정렬만, 추진 안 함" mode 깨고 회전·추진 시퀀스 학습 강제. **v11 26% 천장 단독 돌파의 핵심**.
- 곱셈 보상은 mode collapse — 가산식 정착 (함정 #8).

### Curriculum 학습

Random target full circle은 단일 모터에 어려움. 6단계 자동 진행:

| Stage | tag | theta 범위 | success_radius | ep_sec | max_steps |
|---|---|---|---|---|---|
| 1 | s1_forward | π fixed | 0.08 m | 10s | 400k |
| 2 | s2_anchor | π fixed | 0.04 m | 10s | 400k |
| 3a | s3a_arc15 | π ± 15° | 0.08 m | 30s | 200k |
| 3b | s3b_arc30 | π ± 30° | 0.08 m | 30s | 250k |
| 3c | s3c_arc60 | π ± 60° | 0.08 m | 30s | 350k |
| 3d | s3d_arc90 | π ± 90° | 0.08 m | 30s | **1M (v22~)** |

**진행 규칙 (반드시 준수)**:

- **stage 졸업 조건**: 최근 100 ep `reach_rate ≥ 90%` → 조기 종료 → 다음 stage fine-tune.
- **max_steps는 안전장치**. max_steps 도달했는데 90% 미달이면 강제 진행되지만, **이는 그 stage의 카드가 부족하다는 신호**.
- ⚠ **다음 학습 카드의 우선 대상은 항상 "가장 낮은 미달 stage"**. 예: s3b 69%·s3c 38%·s3d 30%이면 **s3b부터 개선**. s3d만 따로 fine-tune 카드는 의미 없음 (낮은 stage가 천장이면 위 stage도 천장).
- ⚠ **claude는 임의로 더 어려운 stage에 집중하지 말 것**. 분석·카드 후보·release 우선순위 모두 가장 낮은 미달 stage 기준. s3d 천장 분석은 s3c 90% 달성 후에만 본질적 의미.
- **Full circle은 사용자 결정으로 제외**.

```bash
python3 curriculum.py --no-viewer                                         # 전체 자동
python3 curriculum.py --start-stage 6 --seed 0 \
    --tb-tag model3_v29_seed0 --runs-subdir v29_seed0                     # multi-seed s3d
```

---

## 학습 결과 — v1 ~ v30

**현재 가장 낮은 미달 stage = s3b (best 69%, 90% 임계 미달)**. s3c·s3d는 이 위에서 fine-tune이라 s3b 개선이 본질적 다음 카드. 지금까지 v22~v30의 분석(s3d 천장 30%, best metric, multi-seed baseline 등)은 모두 s3b·s3c가 미달인 상태에서의 부수적 정보.

다음 카드 결정용 표·매트릭스만 본 섹션에 유지. **산문 진단 + Release 인덱스 + 카드 후보 상세 → [`docs/training_log.md`](docs/training_log.md)**.

### 작성 규칙 (v26~)

새 카드 추가 시: CLAUDE.md는 표만 갱신 (그룹화·변경점·Stage·s3d 100ep), 종결·돌파 카드면 통찰/함정/매트릭스 한 줄 추가. 산문은 `docs/training_log.md`. CLAUDE.md char 30k 임계 모니터링.

### 결별 그룹화

| 그룹 | 멤버 | 특징 | s3d 천장 |
|---|---|---|---|
| A. Baseline + 가산식 reward | v1·v2·v3 | `+0.02·align` 가산항 | 24% |
| B. 곱셈 보상 실패 | v5·v6·v7 | mode collapse (머리 반대 / 도망) | 16~20% |
| C. 가산식 복귀 + ent_floor | v8·v9·v10 | 균등/차등. ent_floor 카드 종결 | 13~17% |
| **D. 정책 표현력 확장** ⭐ | **v11** | action history N=8→16. ±90° 천장 첫 돌파 | **26%** |
| E. NN 확장 | v12 | net_arch [256,256,128]. s3d catastrophic | 6% ❌ |
| F~I. reward/NN 카드 | v13~v16 | align ep 비례/fix·reach 5→10·NN 회귀 | 19~21% |
| **J. action history N=24** | **v17** | s3c·d 동시 회복, 정점 30% 일시 돌파 | **25%** |
| K. 학습량 ↑ (N=24) | v18 | 500k → 1M — 학습량 부족 가설 reject | 25% |
| L. N 절충 (N=20) | v19 | s3a 96% best + s3c·d mode collapse | 23% ❌ |
| M. v19 + ent_floor ↑ | v20 | mode collapse 해결 ✓ but reach 후퇴 (정렬 best 0.84) | 19% |
| **N. yaw reward** ⭐⭐ | **v21** | `+YAW_W·\|yaw_rate\|·0.005`. **v11 천장 단독 돌파**. 모든 stage 회복 | **29%** |
| **O. v21 + 1M fine-tune** ⭐⭐⭐ | **v22** | 천장 +6%p 추가 돌파. peak 50%. yaw reward가 학습량 lock 해제 | **32%** |
| P. YAW_W 0.007 ❌ | v23-A | 정점 41% best but 700k actor_loss spike → end 23%. **peak만 통계 유의 (+2.8σ)** | 23% ❌ (peak 41% ⭐) |
| Q. signed yaw ◯ | v24-A | 진동 해결 ✓, **final_dist 0.599m best** (옳은 방향 추진). 천장 X. **final_dist만 유의 (−3.0σ)** | 28% ◯ (peak 35% ⭐) |
| R. signed × 0.007 ❌ | v25-A | spike 없음 but W자 점진 catastrophic (970k 12% 최저, end 16%) | 16% ❌ |
| S. v22 reproduce ◯ | v26 | seed default. end 26% — v22 32% 단일 seed 운 첫 입증 | 26% ◯ |
| T·U. soft signed (K=2·0.5) ❌ | v27·v28 | K 변경은 부호 정보 못 없앰. **yaw reward 형태 카드 종결** | 24%·18% ❌ |
| **V. v22 multi-seed** ⭐⭐ | **v29** | 3 seed end 16/14/21% mean 17±2.9. **baseline = 22% ± 7%p (5 seed)** 확립. v22 32% = +1.4σ outlier | mean **17%** |
| **W. best-save 인프라** ⭐⭐⭐ | **v30** | v22 카드 + `model_best.zip` peak 저장, 3 seed. end {16, 14, 21}% (v29 재현, mean 17±2.9%) vs **best {30, 33, 25}% mean 29.3±3.3%**. 5 seed 종합 best = **31.2 ± 4.0** — v22 32% = −0.5σ로 평균 근접. **v22 카드 진짜 천장 ~30% 본격 입증** | end **17%** / best **29.3%** ⭐⭐⭐ |

### 버전별 변경점 (주요만)

| 변경 | v1 | v4 | v6 | v7 | v8~v10 | v11 | v12 | v13 | v14 | v15 | v16 | v17 | v19 | v20 | v21 | v22 | v23-A | v24-A | v25-A | v26 | v27 | v28 | **v29** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| align reward | 없음 | +0.02 | (v4) | (v4) | (v4) | (v4) | (v11) | **0.02·10/ep_sec** | **0.012 fix** | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) |
| reach bonus | 5 | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | **10** | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) |
| **yaw reward** | 없음 | — | — | — | — | — | — | — | — | — | — | — | — | — | **+\|·\|·0.005** | (v21) | **·0.007** ❌ | **+sign·0.005** ◯ | **·sign·0.007** ❌ | **(v22)** ◯ | **·tanh(K=2)** ❌ | **·tanh(K=0.5)** ❌ | **(v22) 3-seed** ⭐ |
| s3 ep | 10s | 10s | **30s** | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) |
| action history | 0 | **8** | (v4) | (v4) | (v4) | **16** | (v11) | (v11) | (v11) | (v11) | (v11) | **24** | **20** | (v19) | (v19) | (v19) | (v19) | (v19) | (v19) | (v19) | (v19) | (v19) | (v19) |
| EntCoefFloor | — | — | — | 0.02 균등 | v8 0.005/v9 0.002/v10 차등 | (v10) | (v10) | (v10) | (v10) | (v10) | (v10) | (v10) | (v10) | **s3c 0.008/s3d 0.010** | (v20) | (v20) | (v20) | (v20) | (v20) | (v20) | (v20) | (v20) | (v20) |
| s3d max_steps | 500k | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | **1M** | (v22) | (v22) | (v22) | (v22) | (v22) | (v22) | (v22) |
| net_arch | [256,256] | (v1) | (v1) | (v1) | (v1) | (v1) | **[256,256,128]** | (v12) | (v12) | (v12) | **[256,256]** | (v16) | (v16) | (v16) | (v16) | (v16) | (v16) | (v16) | (v16) | (v16) | (v16) | (v16) | (v16) |
| seed | default | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | **0/1/2** |

### Stage 진행 비교 (s3a~s3c는 v21에서 학습 후 v22~v29까지 reuse)

| Stage | v8 | v10 | v11 | v17 | v19 | v20 | v21 | v22 |
|---|---|---|---|---|---|---|---|---|
| s3a (±15°) | 74% | 89% | 67% | 66% | 96%* | 63% | **90%** ⭐ | (v21 reuse) |
| s3b (±30°) | 70% | 72% | 44% | 62% | 61% | 53% | **69%** ⭐ | (v21) |
| s3c (±60°) | **46%** | 28% | 38% | 33% | 28% | 21% | **38%** | (v21) |
| s3d (±90°) | 17% | 13% | 26% | 25% | 23% | 19% | 29% | **32%** ⭐⭐⭐ |

*v19 96%는 seed 분산 운. v23~v29는 모두 v21 s3a~c + v22 s3d 카드에서 s3d만 fine-tune.

### s3d (±90°) 결과 — 마지막 100 ep 윈도우 (v21~v29만, 이전은 docs)

| 버전 | reach | avg_align | final_dist | 한 줄 평가 |
|---|---|---|---|---|
| **v11** | 26% ⭐ | +0.58 | 0.57m | N=16 — 천장 첫 돌파 |
| **v17** | 25% | +0.43 | 0.55m | N=24 — s3c·d 동시 회복, 정점 30% 일시 돌파 |
| v20 | 19% | **+0.84** ⭐ | 0.78m | ent_floor ↑ — 정렬 best but 추진 worst |
| **v21** ⭐⭐ | 29% ⭐ | +0.51 | 0.67m | **+yaw reward — v11 천장 단독 돌파**. 정점 38%, ep 21.8s best |
| **v22** ⭐⭐⭐ | **32%** ⭐⭐ | **+0.73** | **0.64m** | **v21 + 1M — peak 50% (단일 seed outlier, v29로 확인)**. ep 21.2s best |
| v23-A ❌ | 23% | +0.71 | 0.76m | YAW_W 0.007 — 정점 41% best but 700k spike → end 23%. **peak만 +2.8σ ⭐** |
| v24-A ◯ | 28% | +0.62 | **0.60m** ⭐ | signed yaw — 진동 해결 ✓, final_dist −3.0σ best. 천장 X |
| v25-A ❌ | 16% | +0.56 | 0.78m | signed × 0.007 — W자 catastrophic, 970k 12% 최저 |
| v26 ◯ | 26% | +0.69 | 0.72m | v22 reproduce — end 26% (−6%p), 32% 단일 seed 운 입증 |
| v27 ❌ | 24% | +0.70 | 0.75m | tanh(K=2) — signed 96% 재현, V자×3 |
| v28 ❌ | 18% | +0.69 | 0.70m | tanh(K=0.5) — 신호 강도만 ↓, 부호 정보 그대로 |
| **v29** ⭐⭐ | **17±2.9** | +0.63 | 0.78m | **v22 카드 3 seed (16/14/21%) — baseline 22%±7%p 확립**. v22 32%=+1.4σ outlier |
| **v30** ⭐⭐⭐ | **end 17% ± 2.9 / best 29.3% ± 3.3** | +0.63 (end) | 0.78m (end) | **v22 카드 + `model_best.zip` 3 seed**. end {16, 14, 21}% (v29 seed 재현). **best {30%@880k, 33%@520k, 25%@865k} mean 29.3 ± 3.3** — 모든 seed 25%+ 도달. **5 seed 종합 best = 31.2 ± 4.0**, v22 32% = −0.5σ (진짜 평균). **v22 카드 진짜 천장 ~30%, end 16~32% 분산은 catastrophic drift 영향**. peak→end −4~−19%p (seed 1 W자 480k 점진 후퇴). 카드 평가 표준 = best metric. |

### 핵심 통찰

- ✅ **Stage 1·2** 60k step에 100% (모든 버전).
- ⭐ **N 축이 천장 카드의 가장 인과 명확한 축** — v11 (+9%p) → v17 (정점 30%) → 1M도 25% 천장.
- ❗ **단일 축 카드(N/ent/reward) 모두 v11 26% 천장 못 깸**. 천장 돌파 = **yaw reward (v21)**, 안정화 = **yaw + 1M (v22)**.
- ⭐⭐ **yaw reward (v21·v22)**: 회전 시도 자체에 +보상 → "정렬만 mode" 깨고 mode catalysis trigger. yaw reward가 학습량 ↑의 lock 해제.
- **yaw reward 변형 카드 (v23-A~v28) 종결**: signed/tanh/YAW_W 변형 모두 \|·\|·0.005 미달. v23-A peak 41% (+2.8σ) / v24-A final_dist 0.599m (−3.0σ best)만 통계 유의 카드 효과.
- ⭐⭐ **v29 multi-seed baseline 확립**: v22 카드 5 seed mean **22% ± 7%p** (std 6.86). v22 32%은 +1.4σ outlier. **v11·v17·v22 모두 baseline noise 안 → v11 이후 카드 사실들상 동급, 단일 seed 비교 무의미**.
- ⭐⭐ **v29 late catastrophic forgetting 발견**: 3 seed **모두 s3d 800~900k peak 후 1M까지 후퇴** (seed 0: 30%→16% −14%p, seed 1: 20%→14%, seed 2: W자 후 25%→21%). 학습 끝 model.zip은 진짜 best 아님. v30~ `model_best.zip` checkpoint로 peak 보존 — 카드 자체 변경 없이 reach 기대 +5~10%p.
- ⭐⭐⭐ **v30 best-save 3-seed 완성 — v22 카드 진짜 천장 본격 확립**: 3 seed end {16, 14, 21}% (v29 재현, mean 17 ± 2.9) vs **best {30, 33, 25}% mean 29.3 ± 3.3**. 5 seed 종합 best = **31.2 ± 4.0** — v22 32%은 −0.5σ로 평균에 매우 가까움 (v29 end 기준 +1.4σ outlier 결론은 metric 선택의 artifact). **v22 카드 진짜 천장 ~30%**, end 16~32% 분산은 1M까지 가는 동안 catastrophic drift 영향. **카드 평가 표준 = best metric** (end는 noise). v23~v28 reject 결론들도 best metric으로 재평가 필요.

### 핵심 yaw_test.py 발견

| 측정 | yaw rate |
|---|---|
| 학습된 v3 정책 | 0.94°/s |
| 대칭 sine | 0.49°/s |
| **D2 (75% 음 + 25% 양 비대칭)** | **4.60°/s** ✓ |

→ ±90° 회전이 20s 안에 물리적 가능 (D2 × 20s = 92°). RL이 비대칭 패턴 발견 못 함이 본질적 병목 — v11 N=16에서 부분 해결.

---

## TensorBoard 메트릭

`tb_logs/model3_vN/` 로 분리 (v29는 `model3_v29_seed{0,1,2}/`).

| 태그 | 정의 |
|---|---|
| `rollout/ep_rew_mean`·`ep_len_mean` | 최근 100 ep 보상·step |
| `train/critic_loss`·`actor_loss`·`ent_coef` | Q MSE·정책 loss·α (auto-tuned) |
| `fish/success_rate`·`final_distance`·`episode_seconds`·`avg_align` | 도달률·종료 거리·ep 길이·정렬도 |

진단: α 5만 step 안 0.001 이하 → 탐색 부족 / ep_len_mean=max → 도달 못함 / avg_align 높 + success 낮 → "정렬만" mode.

---

## SAC 메모

`stable-baselines3.SAC` 기본 (Maximum-Entropy off-policy actor-critic). Twin Q, auto α, ε_target=−1, off-policy + replay 200k. **HP**: LR=3e-4, batch=256, γ=0.99, τ=0.005, `ent_coef="auto_0.1"` (초기값, floor는 EntCoefFloorCallback).

---

## 작업 진행 규칙

- 다단계 작업은 **한 단계씩 분리**해서 진행. 사용자가 결과 보고 결정 내릴 수 있는 지점에서 끊는다. TaskCreate로 미리 나열 OK, 한 번에 한 task만 in_progress.
- ⚠ **학습 분석·카드 후보 우선순위는 가장 낮은 미달 stage 기준** (`Curriculum 학습` 섹션의 진행 규칙). 사용자가 명시적으로 다른 stage를 지정하지 않는 한, claude는 임의로 더 어려운 stage(s3d 등)에 집중하거나 학술적 분석으로 우회하지 말 것.

---

## 알아둘 함정 (이미 발견·해결된 것)

**물리·모델 (v1 이전)**:
1. **6DOF freejoint**: pitch cross-coupling으로 추진 방향 뒤집힘. → 3DOF planar.
2. **fluidshape**: MuJoCo `none`/`ellipsoid` 둘뿐.
3. **단일 passive fin (k=4e-7)** / **다단 passive fin** / **CPG 액션 (MODE_2)** / **F1 STEP 다중 ellipsoid**: 모두 추진 못 만듦. → MODE_3 + 단일 motor + Ecoflex passive fin 정착.

**RL 카드 (v1~v29)**:

7. **곱셈 보상 (v5/v7)**: mode collapse (머리 반대·도망). 가산식이 정착.
8. **차등 N / 차등 NN**: SAC.load weight transfer 불가 (obs Box mismatch 또는 layer random init). **curriculum init_from 무력화**.
9. **단일 축 reward/NN 카드 (v12~v16)**: align ratio·reach bonus·NN 확장/회귀 — 모두 v11 26% 천장 미돌파. 단독 카드 한계.
10. **N=24 + 학습량 ↑ (v18)**: 1M도 end 25% — 학습량 부족 가설 reject. N=24 본질적 진동.
11. **N=20 단독 (v19)**: s3a 96% best but s3c·d mode collapse (align −0.37, final_dist 0.35m). narrow mode 매개.
12. **ent_floor 강화 (v20)**: collapse 해결 ✓ but reach 후퇴 ❌. entropy ↑이 정렬에만 활용.
13. **YAW_W 변형 카드 (v23-A·v24-A·v25-A·v27·v28)**: 모두 v22 \|·\|·0.005 미달. **v22가 sweet spot, 단순 변형으론 천장 못 깸**. v23-A peak 41%·v24-A final_dist 0.599m만 통계 유의 (multi-seed 재검증 후보).
14. **v22 단일값 baseline 신뢰 (v29 3-seed로 입증)**: v22 카드 mean 17% ± 2.9%, 5 seed 종합 22% ± 7%p (end 기준). **v22 32%은 +1.4σ outlier**. v25-A 16%·v27 24%·v28 18% 모두 noise 안 → 카드 무효 결정적 증거 부족. **카드 비교는 최소 3 seed 평균으로**.
15. **end metric으로 카드 평가 (v30 best-save 3-seed로 입증)**: late catastrophic drift (peak→end −4~−19%p)가 end-100ep을 noise화함. **best metric으로 다시 보면 v22 카드 천장 = 29.3 ± 3.3% (v30 3 seed) / 31.2 ± 4.0 (v22+v26+v30 5 seed)** — v22 32% = −0.5σ로 평균 근접. v29 end 기준 outlier 결론은 metric 선택 artifact. **카드 평가 표준 = best metric**, end는 안정성·forgetting 진단용으로만.

→ 정착 조합 = **MODE_3 + Ecoflex passive fin + 3DOF planar + v22 카드 (N=20·ent_floor 차등·yaw \|·\|·0.005·s3d 1M)**.

### 가장 결정적이었던 변경 (회고)

1. **3DOF planar** — pitch wobble 제거 (부호 뒤집기 본질)
2. **MATLAB CG.m 값** — sim2real 정합성
3. **MODE_3 fin + Ecoflex 1070** — wave-thrust
4. **action history N=16 (v11)** — ±90° 천장 첫 돌파
5. **yaw_rate 보상 (v21)** ⭐⭐ — v11 천장 단독 돌파
6. **v21 + 1M (v22)** ⭐⭐⭐ — 천장 +6%p 추가 돌파 (32%/peak 50%)
7. **v29 multi-seed (v22 baseline 본격 확립)** ⭐⭐ — 22% ± 7%p (end 기준), 32%은 outlier로 보였음. 카드 비교는 3 seed 평균.
8. **v30 best-save + 3-seed (best metric 카드 평가 표준)** ⭐⭐⭐ — best 29.3 ± 3.3 (3 seed) / 5 seed 31.2 ± 4.0. **v22 32%은 −0.5σ로 평균 근접** — v29의 outlier 결론은 metric 선택 artifact. **v22 카드 진짜 천장 ~30%, end 분산은 catastrophic drift 영향**. 모든 카드 평가는 best metric 표준.

---

## RL 학습 시 주의

- **fluidshape="ellipsoid" 한계**. vortex shedding 못 모델 → sim2real 격차. 개선은 Lighthill `mjcb_passive` 또는 ANN surrogate.
- **보상 forward = world −x**. 정책이 +x로 가면 환경 인덱스 잘못된 것.
- **3DOF yaw 누적**: fluid asymmetry로 한쪽 도는 경향. RL이 보정 가능.
- **Curriculum init_from**: 이전 stage "직진 편향"이면 Stage 3에서 한쪽 mode만 학습 위험.

---

## 백업·릴리스

```bash
# 새 vN 학습 후
mkdir -p /tmp/models-vN/{runs,plots}
cp sim/runs/{tag}/model.zip /tmp/models-vN/runs/{tag}/
tar -czf /tmp/runs-models-vN.tar.gz -C /tmp/models-vN .
gh release create models-vN --target <branch> --title "..." --notes "..." /tmp/runs-models-vN.tar.gz

# 복원
gh release download models-vN -p '*.tar.gz' && tar -xzf runs-models-vN.tar.gz -C sim/
```

→ **release 인덱스 / 모델 손실 / 복원 상세는 [`docs/training_log.md`](docs/training_log.md)**.

---

## 다음 후보 (미해결)

**curriculum 진행 상태**: s1·s2·s3a 90% ✓ / **s3b 69% (가장 낮은 미달) / s3c 38% / s3d best 30%**. 다음 카드 우선순위는 **s3b 90% 달성**. s3c·s3d 카드는 s3b 졸업 후에만 본질적 의미 (낮은 stage가 천장이면 위 stage도 천장).

**v30 인프라 정착**: `model_best.zip` 자동 저장 — peak reach_rate 시점 모델 보존. 모든 stage에 적용.

### 우선순위 1: s3b 90% 달성 (현 미달 stage)

1. **v31 s3b 본격 학습 (max_steps ↑ + best-save + 3 seed)** ⭐⭐⭐ — 현재 v21 단일 seed 69%이라 분산도 모름. s3b max_steps 250k → 1M, best-save, 3 seed. ~3시간.
   - 90%+ 도달 → s3c 단계로
   - 천장 70~80% → 카드 변경 (yaw reward 강화 / ent_floor / 후반 안정화)
2. **v31-B s3b 천장이 카드 한계로 입증되면** — yaw reward 강화 / ent_floor 변경 / N 조정 등 s3b 위주 카드 매핑.

### 우선순위 2: s3b 졸업 후 s3c (38% 미달)

s3b가 90% 달성됐다는 가정 하에:

3. s3c 본격 학습 (max_steps 350k → 700k+ / best-save / 3 seed) — s3b와 같은 방법.

### 우선순위 3: s3d (s3c 90% 달성 후에만)

지금까지의 s3d 천장 30% / best 31.2±4.0% 분석은 s3b·s3c 미달 상태에서의 부수적 정보. s3c 90% 달성 후 다시 본격 측정.

### s3 외 인프라/메타 카드 (낮은 우선순위)

4. **HER (Hindsight Experience Replay)** — env Dict obs 큰 변경. 강력하나 구현 비용 큼.
5. **학습 후반 안정화 카드** — catastrophic drift 자체를 늦추는 축. s3b/c/d 모두 공통 적용 가능.
6. **fin actuator 추가** — 단일 모터 한계 풂. (사용자 명시 제외)
7. **ANN surrogate (Lighthill / Zhong)** — fluid model 우회. 실물 motion capture 필요.

### 단일 지느러미의 천장

yaw_test.py로 **yaw rate 물리 상한 ~4.6°/s** (D2 × 20s = 92°, ±90° 마진 작음).

**카드 매트릭스 (v30 시점)**:
- N (시간적 표현력) v11/v17/v19 — 정점 30% 가능
- entropy v8~v10/v20 — collapse 해결
- align/reach 가중치 v12~v15 — 비율 카드 한계
- **yaw reward v21~v22** ⭐⭐⭐ — 천장 돌파 (best 30%) + 안정화
- 학습량 ↑ × yaw reward — v18 reject ≠ v22 성공
- YAW_W 변형 (v23~v28) — v23-A·v24-A 통계 유의 후보, 나머지 noise
- **multi-seed end (v29)** — baseline 22%±7%p (end), drift 영향 큼
- **best-save 3 seed (v30)** ⭐⭐⭐ — 진짜 천장 31.2±4.0% (best, 5 seed), **카드 평가 표준**

→ **카드 비교는 best metric × 3 seed 평균으로**. end metric은 안정성 진단용. v22 단일값 baseline 시대 완전 종료.

⚠ **단, 카드 평가는 항상 "가장 낮은 미달 stage" 위에서**. 현재 s3b 69%이므로 s3d 천장 분석은 부수적 — s3b 90% 달성 후에야 본격 의미.
