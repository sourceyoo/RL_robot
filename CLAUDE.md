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

s1: ~100초 / s3a~c: 11~19분 / s3d 500k: ~28분 / s3d 1M: ~55분 / 전체: 2~3시간.

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

- `fish_env.py` — Gymnasium 환경 클래스 `FishSwimEnv`. 주요 인자: `target_theta_range`, `success_radius`, `episode_seconds`, `action_history_n`. obs **(11 + action_history_n)D**: tail/fin qpos·qvel + world v + sin/cos yaw + target rel + 최근 N step ctrl 이력.
- `train.py` — SAC 학습 + 별도 thread viewer. `PolicySnapshotCallback` race-free, `CurriculumStopCallback` 자동 조기 종료, **best-save callback** (reach_rate peak에 `model_best.zip` 저장).
- `curriculum.py` — 6단계 자동 진행 + `--seed` `--tb-tag` `--runs-subdir` (multi-seed 지원) + `--end-stage` (1 stage만 실행).
- `eval_stages.py` — 모델 1개를 모든 stage 분포에서 **deterministic eval** (6 stage × 100 ep, CPU, ~20분). TB callback의 random eval 진동 noise(±10%p)를 회피하고 진짜 졸업 여부 확인. **catastrophic forgetting 측정 필수**.
- `view_policy.py` — 저장된 정책 viewer rollout.
- 진단: `freq_sweep.py` (추진 방향), `yaw_test.py` (회전 능력 — v4 핵심).

### Action history obs (v4~)

`action_history_n`. obs 끝에 최근 N step ctrl stack. v4 N=8, **v11 N=16** (1 wag cycle), v17 N=24, **v19~v21 N=20**. 비대칭 wagging 패턴(D2 등) 학습 enable. yaw_test.py: 대칭 sine 0.49°/s vs D2 4.6°/s.

### 보상 함수 (현재 코드, v21 이후)

```python
reward = progress·10 + (10 if reached else 0) - ctrl_cost
       + align_weight·align + YAW_W·|yaw_rate|
# align_weight = 0.02 (10s ep) / 0.012 (30s ep) — v14
# YAW_W = 0.005 — v21 도입 (회전 시도 보상으로 mode catalysis trigger)
```

- **progress**: delta dist × 10. align ∈ [-1, +1] (cos(머리, 목표)).
- **yaw_rate 보상 (v21)**: 회전 시도 자체에 +보상. v20 "정렬만, 추진 안 함" mode 깨고 회전·추진 시퀀스 학습 강제. **v11 26% 천장 단독 돌파의 핵심**.
- 곱셈 보상은 mode collapse — 가산식 정착 (함정 #7).

### Curriculum 학습

Random target full circle은 단일 모터에 어려움. 6단계 자동 진행:

| Stage | tag | theta 범위 | success_radius | ep_sec | max_steps |
|---|---|---|---|---|---|
| 1 | s1_forward | π fixed | 0.08 m | 10s | 400k |
| 2 | s2_anchor | π fixed | 0.04 m | 10s | 400k |
| 3a | s3a_arc15 | π ± 15° | 0.08 m | 30s | 200k |
| 3b | s3b_arc30 | π ± 30° | 0.08 m | 30s | **1M (v22~)** |
| 3c | s3c_arc60 | π ± 60° | 0.08 m | 30s | 350k |
| 3d | s3d_arc90 | π ± 90° | 0.08 m | 30s | 500k |

**진행 규칙 (반드시 준수)**:

- **stage 졸업 조건 (1차)**: TB `CurriculumStopCallback` 최근 100 ep `reach_rate ≥ 90%` → 조기 종료.
- ⚠ **stage 졸업 확정 (필수)**: `python3 sim/eval_stages.py <model.zip>` **deterministic eval** 100 ep에서 해당 stage reach ≥ 90%. TB callback은 random action sample 진동(±10%p)이라 진동 정점 시점 false positive 가능 (예: v22 s3b TB end 90%/peak 91% vs deterministic 79%, 실제 미달). **TB 졸업 신호만 보고 다음 stage 진행 금지**.
- ⚠ **학습 시작 전 진짜 미달 stage 확인**: init 모델(이전 학습 final 또는 best)을 `eval_stages.py`로 6 stage 전부 측정. 가장 낮은 deterministic 미달 stage가 진짜 학습 대상. TB end metric으로 졸업 단정 X.
- ⚠ **학습 종료 후 6 stage eval 필수**: 학습한 stage 결과뿐 아니라 모든 이전 stage(s1~) deterministic 변화량(Δreach) 측정 — catastrophic forgetting 점검. 이전 stage 후퇴 발견 시 mixed sampling / rehearsal 후속 카드 검토.
- **max_steps는 안전장치**. max_steps 도달했는데 90% 미달이면 강제 진행되지만, **이는 그 stage의 카드가 부족하다는 신호**. 다만 `--end-stage X`로 1 stage만 실행 시 자동 진행 안 함.
- ⚠ **다음 학습 카드의 우선 대상은 항상 "가장 낮은 미달 stage"** (deterministic eval 기준). 예: s3b 79% (deterministic)·s3c 42%·s3d 26%이면 **s3b부터 개선**. s3d만 따로 fine-tune 카드는 의미 없음 (낮은 stage가 천장이면 위 stage도 천장).
- ⚠ **claude는 임의로 더 어려운 stage에 집중하지 말 것**. 분석·카드 후보·release 우선순위 모두 가장 낮은 미달 stage 기준. s3d 천장 분석은 s3c 90% 달성 후에만 본질적 의미.
- ⚠ **다음 stage 자동 진행 X**: `curriculum.py --start-stage N --end-stage N`으로 현 미달 stage만 학습. default가 끝(s3d)까지 자동 진행이라 명시 필요.
- **Full circle은 사용자 결정으로 제외**.

```bash
python3 curriculum.py --no-viewer                                         # 전체 자동
python3 curriculum.py --start-stage 4 --seed 0 \
    --tb-tag model3_v22_seed0 --runs-subdir v22_seed0                     # multi-seed s3b
```

---

## 학습 결과 — main flow v1 ~ v27 (+ Step 0 eval)

**현재 미달 stage 재판정 = s3b** (Step 0 100 ep eval로 mean 85.7% — 50 ep 졸업 측정 90.7%과 5%p gap 발견). 학습 졸업 시 50 ep det check 통과(90/92/90)였지만 100 ep로 다시 측정하니 90/81/86% — sample size noise였음. **forgetting은 없음** (s1·s2·s3a 100/100/100% 3 seed 모두). 다음 카드 = **v28 (s3b 안정화, `--det-episodes 50 → 100`)**.

다음 카드 결정용 표·매트릭스만 본 섹션에 유지. **산문 진단 + Release 인덱스 + 카드 후보 상세 → [`docs/training_log.md`](docs/training_log.md)**.

### 결별 그룹화 (main flow)

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
| **O. s3b 학습량 ↑** ⭐⭐⭐ | **v22** | s3b max_steps 250k → **1M**, v21 정책 이어받기 (`--start-stage 4`). **s3b TB 90% 졸업** (654k 조기 종료, det 79% — 함정 #13) | 32% (s3b TB 90%, det 79%) |
| P. ent_floor schedule | **v25-A** | s3b `ent_floor 0.003 → 0` linear decay (1M). **stochastic-det gap 12%p → 8%p**로 감소 ✓ but det 84%로 90% 미달 | s3b det **84%** |
| Q. det check + 학습량 추가 ❌ | **v26-A** | v25-A 정책 + det check callback (TB 90% trigger 후 det 50 ep 확인) + 학습 1M 추가. **s3b det 80%로 후퇴** — single seed 결과로 카드 천장 추정 | s3b det **80%** (single) |
| **R. v25-A × multi-seed** ⭐⭐ | **v27** | v25-A 카드 seed 0/1/2 + `--det-check` (50 ep). **학습 졸업 50 ep det 90/92/90% (mean 90.7%)** — 일견 졸업. 그러나 Step 0 후속 100 ep eval로 **90/81/86% (mean 85.7%)** — **50 ep는 sample noise로 운 좋게 측정**. s3b 졸업 가설 partial. forgetting 없음 (s1·s2·s3a 100%) | s3b det **50 ep 90.7%** vs **100 ep 85.7%** ⚠ |

### 버전별 변경점 (main flow v1~v27)

v22까지 변경점은 위 그룹화 표 참조. v22 이후 카드 (s3b 천장 시도):

| 변경 | v22 | v25-A | v26-A | **v27 (× 3 seed)** |
|---|---|---|---|---|
| 기반 카드 | v21 (yaw `|·|`0.005·N=20·ent_floor 차등) | (v22) | (v25-A) | (v25-A) |
| init 정책 | v21 final | v22 final | v25-A final | **v25-A final** (sim/runs/v25/s3b_arc30/) |
| s3b max_steps | **1M** | (v22) | 1M *추가* (누적 ~1.6M) | 1M (--end-stage 4) |
| **s3b ent_floor** | 0.003 fix | **0.003 → 0 linear decay (1M)** | (v25-A) | (v25-A) |
| **det check callback** | — | — | **활성** | **활성** |
| **seed** | default | default | default | **0 / 1 / 2** (multi-seed) |
| s3b TB end | 91% (peak 92%) | 92% (peak) | 90% | 90 / 91 / 94% |
| **s3b det reach** | **79%** | **84%** | **80%** ↓ | **90 / 92 / 90%** ⭐⭐⭐ (mean 90.7 ± 1.2) |
| 졸업 step | 654k (TB false) | 615k (TB false) | 1M 완주 (det 미달) | **705k / 880k / 245k** (det 진짜 졸업) |
| 한 줄 평가 | 학습량 ↑ 졸업 — but TB false positive | gap 좁힘 ✓ but single 84% | single seed 80% (single artifact) | **multi-seed로 카드 valid 확정** ⭐⭐⭐ |

### Stage 진행 비교 (main flow)

**v22까지는 TB stochastic 기준. 함정 #13 발견 후 v22~v26은 deterministic eval로 보정**:

| Stage | v8 | v10 | v11 | v17 | v19 | v20 | v21 | **v22 (TB/det)** | **v25-A (det)** | **v26-A (det)** | **v27 50ep / 100ep (3 seed mean)** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s3a (±15°) | 74% | 89% | 67% | 66% | 96%* | 63% | **90%** ⭐ | 100/100 | 98 | **100** | **100 / 100** ✓ |
| s3b (±30°) | 70% | 72% | 44% | 62% | 61% | 53% | 69% | **90/79** | **84** | **80** ↓ | **90.7 / 85.7** ⚠ (gap 5%p) |
| s3c (±60°) | **46%** | 28% | 38% | 33% | 28% | 21% | 38% | 37/42 | 43 | 42 | — / **44.0** |
| s3d (±90°) | 17% | 13% | 26% | 25% | 23% | 19% | 29% | **32**(peak50)/26 | 30 | 28 | — / **30.7** |

*v19 96%는 seed 분산 운. v22~v26-A는 `--start-stage 4`로 s3b만 학습. v27은 `--start-stage 4 --end-stage 4`로 s3b만 학습 (init=v25-A), seed 0/1/2. **Step 0 결과 (eval_stages.py 100 ep × 3 seed)**: s1·s2·s3a forgetting 없음, **s3b 100 ep로는 mean 85.7% (개별 90/81/86%)** — 50 ep 졸업 측정과 5%p 차이. s3c·s3d는 v25-A에서 init 후 변경 없음 (s3b만 fine-tune).

### s3d (±90°) 결과 — 마지막 100 ep 윈도우 (v11~v22, 이전은 docs)

| 버전 | reach | avg_align | final_dist | 한 줄 평가 |
|---|---|---|---|---|
| **v11** ⭐ | 26% | +0.58 | 0.57m | N=16 — 천장 첫 돌파 |
| **v17** | 25% | +0.43 | 0.55m | N=24 — s3c·d 동시 회복, 정점 30% 일시 돌파 |
| v18 | 25% | +0.57 | 0.57m | N=24 + 1M — 학습량 부족 가설 reject |
| v19 ❌ | 23% | −0.37 | 0.35m | N=20 단독 — s3c·d mode collapse |
| v20 | 19% | **+0.84** ⭐ | 0.78m | ent_floor ↑ — 정렬 best but 추진 worst |
| **v21** ⭐⭐ | 29% | +0.51 | 0.67m | **+yaw reward — v11 천장 단독 돌파**. 정점 38%, ep 21.8s best |
| **v22** ⭐⭐⭐ | **32%** ⭐ / 28%† | +0.73 | 0.64m | s3b 1M 졸업 정책 → s3d 1M. run1 32%/peak 50%, run2 28%/peak 50%. 정착 카드의 s3b 90% 위 첫 결과 |

†v22 s3d 두 번 run (s3d_arc90_1·_2) — 의미 분석 보류.

### 핵심 통찰 (main flow)

- ✅ **Stage 1·2** 60k step에 100% (모든 버전).
- ⭐ **N 축이 천장 카드의 가장 인과 명확한 축** — v11 (+9%p) → v17 (정점 30%).
- ❗ **단일 축 카드(N/ent/reward) 모두 v11 26% 천장 못 깸**. 천장 돌파 = **yaw reward (v21)**.
- ⭐⭐ **yaw reward (v21)**: 회전 시도 자체에 +보상 → "정렬만 mode" 깨고 mode catalysis trigger. v11 천장 단독 돌파.
- ⭐⭐⭐ **s3b 학습량 부족 (v22)**: v21까지 s3b max_steps 250k가 짧았던 것. 250k → 1M으로 **s3b TB 90% 달성** (654k 조기 종료). 다른 stage도 같은 의심 — `max_steps`는 카드 한계 진단 전에 학습량부터 충분히 줘야 한다는 교훈.
- ⚠ **v22 졸업의 false positive (함정 #13)**: TB 90%는 stochastic 진동 정점 운. deterministic eval은 79% — **s3b 90% 졸업 무효**. 졸업 확정은 항상 `eval_stages.py` deterministic.
- ⚠ **stochastic-det gap 본질 (v25-A)**: `ent_floor 0.003 → 0` linear decay로 학습 후반 정책을 deterministic하게 압박 → gap 12%p → 8%p로 감소. 단 det 84%로 90% 미달 (single seed).
- ❌ **v26-A single seed 80%**: v25-A 정책 + det check + 1M 추가 학습 = s3b det 80% (single). v22 79% → v25-A 84% → v26-A 80% 진동을 카드 천장으로 추정했으나 **v27이 single seed artifact임을 반박**.
- ⭐⭐ **v27 multi-seed로 s3b 졸업 임계 통과 (50 ep 기준)**: 같은 v25-A 카드 3 seed → 학습 졸업 시 det **90 / 92 / 90%** (mean 90.7 ± 1.2). v22/v25-A/v26-A 79~84% single seed 결론을 부분 반박. 졸업 step 245~880k (×3.6 분산, seed 운).
- ⚠ **Step 0 100 ep eval — 50 ep는 sample noise로 운 좋은 측정**: 같은 모델 100 ep로 재측정 → 90/81/86% (mean 85.7%, 5%p gap). **seed1 만 정확히 90%**, seed0 81%·seed2 86%로 임계 미달. **50 ep det check는 표본 적어 학습 중 졸업 trigger에는 적합하나 진짜 졸업 확정 측정으로는 부족**. 함정 #16 추가.
- ✓ **forgetting 없음 (Step 0)**: 3 seed 모두 s1·s2·s3a 100/100/100% — s3b fine-tune이 이전 stage 망가뜨리지 않음. mixed sampling/rehearsal 새 축 카드 불필요.
- ⭐ **det check 인프라 실증 (v27 seed2)**: 145k에서 TB 90% trigger → det 84% (false positive) → cooldown after 245k에서 TB 94% / det 90% (진짜 졸업). **`--det-check` 옵션 default 활성 권장**.

### 핵심 yaw_test.py 발견

| 측정 | yaw rate |
|---|---|
| 학습된 v3 정책 | 0.94°/s |
| 대칭 sine | 0.49°/s |
| **D2 (75% 음 + 25% 양 비대칭)** | **4.60°/s** ✓ |

→ ±90° 회전이 20s 안에 물리적 가능 (D2 × 20s = 92°). RL이 비대칭 패턴 발견 못 함이 본질적 병목 — v11 N=16에서 부분 해결.

---

## s3d_90 라인 (분리됨)

구 v22~v30은 별도 파일·release line으로 분리 → **[`docs/s3d_90_line.md`](docs/s3d_90_line.md)** (`models-s3d_90-v1~v9`, 9개 카드). s3b·s3c 미달 상태에서 s3d만 fine-tune했던 잘못된 방향성. main flow의 다음 카드는 새 v22 (s3b 90% 달성)로 새로 시작.

---

## TensorBoard 메트릭

`tb_logs/model3_vN/` 로 분리.

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
- ⚠ **학습 시작 전·후 6 stage deterministic eval 필수** (`sim/eval_stages.py <model.zip>`). TB end reach가 90%여도 진동 noise로 false positive 가능 — init 모델 6 stage eval로 진짜 미달 stage 확정, 학습 종료 후 동일 eval로 catastrophic forgetting 점검. 사전 고지 단계에 init eval 결과 포함.
- ⚠ **새 카드(vN) 시작 전 반드시 방향성 사전 고지**. 학습 시작 전에 사용자에게:
  1. **init 모델 6 stage deterministic eval 결과** — 진짜 baseline 확인
  2. **가설** — 무엇을 알고자 하는가
  3. **카드 구성** — 이전 카드 대비 무엇이 바뀌는가 (표 형식)
  4. **예상 결과** — 가설이 맞을 때/틀릴 때 결과 예측
  5. **예상 시간** — multi-seed 포함
  6. **결과 평가 기준** — 어떤 메트릭으로 성공/실패 판단 (deterministic eval 기준)

  사용자가 읽고 이해·승인한 후에만 학습 실행. 임의로 진행하지 말 것.

---

## 알아둘 함정 (이미 발견·해결된 것)

**물리·모델 (v1 이전)**:
1. **6DOF freejoint**: pitch cross-coupling으로 추진 방향 뒤집힘. → 3DOF planar.
2. **fluidshape**: MuJoCo `none`/`ellipsoid` 둘뿐.
3. **단일 passive fin (k=4e-7)** / **다단 passive fin** / **CPG 액션 (MODE_2)** / **F1 STEP 다중 ellipsoid**: 모두 추진 못 만듦. → MODE_3 + 단일 motor + Ecoflex passive fin 정착.

**RL 카드 (main flow v1~v21)**:

7. **곱셈 보상 (v5/v7)**: mode collapse (머리 반대·도망). 가산식이 정착.
8. **차등 N / 차등 NN**: SAC.load weight transfer 불가 (obs Box mismatch 또는 layer random init). **curriculum init_from 무력화**.
9. **단일 축 reward/NN 카드 (v12~v16)**: align ratio·reach bonus·NN 확장/회귀 — 모두 v11 26% 천장 미돌파. 단독 카드 한계.
10. **N=24 + 학습량 ↑ (v18)**: 1M도 end 25% — 학습량 부족 가설 reject. N=24 본질적 진동.
11. **N=20 단독 (v19)**: s3a 96% best but s3c·d mode collapse (align −0.37, final_dist 0.35m). narrow mode 매개.
12. **ent_floor 강화 (v20)**: collapse 해결 ✓ but reach 후퇴 ❌. entropy ↑이 정렬에만 활용.
13. **TB 졸업 callback false positive** ⭐⭐⭐: `CurriculumStopCallback`은 random eval window(±10%p 진동)에서 진동 정점 시점에 trigger 가능. 진짜 정책 실력보다 1차 졸업 신호가 ↑ 나옴. **v22 s3b 사례**: TB end 90.0%/peak 91% → deterministic eval 79% (마지막 100 measurement mean 85.9%, min 79%, max 91%). callback이 4 측정점 91% 시점에 trigger되어 false 졸업. → **졸업 확정은 항상 `eval_stages.py` deterministic eval**. TB는 학습 중 신호일 뿐. **v26-A에서 det check callback (train.py `CurriculumStopCallback.det_env_fn`) 추가** — TB 90% trigger 후 det 50 ep로 진짜 확인 → false positive 차단. v26-A는 1M까지 det 모두 미달(82·72·82%)로 학습 완주, 카드 천장 입증.
14. **v25-A·v26-A 단일 seed 80~84% 결론은 lower outlier (v27이 반박)**: ent_floor `0.003 → 0` linear decay가 stochastic-det gap 12%p → 8%p로 좁힘 ✓. v22 79% → v25-A 84% → v26-A 80% single seed 결과로 "현 카드 본질 천장 80~84%" 결론 → **v27 multi-seed (3 seed mean 90.7%)로 카드 valid 입증**. **single seed로 카드 천장 판단 X — 최소 3 seed**. s3d_90 라인 교훈이 main flow에도 동일하게 적용 (함정 #14 패턴).
15. **TB false positive 차단 = `--det-check` callback (v27 seed2 실증)**: v22 함정 #13 차단 인프라. v27 seed2가 145k에서 TB 90% trigger → det 84% ❌ → cooldown 100k → 245k에서 진짜 졸업. det check 없었다면 145k에서 잘못 졸업했을 것. **새 학습은 default `--det-check` 활성**.
16. **`--det-episodes` 50은 학습 trigger엔 OK but 졸업 확정 측정으로 부족** (v27 Step 0 발견): v27 학습 중 50 ep det check 90/92/90% (mean 90.7) 통과 → 졸업 → 100 ep eval로 재측정하니 90/81/86% (mean 85.7%, **5%p gap**). 50 ep는 sample 적어 운 좋은 분포에서 trigger 가능. **졸업 확정은 항상 `eval_stages.py` 100 ep** (50 ep는 학습 중 cheap check). **다음 학습부터 `--det-episodes 100` 권장**.

**s3d_90 라인 함정** → [`docs/s3d_90_line.md`](docs/s3d_90_line.md) (YAW_W 변형 / single seed baseline / end metric drift artifact 등).

→ **정착 조합 = MODE_3 + Ecoflex passive fin + 3DOF planar + v21 카드 (N=20·ent_floor 차등·yaw \|·\|·0.005)**.

### 가장 결정적이었던 변경 (회고)

1. **3DOF planar** — pitch wobble 제거 (부호 뒤집기 본질)
2. **MATLAB CG.m 값** — sim2real 정합성
3. **MODE_3 fin + Ecoflex 1070** — wave-thrust
4. **action history N=16 (v11)** — ±90° 천장 첫 돌파
5. **yaw_rate 보상 (v21)** ⭐⭐ — v11 천장 단독 돌파 (s3d 29%, 정점 38%)
6. **v22 s3b 학습량 1M** ⭐⭐ — 250k가 부족이었음 입증 (단 TB false positive였음)
7. **v25-A ent_floor schedule + v27 multi-seed** ⭐⭐ — stochastic-det gap 좁힘 + multi-seed로 s3b 50 ep 졸업 임계 통과 (90.7%). v22~v26 single seed 결론 부분 정정.
8. **Step 0 100 ep eval (v27 후속)** — 50 ep det check sample noise 발견 (5%p gap). 졸업 확정 표준 = 100 ep eval 본격 확립. 함정 #16.

**카드 평가 방법론** (s3d_90 라인 + main flow v27에서 본격 확립):
- **multi-seed × deterministic eval** (single seed로 카드 천장 판단 금지)
- **`--det-check` callback** (TB false positive 차단)
- **`eval_stages.py` 6 stage** (catastrophic forgetting 점검)
- **`model_best.zip` 인프라** (peak 시점 보존)

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

**curriculum 진행 상태 (Step 0 = v27 후 100 ep eval × 3 seed)**: s1·s2·s3a **100%** ✓ (forgetting 없음) / **s3b 85.7% ± 3.7%** (mean) — 90% 임계 미달, seed1만 정확히 90% / s3c 44% (v25-A init 그대로) / s3d 30.7%.

### 우선순위 1: s3b 안정화 — 100 ep 기준 mean 90%+ 달성

50 ep 졸업 측정(90.7%)과 100 ep mean (85.7%) 5%p gap이 sample noise. 핵심 카드 = `--det-episodes 100`으로 학습 중 진짜 졸업까지 학습 지속.

1. **v28 — s3b 안정화 (det-episodes 100)** ⭐⭐⭐
   - init: v25-A 모델 (`sim/runs/v25/s3b_arc30/model.zip`, 깨끗한 비교 위해 v27과 동일)
   - 카드 (ent_floor schedule, yaw `|·|`·0.005, N=20): v25-A 그대로
   - **`--det-episodes 100`** (50 → 100, 본질 변경)
   - `--det-check` 활성, s3b max_steps 1M
   - seed 0/1/2 multi-seed
   - 학습 후 `eval_stages.py` × 3 seed 100 ep 필수
   - 예상 ~6시간 (sequential) 또는 ~2시간 (병렬)
   - 가설 1: 학습 중 100 ep 검증 → 진짜 90% 도달까지 학습 지속, mean 90~92% 졸업
   - 가설 2: 1M 안에 100 ep 90% 도달 못함 (현 카드 진짜 천장 86~89%) → 새 축

### 우선순위 2: v28 결과에 따른 다음 카드

| v28 결과 | 다음 카드 |
|---|---|
| 100 ep mean ≥ 90% | ✓ s3b 졸업 확정 → v29 (s3c 학습) |
| 86~89% | ⚠ 카드 한계 — 새 축 (yaw sign-aware / N 변경 등) |
| < 86% | ❌ v27 결과보다 후퇴 — 카드 변경 + multi-seed |

### 우선순위 3: s3c 졸업 후 s3d

s3b 졸업 후 v29 (s3c 1M + det-episodes 100 + multi-seed). 같은 방법론.

### s3 외 인프라/메타 카드 (낮은 우선순위)

2. **HER (Hindsight Experience Replay)** — env Dict obs 큰 변경. 강력하나 구현 비용 큼.
3. **fin actuator 추가** — 단일 모터 한계 풂. (사용자 명시 제외)
4. **ANN surrogate (Lighthill / Zhong)** — fluid model 우회. 실물 motion capture 필요.

### 새 축 카드 (현재 필요 없음 — v27이 reject)

v22~v26 single seed 결과로 "현 카드 천장 80~84%" 결론에 기반한 새 축 후보들 (yaw sign-aware / N 변경 / progress 가중치 등)은 v27 multi-seed로 카드 valid 입증됨에 따라 **현재 진행 불필요**. s3c·s3d에서 카드 한계 입증되면 그때 검토.

### 단일 지느러미의 천장

yaw_test.py로 **yaw rate 물리 상한 ~4.6°/s** (D2 × 20s = 92°, ±90° 마진 작음).

**카드 매트릭스 (Step 0 시점)**:
- N (시간적 표현력) v11/v17/v19 — 정점 30% 가능
- entropy v8~v10/v20/v25-A — collapse 해결 + stochastic-det gap 좁힘
- align/reach 가중치 v12~v15 — 비율 카드 한계
- **yaw reward v21** ⭐⭐ — 천장 돌파 (s3d 29% / 정점 38%)
- **s3b 학습량 (v22)** — 250k 부족 입증, but single seed TB false positive
- **ent_floor schedule (v25-A) + multi-seed (v27)** ⭐⭐ — s3b 50 ep 졸업 임계 통과 (90.7%)
- **det check callback 50 ep (v27)** ⭐ — TB false positive 차단 ✓ but 졸업 확정 측정으론 sample noise
- **`--det-episodes 100` (v28 예정)** — 100 ep 졸업 측정으로 sample noise 제거

→ **카드 평가 표준**: deterministic eval × 3 seed × 100 ep (`eval_stages.py`). single seed / 50 ep는 lower/upper outlier 위험.

⚠ **카드 평가는 항상 "가장 낮은 미달 stage" 위에서**. Step 0 후 s3b 100 ep mean 85.7%로 임계 미달 — **다음 = v28 s3b 안정화** (--det-episodes 100). s3c·s3d 카드는 s3b 100 ep 90% 달성 후.
