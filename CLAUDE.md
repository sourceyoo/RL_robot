# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

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

## 학습 결과·산문 진단·통찰·RL 카드 함정·다음 카드 후보

→ **모든 학습 history는 [`docs/training_log.md`](docs/training_log.md)에** (결별 그룹화·버전별 변경점·Stage 진행 비교·s3d 결과·핵심 통찰·yaw_test 발견·결정적 변경 회고·RL 카드 함정 #7~·카드 평가 방법론·다음 카드 후보).

→ **s3d_90 라인 (구 v22~v30, 분리됨)** → [`docs/s3d_90_line.md`](docs/s3d_90_line.md). s3b·s3c 미달 상태에서 s3d만 fine-tune했던 잘못된 방향성.

**현재 상태**: → [`docs/training_log.md`](docs/training_log.md) "학습 결과 요약 — main flow" 섹션 첫 줄 (매 카드마다 갱신).

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

## 알아둘 함정 — 물리·모델 (절대 규칙)

1. **6DOF freejoint**: pitch cross-coupling으로 추진 방향 뒤집힘. → 3DOF planar 유지. **freejoint로 바꾸지 마세요**.
2. **fluidshape**: MuJoCo `none`/`ellipsoid` 둘뿐.
3. **단일 passive fin (k=4e-7)** / **다단 passive fin** / **CPG 액션 (MODE_2)** / **F1 STEP 다중 ellipsoid**: 모두 추진 못 만듦. → MODE_3 + 단일 motor + Ecoflex passive fin 정착.
4. **중력**: `<flag gravity="disable"/>` 유지. 켜면 가라앉음.
5. **euler "3.14159 0 0" + axis 부호 보정**: planar joint 축 `(1,0,0)`·`(0,-1,0)`·`(0,0,-1)` 보존. base_link 시작 자세에 맞춤.
6. **Inertial 값**: base는 MATLAB CG.m 실물 측정, tail·fin은 MODE_3 URDF 합의값. **임의 수정 X**.

→ **정착 조합 = MODE_3 + Ecoflex passive fin + 3DOF planar + v21 카드 (N=20·ent_floor 차등·yaw \|·\|·0.005)**.

**RL 카드 함정 (#7~#17) + s3d_90 라인 함정** → [`docs/training_log.md`](docs/training_log.md) (RL 카드 시행착오 회고).

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

→ **다음 카드 후보·우선순위·평가 매트릭스 → [`docs/training_log.md` 다음 카드 후보 상세](docs/training_log.md)**.
