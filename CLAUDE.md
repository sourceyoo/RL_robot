# CLAUDE.md

**행동 가이드라인** (caution > speed; trivial은 판단).

## 1. Think before coding
- 가정 명시, 불확실하면 ask. 다중 해석 시 제시(silent pick X).
- 더 간단한 접근 보이면 push back. 불명확하면 멈춰서 무엇이 confusing한지 명명 후 ask.

## 2. Simplicity first
- 요청 외 기능·single-use 추상화·configurability·불가능 시나리오 에러 핸들링 X.
- "200줄 가능하면 50줄로 다시" — senior engineer가 over-complicated라 할 것 같으면 단순화.

## 3. Surgical changes
- 요청 외 코드·주석·포매팅·refactor X. 기존 스타일 유지(개선하지 말 것).
- 무관한 dead code 발견 시 언급만(삭제 X). 본인 변경으로 생긴 orphan(import·변수)만 정리.
- 변경 line은 사용자 요청에 trace 가능해야 함.

## 4. Goal-driven execution
- 검증 가능 성공 기준 정의 후 loop. "Add validation" → "write failing test, then pass" / "Fix bug" → "reproduce in test, then fix".
- Multi-step은 brief plan: `1. step → verify: check`. 약한 기준("make it work")은 빈번한 clarification 유발.

## 5. Cross-check before claiming done (AcoT critic)
- 학습/실험에 영향 주는 코드 변경 (reward·obs·sim 물리·curriculum stage 정의·SAC HP 등) 후 main이 "완료/학습 시작" 보고 **직전** `Agent(general-purpose)` subagent 호출로 교차 검증.
- subagent role = critic. 검토 항목: ① 사용자 의도 ↔ 변경 일치 ② literature·memory·CLAUDE.md 규칙 정합 ③ bug·side effect (분모 0, scale, dim mismatch, 부호 뒤집힘) ④ docstring·주석과 실 코드 동기화.
- writer (main) ↔ critic (subagent) 분리 — 같은 agent의 self-check는 blind spot이 같아 검출률 ↓ (AcoT 패턴).
- critic 결과를 사용자 보고에 명시. 의견 없으면 그 자체 (e.g. "critic 이슈 없음") 명시.
- skip 가능: typo·rename·comment 단독·1-line bash·docs/*.md 수정. 학습 trigger 가능성 있는 변경만.

**성공 지표**: 불필요 diff ↓, 과복잡 재작성 ↓, 구현 후가 아닌 전에 질문, critic 검출로 학습 trigger 전 bug 차단.

---

이하는 본 repo 작업용 가이드.

## 목적

KUFIsh_III (사용자 자체 CAD) 기반 단일 모터(active tail joint + passive fin hinge) 물고기 로봇의 강화학습. MuJoCo + SAC + Gymnasium. **수면 영법(BCF surface swimming) 가정**으로 단순화하여 학습.

---

## ★ 핵심 규칙 (절대 준수)

### 1. 물리·모델 (수정 금지)

- **3DOF planar 유지**: base_link는 `slide_x` + `slide_y` + `hinge_yaw` 3개 명시 joint. **freejoint 금지** — pitch wobble로 추진 방향 뒤집힘.
- **중력 disable 유지**: `<flag gravity="disable"/>`. 켜면 가라앉음.
- **euler="3.14159 0 0" + axis 부호 보정**: planar joint 축 `(1,0,0)`·`(0,-1,0)`·`(0,0,-1)` 보존. base_link 시작 자세에 맞춤.
- **Inertial 값 임의 수정 X**: base는 MATLAB CG.m 실물 측정, tail·fin은 MODE_3 URDF 합의값.
- **정착 조합 = MODE_3 + 단일 motor + Ecoflex passive fin + 3DOF planar**. MODE_2는 fin 미분리 구버전이라 사용 X.
- **fluidshape**: MuJoCo `none`·`ellipsoid` 둘뿐. 현재 `ellipsoid` 사용 (vortex shedding 못 모델 → sim2real 격차 있음).

### 2. 학습 절차

- **단일 stage 학습만**: `curriculum.py --start-stage N --end-stage N`. default 자동 진행(6 stage 끝까지) **사용 금지**.
- **stage 졸업 (자동)**: `CurriculumStopCallback`이 TB stochastic 100 ep `reach_rate ≥ 90%` trigger 후 deterministic eval 100 ep(`--det-episodes` default 100) 통과 시 학습 종료. TB stochastic 정점 false positive(v22/v25-A: TB 90~91% vs det 79~84%) 차단용.
- ⚠ **다음 stage는 사용자 결정**: callback 종료는 학습 trigger일 뿐. 자동 진행 X.
- ⚠ **학습 전·후 6 stage deterministic eval 필수** (`sim/eval_stages.py`): 사전 → 진짜 가장 낮은 미달 stage 확정 / 사후 → 학습 stage 재확인 + 이전 stage catastrophic forgetting 점검 (forgetting 시 mixed sampling·rehearsal 후속).
- ⚠ **카드 우선 대상은 "가장 낮은 미달 stage"** (deterministic 기준). 낮은 stage가 천장이면 위 stage도 천장. claude는 임의로 더 어려운 stage(s3d 등) 우회 금지.
- **max_steps는 안전장치** (도달 못해도 강제 진행이지만 카드 부족 신호). **Full circle은 제외**.

---

## 빠른 시작

```bash
cd sim
python3 curriculum.py --start-stage N --end-stage N --no-viewer   # 단일 stage 학습 (자동 진행 X)
python3 view_policy.py runs/<tag>/model.zip                       # 정책 viewer / `python3 -m mujoco.viewer --mjcf=rl_fish.xml`로 모델 검수
tensorboard --logdir tb_logs/                                     # 학습 곡선 / SSH 백그라운드: `tmux new -s train` + tee log
```

### 학습 시간 (RTX 3090, fps ~305)

s1: ~100초 / s3a~c: 11~19분 / s3d 500k: ~28분 / s3d 1M: ~55분 / 전체: 2~3시간.

병목은 SAC update + Python overhead 98% (env step만 1.4%). 1M step ≈ 1.8M SAC update.

## 활성 모델 — `sim/rl_fish.xml`

RL이 학습하는 단 하나의 모델. **이 파일이 모든 시뮬·학습의 진입점.**

### 핵심 사실 (한눈에 안 보이는 것들)

- **3DOF planar**: base_link = `slide_x` + `slide_y` + `hinge_yaw`. roll/pitch/z 잠김. (수정 금지 → [§1](#1-물리모델-수정-금지))
- **qpos layout = 5**: `[root_x, root_y, root_yaw, tail_joint, fin_joint]`. env 인덱스가 이 순서 의존.
- **수중**: `<option density="1000" viscosity="0.001">` + `<flag gravity="disable"/>`. added mass·drag는 `fluidshape="ellipsoid"`로 자동.
- **euler="3.14159 0 0"**: base_link x축 180° 회전. 보상용 planar joint 축 `(1,0,0)`·`(0,-1,0)`·`(0,0,-1)`. qpos = world 좌표 (x, y, yaw).
- **추진 방향 규약: world −x = 머리(전진).** 보상 함수 작성 시 *목표를 −x에*. freq_sweep에서 x_disp < 0이면 전진.

### 관절 구조 + 액추에이터

```
world ─[slide_x][slide_y][hinge_yaw]─ base_link (PLA 강체)
                                       ├ tail_joint (active hinge, ±20°)
                                       │   └ tail_link (PLA 강체)
                                       │       └ fin_joint (passive hinge, ±30°)
                                       │           └ fin_1 (Ecoflex 00-30, density 1070)
```

- **액션 = 1D** (단일 motor): `position` actuator on `tail_joint`, ctrlrange `-1..1`, gear=0.349(±20°), kp=100, kv=5, forcerange=±3 Nm (BL4260).
- **fin_joint = passive**: stiffness=1e-2, damping=5e-5. RL이 명령하지 않음.

### Inertial 값의 출처

| body | mass | CoM | 출처 |
|---|---|---|---|
| base_link | 2.4349 kg | (0, 0, 0.004535) | **MATLAB CG.m 검증값**. CAD URDF off-diag inertia는 임의 재질 인공물이라 0으로 정리. |
| tail_link | 0.0699 kg | (0.0511, 0, -0.005349) | MODE_3 URDF, y CoM=0으로 대칭화 |
| fin_1 | 0.01052 kg | (0.0773, -0.0004, 0.006) | MODE_3 URDF, density=1070 (Ecoflex) |

base는 사용자 실물 측정값, 나머지는 사용자 합의 후 대칭화한 것. (수정 금지는 [핵심 규칙 §1](#1-물리모델-수정-금지))

### Tail motor spec (BL4260 V2 + 24V + 6.8:1 평기어)

| 항목 | motor side | output side (×6.8, η≈0.8) |
|---|---|---|
| Rated torque | 95 mNm | 0.52 Nm |
| Stall torque | 409 mNm | 2.23 Nm |
| No-load RPM | 4260 | 626 = 10.4 Hz |
| Rated RPM | 3270 | 481 = 8.0 Hz |
| **사용자 실측 (dry 6Hz)** | **2400 rpm** | 352 rpm = 5.88 Hz |

KT=0.137 Nm/A, R=8 Ω → **back-EMF damping (output) = KT²/R × N²·η ≈ 0.087 Nm·s/rad**.
Rotor inertia 추정 ~3e-6 kg·m² motor → **armature (output) ≈ 1.4e-4 kg·m²**.

xml에서 tail_joint `damping="0.087" armature="1.4e-4"`, actuator `forcerange="-2.2 2.2"`로 spec 직접 반영. wet (현재 fluid on)은 fluid drag 추가로 더 ↓ 자연 cap.

**6Hz freq cap 메커니즘**: PD/armature spec만으론 sim PD 무한 bandwidth로 freq cap 안 됨 (정책이 25Hz까지 ctrl 가능). `fish_env.py` `frame_skip=42` (dt=0.084s = 6Hz peak-to-peak time)로 정책 decision freq 11.9Hz → **Nyquist 5.95Hz cap**. 정책이 매 step alternate해도 max ctrl freq 6Hz. 실모터 closed-loop control rate(~10~12Hz)와 정합.

### 추진 검증 (freq_sweep.py)

ctrl=±1 sine 12초: 1~6 Hz 모든 주파수 −x 전진(1Hz −0.49m, 3Hz −2.17m, 6Hz −3.36m). 0.5Hz만 후진.

---

## CAD 소스

`fish_urdf/RL_SIM_MODE_3_description/` mesh 사용. URDF는 형상만, inertial은 위 표대로. URDF의 fin fixed joint는 MJCF에서 passive hinge로 교체(실리콘 변형). `MODE_2`는 fin 미분리 구버전, 사용 X.

---

## Python 학습 인프라 — `sim/`

- `fish_env.py` — Gymnasium 환경 클래스 `FishSwimEnv`. 주요 인자: `target_theta_range`, `success_radius`, `episode_seconds`, `action_history_n`. obs **(11 + action_history_n)D**: tail/fin qpos·qvel + world v + sin/cos yaw + target rel + 최근 N step ctrl 이력.
- `train.py` — SAC 학습 + 별도 thread viewer. `PolicySnapshotCallback` race-free, `CurriculumStopCallback` 자동 조기 종료, **best-save callback** (reach_rate peak에 `model_best.zip` 저장).
- `curriculum.py` — 6단계 stage 정의 + `--start-stage`·`--end-stage`로 단일 stage 실행 + `--seed` `--tb-tag` `--runs-subdir` (multi-seed 지원). 사용 규칙은 [핵심 규칙 §2](#2-학습-절차) 참조.
- `eval_stages.py` — 모델 1개를 모든 stage 분포에서 **deterministic eval** (6 stage × 100 ep, CPU, ~20분). TB callback의 random eval 진동 noise(±10%p)를 회피하고 진짜 졸업 여부 확인. **catastrophic forgetting 측정 필수**.
- `view_policy.py` — 저장된 정책 viewer rollout.
- 진단: `freq_sweep.py` (추진 방향), `yaw_test.py` (회전 능력 — v4 핵심).

### Curriculum 학습

단일 모터로 full circle은 어려움 → 6단계 curriculum. 진행은 [§2](#2-학습-절차).

| Stage | tag | theta | sr | ep_sec | max_steps |
|---|---|---|---|---|---|
| 1 | s1_forward | π fixed | 0.08 | 10s | 400k |
| 2 | s2_anchor | π fixed | 0.04 | 10s | 400k |
| 3a | s3a_arc15 | π ± 15° | 0.08 | 30s | 200k |
| 3b | s3b_arc30 | π ± 30° | 0.08 | 30s | **1M (v22~)** |
| 3c | s3c_arc60 | π ± 60° | 0.08 | 30s | 350k |
| 3d | s3d_arc90 | π ± 90° | 0.08 | 30s | 500k |

multi-seed 예: `python3 curriculum.py --start-stage 4 --end-stage 4 --seed 0 --tb-tag model3_vN_seed0 --runs-subdir vN_seed0`

---

## 학습 history·다음 카드 후보

→ [`docs/training_log.md`](docs/training_log.md) (버전별 변경·통찰·RL 카드 함정 #7~·카드 후보·현재 상태). s3d_90 라인(구 v22~v30 분리) → [`docs/s3d_90_line.md`](docs/s3d_90_line.md). m4 환경(frame_skip 42 + V2 spec) → [`docs/training_log_m4.md`](docs/training_log_m4.md).

---

## TensorBoard 메트릭 (`tb_logs/model3_vN/`)

`rollout/ep_rew_mean`·`ep_len_mean` / `train/{critic_loss,actor_loss,ent_coef}` / `fish/{success_rate,final_distance,episode_seconds,avg_align}`.

진단: α 5만 step 안 0.001↓ → 탐색 부족 / ep_len_mean=max → 도달 X / avg_align 높 + success 낮 → "정렬만" mode.

---

## SAC

`stable-baselines3.SAC` Twin Q + auto α + replay 200k. HP: LR=3e-4, batch=256, γ=0.99, τ=0.005, `ent_coef="auto_0.1"` (floor는 EntCoefFloorCallback).

---

## RL 카드 시행착오·정착

학습 카드 정착 = v21 (N=20·ent_floor 차등·yaw \|·\|·0.005). 추진 실패 사례·함정 #7~#17·s3d_90 라인 → [`docs/training_log.md`](docs/training_log.md).

---

## RL 학습 시 주의

- **fluidshape="ellipsoid" 한계**. vortex shedding 못 모델 → sim2real 격차. 개선은 Lighthill `mjcb_passive` 또는 ANN surrogate.
- **보상 forward = world −x**. 정책이 +x로 가면 환경 인덱스 잘못된 것.
- **3DOF yaw 누적**: fluid asymmetry로 한쪽 도는 경향. RL이 보정 가능.
- **Curriculum init_from**: 이전 stage "직진 편향"이면 Stage 3에서 한쪽 mode만 학습 위험.

---

## 백업·릴리스

`gh release create models-vN ... runs-models-vN.tar.gz` / 복원 `gh release download models-vN -p '*.tar.gz' && tar -xzf ... -C sim/`. 상세·인덱스 → [`docs/training_log.md`](docs/training_log.md).
