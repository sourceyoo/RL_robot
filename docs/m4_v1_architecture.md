# m4_v1 학습 구조 (관찰·행동·보상·학습 인프라)

m4 환경 (frame_skip 42 + V2 spec actuator + fluidcoef + ep_sec 보정 + progress·3.6) 의 학습 구조 전체 정리. 2026-05.

> 변경 history → [`docs/training_log_m4.md`](training_log_m4.md). 규칙·물리 → [`CLAUDE.md`](../CLAUDE.md).

---

## 1. 환경 (MuJoCo 모델 + FishSwimEnv)

### 1-1. MuJoCo 모델 (`sim/rl_fish.xml`)

| 구조 | 값 |
|---|---|
| timestep | 0.002s |
| integrator | `implicitfast` |
| density | 1000 (water) |
| viscosity | 0.001 |
| gravity | disable (`<flag gravity="disable"/>`) |
| fluidshape | `ellipsoid` (default geom) |
| fluidcoef (5계수) | `[0.4, 3.0, 2.81, 1.0, 0.27]` (blunt·slender·angular drag, Kutta·Magnus lift) |

**관절 구조**:
```
world ─[slide_x][slide_y][hinge_yaw]─ base_link (PLA 강체)
                                       ├ tail_joint (active hinge, ±20°)
                                       │   └ tail_link (PLA 강체)
                                       │       └ fin_joint (passive hinge, ±30°)
                                       │           └ fin_1 (Ecoflex 00-30)
```

- **3DOF planar**: base_link = `slide_x` + `slide_y` + `hinge_yaw` (roll·pitch·z 잠김)
- **qpos layout** (5): `[root_x, root_y, root_yaw, tail_joint, fin_joint]`
- **qvel layout** (5): `[vx, vy, vyaw, vtail, vfin]`
- **부호 규약**: world −x = 머리 방향 = **전진**

**Tail actuator (BL4260 V2 + 24V + 6.8:1 평기어)**:
- `position` actuator on `tail_joint`
- ctrlrange ±1, gear 0.349 (→ ±20°)
- kp=100, kv=5
- forcerange ±2.2 Nm (V2 stall × gear × η0.8)
- back-EMF damping 0.087 Nm·s/rad
- armature 1.4e-4 kg·m² (motor rotor inertia × N²)

### 1-2. FishSwimEnv (`sim/fish_env.py`)

| 인자 | 기본값 | 의미 |
|---|---|---|
| `frame_skip` | **42** | mj_step 횟수/decision. dt = 0.002 × 42 = **0.084s** |
| `episode_seconds` | 10.0 (s1/s2: 20, s3a~d: 60) | ep 시간. max_steps = ep_sec / dt |
| `success_radius` | 0.08 (s2: 0.04) | 도달 판정 거리 |
| `target_radius` | 0.5 | target 위치 반지름 |
| `target_theta_range` | (−π, π) | target 각도 분포 (curriculum 단계별) |
| `action_history_n` | 0 (train: **20**) | 최근 N step ctrl obs 추가 |

**핵심 파생값**:
- decision rate = 1/dt = **11.9Hz** → Nyquist **5.95Hz** (정책 ctrl freq cap)
- max_steps (s1/s2) = 20/0.084 ≈ 238 step/ep
- max_steps (s3a~d) = 60/0.084 ≈ 714 step/ep

---

## 2. 관찰 (Observation, **13 + action_history_n = 33D**)

`fish_env.py:113 _get_obs()` 구현. 학습 시 `action_history_n=20`.

| index | 변수 | 단위 | 의미 |
|---|---|---|---|
| 0 | `tail_qpos` | rad | tail joint 각도 |
| 1 | `tail_qvel` | rad/s | tail 각속도 |
| 2 | `fin_qpos` | rad | passive fin 각도 |
| 3 | `fin_qvel` | rad/s | fin 각속도 |
| 4 | `vx` | m/s | world frame x속도 |
| 5 | `vy` | m/s | world frame y속도 |
| 6 | `vyaw` | rad/s | yaw rate |
| 7 | `sin(yaw)` | — | yaw 인코딩 (cyclic) |
| 8 | `cos(yaw)` | — | |
| 9 | `target_rel_x` | m | world frame target − fish 위치 |
| 10 | `target_rel_y` | m | |
| 11 | `\|yaw_err\|` | rad | head_dir과 target 방향의 각도 차 (절댓값) |
| 12 | `distance` | m | √(rel_x² + rel_y²) |
| 13~32 | `action_history[N]` | — | 최근 20 step ctrl (oldest first → newest last) |

**핵심 설계 결정**:
- `sin/cos(yaw)` — yaw discontinuity (−π ↔ π) 회피
- `|yaw_err|` + `distance` — derived obs 명시 (v32-C, 정책 학습 ↑)
- `action_history_n=20` — 비대칭 wagging pattern 학습용 (yaw_test.py로 비대칭 ctrl이 yaw 회전 핵심임 확인)
  - N=20 step × dt 0.084s = **1.68s** 이력 (~10 wag cycle @ 6Hz)
  - 변경 시 obs Box mismatch (함정 #9) — 학습 처음부터

---

## 3. 행동 (Action, **1D continuous**)

`fish_env.py:158 step()`:
```python
action ∈ [-1, +1]   # 단일 motor (tail_joint)
ctrl = np.clip(action, -1.0, 1.0)
data.ctrl[0] = ctrl
# frame_skip=42 inner mj_step 동안 ZOH (zero-order hold)
for _ in range(frame_skip):
    mujoco.mj_step(model, data)
```

ctrl → position actuator → tail target 각도 `gear × ctrl = ±0.349 rad = ±20°`.

**fin_joint는 passive**: stiffness 1e-2, damping 5e-5 — RL 명령 없이 fluid에 의해 휘어짐.

---

## 4. 보상 (Reward, m4_v1)

`fish_env.py:189` 가산식 (곱셈은 mode collapse — 함정 #7):

```python
reward = (
    progress * 3.6                          # ① 진행
    + (10.0 if reached else 0.0)            # ② 도달
    + align_weight * align                  # ③ 정렬
    + YAW_SIGN_W * yaw_rate * yaw_err_sign  # ④ yaw sign-aware
    - ACTION_DIFF_W * action_diff           # ⑤ smoothness
    - TIME_PEN_W * distance * current_time  # ⑥ 시간 penalty
)
```

### 각 항 의미

#### ① Progress reward
```python
progress = self._prev_distance - distance   # step당 target에 가까워진 거리 (m)
reward_progress = progress * 3.6
```
- **weight 3.6**: dt=0.084s 보정 (m1 dt=0.02s 기준 weight 15 대비 4.2× 축소 — literature 권고)
- m4 step당 progress ~0.005~0.02m → reward ~0.018~0.072

#### ② Reach reward
```python
reach_bonus = 10.0 if (distance < success_radius) else 0.0
```
- ep 종료 시점에 1회. 큰 spike reward (학습 trigger).

#### ③ Alignment reward
```python
head_dir = (-cos(yaw), sin(yaw))           # axis (0,0,-1) 보정 반영
align = head_dir · (target_rel_xy / |rel_xy|)   # cos(머리, 목표) ∈ [-1, 1]
reward_align = align_weight * align
```
- `align_weight`:
  - ep_sec ≤ 10s → **0.02** (작은 ep)
  - ep_sec > 10s → **0.012** (큰 ep — 누적 보상 균형) — m4 환경 (20s/60s) 모두 적용
- align ∈ [-1, +1]. +1 = 정확히 target 방향.

#### ④ Yaw sign-aware reward (v30-A 발견)
```python
yaw_err_sign = sign(head_dir × target_rel)  # +1 = target 왼쪽, -1 = 오른쪽
YAW_SIGN_W = 0.005
reward_yaw_sign = YAW_SIGN_W * yaw_rate * yaw_err_sign
```
- **target 방향으로 회전할 때만** + 보상 (반대 방향 회전은 - 페널티)
- 제자리 회전(직진과 회전 sequential) mode 방지. 이동+정렬 parallel 권장.
- v33에서 `YAW_W·|ω|` 제거 (literature 무근거 + 제자리 회전 incentive).

#### ⑤ Action smoothness (Learning Agile ‖J̇‖²)
```python
ACTION_DIFF_W = 0.001
action_diff = (ctrl_t - ctrl_{t-1}) ** 2
reward_smooth = -ACTION_DIFF_W * action_diff
```
- ctrl 변화 max=4 (−1 → +1) → max penalty 0.004
- v33에서 추가 (이전 ctrl_cost 제거 후 교체)

#### ⑥ Time penalty (Pangasius ϕ·d·t)
```python
TIME_PEN_W = 1e-5
current_time = step * dt
reward_time = -TIME_PEN_W * distance * current_time
```
- ep 후반 + 멀리 있으면 penalty ↑ → 천천히 떠도는 mode 억제
- m4 ep 60s × dist 1m × 1e-5 = 6e-4/step max

### 보상 magnitude 예시

| 항 | 일반 범위 | max |
|---|---|---|
| ① progress (도달 중) | 0.02~0.07 | 0.36 (10cm/step) |
| ② reach | 0 또는 10 | 10 (도달 시점 1회) |
| ③ align | -0.012~+0.012 (ep>10s) | ±0.012 |
| ④ yaw_sign | -0.005~+0.005 | ±0.005 × |yaw_rate| (rad/s) |
| ⑤ smoothness | -0.001~0 | -0.004 |
| ⑥ time_pen | -0.0006~0 | -0.0006 |

→ progress·reach가 dominant signal. align·yaw_sign은 shaping. smoothness·time은 regularization.

---

## 5. SAC 알고리즘 (stable_baselines3)

`sim/train.py` 구성.

### 5-1. Hyperparameter

| HP | 값 |
|---|---|
| algorithm | `stable_baselines3.SAC` |
| policy | `MlpPolicy` (default [256, 256] NN) |
| learning_rate | 3e-4 |
| buffer_size | 200,000 |
| batch_size | 256 |
| gamma (γ) | 0.99 |
| tau (τ) | 0.005 |
| train_freq | 1 (매 env step마다 update) |
| gradient_steps | 1 |
| learning_starts | 1,000 |
| ent_coef | `"auto_0.1"` (자동 + 초기 0.1) |
| device | `cuda` |

### 5-2. Callback

| Callback | 역할 |
|---|---|
| `PolicySnapshotCallback` | viewer thread용 정책 동기화 (race-free, sync_every=500) |
| `CurriculumStopCallback` | **stage 졸업 자동 trigger** (조기 종료) |
| `EntCoefFloorCallback` | ent_coef 하한 강제 + linear decay |

#### CurriculumStopCallback 졸업 조건 (CLAUDE.md §2)

```
1. TB stochastic 100 ep `reach_rate ≥ 90%` → 1차 trigger
2. deterministic eval 100 ep `reach_rate ≥ 90%` → 진짜 졸업 (false positive 차단)
3. min_steps 30k 이전엔 조기 종료 X
4. det eval 실패 시 cooldown 100k step 후 재평가
5. max_steps 도달 시 강제 종료 (안전장치, 카드 부족 신호)
```

**best-save**: TB stochastic peak reach_rate 시점에 `model_best.zip` 별도 저장 (후반 후퇴 대비).

#### EntCoefFloorCallback (stage별 차등)

| Stage | floor | floor_end | decay |
|---|---|---|---|
| s1/s2 | 0.002 | — | 없음 |
| s3a | 0.002 | — | 없음 |
| s3b | 0.003 | **0.0** | linear (0 → max_steps) |
| s3c | 0.008 | — | 없음 |
| s3d | 0.010 | — | 없음 |

- 작은 회전(s1~s3b)은 정확도 → 낮은 floor.
- 큰 회전(s3c~d)은 비대칭 ctrl 탐색 → 높은 floor.
- s3b만 0 → linear decay (학습 후반 deterministic policy 수렴, 함정 #13 대응 — stochastic-det gap 좁힘).
- floor 0.02 이상은 학습 붕괴.

---

## 6. Curriculum (`sim/curriculum.py`)

### 6-1. Stage 정의

| Stage | tag | theta range | sr | ep_sec | max_steps | ent_floor |
|---|---|---|---|---|---|---|
| 1 | s1_forward | π fixed (직진) | 0.08 | 20s | 1M | 0.002 |
| 2 | s2_anchor | π fixed | 0.04 (정밀) | 20s | 1M | 0.002 |
| 3a | s3a_arc15 | π ± π/12 (±15°) | 0.08 | 60s | 1M | 0.002 |
| 3b | s3b_arc30 | π ± π/6 (±30°) | 0.08 | 60s | 1M | 0.003 → 0 |
| 3c | s3c_arc60 | π ± π/3 (±60°) | 0.08 | 60s | 1M | 0.008 |
| 3d | s3d_arc90 | π/2 ~ 3π/2 (±90°) | 0.08 | 60s | 1M | 0.010 |

> max_steps는 모두 1M 안전장치. CurriculumStopCallback이 90% 졸업 시 조기 종료.

### 6-2. 학습 진행 규칙 (CLAUDE.md §2)

- **단일 stage 학습만**: `--start-stage N --end-stage N`
- 다음 stage 자동 진행 X (사용자 결정)
- 학습 전·후 6 stage deterministic eval 필수 (`eval_stages.py`, catastrophic forgetting 점검)
- init 모델: 이전 stage `model.zip` 자동 (또는 `--init-from` 명시)

### 6-3. multi-seed (분야 표준 σ 측정)

```bash
# 3 seed 병렬 background
for s in 0 1 2; do
  python3 curriculum.py --start-stage N --end-stage N --seed $s \
      --tb-tag m4_v1_seed${s} --runs-subdir m4_v1_seed${s} --no-viewer \
      --init-from runs/m4_v1_seed${s}/<prev_tag>/model.zip \
      > /tmp/m4_v1_seed${s}_<tag>.log 2>&1 &
done
wait
```

자원 (RTX 3090 24GB + Ryzen 9 32 thread):
- per-seed: GPU ~2GB, CPU ~2 thread, fps 124~193 (병렬 시 GPU 경합)
- 3 seed 합산: GPU 6GB, CPU 6 thread, 총 처리량 ~500 fps

---

## 7. 데이터 흐름 (학습 1 step)

```
[Environment]
  obs (33D) ──────┐
                  ↓
              [SAC Policy] ────► action (1D, ∈[-1,+1])
                  ↑                       │
                  │                       ▼
                  │           [FishSwimEnv.step]
                  │                       │
                  │            ctrl = clip(action)
                  │                       ↓
                  │      [mj_step × frame_skip=42]  (inner ZOH)
                  │                       │
                  │             new qpos, qvel
                  │                       ↓
                  │              reward 계산 (식 §4)
                  │                       │
                  └──── new_obs ◄─────────┤
                                          ▼
                              {obs, action, reward, done, next_obs}
                                          │
                                          ▼
                              [Replay buffer (200k)]
                                          │
                                          ▼ (every step)
                          [SAC update: 1 gradient step]
                          (critic Q-loss, actor loss, ent_coef auto)
```

---

## 8. 평가 (`sim/eval_stages.py`)

```bash
python3 eval_stages.py runs/m4_v1_seed{N}/<stage>/model.zip
```

- 1 모델 × **6 stage × 100 ep deterministic eval** (~20분, CPU)
- TB stochastic eval (진동·noise 큼)의 false positive 차단용
- **catastrophic forgetting 점검** 필수 (학습 안 한 stage 보존 여부)
- m4 환경 stage ep_sec sync (s1/s2 20s, s3a~d 60s)

평가 metric:
- `reach_rate` (도달률, %)
- `avg_align` (정렬도)
- `final_dist` (마지막 distance)
- `ep_seconds` (도달까지 평균 시간)

---

## 9. m4_v1 학습 결과 요약 (2026-05-16)

| Stage | 졸업 step (mean) | 졸업 시간 (mean) | det 100ep (mean ± σ) | 일반화 (다음 stage transfer) |
|---|---|---|---|---|
| s1_forward | ~48k | ~5분 | 100.0 ± 0.0% | s2 자연 100% |
| s3a_arc15 | 245k | 27분 | 99.0 ± 0.8% | s3b 31.7 → **55.7** (+24%p) |
| s3b_arc30 | 317k | 40분 | 97.0 ± 2.2% | s3c 28.0 → **55.0** (+27%p), s3d 16.7 → **38.7** (+22%p) |

자세한 history → [`docs/training_log_m4.md`](training_log_m4.md).

---

## 10. 참조

- 환경 변경 history: [`docs/training_log_m4.md`](training_log_m4.md)
- 이전 환경 (v1~v33 model3): [`docs/training_log.md`](training_log.md)
- 핵심 규칙·물리 모델: [`CLAUDE.md`](../CLAUDE.md)
- 외부 literature:
  - frame_skip 42 (Tp/2 패턴): `~/research/fish_rl/literature_timestep_fish_rl.md` (ETH Verma 2018 / Novati 2017)
  - fluidcoef sysid: `~/research/fish_rl/literature_mujoco_water_env_fish_rl.md` (fishsim ETH SRL)
  - sim2real DR 6 패턴: `~/research/fish_rl/literature_sim2real_drag_fish_rl.md` (Duraisamy 2022 / Peng 2018)
