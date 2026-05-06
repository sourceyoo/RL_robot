# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 목적

KUFIsh_III (사용자 자체 CAD) 기반 단일 관절 물고기 로봇의 강화학습. MuJoCo + SAC + Gymnasium. **수면 영법(BCF surface swimming) 가정**으로 단순화하여 학습.

---

## 빠른 시작

```bash
cd sim
python3 curriculum.py --no-viewer                       # 3단계 자동 학습 (헤드리스)
python3 view_policy.py runs/s2_anchor/model.zip         # 저장된 정책 보기
python3 -m mujoco.viewer --mjcf=rl_fish.xml             # 모델만 검수
tensorboard --logdir tb_logs/                           # 학습 곡선
```

SSH/원격에서 백그라운드 실행:

```bash
tmux new -s train
python3 curriculum.py --no-viewer 2>&1 | tee curriculum.log
# Ctrl+B, D로 분리. 재진입은 tmux attach -t train
```

---

## 활성 모델 — `sim/rl_fish.xml`

RL이 학습하는 단 하나의 모델. **이 파일이 모든 시뮬·학습의 진입점.**

### 핵심 사실 (한눈에 안 보이는 것들)

- **3DOF planar 운동**. base_link는 freejoint가 아니라 `slide_x` + `slide_y` + `hinge_yaw` 3개 명시 joint. roll/pitch/z는 *수학적으로 잠김*. **freejoint로 바꾸지 마세요** — pitch wobble로 추진 방향이 뒤집힙니다 (함정 #1).
- **qpos layout = 5**: `[root_x, root_y, root_yaw, tail_joint, fin_joint]`. env 코드의 인덱스가 이 순서에 의존.
- **수중 환경**. `<option density="1000" viscosity="0.001">` (실제 물). `<flag gravity="disable"/>`로 중력 끔. 부력은 명시적으로 안 받지만 added mass + drag는 `fluidshape="ellipsoid"`로 자동. **중력 켜지 마세요** — 별도 부력 콜백 없으면 가라앉음.
- **euler="3.14159 0 0" + axis 부호 보정**. base_link가 x축 180° 회전된 상태로 시작(시각 상 위쪽 정렬). 이 회전을 보상하기 위해 planar joint 축이 `(1,0,0)`, `(0,-1,0)`, `(0,0,-1)`. qpos는 *world* 좌표 (x, y, yaw)와 1:1.
- **추진 방향 규약: world −x = 머리 방향(전진).** freq_sweep에서 x_disp < 0이면 전진. 보상 함수 작성 시 *목표 위치를 머리 방향(−x)에 두기*.

### 관절 구조 + 액추에이터

```
world ─[slide_x][slide_y][hinge_yaw]─ base_link (PLA 강체)
                                       ├ tail_joint (active hinge, ±20°)
                                       │   └ tail_link (PLA 강체)
                                       │       └ fin_joint (passive hinge, ±30°)
                                       │           └ fin_1 (Ecoflex 00-30, density 1070)
```

- **액션 공간 = 1차원** (단일 motor): `position` actuator on `tail_joint`. ctrlrange `-1..1`, gear=0.349 → 명령 ±20°. kp=100, kv=5, forcerange=±3 Nm (BL4260 + 평기어 트레인 continuous 영역).
- **fin_joint는 passive**. stiffness=1e-2, damping=5e-5. RL이 명령하지 않음.

### Inertial 값의 출처

| body | mass | CoM | 출처 |
|---|---|---|---|
| base_link | 2.4349 kg | (0, 0, 0.004535) | **MATLAB CG.m 검증값** (xG=xb=0, zG=0.004535). CAD URDF의 off-diag inertia는 임의 재질 인공물이라 0으로 정리. |
| tail_link | 0.0699 kg | (0.0511, 0, -0.005349) | MODE_3 URDF, y CoM=0으로 대칭화 |
| fin_1 | 0.01052 kg | (0.0773, -0.0004, 0.006) | MODE_3 URDF, density=1070 (Ecoflex) |

**임의로 수정 마세요**. base는 사용자 실물 측정값, 나머지는 사용자 합의 후 대칭화한 것.

### 추진 검증 — frequency sweep (sine wave ctrl=±1, 12초 측정)

| f[Hz] | tail | fin | x_disp | 방향 |
|---|---|---|---|---|
| 0.5 | ±21° | ±31° | +0.85 | 후진 |
| **1.0** | ±21° | ±32° | **−0.49** | **전진 ✓** |
| **2.0** | ±22° | ±33° | **−1.72** | **전진 ✓** |
| **3.0** | ±23° | ±31° | **−2.17** | **전진 ✓** |
| **4.0** | ±23° | ±29° | **−2.66** | **전진 ✓** |
| **5.0** | ±21° | ±28° | **−3.00** | **전진 ✓** |
| **6.0** | ±23° | ±28° | **−3.36** | **전진 ✓** |

(x_disp < 0 = 머리 방향 = 전진. 6 Hz: 0.28 m/s — 실제 소형 robotic fish 영역 0.1~0.5 m/s)

→ 1~6 Hz 모든 주파수에서 +x 머리 방향 전진. 0.5 Hz 후진은 저주파 wave-thrust 약함 (RL이 자연 회피).

---

## CAD 소스 — `fish_urdf/RL_SIM_MODE_3_description/`

활성 mesh의 출처. URDF는 *형상만* 사용 (mesh 파일들), inertial 값은 위 표대로 별도 검증.

- mesh: `base_link.stl`, `tail_link_1_1.stl`, `fin_1.stl`
- URDF는 fin을 `<joint type="fixed">`로 부착하지만 우리 MJCF는 **passive hinge로 교체** (실리콘 변형 모델링).
- 이전 버전 `RL_SIM_MODE_2_description/`도 남아있지만 **사용 안 함** — fin 분리 안 된 구버전.
- MODE_2 패키지 `launch/controller.yaml`, `launch/controller.launch`에 한국어 관절명이 CP949 mojibake. ros_control 안 돌아가지만 **RL 학습엔 무관** (URDF만 mesh 출처로 사용).

---

## Python 학습 인프라 — `sim/`

- `fish_env.py` — Gymnasium 환경. obs **(11 + action_history_n)D**: tail/fin qpos·qvel + world v + sin/cos yaw + target rel + 최근 N step ctrl 이력.
- `train.py` — SAC 학습 + 별도 thread viewer. `PolicySnapshotCallback`으로 race-free, `CurriculumStopCallback`으로 reach_rate 임계 자동 조기 종료.
- `curriculum.py` — 6단계 자동 진행 스크립트 (v4~).
- `view_policy.py` — 저장된 정책을 viewer로 rollout.
- 진단 스크립트: `freq_sweep.py` (추진 방향), `yaw_test.py` (회전 능력 — v4 핵심 진단), `freq_sweep_locked.py`, `freq_sweep_norollpitch.py`, `waveform_test.py`, `multiseg_test.py`.

### Action history obs (v4~)

`action_history_n` 인자 (default 0). 0보다 크면 obs 끝에 **최근 N step의 ctrl 값**이 stack되어 추가됨. v4에서 N=8 (3Hz × 17 step/cycle의 1/2 = 약 한 sweep 분량).

목적: 정책이 *시간적 비대칭 ctrl 패턴* (D2 패턴 — 75% 한쪽 stroke + 25% 반대 stroke 등)을 발견 가능. `yaw_test.py` 진단으로 비대칭이 yaw 회전의 핵심임 확인 (대칭 sine 0.49°/s vs D2 패턴 4.6°/s).

### 보상 함수 (`fish_env.py`)

**v5 (현재 코드)** — align을 progress에 곱셈으로 결합:

```
prog = (prev_dist - cur_dist) * 10
if prog > 0:
    prog *= 0.5 + 0.5 * cos(head, target)   # 0(반대) ~ 1(정조준)
reward = prog  +  5 if reached  -  0.001 * a²
```

- **progress** (delta dist × 10) — distance(absolute)로 주면 학습 느림
- **align factor** (v5~): 가까워질 때만 적용. 정렬 안 된 상태로의 진행은 보상 약화 → 회전 우선 정책 유도. 멀어질 때(prog<0)는 풀 페널티 유지 (대칭 비대칭).
- **도달 보너스** 5, ctrl 비용 미미.
- 에피소드 = `episode_seconds / 0.02` step (default 10s = 500 step). 목표는 반지름 0.5m 원 위 랜덤.

**v2~v4 (구버전)** — align은 가산항: `... + align_w · cos(head, target)`. 가중치 균형 제약 (10s ep `align_w ≤ 0.02`, 20s ep `≤ 0.01`)이 있었음. v5에서 `align_weight` 인자 자체를 제거.

### Curriculum 학습

Random target full circle은 단일 모터에 어려움. **6단계 자동 진행** (v4~):

| Stage | tag | theta 범위 | success_radius | episode_seconds | max_steps |
|---|---|---|---|---|---|
| 1 | s1_forward | π fixed | 0.08 m | 10s | 400k |
| 2 | s2_anchor | π fixed | **0.04 m** | 10s | 400k |
| 3a | s3a_arc15 | π ± 15° | 0.08 m | 10s | 200k |
| 3b | s3b_arc30 | π ± 30° | 0.08 m | 10s | 250k |
| 3c | s3c_arc60 | π ± 60° | 0.08 m | 10s | 350k |
| 3d | s3d_arc90 | π ± 90° (=[π/2,3π/2]) | 0.08 m | **20s** | 500k |

(v1~v4에는 stage별 `align_weight` 가중치 — 1·2·3a·3b·3c는 0.02, 3d는 v3에서 0.008. v5에서 `align_weight` 제거하고 곱셈 보상으로 통합.)

자동 진행: 최근 100 ep `reach_rate ≥ 90%` → 학습 조기 종료 → 다음 단계로 fine-tune. max_steps는 안전장치. STAGES 딕셔너리는 stage별로 `episode_seconds` 오버라이드 가능 (`curriculum.py`).

v3 → v4 변화: Stage 3을 4단계로 분화 (점진적 회전량 증가) + action history 8 추가.
v4 → v5 변화: 보상 함수 변경 (align을 progress와 곱). align_weight 인자·s3d 오버라이드 제거.

**Full circle (θ ∈ [-π, π])은 사용자 결정으로 제외** — 단일 모터로 180° 회전 후 추적은 비현실적.

```bash
python3 curriculum.py                    # 전체 단계 자동 + viewer
python3 curriculum.py --no-viewer        # viewer 없이 빠르게
python3 curriculum.py --threshold 0.85   # 85%로 임계 완화
python3 curriculum.py --start-stage 2    # Stage 2부터 (이전 model.zip 있어야)

# 단일 단계 수동
python3 train.py --tag s1_forward \
    --theta-min 3.14159 --theta-max 3.14159 \
    --success-radius 0.08 --success-threshold 0.9 \
    --steps 200000
```

`train.py` curriculum 인자: `--theta-min/max`, `--success-radius`, `--success-threshold`, `--init-from <model.zip>`, `--eval-window` (기본 100), `--check-every` (기본 5000).

---

## 학습 결과 — v1 ~ v10

GitHub Release: [`models-v1`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v1), [`models-v2`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v2), [`models-v5`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v5) (실패 기록). v3·v4·v6·v7·v8·v9·v10 release 안 함.

### 버전별 변경점

| 변경 | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v10 |
|---|---|---|---|---|---|---|---|---|---|---|
| align reward | 없음 | `+0.02·align` | s3d만 0.008 | (v3) | **`prog>0`만 곱** `(0.5+0.5·align)` | (v5) | **대칭 곱** (prog 부호 무관) | **v4 가산식 복귀** `+0.02·align` | (v8) | (v8) |
| Stage 3 분할 | 단일 ±90° | s3a→s3b | (v2) | **s3a/b/c/d (15→30→60→90°)** | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) |
| s3 ep 길이 | 10s | 10s | s3d 20s | (v3) | (v3) | **s3a~d 30s 통일** | (v6) | (v6) | (v6) | (v6) |
| ent_coef init | "auto" (1.0) | `auto_0.1` | `auto_0.1` | `auto_0.1` | `auto_0.1` | (v5) | (v5) | (v5) | (v5) | (v5) |
| **action history obs** | 없음 | 없음 | 없음 | **8 step (19D)** | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) |
| **EntCoefFloorCallback** | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | **균등 0.02** | **균등 0.005** | **균등 0.002** | **stage별 차등 (s1~s3a 0.002 / s3b 0.003 / s3c 0.005 / s3d 0.006)** |

### s3d (±90°) 결과 — 마지막 100 ep 윈도우

| 버전 | reach | avg_align | final_dist | mode | 한 줄 평가 |
|---|---|---|---|---|---|
| v1 | 24% | (미기록) | — | (미기록) | baseline (50 ep det) |
| v2 | 24% | (미기록) | — | (미기록) | align 추가, 변화 X |
| v3 | 24% | (미기록) | — | (미기록) | ep 20s, 변화 X |
| v5 | 20% | −0.50 | 0.32 m | 머리 반대 + 후진 | 곱셈 비대칭 부작용 |
| v6 | 20% | −0.10 | 0.31 m | 모호 | 시간만 늘림, 학습 미수렴 |
| v7 | 16% | −0.73 | **6.27 m** | **도망** ⚠ | 대칭 곱 + ent floor 0.02, 재앙 |
| **v8** | **17%** | **+0.58** | **0.57 m** | **목표 향함** ✓ | **mode 정상화 첫 성공, 천장 미달성** |
| v9 | 13% | +0.52 | 0.42 m | 목표 향함 | floor 0.002 — 큰 회전 entropy 부족, s3c·d 후퇴 |
| v10 | 13% | **+0.61** | 0.65 m | 정렬 best, 추진 worst | stage별 차등 floor — s3c/d 회복 실패. 정렬은 v8/v9/v10 중 최고. |

### Stage 진행 비교

| Stage | v5 | v6 | v7 | v8 | v9 | **v10** |
|---|---|---|---|---|---|---|
| s3a_arc15 (±15°) | 91% (150k 조기) | 90% (185k 조기) | 92% (115k 조기) | 74% (200k max) | 90% (125k 조기) | **89% (89k 조기)** ⭐ |
| s3b_arc30 (±30°) | 78% (max) | 71% (max) | 59% (max) | 70% (max) | 75% (max) | 72% (max) |
| s3c_arc60 (±60°) | 36% (max) | 35% (max) | 31% (max) | **46% (max)** ⭐ | 29% (max) | 28% (max) |
| s3d_arc90 (±90°) | 20% | 20% | 16% | 17% | 13% | 13% |

### v8 핵심 진단 — 병목 재정의

v8 이전(v5~v7): **머리 방향 자체가 잘못됨** (mode collapse) → 보상 함수 문제.
v8 시점: **머리 방향 OK (avg_align +0.58), 그러나 fish가 회전+추진 동작 시퀀스를 못 만듦**:
- ep 26s 동안 final_distance 0.57m → 거의 못 움직임
- s3c 46% (이전 best 36%) → ±60°까진 진보. ±90°에서 천장.
- → 진짜 병목은 **정책이 yaw_test.py D2 같은 시간적 비대칭 ctrl 패턴을 발견 못 함**. 보상은 옳은데 policy가 동작을 못 만듦.

### v9 핵심 진단 — ent_floor의 stage 의존성

v8 가설("s3a 74% = floor 0.005 entropy 과잉")은 **확인됨**: floor 0.002로 낮추니 **s3a 74→90% (125k 조기 종료)**. 작은 회전·정확도 stage엔 결정적 진전.

그러나 큰 회전 stage에서 역효과:
- s3c 46→29% (−17%p), s3d 17→13% (−4%p)
- ent_coef 추이: s3a까진 0.004대 (자동 감소 중, floor 미도달) → s3b부터 0.002 floor 도달 → s3c·d 내내 0.002 고정
- 하필 **floor 강제 시점이 큰 회전 stage 진입과 겹쳐** 탐색 죽임. 0.002는 ±60°/±90° mode 발견에 entropy 부족.

→ **단일 floor로 전체 cover 불가**. 작은 회전(s3a~b)은 0.002~0.003, 큰 회전(s3c~d)은 0.005 이상이 적합.

부수 관찰: v9 s3d `ep_rew_mean` +14.24 (v8 +12.20) — reach 더 낮은데 보상 더 높음. align 누적(ep 26초)으로 stay-aligned 정책 강화. final_dist 0.42m (v8 0.57m)로 더 가까이 가지만 도달은 못 함.

### v10 핵심 진단 — Stage별 차등 floor의 한계

가설: stage별 차등 floor (s1~s3a 0.002 / s3b 0.003 / s3c 0.005 / s3d 0.006)로 v9의 작은 회전 진전 + v8의 큰 회전 mode 탐색을 둘 다 회복.

메커니즘 검증 ✓: stage 진입 시 ent_coef가 정확히 새 floor에 머무름 (s3b 0.0030, s3c 0.0050, s3d 0.0060).

그러나 결과적 가설은 **부분 실패**:
- ✅ s3a **89% (89k 조기)** — v9 90% (125k)와 동등하지만 더 빨리 수렴
- ❌ s3c **28%** (v9 29%, v8 46%) — v8 수준 회복 실패
- ❌ s3d **13%** (v9 동률) — v8 17% 못 넘음

원인 추정 (단일 floor 가설로는 설명 안 됨):
- v8은 **모든 stage 처음부터 floor 0.005**. multi-modal 탐색 가능성이 *처음부터* 보장됨.
- v10은 s3a/b에서 **낮은 entropy(0.002~0.003)로 정책 narrow 수렴** → s3c 진입 시 floor를 0.005로 끌어올려도 정책의 표현 분포 자체가 이미 좁아짐. callback과 SAC `log_ent_coef.grad`의 줄다리기로 effective entropy는 0.005에 머무르지만, **정책이 그 entropy를 활용한 탐색을 못 함**.
- 즉 entropy floor는 step별 양은 보장하지만, *이전 stage가 만든 정책 mode*를 깨뜨리진 못함.

부수 관찰 — v10 s3d "정렬 best, 추진 worst":
- avg_align **+0.608** (v8/v9/v10 중 최고). 머리는 가장 잘 맞춤.
- final_dist **0.645m** (v8 0.57m, v9 0.42m). *더 멈춤*.
- 정책이 "정렬 정확 / 추진 무능" mode. 이전 stage의 정확도 편향 잔재.

→ **ent_floor 카드 종결**. v7~v10이 4가지 floor 전략(균등 0.02 / 균등 0.005 / 균등 0.002 / 차등) 시도했고 큰 회전 천장(±90° 13~24%) 못 뚫음. 진짜 병목은 entropy 양이 아니라 **정책 표현력** — 시간적 비대칭 ctrl 패턴(D2 등)을 **obs/architecture가 지원하지 못함**.

### 핵심 통찰 — v3 진단의 결정적 발견 (`yaw_test.py`)

| 측정 대상 | yaw rate |
|---|---|
| 학습된 v3 정책 (50 ep 평균 max yaw) | **0.94°/s** |
| 대칭 sine 인가 (baseline) | 0.49°/s |
| **D2 패턴 (75% 음 stroke + 25% 양)** | **4.60°/s** ✓ |
| B2/B4 (DC offset sine) | 3.2°/s |
| C2 (비대칭 진폭 sine) | 2.9°/s |

→ **물리적으로 ±90° 회전이 20s 안에 충분히 가능** (D2 × 20s = 92°). 단 RL이 이런 *시간적 비대칭 패턴*을 발견 못 함. 진짜 병목 = **정책의 표현력 부족** (현재 obs는 즉각 상태만 받음, 시간적 패턴 학습 어려움).

### 다른 통찰

- ✅ **Stage 1·2** s1 30k + s2 30k = 60k step에 100% (모든 버전 동일). align reward 누적이 ep_rew를 정확히 +1.2 (= 60 step × 0.02 × 평균 align) 올림 — 보상 일관성 (v2~v4·v8 한정, v5~v7은 곱셈식이라 ep_rew 절대값 다름).
- ✅ **±15° head arc** — v2 84%, v5/v6/v7 90%대. s2→s3a fine-tune 효율적.
- ✅ **v8 mode 정상화** — avg_align 처음 양수(+0.58), 도망/반대 mode 사라짐. 가산식 + EntCoefFloorCallback 조합 효과.
- ✅ **v8 s3c 46%** — 이전 best 36% 대비 +10%p. ±60°까진 진보.
- ❌ **±90° head arc 정체** — v1~v10 모두 13~24%. 보상·algorithm·obs·ent floor(균등/차등) 어느 변경도 천장 못 뚫음.
- ❌ **v5/v7 곱셈 보상의 역효과** — 비대칭(v5)은 머리 반대 + 후진, 대칭(v7)은 도망 mode. 함정 #8.
- ❗ **ent_coef collapse 해결** — v2~v6 모두 0.0004~0.0007로 죽었으나 v7 floor 0.02는 결정적 학습 방해, v8 floor 0.005가 mode 정상화 균형점.
- ❗ **v9 s3a 90% 회복** — v8 가설(floor 0.005 entropy 과잉) 입증. floor 0.002로 낮추니 작은 회전 결정적 학습 회복(125k 조기). 그러나 같은 floor가 s3c·d에선 entropy 부족으로 −17%p / −4%p 후퇴 → **floor는 stage 의존적**.
- ❗ **v10 stage별 차등 floor 메커니즘 OK / 결과 부분 실패** — ent_coef가 stage별 정확히 floor에 머무름(메커니즘 정상). s3a 89% (89k) 더 빨리 수렴. 그러나 s3c 28% (v8 46% 회복 실패), s3d 13% (v9 동률). 정렬 best (+0.61) but 추진 worst (final_dist 0.65m). **이전 stage의 낮은 entropy가 만든 narrow mode를 다음 stage의 높은 floor가 깨뜨리지 못함** — entropy 양 ≠ 정책 표현력.

---

## TensorBoard 메트릭 가이드

`tb_logs/` 구조는 **버전 폴더로 분리** (`model3_v1/`, `model3_v2/`, ...). TB UI에서 폴더 prefix가 run 이름에 들어와 v별 비교 용이.

### SB3 기본 (이름 못 바꿈, 라이브러리 internal)

| 태그 | 정의 | 우리 환경 좋은 값 |
|---|---|---|
| `rollout/ep_rew_mean` | 최근 100 ep 평균 보상 | Stage 1·2: +9~+11 (align 포함), Stage 3a: +7~ |
| `rollout/ep_len_mean` | 최근 100 ep 평균 step | 도달 시 <max, 못 하면 max |
| `train/actor_loss` | $J_\pi(\phi) = \alpha\log\pi - Q$ | 음수 보통, 절댓값 추세 중요 |
| `train/critic_loss` | Q-function MSE | 0.001~10, 발산 X |
| `train/ent_coef` (α) | auto-tuned entropy 가중치 | 0.005~0.05 적당. <0.001 일찍 가면 collapse |

### v2 신설 — `fish/*` namespace (직관적 이름)

| 태그 | 정의 |
|---|---|
| `fish/success_rate` | 최근 N ep 도달률 (0~1) |
| `fish/final_distance` | 최근 N ep 평균 종료 거리 (m) |
| `fish/episode_seconds` | 최근 N ep 평균 ep 길이 (초, ep_len × dt) |
| `fish/avg_align` | 최근 N ep 평균 정렬도 (-1 반대 ~ +1 정조준) |

(v1에는 없음 — `CurriculumStopCallback`이 추가하기 시작했음. v1 학습 곡선 비교 시 SB3 default만 사용)

### 진단 신호

- **α가 5만 step 내 0.001 이하** → 탐색 부족, ep_rew 정체. `auto_0.1`로 늦춰지긴 하나 floor는 아님.
- **ep_len_mean = max 유지** → 도달 못 하는 정책 (보상 신호 없음).
- **critic_loss 발산** → reward scaling/LR 문제.
- **fish/avg_align 높지만 success_rate 낮음** → "정렬만 추구 + 진행 안 함" 정책 (align_weight 너무 큼).

TB 띄우기 (SSH 원격 → Mac 브라우저는 VS Code Remote SSH가 자동 포워딩):

```bash
tensorboard --logdir tb_logs/ --bind_all
```

---

## SAC 알고리즘 메모

`stable-baselines3.SAC` 기본 설정. **Maximum-Entropy off-policy actor-critic**.

핵심:
- **목적함수**: $\mathbb{E}[\sum r_t + \alpha \mathcal{H}(\pi)]$ — 보상 + 정책 entropy.
- **Twin Q-network**: $\min(Q_1, Q_2)$로 overestimation 방지.
- **Reparametrization**: $a = \mu + \sigma \cdot \epsilon$로 stochastic 정책 backprop. tanh squash로 [-1,1].
- **Auto α tuning**: 목표 entropy $\mathcal{H}_{target} = -\dim(A) = -1$로 자동 조정.
- **Off-policy + replay 200k**: 같은 transition 수십 번 재사용 → PPO 대비 sample-efficient.

**왜 SAC인가** (PPO 아님): 1D 연속 ctrl, 시뮬 비싸지 않음, multi-modal optimum 가능성. 학계 fish RL 표준.

하이퍼파라미터 (`train.py:262`, `curriculum.py`): LR=3e-4, batch=256, γ=0.99, τ=0.005, buffer=200k, **`ent_coef="auto_0.1"` (v2~)** — *초기값* 0.1, floor 아님. 진짜 floor는 custom callback 필요 (구현 안 됨).

---

## 작업 진행 규칙 — 한 단계씩 분리

다단계 작업은 **한 단계씩 분리해서** 진행. 한 단계 결과를 보고하고 사용자 확인을 받은 뒤 다음으로 넘어간다.

- "한 단계"의 단위는 **사용자가 결과를 보고 다음 결정을 내릴 수 있는 지점**.
- 예: 모델 변경 → 멈춤. viewer 검증 → 멈춤. 정확성 테스트 → 멈춤. env 갱신 → 멈춤. smoke test → 멈춤.
- TaskCreate로 전체 단계를 미리 나열해 두는 건 OK. status는 한 번에 한 task만 in_progress.
- 자명하게 묶이는 미세 작업은 쪼개지 않아도 됨. 핵심은 **사용자가 검수·중단할 기회를 주는 것**.

---

## 알아둘 함정 (이미 발견·해결된 것)

이미 시도해본 막다른 길. **반복하지 마세요**.

1. **6DOF freejoint**: yaw 진동의 pitch cross-coupling으로 fish가 60° 기울며 추진 방향 뒤집힘. → 3DOF planar로 결정.
2. **fluidshape 변경 시도**: MuJoCo는 `none`/`ellipsoid` 둘뿐. 다른 도형 없음.
3. **단일 passive fin (k=4e-7)**: 1자유도 spring은 wave 못 만듦. 단일 passive vs 강체 fin 추진 동일 (소수점 셋째 자리까지).
4. **다단 passive fin (segment chain)**: wave 패턴은 발생하지만 ellipsoid fluid 한계로 +x 추진 못 만듦.
5. **CPG/Fourier 액션공간 (MODE_2 시절)**: 단일 관절 본질적 한계 — 14가지 waveform 모두 후진. 학습할 +x 패턴이 모델에 존재하지 않음.
6. **F1 STEP 분해 → 다중 fluid ellipsoid**: z축으로 +0.56 m 발산. 비대칭 lift 폭주. 롤백.
7. **단일 관절 + ellipsoid + 6DOF 조합**: 위 1~6의 종합 결론 — 부호 못 뒤집음.
8. **곱셈 보상 (`prog × align_factor`)**: v5(비대칭, `prog>0`만)는 머리 반대 + 후진 mode (avg_align −0.5). v7(대칭, prog 부호 무관)은 도망 mode (final_distance 6m, ±90° reach 16%). 둘 다 v3 가산식(`+ 0.02·align`) 24%보다 나쁨. **가산식으로 가야 함**.

→ 정착한 조합 = **MODE_3 + Ecoflex passive fin + 3DOF planar**.

가장 결정적이었던 변경 (회고):
1. **3DOF planar 가정** — pitch wobble 제거가 부호 뒤집기의 본질
2. **MATLAB CG.m 값 적용** (xG=xb=0) — sim2real 정합성
3. **MODE_3 fin 분리 + Ecoflex density=1070** — wave-thrust 발생

---

## RL 학습 시 주의

- **fluidshape="ellipsoid" 한계 인지**. vortex shedding 못 모델 → sim2real 본질적 격차. 추후 개선은 Lighthill slender body theory 콜백(`mjcb_passive`) 또는 ANN surrogate (Zhong 2026 방식).
- **보상 함수의 forward 방향**: world −x. 학습 정책이 +x로 가면 보상 부호 또는 환경 인덱스 잘못된 것.
- **3DOF planar에서 yaw 누적**: fluid asymmetry 잔여로 한쪽으로 도는 경향. RL이 좌우 균형 학습으로 보정 가능. yaw 많이 누적되면 보상 하락 → 자연 보정.
- **Curriculum init_from의 부작용**: 이전 단계가 "직진" 편향이면 Stage 3 head arc에서 한쪽 mode만 학습 가능. 보상에 yaw 정렬 항 추가 검토.

---

## 백업·릴리스 패턴

학습 결과 binary는 GitHub Release로 버전 관리 (`models-v1`, `models-v2`, ...).

```bash
# 새 학습 후 v(N)→v(N+1)
mkdir -p /tmp/models-vN/runs /tmp/models-vN/plots
cp sim/runs/{tag1,tag2,...}/model.zip /tmp/models-vN/runs/<tag>/
cp sim/plots/*.png /tmp/models-vN/plots/
tar -czf /tmp/runs-models-vN.tar.gz -C /tmp/models-vN .

gh release create models-vN --target <branch> \
    --title "<제목>" --notes "<내용>" \
    /tmp/runs-models-vN.tar.gz
```

복원:
```bash
gh release download models-vN -p '*.tar.gz'
tar -xzf runs-models-vN.tar.gz -C sim/
```

---

## 다음 후보 (미해결)

### v10까지 시도된 카드 (요약)

| 카드 | 결과 |
|---|---|
| v2 — align 가산 reward (`+0.02·align`) | s3 ±90° 24% (변화 없음) |
| v3 — s3d만 ep 20s + align_weight 0.008 | 24% (변화 없음) |
| v4 — action history obs (N=8) + Stage 3 4단계 분화 | 평가 미실시 (v5로 이행) |
| v5 — align을 progress와 곱셈 (비대칭, prog>0만) | **20%로 악화**. avg_align −0.5 (머리 반대+후진) |
| v6 — v5 + ep 30s 통일 | 20% (변화 없음). 시간 부족 가설 reject. |
| v7 — 대칭 곱 + EntCoefFloorCallback floor 0.02 | **16%로 더 악화**. avg_align −0.73, final_dist 6m (도망 mode) |
| v8 — v4 가산식 복귀 + ent floor 0.005 | 17%, avg_align +0.58 (mode 정상화). s3c 46%로 +10%p |
| v9 — ent floor 0.005 → 0.002 | s3a 74→90% (가설 입증), s3c 46→29% / s3d 17→13% 후퇴 |
| **v10 — stage별 차등 floor (s3a 0.002 / s3b 0.003 / s3c 0.005 / s3d 0.006)** | **메커니즘 OK (ent_coef stage별 정확히 floor) / s3a 89% (89k 조기) ⭐ / 그러나 s3c 28%·s3d 13% — v8 회복 실패. ent_floor 카드 종결.** |

### 다음 후속 카드

v10 결과로 진단 확정: **ent_floor 카드 종결**. 균등(0.02/0.005/0.002)과 차등 4가지 다 시도했고 ±90° 천장(13~24%) 못 뚫음. 진짜 병목 = **정책 표현력**, 즉 시간적 비대칭 ctrl 패턴(yaw_test.py D2)을 obs/architecture가 지원 못 함.

1. **action history N 확대 (8 → 16 또는 24)** — v4 이후 미변경, **현재 가장 유망**. D2 패턴(75/25 비대칭 stroke) 표현엔 1 wag cycle 이상 필요 (3Hz × 17step = 0.5 cycle/N=8). N=16이면 1 cycle, N=24면 1.5 cycle. obs dim 19 → 27/35.
2. **Entropy schedule (cosine/linear decay)** — floor 대신 stage 시작 시 0.05 → 종료 시 floor로 감쇠. v10 진단(이전 stage가 narrow mode 만듦)에 직접 대응 — fine-tune 시작 시 entropy 부활로 mode 깨뜨림.
3. **HER (Hindsight Experience Replay)** — env Dict obs 큰 변경. 가장 큰 카드. v10 위에 얹기.
4. **align_weight ep에 비례** — 0.02 → 0.0067 (30s ep). stay-still 위험 정량적 감소. v8~v10에서 stay-still mode 안 보이니 우선순위 낮음.
5. **CrossQ (BatchNorm + target net 제거)** — SAC 변형, sample efficiency. SB3 native 미지원.
6. **Custom 보상 — yaw 변화 자체를 보상** — 회전 시도 자체에 인센티브. 다만 신호 noise 우려.
7. **fin actuator 추가** — 단일 모터 한계 자체를 풂. (사용자 명시 제외)
8. **ANN surrogate (Lighthill / Zhong 2026)** — fluid model 한계 우회. 실물 motion capture 필요.

### 단일 지느러미의 천장

`yaw_test.py` 측정으로 **단일 모터+passive fin의 yaw rate 물리 상한 ~4.6°/s** 확인. ±90° 회전은 20s 안에 가능. v1~v10 모두 13~24%로 천장 못 뚫음. v8에서 mode 정상화, v9에서 s3a 회복, v10에서 차등 floor로 작은 회전 빠른 수렴 — 그러나 큰 회전 천장은 ent_floor 카드(균등·차등 4종) 어느 것도 못 뚫음. 다음 카드는 **obs/architecture 확장**(action history N↑, HER) 방향이 핵심. entropy schedule도 이전 stage의 narrow mode 깨뜨리는 보조 카드.
