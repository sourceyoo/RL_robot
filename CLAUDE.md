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

- `fish_env.py` — Gymnasium 환경. obs 11D: tail/fin qpos·qvel + world v + sin/cos yaw + target rel.
- `train.py` — SAC 학습 + 별도 thread viewer. `PolicySnapshotCallback`으로 race-free, `CurriculumStopCallback`으로 reach_rate 임계 자동 조기 종료.
- `curriculum.py` — 3단계 자동 진행 스크립트.
- `view_policy.py` — 저장된 정책을 viewer로 rollout.
- 진단 스크립트: `freq_sweep.py`, `freq_sweep_locked.py`, `freq_sweep_norollpitch.py`, `waveform_test.py`, `multiseg_test.py`. 추진 방향·대칭성 검증용.

### 보상 함수 (`fish_env.py`)

```
reward = (prev_dist - cur_dist) * 10  +  5 if reached  -  0.001 * a²  +  align_w * cos(head, target)
```

- **progress** (delta dist × 10) — distance(absolute)로 주면 학습 느림
- **도달 보너스** 5, ctrl 비용 미미
- **align term** (v2~): `head_dir·target_dir`. yaw=0이면 head=(-1,0). 정조준 +1, 반대 -1.
- 가중치 균형: stay-still 누적(N×align_w)이 reach 누적(~10+k·align_w)보다 작아야 함.
  - 10s ep (N=500): `align_w ≤ 0.02`
  - 20s ep (N=1000): `align_w ≤ 0.01`
- 에피소드 = `episode_seconds / 0.02` step (default 10s = 500 step). 목표는 반지름 0.5m 원 위 랜덤.

### Curriculum 학습

Random target full circle은 단일 모터에 어려움. **4단계 자동 진행** (v2~):

| Stage | tag | theta 범위 | success_radius | episode_seconds | align_weight | max_steps |
|---|---|---|---|---|---|---|
| 1 | s1_forward | π fixed | 0.08 m | 10s | 0.02 | 400k |
| 2 | s2_anchor | π fixed | **0.04 m** | 10s | 0.02 | 400k |
| 3a | s3a_arc15 | π ± 15° | 0.08 m | 10s | 0.02 | 300k |
| 3b | s3b_arc90 | π ± 90° (=[π/2,3π/2]) | 0.08 m | **20s** (v3~) | **0.008** (v3~) | 500k |

자동 진행: 최근 100 ep `reach_rate ≥ 90%` → 학습 조기 종료 → 다음 단계로 fine-tune. max_steps는 안전장치. STAGES 딕셔너리는 stage별로 `episode_seconds`, `align_weight` 오버라이드 가능 (`curriculum.py`).

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

## 학습 결과 — v1, v2, v3

GitHub Release로 버전 백업: [`models-v1`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v1), [`models-v2`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v2). v3은 진행 중/예정.

### v1 → v2 → v3 변경점

| 변경 | v1 | v2 | v3 (진행 중) |
|---|---|---|---|
| align reward | 없음 | `align_weight=0.02` | s3b만 0.008 |
| Stage 3 분할 | 단일 ±90° | s3a (±15°) → s3b (±90°) | (v2 유지) |
| s3b episode 길이 | 10s | 10s | **20s** |
| ent_coef init | "auto" (=1.0 init) | `auto_0.1` | `auto_0.1` |
| TB metric | SB3 default | + `fish/*` (success_rate, final_distance, episode_seconds, avg_align) | (v2 유지) |

### Rollout 평가 (50 ep, deterministic)

| Run | v1 | v2 | 비고 |
|---|---|---|---|
| mode3-planar (random 500k baseline) | 6% / +0.95 | — | curriculum 가치 입증용 |
| s1_forward (직진) | 100% / +9.22 | **100% / +10.38** | ep_rew +1.2 = align 누적분 정확 |
| s2_anchor (radius 0.04) | 100% / +9.64 | **100% / +10.81** | 동일 |
| **±15° head arc (s3a, 신규)** | — | **84% / +7.99** | sub-staging 효과 입증 |
| **±90° head arc (s3b vs v1 s3)** | **24%** | **24%** | v2도 정복 못 함 → v3 (20s ep)로 시도 |

### 핵심 통찰

- ✅ **Stage 1·2** s1 30k + s2 30k = 60k step에 100%. align reward 누적이 ep_rew를 정확히 +1.2 (= 60 step × 0.02 × 평균 align) 올림 — 보상 일관성.
- ✅ **±15° head arc 84%** — s2→s3a fine-tune이 효율적.
- ❌ **±90° head arc 24% (v1 = v2)** — align reward로도 못 풂. 추정 원인: 10s ep로는 ±90° 회전(yaw rate ~5°/s 추정 × 18s) + 추진을 못 끝냄. v3에서 ep 길이 20s로 확장 시도.
- ❗ **ent_coef collapse** v2도 동일 (0.1 init → 0.0007). `auto_0.1`은 floor 아닌 초기값. 진짜 floor가 필요하면 custom callback.

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

### 진행 중 — v3 (Option A: ep 길이 확장)

s3b ±90°가 v1·v2 모두 24%에서 정체. 진단: 단일 모터 yaw rate 한계로 **10s 안에 ±90° 회전+추진 물리적 불가능**. 처방:

| 변경 | s3b만 | 이유 |
|---|---|---|
| `episode_seconds` | 10 → **20s** | 회전+추진 시간 확보 |
| `align_weight` | 0.02 → **0.008** | ep 두 배 → stay-still 위험 비례 증가 → 가중치 감소 |
| `max_steps` | 500k 유지 | 데이터 부족 아닌 환경 한계가 본질 |

### v3로도 안 풀리면 후속 카드

1. **HER (Hindsight Experience Replay)** — 실패 ep도 "그때 닿은 곳을 목표였다고" 라벨링해 성공 경험으로. SB3 `HerReplayBuffer` 지원. env interface 수정 필요 (Dict obs).
2. **fin actuator 추가** — 단일 모터 한계 자체를 풂. 모델·env 큰 변경.
3. **Custom EntCoefFloorCallback** — `log_ent_coef` 강제 floor. SB3 native 불가능 → callback 자작.
4. **ANN surrogate (Lighthill 콜백 또는 Zhong 2026 방식)** — fluid model 한계 우회. 실물 motion capture 필요.

### 단일 지느러미의 천장

학계 사례·진단 종합: **단일 모터로는 ~70~80%가 ±90° 천장**. 90% 도달이 목표면 fin actuator 추가가 필수. 사용자 결정으로 fin은 passive 유지 → v3까지가 가용 카드.
