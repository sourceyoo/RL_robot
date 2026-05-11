# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 목적

KUFIsh_III (사용자 자체 CAD) 기반 단일 관절 물고기 로봇의 강화학습. MuJoCo + SAC + Gymnasium. **수면 영법(BCF surface swimming) 가정**으로 단순화하여 학습.

---

## 빠른 시작

```bash
cd sim
python3 curriculum.py --no-viewer                       # 6단계 자동 학습 (헤드리스)
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

### 학습 시간 기준 (RTX 3090, fps ~305)

| 작업 | 시간 |
|---|---|
| s1_forward (30k 조기 종료) | ~100초 |
| s3a~c stage 1개 (200~350k) | 11~19분 |
| s3d 500k | ~28분 |
| **s3d 1M (v22/v23)** | **~55분** |
| 전체 curriculum (s1~s3d, 종합) | 2~3시간 |

학습이 빠른 이유 (v23 측정):
- **환경 step 0.064 ms/step (15.6k fps)** — qpos 5·action 1·MuJoCo planar 3DOF + ellipsoid fluid라 매우 가벼움. 전체 시간의 1.4%.
- **SAC update + Python overhead 98%** — [256,256] MLP는 RTX 3090에서 under-utilize, GPU/CPU/CUDA launch overhead가 bottleneck.
- 1M step ≈ 1.8M SAC update (n_updates) → 시간 추정 시 step이 아닌 update 수 기준이 정확.

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

| f[Hz] | x_disp | 방향 |
|---|---|---|
| 0.5 | +0.85 | 후진 |
| 1.0 | −0.49 | 전진 ✓ |
| 3.0 | −2.17 | 전진 ✓ |
| 6.0 | −3.36 | 전진 ✓ (0.28 m/s) |

→ 1~6 Hz 모든 주파수에서 +x 머리 방향 전진. 0.5 Hz 후진은 저주파 wave-thrust 약함 (RL이 자연 회피).

---

## CAD 소스 — `fish_urdf/RL_SIM_MODE_3_description/`

활성 mesh의 출처. URDF는 *형상만* 사용 (mesh 파일들), inertial 값은 위 표대로 별도 검증.

- mesh: `base_link.stl`, `tail_link_1_1.stl`, `fin_1.stl`
- URDF는 fin을 `<joint type="fixed">`로 부착하지만 우리 MJCF는 **passive hinge로 교체** (실리콘 변형 모델링).
- 이전 버전 `RL_SIM_MODE_2_description/`도 남아있지만 **사용 안 함** — fin 분리 안 된 구버전.

---

## Python 학습 인프라 — `sim/`

- `fish_env.py` — Gymnasium 환경. obs **(11 + action_history_n)D**: tail/fin qpos·qvel + world v + sin/cos yaw + target rel + 최근 N step ctrl 이력.
- `train.py` — SAC 학습 + 별도 thread viewer. `PolicySnapshotCallback`으로 race-free, `CurriculumStopCallback`으로 reach_rate 임계 자동 조기 종료.
- `curriculum.py` — 6단계 자동 진행 스크립트.
- `view_policy.py` — 저장된 정책을 viewer로 rollout.
- 진단 스크립트: `freq_sweep.py` (추진 방향), `yaw_test.py` (회전 능력 — v4 핵심 진단).

### Action history obs (v4~)

`action_history_n` 인자. obs 끝에 **최근 N step의 ctrl 값**이 stack. v4에서 N=8, **v11에서 N=16** (3Hz × 17 step/cycle ≈ 1 wag cycle 커버).

목적: 정책이 *시간적 비대칭 ctrl 패턴* (D2 패턴 — 75% 한쪽 stroke + 25% 반대 stroke 등) 발견. `yaw_test.py`로 비대칭이 yaw 회전의 핵심임 확인 (대칭 sine 0.49°/s vs D2 패턴 4.6°/s).

### 보상 함수 (현재 코드, v24-A)

```python
reward = progress·10 + (10 if reached else 0) - ctrl_cost + ALIGN_W·align + YAW_W·yaw_rate·sign(yaw_dir)
# ALIGN_W = 0.02 (10s ep) / 0.012 (30s ep) — v14
# YAW_W = 0.005 — v22 best 균형값
# v24-A signed yaw: |yaw_rate|·sign(d(align)/d(yaw)) — 옳은 방향 +, 틀린 −
#   d(align)/d(yaw) = head_perp · target_dir, head_perp = (sin(yaw), cos(yaw))
# v23-A 700k actor_loss spike 진단(좌우 흔들기) 직접 대응. **진동 해결 ✓
# but 천장 돌파 못 함** (end 28%, v22 32% −4%p). final_dist 0.599m 모든 v best.
```

- **progress**: delta dist × 10. align ∈ [-1, +1] (cos(머리, 목표)).
- **yaw_rate 보상 (v21)**: 회전 시도 자체에 +보상. v20 "정렬만, 추진 안 함" mode 깨고 회전·추진 시퀀스 학습 강제. **v11 26% 천장 단독 돌파의 핵심**.
- 곱셈 보상은 mode collapse (v5/v7) — 가산식이 정착 (함정 #8).

### Curriculum 학습

Random target full circle은 단일 모터에 어려움. **6단계 자동 진행**:

| Stage | tag | theta 범위 | success_radius | episode_seconds | max_steps |
|---|---|---|---|---|---|
| 1 | s1_forward | π fixed | 0.08 m | 10s | 400k |
| 2 | s2_anchor | π fixed | 0.04 m | 10s | 400k |
| 3a | s3a_arc15 | π ± 15° | 0.08 m | **30s** | 200k |
| 3b | s3b_arc30 | π ± 30° | 0.08 m | 30s | 250k |
| 3c | s3c_arc60 | π ± 60° | 0.08 m | 30s | 350k |
| 3d | s3d_arc90 | π ± 90° | 0.08 m | 30s | 500k |

자동 진행: 최근 100 ep `reach_rate ≥ 90%` → 학습 조기 종료 → 다음 단계로 fine-tune. max_steps는 안전장치.

**Full circle (θ ∈ [-π, π])은 사용자 결정으로 제외** — 단일 모터로 180° 회전 후 추적은 비현실적.

```bash
python3 curriculum.py                    # 전체 단계 자동 + viewer
python3 curriculum.py --no-viewer        # viewer 없이 빠르게
python3 curriculum.py --threshold 0.85   # 임계 완화
python3 curriculum.py --start-stage 2    # Stage 2부터 (이전 model.zip 있어야)

# 단일 단계 수동
python3 train.py --tag s1_forward --theta-min 3.14159 --theta-max 3.14159 \
    --success-radius 0.08 --success-threshold 0.9 --steps 200000
```

---

## 학습 결과 — v1 ~ v25

GitHub Release: [`models-v1`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v1), [`models-v2`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v2), [`models-v5`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v5) (실패), [`models-v10`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v10), [`models-v12`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v12), [`models-v13`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v13), [`models-v17`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v17) (N=24, 정점 30% 천장 일시 돌파), [`models-v21`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v21) (yaw reward 천장 단독 돌파, s3d 29%/peak 38%) ⭐, [`models-v22`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v22) (**v21 + 1M fine-tune, s3d 32%/peak 50%, 천장 +6%p**) ⭐⭐. **v11 binary 영구 손실** (v12 학습이 덮어씀). v16·v19·v20은 `/tmp/models-v{16,19,20}-backup/`, v23·v24·v25는 `/tmp/models-v{23,24,25}/` 로컬 백업 (v25는 `/tmp/runs-models-v25.tar.gz`도). v18은 v17 s3d만 fine-tune (1M step) — `sim/runs/s3d_arc90/`이 덮어써진 상태, 별도 백업 없음. v3·v4·v6~v9·v11·v14~v16·v18~v20·v23~v25 release 없음.

### 결별 그룹화

| 그룹 | 멤버 | 특징 | s3d 천장 |
|---|---|---|---|
| **A. Baseline + 가산식 reward** | v1·v2·v3 | `+0.02·align` 가산항 시도 | 24% |
| **B. 곱셈 보상 실패** | v5·v6·v7 | mode collapse (머리 반대 / 도망) | 16~20% |
| **C. v4 가산식 복귀 + ent_floor** | v8·v9·v10 | 균등(0.005/0.002) + 차등. ent_floor 카드 종결 | 13~17% |
| **D. 정책 표현력 확장** | **v11** | action history N=8→16. **±90° 천장 첫 돌파** | **26%** ⭐ |
| **E. NN 확장** | v12 | net_arch [256,256,128]. **s3d catastrophic** | 6% ❌ |
| **F. align reward ep 정규화** | v13 | `align_weight = 0.02·10/ep_seconds` (30s → 0.0067) | 19% |
| **G. align reward 30s 직접 fix** | v14 | `align_weight = 0.012` 30s | 20% |
| **H. reach 보너스 강화** | v15 | reach 5→10 (s3c 회복 입증) | 21% |
| **I. NN default 회귀** | v16 | net_arch [256,256] 회귀 (작은 회전 회복) | 19% |
| **J. action history N 확장** | **v17** | N=16→24 (1.5 wag cycle). **s3c·d 동시 회복**, 정점 30% (천장 일시 돌파) | **25%** |
| **K. 학습량 ↑** | **v18** | v17 s3d만 max_steps 500k → 1M (fine-tune). **end 25% 동일** (학습량 부족 가설 reject), 정렬만 회복 (+0.43→+0.57) | **25%** |
| **L. N 절충 (N=20)** | **v19** | N=24 → 20 (1.25 wag cycle). s3a 96% (170k 조기, but seed 분산) + s3c·d mode collapse (avg_align −0.17/−0.37). N=20 작은 회전 가속이 큰 회전 mode 안정성 희생 | **23%** ❌ |
| **M. v19 + ent_floor ↑** | **v20** | s3c·d ent_floor 0.005·0.006 → **0.008·0.010**. **mode collapse 완전 해결 ✓** (align s3d −0.37 → **+0.84 모든 버전 best** ⭐). but reach 후퇴 (s3a 63%, s3d 19%) — v10 "정렬 best, 추진 worst" 재현 (final_dist 0.78m, 시작 0.5m보다 멀어짐) | **19%** |
| **N. yaw reward** ⭐⭐ | **v21** | `+ YAW_W·\|yaw_rate\|` (YAW_W=0.005) 추가. v20 "정렬만, 추진 안 함" mode 직접 대응. **v11 26% 천장 단독 돌파**. 모든 stage 동시 회복 (s3a 90% 159k 조기·s3b 69% best·s3c 38%·s3d 29%/정점 38%). ep 21.8s (모든 v best). final_dist 0.78→0.67m (추진 회복) | **29%** ⭐⭐ |
| **O. v21 + 학습량 ↑** ⭐⭐ | **v22** | s3d max_steps 500k → 1M (v21 s3c model에서 fine-tune). **천장 +6%p 추가 돌파**. v18(N=24+1M)은 reject였는데 **yaw reward 있으면 학습량 ↑가 효과**. peak 50% (6.4k init), end 32%, align +0.73 (v21 +0.51 회복), ep 21.2s (모든 v best). 마지막 10% 26~30% 안정 진동. | **32%** ⭐⭐⭐ |
| **P. yaw reward 강화** ❌ | **v23-A** | YAW_W 0.005 → **0.007** (1.4배). v22 카드(N=20·ent_floor 차등·1M·s3c init) 그대로. **정점 41% (window 100, 모든 v best)** 625k~650k 도달하나 700k actor_loss −2.5 + ent_coef 0.018 spike instability event → 정책 진동 → **end 23%로 catastrophic 후퇴**. final_dist 0.764m (v20 패턴 부분 재현). **trade-off: 정점 +3%p / 평균 −9%p — reject**. | **23%** ❌ (peak 41%) |
| **Q. signed yaw (방향 가산)** ◯ | **v24-A** | `+YAW_W·yaw_rate·sign(d(align)/d(yaw))`. 옳은 방향 +, 틀린 −. v23-A 진동 진단(\|yaw_rate\| 좌우 흔들기) 직접 대응. **진동 해결 ✓** (700k actor_loss spike 사라짐, 마지막 10% 26~28% 안정). final_dist **0.599m 모든 v best ⭐** (옳은 방향 추진 시퀀스 학습). but **천장 돌파 못 함** (end 28%, peak 35% — v22 32%/peak ~38% 미달). avg_align +0.615 (v22 +0.73 −0.115) — signed reward가 회전 강도 ↓ → mode 전환 catalysis 약함. **v22 \|yaw_rate\|의 "노이즈 보상"이 사실 학습 dynamic에 도움이 됐다는 역설 입증**. | **28%** ◯ (peak 35%) |
| **R. signed yaw + 강한 weight** ❌ | **v25-A** | v24-A signed × v23-A YAW_W 0.005 → **0.007** 결합. 가설: signed면 좌우 흔들기 페널티 작동 → weight ↑해도 진동 안 일어남. ✅ **spike 없음 진단 부분 입증** (actor_loss V자×3, v23-A −2.5 폭락 없음). but ❌ **W자 점진 후퇴**: peak **33% (520~535k)** 도달 후 catastrophic drift — 685k 18%, 970k **12% (모든 v 후반 최저)**, end **16%**. avg_align +0.558, final_dist 0.775m, ep 25.5s (모든 v worst). v23-A는 700k 폭락이라면 v25-A는 470k 동안 점진 붕괴. **결론**: signed × 강한 weight는 spike는 막아도 학습 dynamic 자체를 천천히 망침. **v22 0.005가 sweet spot 재확인**. | **16%** ❌ (peak 33%) |

### 버전별 변경점 (주요만)

| 변경 | v1 | v4 | v6 | v7 | v8~v10 | v11 | v12 | v13 | v14 | v15 | v16 | v17 | v19 | v20 | v21 | v22 | v23-A | v24-A | **v25-A** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| align reward | 없음 | +0.02 가산 | (v4) | (v4) | (v4) | (v4) | (v11) | **0.02·10/ep_sec** | **0.012 fix** | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) | (v14) |
| reach bonus | 5 | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | **10** | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) | (v15) |
| **yaw reward** | 없음 | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | **+\|yaw_rate\|·0.005** | (v21) | **·0.007** ❌ | **+yaw_rate·sign·0.005** ◯ | **·sign·0.007** ❌ |
| Stage 3 분할 | 단일 ±90° | **s3a/b/c/d** | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) | (v4) |
| s3 ep 길이 | 10s | 10s | **30s 통일** | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) | (v6) |
| action history | 0 | **8 (19D)** | (v4) | (v4) | (v4) | **16 (27D)** | (v11) | (v11) | (v11) | (v11) | (v11) | **24 (35D)** | **20 (31D)** | (v19) | (v19) | (v19) | (v19) | (v19) | (v19) |
| EntCoefFloor | 없음 | 없음 | 없음 | 균등 0.02 | v8 0.005 / v9 0.002 / v10 차등 | (v10) | (v10) | (v10) | (v10) | (v10) | (v10) | (v10) | (v10) | **s3c 0.008 / s3d 0.010** | (v20) | (v20) | (v20) | (v20) | (v20) |
| s3d max_steps | 500k | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | (v1) | **1M** | (v22) | (v22) | (v22) |
| net_arch | [256,256] | (v1) | (v1) | (v1) | (v1) | (v1) | **[256,256,128]** | (v12) | (v12) | (v12) | **[256,256] 회귀** | (v16) | (v16) | (v16) | (v16) | (v16) | (v16) | (v16) | (v16) |

### Stage 진행 비교

| Stage | v8 | v10 | v11 | v12 | v15 | v16 | v17 | v19 | v20 | v21 | v22 | v23-A | v24-A | **v25-A** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3a_arc15 (±15°) | 74% | 89% (89k 조기) | 67% | 84% | 69% | 80% | 66% | 96% (분산) | 63% | **90% (159k 조기)** ⭐ | (s3a~c는 v21) | — | — | — |
| s3b_arc30 (±30°) | 70% | 72% | 44% | 67% | 62% | 70% | 62% | 61% | 53% | **69%** ⭐ | — | — | — | — |
| s3c_arc60 (±60°) | **46%** ⭐ | 28% | 38% | 31% | 27% | 21% | 33% | 28% | 21% | **38%** | — | — | — | — |
| s3d_arc90 (±90°) | 17% | 13% | 26% | 6% ❌ | 21% | 19% | 25% | 23% | 19% | 29% | **32%** ⭐⭐⭐ | 23% ❌ (peak 41%) | 28% ◯ (peak 35%) | **16%** ❌ (peak 33%) |

### s3d (±90°) 결과 — 마지막 100 ep 윈도우

| 버전 | reach | avg_align | final_dist | 한 줄 평가 |
|---|---|---|---|---|
| v1~v3 | 24% | (미기록) | — | baseline (+align 가산은 변화 X) |
| v5/v6/v7 | 20/20/16% | −0.5/−0.1/−0.73 | 0.32/0.31/**6.27**m | 곱셈 보상 mode collapse (머리 반대 / 도망) |
| v8 | 17% | +0.58 | 0.57m | mode 정상화 첫 성공 (가산식 복귀 + ent floor) |
| v9·v10 | 13/13% | +0.52/+0.61 | 0.42/0.65m | ent_floor 차등 — 큰 회전 entropy 부족 / mode 못 깸 |
| **v11** | **26%** ⭐ | +0.58 | 0.57m | **N=16 — 천장 첫 돌파**. ep_seconds 22.7s |
| v12 | 6% ❌ | +0.60 | 0.70m | NN [256,256,128] catastrophic (50→6%) |
| v13 | 19% | +0.19 | 0.53m | align ep 비례 — catastrophic 해결, 정렬 부족 |
| v14 | 20% | +0.50 | 0.58m | align 0.012 fix — 정렬 회복, 천장 미돌파 |
| v15 | 21% | +0.53 | 0.62m | reach 5→10 — s3c 회복 ✓, V자×2 변동 |
| v16 | 19% | +0.53 | 0.56m | NN 회귀 — 작은 회전 회복 ✓, 천장 미돌파 |
| **v17** | **25%** | +0.43 | 0.55m | **N=24 — s3c·d 동시 회복, 정점 30%(천장 일시 돌파), end 25%로 v11 −1%p 미달**. ep 23.1s (효율). 작은 회전 후퇴 (s3a 66%·s3b 62%) — N-stage trade-off 재현. |
| **v18** | **25%** | **+0.57** | 0.57m | **v17 s3d 1M fine-tune — 학습량 부족 가설 reject**. 정점 30% 3회 도달(22.9k·312k·944k)하나 100ep window 안정화 실패. 정렬은 v11 동급 회복 (+0.43→+0.57). ep_rew_mean 6.6→8.2 ↑. **N=24 카드의 본질적 진동 입증**. |
| **v19** | **23%** | **−0.37** ❌ | 0.35m | **N=20 — s3a 96% best ⭐ but s3c·d mode collapse**. align +0.58(s3a)→+0.10(s3b)→−0.17(s3c)→**−0.37(s3d)** 단조 감소. ep_rew −2.5 (음수). 정책이 "후진하면서 일부 도달" mode 학습 (final_dist 0.35m < 시작 0.5m). N=20 작은 회전 가속이 narrow mode 강화 → 큰 회전 collapse trigger. |
| **v20** | **19%** | **+0.84** ⭐ | 0.78m | **v19 + ent_floor s3c·d 강화 (0.005·0.006 → 0.008·0.010). mode collapse 완전 해결 ✓** (align −0.37 → +0.84, 모든 버전 best). but reach 후퇴 (s3a 96→63%, s3d 23→19%) — v10 패턴(정렬 best, 추진 worst) 더 심하게 재현. final_dist 0.78m (시작 0.5m보다 멀어짐). entropy ↑가 정렬에만 활용되고 추진 학습 못 함. |
| **v21** ⭐⭐ | **29%** ⭐ | +0.51 | **0.67m** | **v20 + yaw reward (`+YAW_W·\|yaw_rate\|`, YAW_W=0.005). v11 26% 천장 단독 돌파**. 정점 38% (모든 버전 best, v17 30% 능가). align s3d 단조 우상향(−0.13→+0.58→+0.51 end). final_dist 0.78→0.67m (추진 회복). ep 21.8s (모든 v best). 모든 stage 동시 회복 (s3a 90% 159k 조기·s3b 69% best·s3c 38%). |
| **v22** ⭐⭐⭐ | **32%** ⭐⭐ | **+0.73** | **0.64m** | **v21 s3d 1M fine-tune. 천장 +6%p 추가 돌파 (v11 26% 대비)**. peak **50%** (6.4k init, 모든 v best, v21 38% 능가). 마지막 10% 26~30% 안정 진동. ep 21.2s (모든 v best). **v18(N=24+1M) reject ≠ v22(yaw+1M) 성공** — yaw reward가 학습량 ↑의 lock 해제. 학습 곡선: 25→18→13(저점)→16→17→**저점 후 우상향 → 30~32 (end)**. |
| **v23-A** ❌ | **23%** | +0.713 | 0.764m | **v22 + YAW_W 0.005 → 0.007 (1.4배)**. v22 카드 그대로(N=20·ent_floor 차등·1M·s3c init). **정점 41% (window 100, 모든 v best)** 625k~650k 도달. 그러나 700k에서 actor_loss 0.7 → **−2.5 폭락** + ent_coef 0.010 → **0.018 spike** (instability event) → reach 41→34% → **940k부터 21~23% 마지막 후퇴** → end 23% (v22 32% 대비 **−9%p**). final_dist 0.764m (v20 패턴 부분 재현). 학습 곡선 V자×3: 17→13(저점)→33→27→23(저점)→**41(정점)**→34→24(저점)→31→**23(end)**. **trade-off: 정점 +3%p / 평균 −9%p — reject**. v22 best 카드(YAW_W=0.005) 회귀 권장. |
| **v24-A** ◯ | **28%** | +0.615 | **0.599m** ⭐ | **v22 + signed yaw**: `+YAW_W·yaw_rate·sign(d(align)/d(yaw))` (옳은 방향 +, 틀린 −). YAW_W=0.005 그대로, 다른 카드 v22 그대로. **v23-A 진동 진단(\|·\| 좌우 흔들기) 직접 검증**: ✓ **actor_loss spike 사라짐** (700k 부근 안정), ✓ **마지막 10% 26~28% 안정** (v23-A 21~23% 후퇴 회복), ✓ **final_dist 0.599m 모든 v best** (옳은 방향 추진 시퀀스 학습 ⭐). but **천장 돌파 못 함** (end 28%, peak 35% — v22 32%/peak ~38% 미달, −4%p). avg_align +0.615 (v22 +0.73 −0.115). 학습 곡선 V자×2: 17→24→18(저점)→**35(정점 405k)**→22(저점 575k)→16(저점 675k)→28(end). **역설**: v22 \|yaw_rate\|의 "좌우 모두 보상" 노이즈가 사실 회전 빈도 ↑ → mode 전환 catalysis ↑ → 천장 돌파 유리. signed로 진동은 막았으나 학습 dynamic 보수화. |
| **v25-A** ❌ | **16%** | +0.558 | 0.775m | **v24-A signed × v23-A YAW_W 0.007 결합**. 다른 카드 v22 그대로 (N=20·ent_floor 차등·1M·s3c init). 가설: signed면 좌우 흔들기 페널티 작동 → weight ↑해도 진동 안 일어남. ✅ **spike 없음 (진단 부분 입증)** — actor_loss V자×3 있으나 v23-A −2.5 폭락 없음. ❌ **W자 점진 후퇴**: 23→25.6(90k 초기 peak)→13(225k 저점#1)→17→21(310k)→30(490k)→**33% (520~535k peak)** ⭐→28(580k)→20(690k 저점#2)→27(755~820k 회복)→18(890k)→**12% (970k 모든 v 후반 최저 ❌)**→16(end). avg_align +0.558 (v24-A +0.615 −0.057, v22 +0.73 −0.17), ep 25.5s (v22 21.2 +4.3s, 모든 v worst). **v23-A 700k 폭락이 470k 동안 점진 붕괴로 변형**. signed × 강한 weight: spike는 막아도 dynamic 천천히 망침. **v22 0.005가 sweet spot 재확인**. |

---

### 주요 진단 (요약)

**v8 (mode 정상화)**: v5~v7 곱셈 보상 mode collapse → 가산식 복귀 + ent floor 0.005. avg_align 처음 양수(+0.58). s3c 46% (이전 best). 그러나 s3d 17% — fish가 회전+추진 시퀀스 못 만듦 (정책 표현력 부족 가설).

**v9 (ent_floor stage 의존성)**: floor 0.005 → 0.002. s3a 74→90% (가설 입증) but s3c 46→29%, s3d 17→13% 후퇴. **단일 floor로 전체 cover 불가**.

**v10 (차등 floor 한계)**: stage별 차등 (s1~s3a 0.002 / s3b 0.003 / s3c 0.005 / s3d 0.006). 메커니즘 OK (ent_coef 정확히 floor에 머무름) but s3c 28%·s3d 13%. **이전 stage의 낮은 entropy가 만든 narrow mode를 다음 stage 높은 floor가 못 깸** — entropy 양 ≠ 정책 표현력. **ent_floor 카드 종결**.

**v11 (천장 첫 돌파) ⭐**: action history N=16으로 1 wag cycle 커버 (obs 19→27D). **±90° 26% — v1~v10 24% 천장 +9%p 돌파**. ep_seconds 22.7s (효율). 단 작은 회전 후퇴 (s3a 67%·s3b 44%) — N과 stage 복잡도 매칭 trade-off. **진짜 병목이 정책 표현력(시간적 패턴)이었음 입증**.

**v12 (NN 확장 catastrophic ❌)**: net_arch [256,256] → [256,256,128]. 작은 회전 회복(s3a 84%·s3b 67%) but **s3d 26→6% catastrophic interference** — 학습 시작 50% (s3c 정책) → 끝 6%. v10 패턴(정렬 best, 추진 worst) 재현. 진단: align 누적 +18 vs reach +5 = **3.6:1로 align dominance** → "정렬만" mode. **차등 N/NN은 SAC.load 호환으로 막힘** (함정 #9).

**v13 (catastrophic 해결, 정렬 부족)**: `align_weight = 0.02·10/ep_seconds` (30s → 0.0067). **catastrophic 완전 해결** (학습 우상향, s3d 6→19%) but align +0.19로 정렬 신호 부족. 비율 1.2:1 — 적정값 0.012 (비율 2.2:1) 추정.

**v14 (정렬 회복, 천장 미돌파)**: 30s ep 직접 fix `0.012`. 정렬 회복 (avg_align +0.19→+0.50, v11 +0.58 근접). catastrophic 안 일어남 (15→20% 우상향). 단 s3d 20% / **s3c 모든 버전 중 worst (20%, v8 best 46% 대비 −26%p)**. 실측 비율 7.4:1로 align dominance 여전.

**v15 (s3c 회복 ✓, 천장 여전)**: reach 보너스 5→10. **s3c 20→27% 회복 (가설 검증 ✓)** + s3b 59→62%. 단 s3d 21% — v11 26% 천장 미돌파. **s3d 학습 곡선 V자×2 변동** (50→12→21%, v12 직전 패턴). 비율 3.6:1(=v12) but reach가 sparse라 catastrophic 안 일어남 — **신호 형태(dense vs sparse) > 비율**.

**v16 (NN default 회귀 — 작은 회전 회복 ✓, 천장 미돌파)**: net_arch [256,256] 회귀. **가설 입증**: 작은 회전 회복 (s3a 69→80%·s3b 62→70%, v15 대비 +11/+8%p) — NN [256,256,128]이 narrow mode 학습 가속한 부작용 제거. 단 s3c 27→21% (-6%p), s3d 21→19% (-2%p) — NN 회귀가 큰 회전엔 표현력 부족. **s3d 학습 곡선 V자×2 사라짐** (50→27→28→19% 안정 후 점진 후퇴) — V자×2 변동은 NN 확장 + reach=10 조합 부작용이었음. v12 best (s3a 84%·s3b 67%) 못 따라감 — NN 확장이 그만큼 강력한 효과였음을 역으로 입증. **v11 26% 천장은 NN 카드도 못 깸**: reward 4종 + NN 1종 = 5장 모두 미돌파.

**v17 (N=24 — 큰 회전 회복 ✓, 천장 거의 근접) ⭐**: action history N=16 → 24 (1.5 wag cycle 커버, obs 27→35D). NN default + reward 카드 모두 v16 그대로. 가설(시간적 표현력 추가 확장)에 대응. **결과**:

- ✅ **s3c·s3d 동시 회복**: s3c 21→33% (+12%p, v8 46% 빼면 best), s3d 19→25% (+6%p, v11 26%에 -1%p 근접).
- ✅ **s3d 정점 30%로 v11 26% 천장 일시 돌파** (270~310k 구간) — **동일 카드 조합에서 N 축만으로 천장 가능성 입증**. 단 후반 25%로 안정 (학습량 부족? max_steps 500k).
- ✅ **ep_seconds 23.1s — v11 22.7s 다음 효율** (N=24 표현력이 더 빠른 도달 정책).
- ✅ **s3d 학습 곡선 단조 우상향** (0→14→27→30→25, V자 없음).
- ❌ 작은 회전 후퇴: s3a 80→66% (-14%p), s3b 70→62% (-8%p) — v11 패턴 재현 (N과 stage 복잡도 매칭 trade-off).
- ⚠ avg_align +0.43 (v16 +0.53 대비 -0.10) — 정렬 신호 양보, 큰 회전 reach 우선 정책 분배.

→ **N 축이 천장 카드의 가장 인과 명확한 축임 재입증**. v11이 N=8→16으로 24%→26% 돌파했고, v17이 N=16→24로 19%→25% (정점 30% 일시 돌파). 다만 학습 안정성·작은 회전 trade-off는 여전. 다음은 **max_steps ↑ 또는 N=20 절충**.

**v18 (v17-B: s3d 1M fine-tune — 학습량 부족 가설 reject)**: v17 s3c model에서 init, s3d만 max_steps 500k → 1M. 다른 변수 모두 v17과 동일.

- ❌ **학습량 가설 reject**: end 25% (v17 동일). 정점 30% 3회 도달(22.9k·312k·944k)하나 100ep window 안정화 실패.
- ✅ **정렬 회복**: avg_align +0.43 → +0.57 (v11 +0.58 동급). v17의 "정렬 양보, 큰 회전 우선" mode가 학습량 ↑로 양쪽 강화.
- ✅ ep_rew_mean 6.6 → 8.2 — 정렬 누적 reward 증가.
- ❌ 22.9k에 정점 30% 도달 (v17 312k와 매우 다른 시점) — early peak 후 진동, **학습량과 무관한 본질적 진동**.

→ **단일 모터 + N=24 + 현 reward 카드의 이론적 천장 ≈ 25~30%, 평균 25%**. v11 26% 천장이 카드 변경 없이 깨질 가능성 낮음. 진단: 정책이 mode 전환(정렬↔추진)을 못 하고 25% 부근에서 진동 — `yaw_test.py` 4.6°/s × 20s = 92°라 마진 작아 반복 안정 어려움. 다음은 **N=20 절충(v17-C) 또는 ent_floor 강화(v17-D) 또는 더 근본적 카드(HER, custom yaw reward)**.

**v19 (v17-C: N=20 절충 — s3a 96% best ⭐ + s3c·d mode collapse ❌)**: N=24 → 20 (1.25 wag cycle, obs 35→31D). N=16(v11 s3a 67%)·N=24(v17 s3a 66%) 사이 절충 가설.

- ⭐ **s3a 96% (170k 조기 종료)** — v10 89% 동급 best 능가, **모든 버전 중 best**. N=20이 작은 회전 학습엔 강력.
- ❌ **s3c·d mode collapse**: avg_align s3a +0.58 → s3b +0.10 → s3c **−0.17** → s3d **−0.37** 단조 감소. ep_rew_mean −2.5 (음수).
- final_dist 0.35m (< 시작 0.5m) — 도달은 일부 하지만 **머리 반대 방향**. 정책이 "거꾸로 가면서 가까워지면 prog 양수" mode 발견 → align 음수 + prog 양수 + reach 일부에 갇힘.
- v17(N=24) align +0.43과 비교 — 같은 카드에서 N 4 차이로 **mode 양→음 반전**. **N=20이 작은 회전 학습 가속하면서 narrow mode 잔재가 큰 회전 entropy floor(0.005/0.006)로도 못 깨짐**.

→ **N=20은 단독 카드로 부적합** — s3a 96% best는 의미 크지만 큰 회전에서 v5/v7 함정 #8 패턴 재현 (mode collapse). 다음은 **v19-D (s3c/d ent_floor ↑로 mode collapse 막기)** 또는 **N=20 + reward 변경 (align dominance 약화로 collapse 막기)**, 또는 **v11(N=16)로 회귀**.

**v20 (v19 + ent_floor s3c·d 강화 — mode collapse 해결 ✓ + reach 후퇴 ❌)**: s3c 0.005 → 0.008, s3d 0.006 → 0.010. 다른 변수 v19 그대로.

- ✅ **mode collapse 완전 해결**: avg_align s3d −0.37 → **+0.84** (모든 버전 best, v10 +0.61 능가). 가설(narrow mode 잔재 + entropy 부족) 정확히 입증.
- ✅ ent_coef 정확히 floor에 머무름 (0.008/0.010).
- ❌ **reach 큰 후퇴**: s3a 96→63% (-33%p, but v19 96%는 seed 분산이라 의미 약함), s3c 28→21%, s3d 23→19%.
- ❌ **final_dist 0.78m** (시작 0.5m보다 멀어짐) — v10 "정렬 best, 추진 worst" 패턴 더 심하게 재현. ep_rew_mean 3.4 (v17 6.6의 절반).
- 진단: ent_floor 0.010이 entropy 강제하나 SAC actor가 그 entropy를 *추진* 학습엔 못 쓰고 *정렬* 정확도에만 사용. **이전 stage(s3a 0.002 narrow mode)가 만든 정책 분포가 entropy↑로도 못 깨짐 — v10 함정 재발견**.

→ **ent_floor 카드 한계 입증**: collapse는 막지만 reach 천장 못 깸. **N·entropy·reward 단일 축 카드 모두 한계 도달**. 다음은 근본 카드 (HER / Custom yaw reward) 또는 N=16(v11) 안전 회귀.

**v21 (yaw reward — v11 26% 천장 단독 돌파) ⭐⭐**: `+YAW_W·|yaw_rate|` (YAW_W=0.005) 추가. v20 진단 직접 대응 — 회전 시도 자체에 인센티브로 정책의 "정렬만, 추진 안 함" mode 깨고 회전·추진 시퀀스 학습 강제.

- ⭐⭐ **v11 26% 천장 단독 돌파**: s3d end **29%** (+3%p), 정점 **38%** (모든 버전 best, v17 30% 능가).
- ⭐ **모든 stage 동시 회복**: s3a 90% (159k 조기, v10 동급) · s3b 69% (모든 v best, v12 67% 능가) · s3c 38% (v8 46% 다음) · s3d 29%.
- ⭐ **ep_seconds 21.8s** — 모든 v best (v11 22.7s 능가). 정책이 더 빨리 도달.
- ✅ **추진 회복**: final_dist 0.78m (v20) → 0.67m. avg_align +0.51 (양수 안정, collapse 회피).
- ✅ **align 단조 우상향**: s3d −0.13 (init) → 0.27 → 0.38 → **0.52 → 0.58 → 0.57 → 0.51 (end)** — 학습할수록 정렬 강화.

→ **천장 돌파의 본질 = "회전 시도 자체에 보상"**. 진단 정확:
- N (시간적 표현력, v11/v17/v19) — 정점 30% 가능
- entropy (mode 전환, v8~v10/v20) — collapse 해결
- **yaw reward (회전·추진 시퀀스, v21)** — **천장 돌파** ⭐

s3b/c init 음수 align (-0.98, -0.40)에서 학습할수록 양수로 회복 — yaw reward 받으려 회전만 하다가 align reward가 옳은 방향 강제. 두 신호의 균형이 핵심. 다음은 v21 안정화 (학습량 ↑) 또는 yaw reward 가중치 튜닝.

**v22 (v21 s3d 1M fine-tune — 천장 +6%p 추가 돌파) ⭐⭐⭐**: max_steps 500k → 1M, 다른 변수 v21 그대로.

- ⭐⭐⭐ **천장 +6%p 추가 돌파**: end 29% → **32%** (v11 26% 대비 +6%p).
- ⭐⭐ **peak 50%** (6.4k init step, 모든 v best, v21 38% 능가) — N=20 + yaw reward + ent_floor ↑ 조합이 학습 초기 매우 강력.
- ⭐ **align +0.51 → +0.73** 회복 (v20 +0.84 다음, but v22는 추진도 됨).
- ⭐ **ep_seconds 21.2s** — 모든 v best 갱신 (v21 21.8s, v11 22.7s 능가).
- ✅ **마지막 10% 26~30% 안정**: 900k 28% / 929k 29% / 955k 28% / 994k 30% / end 32%. v21 정점 후 단기 후퇴 패턴 사라짐.

**v18 vs v22 — 핵심 인사이트**:
- v18 (N=24, no yaw, 1M): 50→27→23→22→20→19 (점진 후퇴, end 25%) ❌
- v22 (N=20, yaw, 1M): 25→18→13(저점)→16→21→**30→32 (저점 후 우상향, end 32%)** ⭐⭐
- → **yaw reward 있으면 학습량 ↑가 정점→평균 안정화에 효과**. yaw reward가 mode 전환 catalysis로 작동, 학습량이 본격 활용.

→ **누적 천장 진전**: v11 26% → v17 정점 30% → v21 29% (단독 돌파) → **v22 32% (peak 50%)**. 다음은 yaw reward 가중치 튜닝(v22-B) 또는 N=24+yaw reward 조합(v22-C).

**v23-A (yaw reward 강화 — 정점 ↑ but 평균 ❌)**: YAW_W 0.005 → **0.007** (1.4배). v22 best 카드(N=20·ent_floor 차등·1M·s3c init) 그대로. 가설: 회전 신호 강화로 v22 peak 50% 평균값 추가 향상.

- ⭐ **정점 41% (window 100, 모든 v best)**: 625k~650k 약 25k step 동안 38~41% 유지. v22 peak 50%은 6.4k init 시점 평균이라 실제 안정 정점은 v22도 ~38% 가능성 — **v23-A가 안정 정점에선 +3%p**.
- ❌ **end 23% (v22 32% 대비 −9%p)**: 700k 부근 instability event (actor_loss 0.7→−2.5, ent_coef 0.010→0.018 spike) 후 reach 41→34→27→23으로 단조 후퇴.
- ❌ **마지막 10% 21~23% 후퇴** (v22 26~30% 안정 패턴과 정반대) — 940k부터 평균 catastrophic.
- final_dist 0.764m (v22 0.64 대비 +0.12m, v20 0.78 가까움) — 추진 양보, **v20 "정렬만, 추진 안 함" mode 부분 재현**.
- ep 23.8s (v22 21.2 대비 +2.6s) — 효율 후퇴.

→ **YAW_W 1.4배 trade-off: 정점 +3%p / 평균 −9%p / 진동 ↑**. 이론적으로 YAW_W 강화는 회전 신호 dominance를 만들고 그게 정책 진동 trigger — v5/v7 곱셈 보상 mode collapse와 다른 mechanism이지만 비슷한 결과. **v22 best 카드(YAW_W=0.005)가 균형점**. 다음은 signed yaw (`yaw_rate × sign(yaw_error)`) 또는 N=24+yaw 0.005 조합.

**v24-A (signed yaw — 진동 해결 ✓ but 천장 돌파 못 함 ◯)**: `+YAW_W·yaw_rate·sign(d(align)/d(yaw))` (옳은 방향 +, 틀린 −). YAW_W=0.005, 다른 카드 v22 그대로. v23-A 진동 진단(`|yaw_rate|`가 좌우 모두 보상 → 흔들기 trigger) 직접 검증.

- ✅ **진동 해결**: 700k actor_loss spike (v23-A: −2.5) **사라짐** ⭐. ent_coef 0.018 spike도 없음 (시작 직후 0.017→0.010 floor 빠른 수렴 후 평탄).
- ✅ **마지막 10% 안정**: 26~28% (v22 26~30% 동급, v23-A 21~23% 후퇴 회복).
- ✅ **final_dist 0.599m — 모든 v best ⭐**: v22 0.64m, v23-A 0.764m, v20 0.78m 능가. signed reward가 옳은 방향 회전·추진 시퀀스 명시적으로 학습.
- ✅ **학습 곡선 단조 우상향**: ep_rew_mean −19→+3 (저점 후 회복), actor_loss 2.1→0.8 단조 감소, V자 있으나 spike 없음.
- ❌ **천장 돌파 못 함**: end 28% (v22 32% −4%p), peak 35% at 405k (v23-A peak 41% −6%p).
- ❌ **avg_align +0.615** (v22 +0.73 대비 −0.115) — signed reward가 회전 시도 자체를 보수화 → 정렬 학습 약화.

→ **역설 — `|yaw_rate|`의 "좌우 모두 보상"이 사실 학습 dynamic에 도움**: v22의 좌우 무차별 보상은 회전 빈도를 ↑ → mode 전환 catalysis(정렬↔추진) 강화 → 천장 돌파. signed yaw는 회전을 "옳은 방향만"으로 제한 → 진동은 막았으나 학습 dynamic 보수화 → 천장 미돌파. 즉 v22 \|·\|0.005가 안정성·천장 돌파의 sweet spot, v23-A 0.007은 신호 너무 강해 진동, v24-A signed는 신호 너무 보수적이라 dynamic 부족. **천장 돌파 = 회전 신호 양 + 진동 회피 모두 필요**.

**v25-A (signed yaw + YAW_W 0.007 — spike 진단 부분 입증 but 천장 catastrophic ❌)**: v24-A signed × v23-A YAW_W 0.007 결합. 가설(signed면 weight ↑해도 진동 없음)을 직접 검증.

- ✅ **spike 없음 (진단 부분 입증)**: actor_loss V자×3 변동 있으나 v23-A의 −2.5 폭락 없음. signed가 좌우 흔들기 페널티 작동하는 것 일부 확인.
- ⭐ **peak 33% (520~535k)**: v22 안정 정점(~38%)·v23-A 41% 미달이지만 v24-A 35%와 동급.
- ❌ **W자 점진 후퇴 — 모든 v 중 worst end**: 학습 곡선 23(60k init)→25.6(90k 초기 peak)→13(225k 저점#1)→17→21(310k)→30(490k)→**33(520~535k peak)** ⭐→28(580k)→20(690k 저점#2)→27(755~820k 회복)→18(890k)→**12 (970k 모든 v 후반 최저)** ❌→16(end). 530k peak 후 470k 동안 −21%p 점진 catastrophic.
- ❌ **avg_align +0.558**: v22 +0.73 −0.17, v24-A +0.615 −0.057. signed weight ↑가 정렬도 후퇴.
- ❌ **final_dist 0.775m**: v22 0.64 +0.135, v24-A 0.599 +0.176, v20 0.78 근접 — 추진 학습 후퇴 ("정렬만 mode" 잔재).
- ❌ **ep 25.5s**: 모든 v worst (v22 21.2 +4.3s, v24-A 미보고 but v22보다 길었음). 도달 효율 후퇴.

→ **결론**: v23-A 700k 즉시 폭락이 v25-A에서는 470k 동안 점진 붕괴로 변형. signed × 강한 weight는 **spike는 막아도 학습 dynamic 자체를 천천히 망침**. 가설(signed의 안정성 × 강한 신호 결합) 부분 입증(spike 없음) but 천장 돌파 실패(weight 0.007이 signed의 보수성 깨뜨려 mode drift trigger). **v22 \|·\|0.005가 sweet spot 재확인** — yaw reward의 두 축(신호 형태·강도)에서 모두 v22 기본값이 최적.

다음 후보: **v22 reproduce (재현성 검증)** 또는 **N=24 + signed yaw 0.005** (v17 표현력 + v24-A 안정성, s1부터 새 학습 필요), 또는 **soft signed (`tanh(·)` 부드러운 부호)**. 또는 카드 축 자체 변경 (HER / fin actuator 추가).

### 핵심 통찰 — `yaw_test.py`의 결정적 발견

| 측정 대상 | yaw rate |
|---|---|
| 학습된 v3 정책 | 0.94°/s |
| 대칭 sine | 0.49°/s |
| **D2 (75% 음 + 25% 양 비대칭 stroke)** | **4.60°/s** ✓ |

→ **물리적으로 ±90° 회전이 20s 안에 충분히 가능** (D2 × 20s = 92°). 단 RL이 이런 *시간적 비대칭 패턴*을 발견 못 함이 본질적 병목. v11에서 N=16으로 부분 해결.

### 종합 통찰

- ✅ **Stage 1·2** s1 30k + s2 30k = 60k step에 100% (모든 버전 동일).
- ⭐ **v11 ±90° 천장 첫 돌파** — N=16이 D2 같은 시간적 비대칭 ctrl 표현 가능케 함.
- ❗ **30s ep의 부작용** — v6에서 도입한 ep 30s가 align 누적(+18) vs reach(+5) 3.6:1 비율 만듦 → v10·v12 "정렬만" mode 본질적 원인.
- ❗ **차등 N / 차등 NN 막힘** — obs space / net_arch 변화 시 SAC.load weight transfer 불가 (함정 #9).
- ⚠ **reward shaping 카드 4종(v12~v15) 모두 천장 미돌파** — NN [256,256] 회귀(v16)도 미돌파.
- ⚠ **신호 형태(dense vs sparse) > 비율** — v12·v15 비율 동일(3.6:1) but v12 catastrophic / v15 안정. align dense vs reach sparse.
- ⭐ **v17 N=24로 천장 일시 돌파 (정점 30%, end 25%)** — N 축이 가장 인과 명확. v11 N=8→16(+9%p) → v17 N=16→24(s3c +12%p, s3d 정점 +4%p)로 일관된 효과.
- ❗ **v18 학습량 ↑(1M)도 천장 못 깸** — end 25% 동일, 정점 30% 3회 도달하나 평균 안정화 실패. **단일 모터 + N=24 + reward 카드의 이론적 천장 ≈ 25~30%, 평균 25%**. align만 v11 동급(+0.57) 회복.
- ⭐❌ **v19 N=20 — s3a 96% (seed 분산) + s3c·d mode collapse** — N=20이 작은 회전엔 강력하나 큰 회전에서 narrow mode 잔재가 align 음수 mode 만듦. align +0.58→−0.37 단조 반전. **N과 stage 복잡도의 비대칭이 가산식 reward에서도 mode collapse 만들 수 있음** 입증 — v5/v7 함정 #8과 다른 mechanism (entropy/narrow mode 매개).
- ❗ **v20 ent_floor 강화 — collapse 해결 ✓ but reach 후퇴 ❌** — s3c·d floor 0.008/0.010으로 v19 collapse 완전 해결 (align +0.84 모든 버전 best). but reach 후퇴, final_dist 0.78m로 멀어짐. **entropy 양 ↑이 정책 mode 깨지만 추진 학습엔 못 쓰임** — v10 패턴 재현. ent_floor 카드 한계.
- ⭐⭐ **v21 yaw reward — v11 26% 천장 단독 돌파** — `+YAW_W·|yaw_rate|` (0.005) 추가. s3d end 29% (+3%p), **정점 38% (모든 v best)**, ep 21.8s (모든 v best). 모든 stage 동시 회복 (s3a 90%·s3b 69% best·s3c 38%·s3d 29%). 진단 정확: 회전 시도 자체에 보상 → 정책이 "정렬만 mode" 탈출, 회전·추진 시퀀스 학습. **천장 돌파의 본질 = yaw 변화 보상**.
- ⭐⭐⭐ **v22 v21 s3d 1M fine-tune — 천장 +6%p 추가 돌파** — end 32% (v11 26% +6%p), **peak 50%** (모든 v best). 마지막 10% 26~30% 안정. **v18(N=24+1M reject) ≠ v22(yaw+1M 성공)** — yaw reward가 학습량 ↑의 lock 해제. yaw reward = mode 전환 catalysis, 학습량 ↑가 본격 활용 가능.
- ❌ **v23-A YAW_W 0.005 → 0.007 (1.4배) — 정점 ↑ but 평균 ❌** — 정점 41% (window 100 모든 v best) 625k~650k 도달. but 700k actor_loss −2.5 폭락 + ent_coef 0.018 spike (instability event) → end 23% (v22 32% 대비 -9%p). 마지막 10% 21~23% 후퇴 (v22 26~30% 안정과 정반대). final_dist 0.764m (v20 패턴 부분 재현). **YAW_W 강화는 정점 +3%p / 평균 -9%p / 진동 ↑ trade-off — 회전 신호 dominance가 정책 진동 trigger. v22 0.005가 균형점**.
- ◯ **v24-A signed yaw — 진동 해결 ✓ but 천장 돌파 못 함** — `yaw_rate·sign(d(align)/d(yaw))`. ✓ actor_loss spike 사라짐, ✓ 마지막 10% 26~28% 안정, ✓ **final_dist 0.599m 모든 v best** (옳은 방향 추진 시퀀스 학습). but end 28% / peak 35% (v22 32% −4%p, v23-A peak 41% −6%p). avg_align +0.615 (−0.115). **역설**: v22 \|yaw_rate\|의 좌우 무차별 보상이 회전 빈도 ↑ → mode 전환 catalysis 강화 → 천장 돌파 유리. signed로 신호 보수화 → 진동 ↓ but 학습 dynamic ↓. **천장 돌파 = 회전 신호 양 + 진동 회피 동시 필요**.
- ❌ **v25-A signed yaw + YAW_W 0.007 (v24-A × v23-A 결합) — spike 없음 but 천장 catastrophic** — 가설(signed면 weight ↑해도 진동 없음) 부분 입증: ✓ actor_loss V자×3 있으나 v23-A −2.5 폭락 없음. but ❌ **W자 점진 후퇴 — 모든 v 중 worst end**: peak 33% (520~535k) 후 catastrophic drift → 970k 12% (모든 v 후반 최저) → end **16%** (v22 32% **−16%p**). avg_align +0.558, final_dist 0.775m, ep 25.5s (모든 v worst). **v23-A 700k 즉시 폭락이 v25-A에서는 470k 동안 점진 붕괴로 변형** — signed × 강한 weight는 spike는 막아도 학습 dynamic 천천히 망침. **v22 \|·\|0.005가 sweet spot 재확인**: yaw reward의 두 축(신호 형태·강도)에서 모두 v22 기본값이 최적, 단순 weight 카드 종결.

---

## TensorBoard 메트릭 가이드

`tb_logs/` 구조는 **버전 폴더로 분리** (`model3_v1/` ~ `model3_v25/`).

### SB3 기본

| 태그 | 정의 |
|---|---|
| `rollout/ep_rew_mean` | 최근 100 ep 평균 보상 |
| `rollout/ep_len_mean` | 최근 100 ep 평균 step |
| `train/critic_loss` | Q-function MSE (발산 X) |
| `train/ent_coef` (α) | auto-tuned entropy 가중치 |

### v2 신설 — `fish/*` namespace

| 태그 | 정의 |
|---|---|
| `fish/success_rate` | 최근 N ep 도달률 (0~1) |
| `fish/final_distance` | 최근 N ep 평균 종료 거리 (m) |
| `fish/episode_seconds` | 최근 N ep 평균 ep 길이 (초) |
| `fish/avg_align` | 최근 N ep 평균 정렬도 (-1 반대 ~ +1 정조준) |

### 진단 신호

- **α가 5만 step 내 0.001 이하** → 탐색 부족, ep_rew 정체.
- **ep_len_mean = max 유지** → 도달 못 하는 정책.
- **avg_align 높지만 success_rate 낮음** → "정렬만 추구 + 진행 안 함" mode.

```bash
tensorboard --logdir tb_logs/ --bind_all
```

---

## SAC 알고리즘 메모

`stable-baselines3.SAC` 기본 — Maximum-Entropy off-policy actor-critic.

핵심:
- **목적함수**: $\mathbb{E}[\sum r_t + \alpha \mathcal{H}(\pi)]$
- **Twin Q-network**: $\min(Q_1, Q_2)$로 overestimation 방지.
- **Auto α tuning**: $\mathcal{H}_{target} = -\dim(A) = -1$.
- **Off-policy + replay 200k**: PPO 대비 sample-efficient.

하이퍼파라미터: LR=3e-4, batch=256, γ=0.99, τ=0.005, buffer=200k, **`ent_coef="auto_0.1"`** — *초기값* 0.1, floor 아님 (진짜 floor는 EntCoefFloorCallback).

---

## 작업 진행 규칙 — 한 단계씩 분리

다단계 작업은 **한 단계씩 분리해서** 진행. 한 단계 결과를 보고하고 사용자 확인을 받은 뒤 다음으로 넘어간다.

- "한 단계"의 단위는 **사용자가 결과를 보고 다음 결정을 내릴 수 있는 지점**.
- TaskCreate로 전체 단계 미리 나열 OK. status는 한 번에 한 task만 in_progress.
- 자명하게 묶이는 미세 작업은 쪼개지 않아도 됨. 핵심은 **사용자가 검수·중단할 기회를 주는 것**.

---

## 알아둘 함정 (이미 발견·해결된 것)

이미 시도해본 막다른 길. **반복하지 마세요**.

1. **6DOF freejoint**: yaw 진동의 pitch cross-coupling으로 fish가 60° 기울며 추진 방향 뒤집힘. → 3DOF planar로 결정.
2. **fluidshape 변경 시도**: MuJoCo는 `none`/`ellipsoid` 둘뿐.
3. **단일 passive fin (k=4e-7)**: 1자유도 spring은 wave 못 만듦.
4. **다단 passive fin (segment chain)**: wave 패턴은 발생하지만 ellipsoid 한계로 +x 추진 못 만듦.
5. **CPG/Fourier 액션공간 (MODE_2)**: 단일 관절 본질적 한계 — 14가지 waveform 모두 후진.
6. **F1 STEP 분해 → 다중 fluid ellipsoid**: z축으로 +0.56 m 발산. 비대칭 lift 폭주.
7. **단일 관절 + ellipsoid + 6DOF**: 위 1~6의 종합 결론.
8. **곱셈 보상 (`prog × align_factor`)**: v5(비대칭)는 머리 반대+후진 mode (avg_align −0.5). v7(대칭)은 도망 mode (final_dist 6m). **가산식으로 가야 함**.
9. **차등 N / 차등 NN**: stage별 obs dim이나 net_arch가 다르면 SAC.load weight transfer 불가. obs Box mismatch는 즉시 AssertionError, layer 개수 mismatch는 새 layer가 random init돼서 정책 손실. **curriculum init_from 무력화**.
10. **NN 확장만 (v12)**: 27D obs 흡수 위해 [256,256,128] 시도. 작은 회전 회복(+17~23%p) but **s3d catastrophic interference (26→6%)**. NN 확장 단독으로는 v11 못 넘음.
11. **align_weight 단순 ep 반비례 (v13)**: `0.02·10/ep_seconds` (30s → 0.0067). v12 catastrophic 해결 but 정렬 신호 부족 (align +0.19). 비율 1.2:1는 reach와 동급이라 정렬 학습 거의 못 함. 적정값 ~0.012 (비율 2.2:1) — ep 비례보다 *직접 fix*가 정확.
12. **align_weight 단독 카드 (v13·v14)**: 0.020 / 0.0067 / 0.012 — reward 비율 카드 3종 모두 v11 26% 천장 못 깸. **align_weight 단독으로는 천장 못 깸**.
13. **reach 보너스 강화 (v15)**: align 0.012 + reach 5→10. s3c 20→27% 회복 ✓ but s3d 21%로 천장 여전. s3d V자×2(50→12→21%) 위험 신호. **reward shaping 4종(v12~v15) 모두 천장 못 깸**.
14. **NN default 회귀 (v16)**: net_arch [256,256,128] → [256,256] 회귀. 작은 회전 회복 ✓ (s3a 69→80%·s3b 62→70%) — NN 확장이 narrow mode 가속한 부작용 입증. but s3c 21%·s3d 19%로 천장 미돌파. **NN 카드도 천장 못 깸 — reward 4종 + NN 1종 = 5장 모두 미돌파**.
15. **action history N=24 (v17)**: 16 → 24 (1.5 wag cycle, obs 27→35D). **s3c·d 동시 회복** (s3c 21→33% +12%p, s3d 19→25% +6%p), **s3d 정점 30%로 v11 26% 천장 일시 돌파** (270~310k). end 25%로 v11 −1%p 근접하지만 단독 돌파는 못 함 — max_steps 500k 부족 가능성. 작은 회전 후퇴 재현 (s3a 80→66%, s3b 70→62%) — N-stage 매칭 trade-off는 N=16(v11)에서도 보였던 패턴, N 축의 본질적 비용. **N 축이 천장 카드의 가장 인과 명확한 축임 재확인**.
16. **학습량 ↑ (v18, v17-B)**: v17 s3d만 500k → 1M fine-tune. **end 25% 동일** — 학습량 부족 가설 reject. 정점 30% 3회 도달(22.9k·312k·944k)하나 100ep window 안정화 실패. 정렬만 v11 동급 회복 (+0.43→+0.57), ep_rew 6.6→8.2 ↑. **단일 모터 + N=24 + 현 reward의 이론적 천장 ≈ 25~30%, 평균 25%**가 잠정 결론. 정책이 mode 전환(정렬↔추진) 못 하고 25% 부근 진동.
17. **N=20 절충 (v19, v17-C)**: 24 → 20 (1.25 wag cycle). s3a 96% (170k 조기, but seed 분산 — v20 동일 setup에서 63%) + **s3c·d mode collapse** — avg_align s3a +0.58 → s3d **−0.37** 단조 반전, ep_rew −2.5 (음수). 정책이 "거꾸로 가면서 가까워지면 prog 양수" mode 학습 (final_dist 0.35m < 시작 0.5m). N=20이 작은 회전 narrow mode 강화 → 큰 회전 entropy floor로 못 깨짐 → align 음수 mode 학습. **N=20 단독 부적합** — 큰 회전 안정성 희생.
18. **ent_floor 강화 (v20)**: s3c·d floor 0.005·0.006 → **0.008·0.010**. v19 mode collapse **완전 해결 ✓** (align s3d −0.37 → **+0.84 모든 버전 best**). but reach 후퇴 (s3a 63%, s3d 19%) + final_dist 0.78m (시작 0.5m보다 멀어짐). v10 "정렬 best, 추진 worst" 패턴 더 심하게 재현. **entropy 양 ↑이 정책 mode 깨지만 추진 학습엔 못 쓰임 — ent_floor 카드 한계 입증**. SAC actor가 entropy를 정렬에만 활용. 다음은 N·entropy·reward 단일 축 카드 종결, 근본 카드 (HER / Custom yaw reward) 또는 v11 회귀.
19. **YAW_W 강화 (v23-A)**: v22 best 카드 + YAW_W 0.005 → **0.007** (1.4배). **정점 41% (window 100 모든 v best, 625k~650k)** but 700k actor_loss 0.7 → **−2.5 폭락** + ent_coef 0.010 → **0.018 spike** (instability event) → end 23% (v22 32% −9%p). 마지막 10% 21~23%로 catastrophic 후퇴 (v22 26~30% 안정과 정반대). final_dist 0.764m로 v20 패턴 부분 재현. **YAW_W 강화 trade-off: 정점 +3%p / 평균 −9%p / 진동 ↑** — 회전 신호 dominance가 정책 진동 trigger. v5/v7 곱셈 보상 mode collapse와 다른 mechanism (정점에 도달은 하지만 유지 못 함). **v22 YAW_W=0.005가 균형점**, weight 단순 ↑로는 천장 못 깸.
20. **signed yaw 단독 (v24-A)**: v22 best 카드 + `+YAW_W·yaw_rate·sign(d(align)/d(yaw))` (옳은 방향 +, 틀린 −). YAW_W=0.005 그대로. v23-A 진동 진단 직접 검증. ✅ **진동 해결**: 700k actor_loss spike 사라짐, 마지막 10% 26~28% 안정, **final_dist 0.599m 모든 v best ⭐** (옳은 방향 추진 시퀀스 학습). ❌ **천장 돌파 못 함**: end 28% / peak 35% (v22 32% −4%p, v23-A peak 41% −6%p). avg_align +0.615 (v22 +0.73 −0.115). **역설**: v22 \|yaw_rate\|의 "좌우 모두 보상" 노이즈가 사실 회전 빈도 ↑ → mode 전환 catalysis 강화 → 천장 돌파 유리. signed로 신호 보수화 → 진동 ↓ but 학습 dynamic ↓. **천장 돌파 = 회전 신호 양 + 진동 회피 동시 필요** — signed 단독으론 부족, **signed + YAW_W ↑ 조합 필요**.
21. **signed yaw + 강한 weight (v25-A)**: v22 best 카드 + signed yaw × YAW_W 0.005 → **0.007** (v24-A 안정성 × v23-A 신호 강도 결합). 가설: signed면 weight ↑해도 좌우 흔들기 페널티 작동 → 진동 안 일어남. ✅ **spike 없음 (가설 부분 입증)**: actor_loss V자×3 있으나 v23-A −2.5 폭락 없음. but ❌ **W자 점진 catastrophic — 모든 v 후반 최저**: peak 33% (520~535k, v24-A 35% 동급) 후 470k 동안 −21%p 점진 붕괴 → 970k **12%** → end **16%** (v22 32% −16%p, v24-A 28% −12%p). avg_align +0.558, final_dist 0.775m, ep 25.5s (모든 v worst). **v23-A 700k 즉시 폭락이 v25-A에서는 천천히 진행되는 형태로 변형** — signed × 강한 weight는 spike는 막아도 학습 dynamic 천천히 망침. **v22 \|·\|0.005가 sweet spot 재확인** — yaw reward 두 축(신호 형태·강도)에서 모두 기본값이 최적. **YAW_W 단순 ↑ 카드 완전 종결 (v23-A·v25-A 모두 reject)**, signed × 강한 weight도 reject.

→ 정착한 조합 = **MODE_3 + Ecoflex passive fin + 3DOF planar**.

가장 결정적이었던 변경 (회고):
1. **3DOF planar 가정** — pitch wobble 제거가 부호 뒤집기의 본질
2. **MATLAB CG.m 값 적용** (xG=xb=0) — sim2real 정합성
3. **MODE_3 fin 분리 + Ecoflex density=1070** — wave-thrust 발생
4. **action history N=16 (v11)** — ±90° 천장 첫 돌파 (24%→26%)
5. **action history N=24 (v17)** — 큰 회전 동시 회복 + 정점 30% 일시 돌파
6. **yaw_rate 보상 (v21)** ⭐⭐ — **v11 26% 천장 단독 돌파** (29%, 정점 38%). 회전 시도 인센티브가 mode 전환 trigger. 모든 stage 동시 회복.
7. **v21 s3d 1M fine-tune (v22)** ⭐⭐⭐ — **천장 +6%p 추가 돌파** (32%, peak 50%). yaw reward가 학습량 ↑의 lock 해제 — v18(no yaw, 1M)은 reject였으나 v22(yaw, 1M)는 성공.
8. **v23-A YAW_W 1.4배 reject** ❌ — 정점은 41%로 ↑되나 700k actor_loss spike 후 end 23%로 후퇴. **YAW_W=0.005가 균형점** 입증, 단순 weight ↑ 카드 종결.
9. **v24-A signed yaw — 진동 해결 ✓ but 천장 ◯** — `yaw_rate·sign(d(align)/d(yaw))`. v23-A 진동 진단 직접 검증, **actor_loss spike 사라짐 + 마지막 10% 26~28% 안정 + final_dist 0.599m 모든 v best**. but end 28% / peak 35% (v22 −4%p). **역설**: \|·\|의 좌우 무차별 보상이 회전 빈도 ↑로 mode catalysis 강했던 것. 천장 돌파 = 신호 양 + 진동 회피 동시 필요.
10. **v25-A signed × 강한 weight reject** ❌ — signed × YAW_W 0.007 결합. spike는 막혔으나(가설 부분 입증) **W자 점진 catastrophic — end 16% (모든 v worst)**, 970k 12%로 후반 최저. 530k peak 33% 후 470k 동안 점진 붕괴. **v23-A 700k 즉시 폭락의 슬로 모션 버전**. signed × 강한 weight도 reject — **v22 \|·\|0.005가 sweet spot 재확인, YAW_W 단순 ↑ 카드 완전 종결**.

---

## RL 학습 시 주의

- **fluidshape="ellipsoid" 한계 인지**. vortex shedding 못 모델 → sim2real 본질적 격차. 추후 개선은 Lighthill slender body theory 콜백(`mjcb_passive`) 또는 ANN surrogate.
- **보상 함수의 forward 방향**: world −x. 학습 정책이 +x로 가면 보상 부호 또는 환경 인덱스 잘못된 것.
- **3DOF planar에서 yaw 누적**: fluid asymmetry 잔여로 한쪽으로 도는 경향. RL이 좌우 균형 학습으로 보정 가능.
- **Curriculum init_from 부작용**: 이전 단계가 "직진" 편향이면 Stage 3 head arc에서 한쪽 mode만 학습 가능.

---

## 백업·릴리스 패턴

```bash
# 새 학습 후 v(N)→v(N+1)
mkdir -p /tmp/models-vN/runs /tmp/models-vN/plots
cp sim/runs/{tag1,tag2,...}/model.zip /tmp/models-vN/runs/<tag>/
cp sim/plots/*.png /tmp/models-vN/plots/
tar -czf /tmp/runs-models-vN.tar.gz -C /tmp/models-vN .

gh release create models-vN --target <branch> \
    --title "<제목>" --notes "<내용>" \
    /tmp/runs-models-vN.tar.gz

# 복원
gh release download models-vN -p '*.tar.gz'
tar -xzf runs-models-vN.tar.gz -C sim/
```

---

## 다음 후보 (미해결)

**v25까지 결론**: 9장(v12~v20) 미돌파 → **v21 yaw reward 천장 단독 돌파** (29%) → **v22 v21+1M으로 천장 +6%p 추가 돌파** (32%, peak 50%) ⭐⭐⭐ → **v23-A YAW_W 강화 reject** (정점 +3%p but 평균 -9%p, 700k 즉시 spike) → **v24-A signed yaw 진동 해결 ✓ but 천장 미돌파** (end 28%, peak 35%, final_dist best 0.599m) → **v25-A signed × YAW_W 0.007 reject** (spike 없음 but W자 점진 catastrophic, end 16% 모든 v worst). 핵심 통찰: **천장 돌파 = 회전 신호 양 + 진동 회피 동시 필요. yaw reward 두 축(신호 형태·강도)에서 v22 기본값(`|·|`·0.005)이 sweet spot**. YAW_W 단순 ↑ 카드 완전 종결 (`|·|`·0.007 즉시 spike / signed·0.007 점진 catastrophic 모두 reject). 현재 best 카드 = N=20 + ent_floor 차등 + reward (align 0.012 + reach 10 + yaw `|·|`0.005) + s3d 1M (= v22).

1. **v22 reproduce (재현성 검증)** — v22 32% / peak 50%이 단일 seed인지. v23-A·v24-A·v25-A 모두 v22 reject → v22가 정말 baseline인지 재현성 확인 필요. **다음 권장 (현재 best 카드 신뢰도 확보)**.
2. **v26 — soft signed yaw** — `sign` 대신 `tanh(d(align)/d(yaw) × scale)`로 부드러운 부호. v24-A signed 단절 + v22 `|·|` 노이즈의 중간. 0 부근 단절 제거, mode catalysis 일부 보존.
3. **v26 — N=24 + signed yaw 0.005** — v17 큰 회전 표현력(정점 30% 일시) + v24-A 안정성. N 변경은 s1부터 새 학습 (함정 #9, 2~3시간).
4. **v26 — v22 + s3d 2M fine-tune** — v22가 1M 안에서 peak 50% 후 점진 안정 32%. 학습량 2배로 추가 안정화 가능성 검토 (다만 v18 패턴 재현 위험).
5. **HER (Hindsight Experience Replay)** — env Dict obs 큰 변경. 가장 강력하나 구현 비용 큼.
6. **fin actuator 추가** — 단일 모터 한계 자체를 풂. (사용자 명시 제외)
7. **ANN surrogate (Lighthill / Zhong 2026)** — fluid model 한계 우회. 실물 motion capture 필요.

### 단일 지느러미의 천장

`yaw_test.py`로 **단일 모터+passive fin yaw rate 물리 상한 ~4.6°/s** 확인 (D2 패턴 × 20s = 92°, ±90° 도달은 가능하나 마진 작음). v1~v10 13~24% → v11 N=16으로 26% 첫 돌파 → v12~v15 reward 4종 + v16 NN 회귀 모두 미돌파 → v17 N=24 정점 30% 일시 → v18 1M 학습량도 reject → v19 N=20 s3c·d mode collapse → v20 ent_floor ↑로 collapse 해결 but reach 19% → **v21 yaw reward로 천장 단독 돌파 (29%)** ⭐⭐ → **v22 v21+1M으로 +6%p 추가 돌파 (32%, peak 50%)** ⭐⭐⭐ → **v23-A YAW_W 1.4배 reject** (정점 41% but 평균 23%, 700k 즉시 spike) → **v24-A signed yaw 진동 해결 but 천장 미돌파** (end 28%, peak 35%, final_dist 0.599m best) → **v25-A signed × YAW_W 0.007 reject ❌** (spike 없음 but W자 점진 catastrophic, end 16% 모든 v worst, 970k 12% 후반 최저).

**카드 분류 매트릭스 (최종)**:
- **N (시간적 표현력)** v11/v17/v19 — 정점 30% 가능
- **entropy (mode 전환)** v8~v10/v20 — collapse 해결
- **align/reach 가중치** v12~v15 — reward 비율 카드 한계
- **yaw reward (회전·추진 시퀀스)** **v21~v22** — **천장 돌파 + 안정화 ⭐⭐⭐**
- **학습량 ↑ × yaw reward** — yaw reward가 학습량 lock 해제 (v18 reject ≠ v22 성공)
- **YAW_W 강화 (v23-A: \|·\|·0.007)** ❌ — 정점 +3%p / 평균 −9%p / 진동 ↑. 700k actor_loss 즉시 폭락.
- **signed yaw 단독 (v24-A: signed·0.005)** ◯ — 진동 해결 ✓ but 천장 미돌파 (end 28%, v22 −4%p). final_dist 0.599m best. avg_align 0.615로 ↓ — 신호 보수화의 대가.
- **signed × 강한 weight (v25-A: signed·0.007)** ❌ — spike는 막힘 (가설 부분 입증) but **W자 점진 catastrophic** — end 16% (모든 v worst), 970k 12% 후반 최저. v23-A 즉시 폭락의 슬로 모션 버전.

→ **천장 돌파의 본질 = "회전 신호 양" + "진동 회피" 동시 필요**. v22 \|·\|0.005가 sweet spot — yaw reward 두 축(신호 형태·강도)에서 v22 기본값이 최적. v23-A·v25-A 모두 weight ↑ 카드 reject, v24-A signed 단독은 보수적이라 천장 미돌파. **YAW_W 단순 변형 카드 완전 종결**. 다음은 **v22 reproduce (재현성 검증)** 또는 **soft signed (`tanh(·)`)** 또는 **N=24 + signed 0.005** (s1부터 새 학습).
