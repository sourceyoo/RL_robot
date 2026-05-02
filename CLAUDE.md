# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 목적

KUFIsh_III (사용자 자체 CAD) 기반 단일 관절 물고기 로봇의 강화학습. MuJoCo + SAC + Gymnasium. **수면 영법(BCF surface swimming) 가정**으로 단순화하여 학습.

## 활성 모델 — `sim/rl_fish.xml`

RL이 학습하는 단 하나의 모델. **이 파일이 모든 시뮬·학습의 진입점.**

### 핵심 사실 (한눈에 안 보이는 것들)

- **3DOF planar 운동**. base_link는 freejoint가 아니라 `slide_x` + `slide_y` + `hinge_yaw` 3개 명시 joint. roll/pitch/z는 *수학적으로 잠김*. **freejoint로 바꾸지 마세요** — pitch wobble로 추진 방향이 뒤집힙니다 (PROGRESS.md 13장 참고).
- **qpos layout = 5**: `[root_x, root_y, root_yaw, tail_joint, fin_joint]`. env 코드의 인덱스가 이 순서에 의존.
- **수중 환경**. `<option density="1000" viscosity="0.001">`(실제 물). `<flag gravity="disable"/>`로 중력 끔. 부력은 명시적 안 받지만 added mass + drag는 `fluidshape="ellipsoid"`로 자동. **중력 켜지 마세요** — 별도 부력 콜백 없으면 가라앉음.
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

## CAD 소스 — `fish_urdf/RL_SIM_MODE_3_description/`

활성 mesh의 출처. URDF는 *형상만* 사용 (mesh 파일들), inertial 값은 위 표대로 별도 검증.

- mesh: `base_link.stl`, `tail_link_1_1.stl`, `fin_1.stl`
- URDF는 fin을 `<joint type="fixed">`로 부착하지만 우리 MJCF는 **passive hinge로 교체** (실리콘 변형 모델링).
- 이전 버전(`RL_SIM_MODE_2_description/`)도 남아있지만 **사용 안 함** — fin 분리 안 된 구버전.

### MODE_2 패키지 깨진 파일 (참고)

`launch/controller.yaml`, `launch/controller.launch`에 한국어 관절명이 CP949 mojibake (`ȸ�� 3`). URDF 실제 관절명 `tail_joint`와도 다름. ros_control 안 돌아감. **수정 안 해도 RL 학습엔 무관** (URDF만 mesh 출처로 사용).

## Python 학습 인프라 — `sim/`

- `fish_env.py` — Gymnasium 환경. obs 차원 = 7 (x, y, yaw, tail_qpos, tail_qvel, fin_qpos, target 상대좌표) — *5DOF에 맞춰 갱신 필요*.
- `train.py` — SAC 학습 + 별도 thread viewer. PolicySnapshotCallback으로 race-free.
- `view_policy.py` — 저장된 정책을 viewer로 rollout.
- 진단 스크립트: `freq_sweep.py`, `freq_sweep_locked.py`, `freq_sweep_norollpitch.py`, `waveform_test.py`, `multiseg_test.py`. 추진 방향·대칭성 검증용.

### 학습 명령

```bash
cd sim
python3 train.py --steps 20000 --tag smoke    # 5~10분 시운전
python3 train.py                               # 본 학습 50만 step (~30분~1시간)
python3 train.py --no-viewer                   # viewer 없이 빠르게
python3 view_policy.py runs/<tag>/model.zip    # 저장된 정책 보기
python3 -m mujoco.viewer --mjcf=rl_fish.xml    # 모델만 검수
tensorboard --logdir runs/                     # 학습 곡선
```

## 작업 진행 규칙 — 한 단계씩 분리

다단계 작업은 **한 단계씩 분리해서** 진행. 한 단계 결과를 보고하고 사용자 확인을 받은 뒤 다음으로 넘어간다.

- "한 단계"의 단위는 **사용자가 결과를 보고 다음 결정을 내릴 수 있는 지점**.
- 예: 모델 변경 → 멈춤. viewer 검증 → 멈춤. 정확성 테스트 → 멈춤. env 갱신 → 멈춤. smoke test → 멈춤.
- TaskCreate로 전체 단계를 미리 나열해 두는 건 OK. status는 한 번에 한 task만 in_progress.
- 자명하게 묶이는 미세 작업(같은 파일 수정 + 같은 줄 검증)은 쪼개지 않아도 됨. 핵심은 **사용자가 검수·중단할 기회를 주는 것**.

## 알아둘 함정 (이미 발견·해결된 것)

이미 시도해본 막다른 길. **반복하지 마세요** (PROGRESS.md에 상세 기록).

1. **6DOF freejoint**: yaw 진동의 pitch cross-coupling으로 fish가 60° 기울며 추진 방향 뒤집힘. → 3DOF planar로 결정.
2. **fluidshape 변경 시도**: MuJoCo는 `none`/`ellipsoid` 둘뿐. 다른 도형 없음.
3. **단일 passive fin (k=4e-7)**: 1자유도 spring은 wave 못 만듦. 단일 passive vs 강체 fin 추진 동일 (소수점 셋째 자리까지).
4. **다단 passive fin (segment chain)**: wave 패턴은 발생하지만 ellipsoid fluid 한계로 +x 추진은 못 만듦. MODE_2 시절 14가지 waveform 모두 후진.
5. **CPG/Fourier 액션공간**: 단일 관절 본질적 한계 — 학습할 +x 패턴이 모델에 존재하지 않음.

## RL 학습 시 주의

- **fluidshape="ellipsoid" 한계 인지**. vortex shedding을 못 모델하므로 sim2real에 본질적 격차. 추후 개선은 Lighthill slender body theory 콜백(`mjcb_passive`) 또는 ANN surrogate (Zhong 2026 방식).
- **보상 함수의 forward 방향**: world −x. 학습 정책이 +x로 가면 보상 부호 또는 환경 인덱스 잘못된 것.
- **3DOF planar에서 yaw 누적**: fluid asymmetry 잔여로 한쪽으로 도는 경향. RL이 좌우 균형 학습으로 보정 가능. 단 yaw가 많이 누적되면 보상 하락 → 학습이 자연 보정.
