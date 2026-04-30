# 물고기 로봇 RL — 진행 보고서

> 작업일: 2026-04-30
> 목표: 자체 CAD 물고기 로봇(`fish_urdf/RL_SIM_MODE_2`)을 MuJoCo로 시뮬레이션해서 강화학습 시키기.

---

## 0. 큰 그림

강화학습으로 물고기를 헤엄치게 만든다 = 다음 4가지를 만든다:

1. **시뮬레이터 모델** (.xml) — MuJoCo가 읽을 물고기 + 물 + 목표
2. **환경 wrapper** (.py) — "한 step 진행하면 보상 얼마"를 정의하는 Python 클래스
3. **학습 알고리즘** (SAC) — 환경에서 시행착오로 정책(policy)을 만드는 코드
4. **시각화** — 학습된 정책이 실제로 어떻게 헤엄치나 확인

이번 작업으로 1~3번 인프라가 완성되었고, 4번 시각화는 영상 파일로 진행 중.

---

## 1. 환경 점검 (시작 전)

| 항목 | 결과 |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | RTX 3090 24GB (CUDA 13.1 드라이버) |
| Python | 시스템 `/usr/bin/python3` 3.10 (conda·uv 없음) |
| 이미 설치됨 | `mujoco==3.8.0`, `torch==2.11.0+cu130` |
| 추가 설치 | `gymnasium`, `stable-baselines3`, `tensorboard`, `mediapy` |

CUDA가 잘 동작하는 걸 확인했고, GPU 학습 가능한 환경입니다.

---

## 2. 갈림길 — 왜 C-2를 선택했나

### 두 개의 모델이 같이 들어 있음

저장소엔 두 가지 물고기 모델이 있습니다 — 이걸 정리하는 게 첫 단계였습니다.

- **`robotic_fish.xml`** — DeepMind Control Suite의 "fish" 데모를 그대로 가져온 모델. 꼬리 2단 + 양 가슴지느러미 + 5개 액추에이터로 잘 헤엄치도록 이미 튜닝되어 있음.
- **`fish_urdf/RL_SIM_MODE_2_description/`** — 사용자가 Fusion 360에서 직접 디자인한 CAD 모델. 관절은 `tail_joint` **하나**뿐.

### 3가지 길

| 길 | 학습 대상 | 작업량 | 의미 |
|---|---|---|---|
| **B** | 기성 dm_control 모델 | 적음 | 데모 재현, "내 로봇" 느낌 약함 |
| **C-1** | 자체 CAD + 가슴지느러미 추가 | 많음 (CAD 재모델링) | 본격적, sim2real 가능 |
| **C-2** | **자체 CAD 그대로 (1관절)** | 중간 | "내 로봇으로 RL", 단순함 우선 |

→ **C-2 선택**. 자체 CAD를 쓰되 모델은 간략하게.

---

## 3. URDF → MuJoCo 변환 (왜 그냥 안 됐나)

URDF 파일(`RL_SIM_MODE_2.urdf`)은 ROS·Gazebo용 표준 포맷이고 MuJoCo도 직접 읽을 수 있습니다. 그래서 처음엔:

```python
mujoco.MjModel.from_xml_path("RL_SIM_MODE_2.urdf")
```

한 줄로 끝낼 수 있을 줄 알았는데 — **두 가지 문제**가 발견되었습니다.

### 문제 1: 몸체가 공중에 박혀 있다

URDF 표준은 root link(여기선 `base_link`)가 **세계에 고정**된다고 가정합니다. 매니퓰레이터(고정 받침대 위 로봇팔)를 위한 표준이라 그렇습니다. 즉 그대로 import하면 *몸은 공중에 박혀 있고 꼬리만 흔들리는* 상태가 됩니다. 헤엄칠 수 없습니다.

해결: MuJoCo의 **freejoint**를 `base_link`에 명시적으로 추가. freejoint는 6자유도(이동 3 + 회전 3)를 모두 부여합니다.

```xml
<body name="base_link" ...>
  <freejoint name="root"/>   <!-- 이 한 줄로 자유 물체가 됨 -->
  ...
</body>
```

### 문제 2: 액추에이터가 0개

URDF에 있는 `<transmission>` 태그는 ROS의 ros_control 시스템용이라 MuJoCo는 무시합니다. 그래서 import 직후 `nu=0` (제어 가능한 액추에이터 0개). RL은 행동을 줄 곳이 없으면 못 합니다.

해결: `<actuator>` 섹션을 명시적으로 추가.

```xml
<actuator>
  <motor name="tail" joint="tail_joint" ctrlrange="-1 1" gear="0.3"/>
</actuator>
```

`motor` = 토크 제어 액추에이터. ctrl 값(-1~1)에 gear를 곱한 만큼의 토크가 `tail_joint`에 가해집니다. RL이 매 스텝마다 -1~1 사이 숫자를 출력하면 그게 꼬리 토크가 됩니다.

### 그 외에 추가한 것

원본 URDF엔 RL에 필요한 게 더 빠져 있었습니다:

| 추가한 것 | 왜 |
|---|---|
| `<option density="5000">` | **수중 환경**. MuJoCo는 매질의 density로 부력·항력을 자동 계산. dm_control fish와 동일값. |
| `gravity="disable"` | 중력은 끔. 부력으로 떠 있다고 가정 (수중 호버링) |
| `fluidshape="ellipsoid"` | 물이 mesh를 ellipsoid로 근사해서 추진력 계산. 이게 없으면 꼬리쳐도 안 나아감 |
| `<site name="torso">` | 센서 부착점 (눈에 안 보이는 가상의 점) |
| `<sensor> velocimeter, gyro` | 물고기가 자기 속도·회전속도를 알 수 있게 |
| `<geom name="target">` | 헤엄쳐 갈 빨간 구. 매 에피소드 위치 랜덤화 |
| `integrator="implicit"` | 안정적인 통합기. explicit는 freejoint에서 NaN 나기 쉬움 |
| `damping="0.05" armature="0.001"` | 진동 억제 — 처음 NaN 폭주 났던 원인 |

→ 결과물: **`sim/rl_fish.xml`**

---

## 4. Gymnasium 환경 wrapper (`sim/fish_env.py`)

MuJoCo는 물리만 시뮬레이션합니다. RL을 돌리려면 "관측 / 행동 / 보상"을 정의하는 환경이 필요합니다 — 이게 **Gymnasium**의 `Env` 클래스입니다. SB3의 SAC가 직접 Gymnasium을 받아주기 때문에 이 표준을 따릅니다.

### 관측 공간 (observation, 11차원)

| 인덱스 | 내용 | 차원 |
|---|---|---|
| 0 | 꼬리 관절 각도 (qpos) | 1 |
| 1~7 | 모든 관절 속도 (qvel) — freejoint 6 + tail 1 | 7 |
| 8~10 | torso → target 상대 벡터 (x,y,z) | 3 |

→ 정책은 *내 위치/속도/꼬리 상태/목표가 어디 있는지*를 입력으로 받습니다. 절대 좌표가 아니라 *상대 위치*를 주는 게 핵심 — 물고기가 어디 있어도 같은 행동 학습 가능.

### 행동 공간 (action, 1차원)

`tail_joint`의 motor ctrl 한 값. 범위 `[-1, 1]`. SAC가 매 step마다 이 한 숫자를 출력합니다.

### 보상 (reward) — 이게 학습의 신호

```python
progress = prev_distance - current_distance          # 가까워진 만큼
ctrl_cost = 0.001 * action²                          # 헛힘 미세 페널티
reached_bonus = 5.0 if distance < 0.08 else 0.0     # 도달 보너스

reward = progress * 10  +  reached_bonus  -  ctrl_cost
```

**왜 progress 방식인가:** 거리(absolute)를 보상으로 주면 멀리 있을 때 보상이 항상 작아서 학습이 느려집니다. *얼마나 가까워졌는지*(delta)를 주면 매 step마다 즉각적인 신호가 나와 학습이 빠릅니다.

### 에피소드 구조

- timestep = 0.002s × frame_skip 10 = **dt 0.02s** (정책은 50Hz로 결정 내림)
- 한 에피소드 = 10초 = **500 step**
- 종료: 목표 도달 (distance < 8cm) **또는** 10초 경과
- reset 시 목표가 반지름 0.5m 원 위에서 랜덤 방향으로 재배치

### Smoke test 결과

랜덤 정책으로 한 에피소드 굴려봤더니 obs 11차원 정상, NaN 없음, 시작거리 0.5m → 끝거리 0.4m (랜덤이니 약간만 가까워짐). 환경 완성.

---

## 5. SAC 학습 (`sim/train.py`)

**SAC (Soft Actor-Critic)** — 연속 행동 공간(우리처럼 [-1,1] 실수)에서 가장 잘 동작하는 RL 알고리즘 중 하나. stable-baselines3의 기본 SAC를 거의 그대로 사용.

### 핵심 하이퍼파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| learning_rate | 3e-4 | Adam 학습률 (SB3 기본) |
| buffer_size | 200,000 | replay buffer 크기 |
| batch_size | 256 | 한 번 업데이트할 때 샘플 |
| gamma | 0.99 | 미래 보상 할인율 |
| tau | 0.005 | target network soft update |
| device | cuda | GPU 사용 |

### Smoke test (20,000 step, 57초)

| 지표 | 값 | 해석 |
|---|---|---|
| FPS | ~350 | GPU 잘 활용됨 (3090 1대) |
| ep_rew_mean | -0.149 → -0.325 | 보상 음수로 떨어짐 — **정책이 헛 ctrl만 내고 진행 못 함** |
| ent_coef | 0.058 → 0.003 | 자동 entropy tuning이 너무 빨리 수축 |
| eval reward | -0.017 ± 0.030 | 사실상 0, 학습 미완 |

→ 파이프라인은 끝까지 돕니다. 하지만 20k step은 SAC에 짧고, 보상 스케일도 작아서 본 학습 전에 한 번 손볼 여지가 있음.

---

## 6. 시각화 — MuJoCo viewer 워크플로

**원칙: 영상 녹화(mp4) 없음. 모든 시각화는 MuJoCo viewer GUI로 직접 본다.**

이유: 영상은 정적이고 카메라가 고정. viewer는 마우스로 카메라 자유 조작 + 시뮬을 일시정지·재개 + 좌측 패널에서 ctrl 슬라이더 직접 조작 가능. 학습 진행 모니터링과 디버깅 모두 viewer가 더 직관적.

### 6.1 학습 중 실시간 viewer (기본)

`train.py`를 viewer와 함께 실행하면 **학습이 진행되는 동안 별도 thread에서 viewer가 떠서 정책이 점점 똑똑해지는 모습이 실시간으로 보임**.

```bash
cd /home/yoo/RL_robot/sim
python3 train.py                       # 기본 50만 step + viewer
python3 train.py --steps 20000 --tag smoke   # 짧은 smoke run + viewer
python3 train.py --no-viewer           # viewer 없이 빠르게 학습만
```

**동작 방식:**
- 메인 thread: SAC 학습 (CPU+GPU full-throttle)
- viewer thread: 매 50Hz로 **현재까지 학습된 정책**을 evaluation env에서 rollout, viewer.sync()로 화면 갱신
- 정책 weights는 학습 thread가 매 step 업데이트, viewer thread는 그 최신 weights로 forward
- viewer 창을 닫아도 학습은 계속됨. 학습이 끝나면 viewer thread도 자동 종료

**콘솔에 2초마다 출력**: `[viewer] ep3 step=147 dist=0.241 reward=2.31` 같은 식으로 viewer 안에서 진행 중인 에피소드 상태를 알려줌.

**학습 완료 후**: 자동으로 최종 정책의 무한 rollout viewer가 다시 뜸 (`--no-viewer`이면 생략). 사용자가 창 닫거나 Ctrl+C 칠 때까지.

### 6.2 학습 끝난 정책만 보기

`view_policy.py`로 저장된 모델을 언제든 다시 불러와 viewer로 볼 수 있음.

```bash
python3 view_policy.py                       # 기본: runs/smoke/model.zip
python3 view_policy.py runs/sac/model.zip    # 다른 태그의 모델
python3 view_policy.py --sine                # 정책 대신 2Hz 사인파
python3 view_policy.py --zero                # ctrl=0 정지 상태
```

### 6.3 모델만 잠깐 보기 (학습 무관)

```bash
python3 -m mujoco.viewer --mjcf=/home/yoo/RL_robot/sim/rl_fish.xml
```

ctrl=0이라 가만히 있지만, 좌측 `Control` 탭의 슬라이더로 꼬리 직접 조작 가능. 모델 형상 검수용.

### 6.4 viewer 조작

| 조작 | 효과 |
|---|---|
| 좌클릭 드래그 | 카메라 회전 |
| 우클릭 드래그 | 카메라 평행 이동 |
| 휠 | 줌 |
| Space | 시뮬 일시정지/재개 |
| Tab | 좌측 사이드바 토글 |
| Control 탭 슬라이더 | ctrl 직접 조작 (수동 제어) |
| 창 닫기 | viewer 종료 (학습 thread는 영향 없음) |

---

## 7. 디렉토리 정리

```
RL_robot/
├── PROGRESS.md                   # ← 이 문서
├── CLAUDE.md                     # 미래 Claude 세션용 가이드
├── robotic_fish.xml              # 레퍼런스 (dm_control fish 사본)
├── common/                       # robotic_fish.xml이 include하는 공용 에셋
├── fish_urdf/RL_SIM_MODE_2_description/
│   ├── urdf/RL_SIM_MODE_2.urdf   # 사용자 CAD 원본
│   └── meshes/                   # base_link.stl, tail_link_1_1.stl
└── sim/                          # ← RL 작업 공간
    ├── rl_fish.xml               # 학습용 MJCF (URDF 변환 + 보강)
    ├── fish_env.py               # Gymnasium 환경
    ├── test_env.py               # 환경 smoke test
    ├── train.py                  # SAC 학습 스크립트 (viewer 통합)
    ├── view_policy.py            # 저장된 정책을 viewer로 보기
    └── runs/<태그>/              # 학습 산출물 (model.zip + tensorboard)
```

---

## 8. 다음 단계 — 결정해야 할 것

1. **보상함수 개선** — dm_control fish의 `tolerance` 함수 스타일로 [0,1] 보상으로 교체 (학습 효율↑)
2. **본 학습** — 50만~100만 step (30분~1시간). viewer로 진화 관찰 + tensorboard로 수치 모니터링
3. **추진력 보강** — gear, density 조정 (간략 우선과 약간 상충)

추천 순서: **보상함수 개선 → 본 학습 (viewer로 관찰)**.
