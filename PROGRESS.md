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

## 8. 후속 진행 — 추진 방향 문제와 시도들

> 인프라가 도는 것을 확인한 뒤, **본 학습 전에 발견한 근본 문제**와 그 해결을 위해 시도한 것들 정리.

### 8.1 GitHub 업로드 (해결)

- `gh` CLI 설치 → `gh auth login` (브라우저) → repo 생성 + push 완료
- 의미 있었음: 각 단계(`gitignore` → `commit` → `repo create` → `push`)를 분리해서 진행. 이후 `CLAUDE.md`에 **단계별 진행 규칙**으로 명문화 (사용자 피드백: "테스크가 여러개일 때는 하나씩 분리되어 진행").

### 8.2 학습 중 viewer race condition (해결)

- 첫 `train.py`가 viewer thread에서 `model.predict`를 호출하면서 학습 thread와 PyTorch 자원을 공유 → 가끔 cuda assertion 폭주.
- **해결 (의미 있었음): `PolicySnapshotCallback`** — 학습 thread가 500 step마다 정책을 `deepcopy().cpu().eval()`로 스냅샷, viewer thread는 그 스냅샷만 forward. race-free.

### 8.3 본 문제: 꼬리 방향(-x)으로 전진 (미해결)

> 꼬리를 흔들면 머리(+x) 쪽으로 가야 하는데 정책 학습 결과 꼬리(-x) 쪽으로 슬슬 밀려감.
> 이게 **핵심 블로커**. 본 학습 들어가기 전에 해결 필요.

#### 시도한 것들과 의미

| # | 시도 | 결과 | 의미 |
|---|---|---|---|
| 1 | 모델을 x축 180° 회전 (`euler="3.14159 0 0"`) | 시각상 정상 | △ 시각만 정렬, 추진 방향은 그대로 |
| 2 | density 5000 → **1000** (실제 물) | 추진력은 강해짐 | △ 강도만 증가, 방향은 여전히 -x |
| 3 | 꼬리 주파수 6Hz, 진폭 ±20° 튜닝 | sine wave에서 좌우 wag 잘 됨 | ○ 꼬리 동작 자체는 의도대로 |
| 4 | `motor` → `position` 액추에이터 (kp=50, gear=0.349) | 명령 추종성 향상 | ○ 학습 안정성에 기여 |
| 5 | passive `tail_fin` 추가 (Ecoflex 00-30 실리콘 ellipsoid) | 꼬리 끝이 위상차로 흔들림 | ○ 모델 충실도 ↑, 추진 방향엔 무관 |
| 6 | 사각 모터(BL4260) → 강체 가정으로 단순화 | 모델 정합성 ↑ | ○ 사용자 피드백 반영 ("실리콘은 fin만") |
| 7 | MATLAB CG.m 값 적용 (`pos="0 0 0.004535"`, `xG=xb=0`) | G/B 정렬됨 | ◎ **실물 모델 G/B**라 의미 큼. xG=xb=0이 안정 핵심 |
| 8 | off-diagonal 관성(ixz=0)로 정리 | — | ○ 사후 정당성 확보. 원본 CAD는 임의 재질로 질량 맞춘 거라 ixz가 인공물이었음 |
| 9 | F1: STEP 파일 분해 → 다중 fluid ellipsoid | **z축으로 +0.56m 발산** | ✗ 롤백. fluidshape 다중화는 비대칭 lift 폭주 |
| 10 | Zhong et al. 2026 논문(ANN data-driven) 검토 | 학계 표준 해법 확인 | ○ 추후 옵션. 실물 motion capture 필요해 지금은 부담 |

**결론:** 1~8은 모델 충실도·안정성을 올렸지만 **추진 방향 부호는 못 뒤집었음**. 이는 MuJoCo `fluidshape="ellipsoid"`의 본질적 한계 (전후 대칭 ellipsoid에서 비대칭 추진력 부호가 환경 의존적).

#### 가장 의미 있었던 것 (요약)

1. **MATLAB CG.m 값 적용** (#7) — `xG=xb=0`은 사용자가 실물 모델 안정성을 위해 설계한 값. 이걸 시뮬에 박은 것은 sim2real 정합성에서 가장 큰 의미.
2. **PolicySnapshotCallback** (8.2) — race-free viewer가 가능해지면서 학습 진화 관찰이 신뢰성 있게 작동.
3. **단계 분리 규칙 명문화** (8.1) — 작업 흐름 자체가 안정됨.
4. **off-diag 관성 정당성** (#8) — "임의 재질로 질량 맞춤"이라는 원본 CAD 사실 때문에 off-diag는 인공물 → 0 처리가 맞다는 사후 확증.

#### 의미가 적었던 것

- density·주파수·진폭 단순 튜닝 (#2, #3): 강도만 변할 뿐 방향 부호 문제와 무관.
- F1 다중 ellipsoid 분해 (#9): 더 정교해 보이지만 fluid 안정성 깨져 즉시 롤백.

### 8.4 현재 모델 상태 (`sim/rl_fish.xml`)

- density=1000 (실제 물), viscosity=0.001, gravity 끔
- 꼬리: position 액추에이터(±20°, 6Hz 가능) + passive Ecoflex fin
- 관성: 실물 G/B (`pos="0 0 0.004535"`, off-diag=0)
- 학습 파이프라인은 그대로 동작. **단, 학습 시 정책이 꼬리(-x) 방향으로 진행하는 정책을 찾음.**

---

## 9. 추가 진단 — 주파수 스윕, fin 비대칭, waveform 테스트

> 본 학습 진입 전, "어떤 운전 조건이라면 +x로 갈 수 있는지" 체계적으로 검증.

### 9.1 주파수 스윕 (sine wave, ctrl=±1)

`sim/freq_sweep.py`로 0.25~6Hz 인가, 꼬리/fin 도달 각도 + body 변위 측정.

| 발견 | 데이터 |
|---|---|
| **모든 주파수에서 +x로만 감** (저주파라도 풀리지 않음) | 0.25Hz: +0.10m, 6Hz: +1.37m |
| **저주파(≤2.5Hz)에서 꼬리 ±24°로 오버슈트** (PD 제어 + 유체 외란) | 명령 ±20° → 실제 ±24° |
| **고주파(≥4Hz)에서 꼬리 진폭 미달** | 6Hz에서 명령 ±20° → 실제 ±9° |
| **passive fin이 freq별로 한쪽 쏠림** (비대칭 작동) | 0.5Hz: +35°/-1°, 3Hz: -10°/-44° |

→ 사용자 가설("주파수를 낮추면 +x로 풀릴까?") **부정**.

### 9.2 fin 비대칭의 원인 진단

> "passive fin이 한쪽으로만 휘는" 현상이 추진 방향 문제와 연결되어 있는지 추적.

**진단 1**: tail_link 관성 대칭화 (`y CoM=0`, off-diag quat 제거 = identity)
- 결과: fin 비대칭 거의 그대로. 의미 △.
- 다만 *원본 CAD가 임의 재질로 질량 맞춘 결과*라 off-diag는 어차피 인공물 → 정당성 확보 ◎

**진단 2**: fin을 강체화 (stiffness 4e-7 → 1.0)
- fin은 0°에 고정, body 추진 거리는 **소수점 셋째자리까지 동일** (+1.384m @ 6Hz)
- → **fin 유연성은 추진력에 기여하지 않았음.** 강체화로 잃은 것 없이 fin 대칭만 얻음.

**진단 3**: body free vs body locked vs roll/pitch만 잠금 (`freq_sweep_locked.py`, `freq_sweep_norollpitch.py`)
- body 자유: fin 비대칭 강함
- body 완전 고정: 고주파에선 fin 대칭(±70~93°), 저주파는 여전히 비대칭
- roll/pitch만 잠금: 자유와 거의 동일 → **roll/pitch는 주범 아님**
- → 비대칭의 본질 = **유체 자체의 nonlinear DC bias** (모델 한계). 어떤 stiffness/damping 튜닝으로도 못 풀음.

**적용된 변경**: `tail_fin_joint stiffness 4e-7 → 1.0` (강체화). 
- fin은 tail에 강체로 부착되어 함께 회전. World frame에서 fin 각도 = tail 각도.
- 이로써 "fin을 ±20° 대칭으로 움직이게 하라"는 요구를 **3Hz 운전 시 자연스럽게 달성** (tail이 자연스럽게 ±20° 대칭).

### 9.3 Waveform 테스트 — 어떤 패턴이 +x로 갈까?

`sim/waveform_test.py`로 3Hz에서 14가지 waveform 인가, 부호 검증.

| 카테고리 | 패턴 | x_disp 부호 |
|---|---|---|
| Baseline | 대칭 sine, square wave | + (후진) |
| DC offset | ±0.3, ±0.6 | + (후진) |
| 2nd harmonic | sin±0.3·sin(2x), quadrature | + (후진) |
| 비대칭 stroke | fast+/slow-, fast-/slow+ | + (후진) |
| Burst-and-coast | 0.5초 진동 + 0.5초 정지 | + (후진, 거리 감소) |
| 정지 ref | ctrl=0 | 0 |

**14가지 모두 +x.** DC offset 부호를 뒤집어도 y만 반대로 갈 뿐 x는 동일. **단일 관절로는 어떤 waveform이든 -x 추진 불가능.**

→ CPG/Fourier 액션공간으로 SAC 학습해도 **학습할 +x 패턴이 애초에 존재하지 않음.** 시나리오 (b) 확정.

### 9.4 학계 사례 조사 — 우리 진단 검증

웹 검색으로 비슷한 사례 + 우리 문제와의 비교 정리.

| 출처 | 우리 문제와의 관련성 |
|---|---|
| **MDPI Sensors 2026 종합 review** | *"passive fin can hijack control"*, *"simplified hydrodynamic models cannot capture vortex shedding"* — **우리가 본 현상을 학계가 그대로 인정** |
| **DeepMind dm_control fish** | 같은 ellipsoid fluid model. 단 다중 관절(꼬리 2단 + 가슴지느러미 4 DOF). **단일 관절은 dm_control에도 없음** |
| **FishSim (ETH-SRL)** | MuJoCo + 다중 관절 tendon-driven + system identification. 실측 마커로 5개 fluid coef 역최적화 |
| **Bio-mimetic Fish E2E DRL (arXiv 2506)** | 3-link + 2D CFD + sine wave **pretraining 후** RL 미세조정 ← 정공법 |
| **Zhong et al. 2026 (Ocean Eng)** | ANN surrogate로 dynamics 학습. MuJoCo 한계 우회. 단 실물 motion capture 필요 |

→ 학계 컨센서스: **single-joint는 wave-thrust 학습이 본질적으로 어려움**, multi-joint가 표준.

---

## 10. 결정해야 할 것 — 진짜 갈림길

진단 결과 단일 관절 + ellipsoid fluid model 조합으로는 **어떤 ctrl 패턴으로도 +x 추진 불가능**. 모델 변경이 필수.

| 길 | 작업량 | 의미 | sim2real |
|---|---|---|---|
| **A. 꼬리 2~3 마디 분절** | 중 | 학계 표준. wave-thrust 학습 가능해짐 | 강 |
| **B. Mesh x-mirror** | 소 | 시각만 뒤집기. "후진"이 시각상 "전진" | 약 |
| **C. 보상함수에서 +x 강제** | 소 | 학습 편법. 정책이 비물리적 +x 정책 찾을 가능성 | 약 |
| **D. ANN surrogate (Zhong)** | 대 | 정공법. 실물 motion capture 데이터 필요 | 강 |

**추천 = A.** 사용자 직관(유연한 꼬리가 wave 만들어 추진)과 학계 권고가 일치. 진행 시:
1. tail_link를 2~3 segment로 분절 (각 마디에 active hinge)
2. 액션공간 1D → 2~3D
3. fish_env.py 수정
4. Sine wave traveling pattern으로 빠른 검증 (각 segment에 phase 차이 둔 sine)
5. 검증 후 SAC 학습

---

## 11. 현재 모델 상태 (`sim/rl_fish.xml`)

- density=1000, viscosity=0.001, gravity 끔
- 꼬리: position 액추에이터 (±20° 명령, kp=50, gear=0.349, forcerange ±3)
- **fin 강체화** (stiffness=1.0, damping=0.001) — passive Ecoflex 가정 포기
- 관성 대칭화: tail_link `y CoM=0`, off-diag quat 제거
- base_link 관성: 실물 G (`pos="0 0 0.004535"`, off-diag=0)
- 학습 파이프라인 정상 동작. **단, 정책이 꼬리(-x = world +x) 방향 진행 정책을 학습.**

## 12. 진단 스크립트 일람 (`sim/`)

- `freq_sweep.py` — 주파수 스윕 + body roll/pitch/yaw 로깅
- `freq_sweep_locked.py` — body 강제 고정 (qvel[0:6]=0)
- `freq_sweep_norollpitch.py` — roll/pitch만 잠금
- `waveform_test.py` — 14가지 waveform 부호 검증
- `multiseg_test.py` — 다단 passive fin 응답

---

## 13. 돌파 — MODE_3 모델 + 3DOF planar로 +x 추진 달성 (2026-05-02)

> 단일 관절 한계 진단 후, **CAD 정확성 + 운동 차원 축소**로 +x 추진 부호 뒤집기 성공.

### 13.1 MODE_3 CAD 모델 도입 (`fish_urdf/RL_SIM_MODE_3_description/`)

이전(MODE_2)에는 fin이 CAD에 없어 우리가 placeholder ellipsoid로 임의 추가. MODE_3 도입으로 fin이 정식 메쉬(`fin_1.stl`)로 분리됨.

| body | mass | 출처 | 비고 |
|---|---|---|---|
| base_link | 2.4349 kg | MATLAB CG.m (그대로 유지) | PLA, CoM=(0,0,0.004535), 대칭 inertia |
| tail_link | **0.0699 kg** | MODE_3 새 사양 (fin 분리됨) | PLA, CoM=(0.0511,0,-0.005349), 대칭화 |
| fin_1 | 0.01052 kg | MODE_3 CAD | **Ecoflex 00-30 (density=1070)**, passive hinge |

총 질량 2.515 kg. 시뮬 NaN 없음.

### 13.2 BL4260 + 평기어 트레인 모터 사양 적용

```xml
<position name="tail" joint="tail_joint" ctrlrange="-1 1" gear="0.349"
          kp="100" kv="5" forcerange="-3 3"/>
```

- Peak 토크 ~3 Nm (BL4260 connect + 기어 감속 후 continuous 영역)
- kp/kv는 ±20° 추종 + 오버슈트 억제 균형
- 이전 시도(kp=200, ±6 Nm)는 tail이 ±44°까지 오버슈트했으나 현재는 ±21~23°로 안정.

### 13.3 6DOF freejoint → 3DOF planar 결정적 변경

문제: yaw 진동이 cross-coupling으로 pitch 자세 변화 유발 (최대 +67° → 물고기가 옆으로 누움). 모든 파라미터 조정해도 본질적으로 안 사라짐.

**해법 = freejoint 제거 + 3DOF 명시 joint**:

```xml
<joint name="root_x"   type="slide" axis="1 0 0"/>      <!-- world x -->
<joint name="root_y"   type="slide" axis="0 -1 0"/>     <!-- world y -->
<joint name="root_yaw" type="hinge" axis="0 0 -1"/>     <!-- world yaw -->
```

- z, roll, pitch는 *수학적으로* 잠김 (constraint solver 강제)
- 표면 영법 가정 (BCF surface swimming) — fish RL 학계 표준
- qpos 차원: 9 → 5 (학습 효율 ↑)

### 13.4 추진 방향 부호 뒤집기 — 압도적 결과

`freq_sweep.py`로 1~6Hz sine wave 인가:

| f[Hz] | tail | fin | x_disp | y_disp | yaw | 방향 |
|---|---|---|---|---|---|---|
| 0.5 | ±21° | ±31° | +0.85 | 0 | 0 | 후진 |
| **1.0** | ±21° | ±32° | **−0.49** | 0 | 0 | **전진 ✓** |
| **2.0** | ±22° | ±33° | **−1.72** | −0.04 | +2° | **전진 ✓** |
| **3.0** | ±23° | ±31° | **−2.17** | −0.14 | +6° | **전진 ✓** |
| **4.0** | ±23° | ±29° | **−2.66** | −0.18 | +6° | **전진 ✓** |
| **5.0** | ±21° | ±28° | **−3.00** | −0.43 | +12° | **전진 ✓** |
| **6.0** | ±23° | ±28° | **−3.36** | −0.58 | +14° | **전진 ✓** |

(12초 측정. x_disp < 0 = 머리 방향 = 전진)

- **1~6 Hz 모든 주파수에서 +x 머리 방향 전진** ✓
- **6 Hz: 0.28 m/s** 전진 — 실제 소형 robotic fish 영역 (0.1~0.5 m/s)
- tail/fin 진폭 정상 (±21~33°)

### 13.5 결정적이었던 변경점 (회고)

| 변경 | 효과 | 비고 |
|---|---|---|
| **CAD MODE_3 fin 사양 적용** | fin 형상/관성 정확 | placeholder → 실 사양 |
| **6DOF → 3DOF planar** | pitch wobble 제거, 부호 뒤집힘 | 학계 표준 가정 |
| BL4260 + 기어 모터 사양 | tail 추종성 | continuous 영역 |
| Ecoflex 사양 (density=1070) | wave-thrust 발생 | 강체 fin과 차별화 |
| MATLAB CG.m 값 유지 | 안정성 | xG=xb=0 |

가장 본질적이었던 것은 **3DOF planar 가정**. 단일 관절 + ellipsoid fluid model의 부호 문제는 6DOF에선 안 풀렸지만 3DOF planar에선 자연스럽게 +x로 정렬됨.

### 13.6 잔여 이슈 (학습으로 보정 가능)

- yaw 누적: 6Hz에서 +14° (fluid asymmetry 잔여) — RL 정책이 좌우 균형 학습으로 보정
- y_disp 누적: 동일 메커니즘
- 0.5 Hz는 후진 — 저주파에선 wave-thrust 약하므로 학습이 1~6Hz 영역 선호하도록 자연 학습

### 13.7 환경 갱신 + 본 학습

- `fish_env.py`를 5-DOF qpos layout으로 갱신 (obs 11D: tail/fin qpos·qvel + world v + sin/cos yaw + target rel)
- SAC smoke test 통과 (20k step, ep_rew_mean -23 → -16)
- 본 학습 500k step (random target, tag `mode3-planar`):
  - ep_rew_mean: -20 → +1.78
  - 일부 방향 도달 가능 (ep351 reward 3.56), 일부 방향 실패 (ep350 reward -0.73)
  - 도달 보너스(+5)엔 못 미침 → 모든 방향 일관 정복 안 됨
  - **단일 모터로 random target 전체는 어려움** (180° 회전 필요한 방향이 학습 어려움)

---

## 14. Curriculum 학습 계획 (2026-05-02)

> Random target full circle은 단일 모터에 너무 어려움. **단계적 학습**으로 분리.

### 사용자 의도 (원어)

> "나는 학습을 단계적으로 진행하고 싶어. 먼저 직진 학습을 하고, 직진 목표 지점 위에 안착하는 게 다음 단계, 그리고 목표 지점을 물고기 머리쪽에만 세워 좌우 목표 지점 도달, 이렇게 단계적 학습을 했으면 해."

> "180도 회전 후 추적은 제외."

> "각 학습률이 90% 이상 도달할 때마다 다음 단계로 넘어가도록."

이 의도가 **fixed step 수가 아닌 reach_rate ≥ 90% 자동 진행** 방식의 근거. Stage 4 (full circle)는 사용자 명시적 제외 결정.

### 14.1 단계 설계 — 자동 진행 (도달률 90% 임계)

각 단계는 **고정 step 수가 아니라 reach_rate ≥ 90%에 도달하면 자동 종료**, 다음 단계로. max_steps는 안전장치(도달 실패 시 강제 진행).

| 단계 | 목표 위치 (theta 범위) | success_radius | 시작점 | 학습 내용 | max_steps |
|---|---|---|---|---|---|
| **1** | π fixed (정확히 머리 앞) | 0.08m | fresh | 전진 추진 | 200k |
| **2** | π fixed (동일) | **0.04m** (축소) | Stage 1 정책 | 정밀 안착 | 200k |
| **3** | [π/2, 3π/2] (전방 180°) | 0.08m | Stage 2 정책 | yaw 정렬 + 추적 | 300k |

### 14.2 제외된 단계 — 사용자 결정

**Stage 4 (full circle θ ∈ [-π, π])는 보류**. 180° 회전 후 추적은 단일 모터에 본질적으로 어려우며, sim2real에서도 실용성 낮음 (실물 fish 로봇은 보통 후진/완전 반전 안 함).

대신 baseline `mode3-planar` 모델(random 500k)이 비교 기준으로 보존됨.

### 14.3 구현

**`fish_env.py`** — `target_theta_range`, `success_radius` 인자 추가:
```python
def __init__(self, ..., target_theta_range=(-np.pi, np.pi), success_radius=0.08):
    self.target_theta_range = target_theta_range
    self.success_radius = success_radius

def reset(self, ...):
    theta = rng.uniform(*self.target_theta_range)
    ...
```

**`train.py`** — curriculum CLI 추가:
- `--theta-min`, `--theta-max`: 목표 각도 범위 (라디안)
- `--success-radius`: 도달 판정 거리
- `--init-from <model.zip>`: 이전 단계 정책에서 fine-tuning 시작
- `--success-threshold`: 이 비율 이상 도달 시 학습 조기 종료 (curriculum)
- `--eval-window`: reach_rate 측정용 최근 에피소드 수 (기본 100)
- `--check-every`: 체크 주기 step (기본 5000)

내부에 `CurriculumStopCallback` (rolling window로 reach_rate 추적, threshold 도달 시 `_on_step`이 False 반환).

**`curriculum.py`** — 모든 단계 자동 순차 실행. 각 단계가 90% 도달하면 다음으로.

### 14.4 실행 명령

```bash
# 자동 (모든 단계 순차)
python3 curriculum.py                    # viewer + 자동 진행
python3 curriculum.py --no-viewer        # viewer 없이 빠르게
python3 curriculum.py --threshold 0.85   # 85%로 임계 완화
python3 curriculum.py --start-stage 2    # Stage 2부터 (이전 모델 있어야)

# 수동 (개별 단계)
python3 train.py --tag s1_forward --theta-min 3.14159 --theta-max 3.14159 \
    --success-radius 0.08 --success-threshold 0.9 --steps 200000

python3 train.py --tag s2_anchor --theta-min 3.14159 --theta-max 3.14159 \
    --success-radius 0.04 --success-threshold 0.9 --steps 200000 \
    --init-from runs/s1_forward/model.zip
```

### 14.5 자동 진행 메커니즘

`CurriculumStopCallback`이 매 step의 done 에피소드의 `info["reached"]`를 rolling deque(window=100)에 기록. `check_every` step마다 평균 reach_rate 계산:
- `reach_rate ≥ threshold (0.9)` → callback이 False 반환 → SB3 `model.learn` 종료
- 도달 못해도 `--steps` (max_steps)에 걸리면 강제 종료
- 학습 종료 시 model.zip 저장 → `curriculum.py`가 다음 단계의 `--init-from`으로 전달

### 14.6 다음 단계

이후 작업 (필요 시):
- yaw 보상 추가 (목표 방향과 heading 정렬에 reward)
- fin actuator 추가 (단일 모터 한계 극복)
- ANN surrogate (Lighthill 콜백 또는 Zhong 논문 방식)
