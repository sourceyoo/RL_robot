# KUFIsh_III — 단일 관절 물고기 로봇 RL 실행 가이드

단일 관절 물고기 로봇(KUFIsh_III)의 MuJoCo + SAC 강화학습 코드를 **처음부터 끝까지 실행**하기 위한 가이드. 환경 준비 → 설치 → 학습 → 평가 → 시각화 → 진단 → 백업 순서로 정리한다.

> 이 문서는 코드를 받은 사람이 그대로 재현할 수 있도록 하는 실행 매뉴얼이다.
> 기준 브랜치: **`m4_cpg`** (CPG 기반 — 정책이 action 3D `[freq, amp, offset]`로 꼬리 진동을 합성).

---

## 1. 실행 환경 (검증된 구성)

| 항목 | 사용 값 | 비고 |
|---|---|---|
| OS | Ubuntu 22.04 (Linux 6.8) | macOS/Windows도 MuJoCo 3.x 지원하나 미검증 |
| GPU | NVIDIA RTX 3090 24GB | SAC 학습용. CPU도 가능하나 수 배 느림 |
| GPU 드라이버 | 595.71.05 | CUDA 13 런타임 호환 |
| Python | 3.10.12 (시스템 Python) | 3.10 권장. 가상환경 사용 권장 |

- **DISPLAY 없는 SSH 환경**: 학습·평가·진단은 모두 headless로 동작 (`--no-viewer`, matplotlib `Agg`). viewer(GUI) 기능만 디스플레이 필요.
- GPU 없이 돌릴 경우: 학습/평가에 `--device cpu` 지정 (아래 참조).

---

## 2. 필요한 툴·라이브러리

### 2.1 시스템 툴

| 툴 | 용도 |
|---|---|
| `git` | 코드 클론 |
| `python3` (3.10) + `pip` | 실행·패키지 설치 |
| NVIDIA 드라이버 + CUDA | GPU 학습 (선택) |
| `tmux` | SSH 장시간 학습 세션 유지 (권장) |
| `gh` (GitHub CLI) | 모델 백업/복원 release (선택) |

### 2.2 Python 라이브러리 (검증된 버전)

| 패키지 | 검증 버전 | 역할 |
|---|---|---|
| `mujoco` | 3.8.0 | 물리 시뮬레이터 + 내장 viewer |
| `stable-baselines3` | 2.8.0 | SAC 알고리즘 |
| `gymnasium` | 1.2.3 | 환경 인터페이스 (`FishSwimEnv`) |
| `torch` | 2.11.0+cu130 | SAC 신경망 백엔드 (CUDA 13) |
| `numpy` | 2.2.6 | 수치 연산 |
| `matplotlib` | (시스템) | 학습 곡선·trajectory plot |
| `tensorboard` | (sb3 의존) | 학습 곡선 실시간 모니터 |
| `pillow` (PIL) | (시스템) | 진단 스크립트 이미지 저장 |

> 표준 라이브러리(`argparse`, `pathlib`, `threading`, `subprocess`, `struct`, `datetime` 등)는 별도 설치 불필요.

---

## 3. 설치 절차

```bash
# 1) 코드 클론
git clone https://github.com/sourceyoo/RL_robot.git
cd RL_robot
git checkout m4_cpg             # CPG 기반 브랜치

# 2) (권장) 가상환경
python3 -m venv .venv
source .venv/bin/activate

# 3) GPU(CUDA 13)용 PyTorch — GPU 사용 시
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
#   CPU만 사용할 경우:
#   pip install torch==2.11.0

# 4) 나머지 패키지
pip install mujoco==3.8.0 stable-baselines3==2.8.0 gymnasium==1.2.3 \
            numpy==2.2.6 matplotlib tensorboard pillow
```

> 이 리포에는 `requirements.txt`가 없다. 위 명령이 사실상의 의존성 목록이다.

### 3.1 설치 검증

```bash
cd sim
python3 -c "import mujoco, stable_baselines3, gymnasium, torch; \
print('mujoco', mujoco.__version__, '| sb3', stable_baselines3.__version__, \
'| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# 환경 smoke test (모델 로드 + 한 에피소드)
python3 test_env.py

# MuJoCo 모델 자체 검수 (GUI 필요)
python3 -m mujoco.viewer --mjcf=rl_fish.xml
```

`torch.cuda.is_available()` 가 `True`면 GPU 학습 준비 완료.

---

## 4. 학습 실행 (`curriculum.py`)

학습 진입점은 `sim/curriculum.py`. **단일 stage 단위로만** 실행한다 (자동 다음 진행 금지 — 다음 stage는 사용자가 결정).

### 4.1 기본 사용법

```bash
cd sim
python3 curriculum.py --start-stage N --end-stage N --no-viewer
```

### 4.2 Curriculum stage 정의

| stage | tag | 목표 heading θ | 설명 |
|---|---|---|---|
| 1 | s1_forward | π 고정 | 직진 추진 |
| 2 | s3a_arc15 | π ± 15° | 완만한 호 선회 |
| 3 | s3b_arc30 | π ± 30° | |
| 4 | s3c_arc60 | π ± 60° | |
| 5 | s3d_arc90 | π ± 90° | 큰 각 선회 |

> 추진 방향 규약: **world −x = 머리(전진)**.

### 4.3 주요 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--start-stage` / `--end-stage` | 1 / None | 학습할 stage 범위. **단일 stage는 둘을 같게** |
| `--no-viewer` | off | GUI 끄기 (SSH·서버 필수) |
| `--device` | `cuda` | `cpu` 지정 시 GPU 없이 학습 |
| `--seed` | None | 재현용 시드 (multi-seed 비교 시 0/1/2) |
| `--tb-tag` | `m4_v1` | TensorBoard 로그 태그 |
| `--runs-subdir` | None | 산출물 하위 디렉토리 (seed별 분리) |
| `--init-from` | None | 이전 stage `model.zip` 에서 warm-start |
| `--threshold` | 0.9 | stage 졸업 reach_rate (stochastic) |
| `--det-episodes` | 100 | 졸업 후 deterministic eval 에피소드 수 |
| `--min-steps` | 30,000 | 졸업 판정 전 최소 학습 step |
| `--max-steps` | None | 안전장치 (도달 실패 시 상한). **임의 축소 금지** |

### 4.4 졸업(stage 종료) 규칙

`CurriculumStopCallback`이 TB stochastic 100 ep `reach_rate ≥ 90%`를 trigger한 뒤, deterministic eval 100 ep을 통과해야 stage 종료. (stochastic 정점의 false positive 차단용 2단 게이트.)

### 4.5 multi-seed 실행 예

```bash
# seed 0 으로 stage 1 학습, seed별 nested 디렉토리에 저장
python3 curriculum.py --start-stage 1 --end-stage 1 \
    --seed 0 --tb-tag m4_cpg_seed0 --runs-subdir m4_cpg/seed0 --no-viewer
```

> seed 산출물은 `<card>/seed{N}/` nested 구조 (runs·plots·tb_logs 셋 다)로 정리한다.

### 4.6 SSH 장시간 학습 (tmux 권장)

```bash
tmux new -s train
cd sim
python3 curriculum.py --start-stage 1 --end-stage 1 --no-viewer 2>&1 | tee train_s1.log
# Ctrl+b d 로 detach, 재접속 후 `tmux attach -t train`
```

### 4.7 학습 시간 참고 (RTX 3090)

대부분의 wall-clock은 SAC update + Python overhead(98%). frame_skip=42 (12Hz 결정) 기준 stage별 수 분~수십 분. 정확한 실측은 `docs/training_log_m4.md` 참조.

---

## 5. 평가 실행 (`eval_stages.py`)

학습 **전·후 deterministic eval은 필수**다. (사전: 진짜 미달 stage 확정 / 사후: 학습 stage 재확인 + 이전 stage forgetting 점검.)

```bash
cd sim
python3 eval_stages.py runs/<tag>/model.zip
```

### 주요 인자

| 인자 | 기본값 | 의미 |
|---|---|---|
| `model` (위치 인자) | — | 평가할 `model.zip` 경로 |
| `--episodes` | 100 | stage당 평가 에피소드 |
| `--seed` | 42 | 평가 시드 |
| `--plot-episodes` | 20 | trajectory plot에 그릴 에피소드 수 |
| `--only-stage` | None | 특정 stage만 평가 |
| `--include-future` | off | 학습 stage **이후** stage까지 평가 (사용자 명시 시만) |

> 기본 동작: 학습 stage까지만 평가. catastrophic forgetting 점검을 위해 이전 stage 전부 포함.
> 평가 결과는 mean뿐 아니라 **σ(표준편차)도** 함께 확인 — mean ≥ 90%여도 σ가 크면 안정 졸업 아님.

---

## 6. 시각화

### 6.1 정책 viewer (GUI)

```bash
cd sim
python3 view_policy.py runs/<tag>/model.zip
```

### 6.2 TensorBoard (학습 곡선)

```bash
cd sim
tensorboard --logdir tb_logs/
# SSH면 포트포워딩: ssh -L 6006:localhost:6006 user@host
```

주요 지표:
- `rollout/ep_rew_mean`, `ep_len_mean`
- `train/{critic_loss, actor_loss, ent_coef}`
- `fish/{success_rate, final_distance, episode_seconds, avg_align}`

진단 신호: `ent_coef`가 5만 step 안에 0.001↓ → 탐색 부족 / `ep_len_mean`=max → 미도달 / `avg_align` 높고 `success` 낮음 → "정렬만" mode.

> 생성하는 모든 이미지·plot은 `/home/yoo/RL_robot/images/` 아래에 저장한다 (임시 경로 금지).

---

## 7. 진단 스크립트 (`sim/diagnostics/`)

`rl_fish.xml`만 의존하는 독립 진단 도구. 호출 형식:

```bash
cd sim
python3 diagnostics/<name>.py
```

m4_cpg 핵심 진단:

| 스크립트 | 용도 |
|---|---|
| `freq_sweep.py` | 꼬리 sine 주파수 스윕 → 추진 방향 검증 (−x 전진 확인) |
| `yaw_test.py` | 회전 능력(비대칭 패턴) 진단 |
| `sf_asym_yaw_test.py` | 비대칭 duty 패턴 yaw 진단 |
| `measure_speed.py` | 전진 속도 측정 |

> 선회 진단 시: 숫자(방향 cos) 이전에 **trajectory를 먼저 확인**. 작은 각에서는 cos가 함정이 됨.

---

## 8. 모델 백업 / 복원 (선택)

```bash
# 백업 (GitHub release)
cd sim
tar -czf runs-models-vN.tar.gz runs/
gh release create models-vN runs-models-vN.tar.gz

# 복원
gh release download models-vN -p '*.tar.gz'
tar -xzf runs-models-vN.tar.gz -C sim/
```

---

## 9. 표준 워크플로 요약

```
1. 설치·검증        → python3 test_env.py
2. 사전 eval        → python3 eval_stages.py <이전 model.zip>   (미달 stage 확정)
3. 단일 stage 학습  → python3 curriculum.py --start-stage N --end-stage N --no-viewer
4. 모니터           → tensorboard --logdir tb_logs/
5. 사후 eval        → python3 eval_stages.py runs/<tag>/model.zip  (졸업 + forgetting 점검)
6. 시각화           → python3 view_policy.py runs/<tag>/model.zip
7. (사용자 결정)    → 다음 stage N+1 로 이동
```

---

## 10. 자주 막히는 지점 (트러블슈팅)

| 증상 | 원인 / 해결 |
|---|---|
| `torch.cuda.is_available()` False | CUDA 드라이버 ↔ torch cu 버전 불일치. cu130 휠 재설치 또는 `--device cpu` |
| viewer 실행 시 디스플레이 에러 | SSH headless. 학습·평가는 `--no-viewer`로, viewer는 로컬 GUI에서만 |
| matplotlib Axes3D 경고 | 시스템/pip matplotlib 중복. 동작에는 무해 (무시 가능) |
| 학습이 +x로 전진 | 환경 인덱스 오류. 규약은 **world −x = 전진** |
| stage가 max_steps만 채우고 안 끝남 | 도달 실패 신호 = 카드(보상·탐색) 부족. max_steps 축소 금지, 보상·탐색 재설계 |

---

> 물리·모델 수정 금지 규칙, 보상 설계, 학습 history 등 더 깊은 맥락은 [`CLAUDE.md`](CLAUDE.md)와 [`docs/training_log_m4.md`](docs/training_log_m4.md) 참조.
