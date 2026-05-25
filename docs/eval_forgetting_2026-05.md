# Curriculum Stage 전환 시 Catastrophic Forgetting 평가

> **작성**: 2026-05 / **자매 문서**: `~/research/fish_rl/literature_curriculum_forgetting_fish_rl.md` (외부 literature)
> **선행**: `docs/diagnosis_stagnation_2026-05.md` (정지/전진 진단, 별개 task)
> **목적**: RL_robot 의 stage 전환 forgetting 현 mitigation 현황을 literature 기법과 매핑 + Tier 별 권장사항.

---

## 1. 현상 정의

사용자 보고: 이전 stage 통과 모델이 다음 stage 학습 진입 시 이전 stage 의 skill 을 잊음.

관측 metric 후보 (literature `Continual World §4.1` 정의 채택):
- `forgetting_i = max_t P_i(t) − P_i(T)` — stage i 의 peak reach_rate 와 학습 종료 시점 reach_rate 차이
- RL_robot 의 sub-trigger 메커니즘 (`curriculum.py:91-101, 162-177`) 이 이미 sub_id 별 reach_rate 추적 → forgetting 측정 인프라 일부 존재

---

## 2. RL_robot 현황 매핑

| 항목 | 현 상태 | 평가 | Path:line |
|---|---|---|---|
| **Checkpoint 이어가기** | `SAC.load(prev_model_path, env=env)` weight 그대로 fine-tune | ✅ 좋음 | `sim/curriculum.py:240-247` |
| **Replay buffer** | SB3 `SAC.load` 기본은 buffer 미로드 (별도 file 필요). stage k+1 학습은 빈 buffer + `learning_starts=1000` random | ⚠ **rehearsal 부재** | `sim/curriculum.py:240-258`, `sim/train.py:486` (`buffer_size=200_000`) |
| **Entropy floor** | stage 별 명시 (s1: 0.002, s3a: 0.002, s3b: 0.003→0 decay, s3c: 0.008, s3d: 0.010, s_all_mix: 0.005) + `EntCoefFloorCallback` 매 step `log_ent_coef` clamp | ✅ **이미 잘 구현** | `sim/curriculum.py:50-128 STAGES`, `sim/train.py:241-290 EntCoefFloorCallback`, `sim/curriculum.py:289-296` |
| **LR / γ / τ** | 모든 stage 동일 (`lr=3e-4, tau=0.005, gamma=0.99`). SAC.load 후 Adam optimizer state (m, v moment) 이어짐 | ⚠ **Adam moment 잔류** | `sim/curriculum.py:253-258` |
| **Obs space** | dim 13 + action_history N=20, stage 간 불변 (단 변경 시 `SAC.load Box mismatch — s1부터 새 학습`, `curriculum.py:46 comment 함정 #9`) | ✅ 호환 | `sim/curriculum.py:46`, `sim/fish_env.py` |
| **Reward function** | stage 간 동일 상수. `align_weight` 만 `episode_seconds ≤10 vs >10` 분기 (s1: 0.05, s3a-d: 0.05 — 사실상 일관) | ✅ 일관 | `sim/fish_env.py` (다른 task 의 진단 문서 참조) |
| **Stage progression** | s1 (직진 fixed) → s3a (\|θ-π\| 9.2~15°) → s3b (15~30°) → s3c (30~60°) → s3d (60~90°) → **s_all_mix (5 sub 균등 mixed sampling, sub trigger)** | ✅ 단조 + ✅ **m4_v20 mixed recovery stage 존재** | `sim/curriculum.py:50-128 STAGES` |
| **명시적 forgetting mitigation** | (1) `model_best.zip` 자동 저장 (`curriculum.py:280`, `train.py:103, 459` comment "v29 분석으로 후반 catastrophic forgetting 발견 → peak 모델 보존용") (2) deterministic eval 이중 trigger (`train.py:107-114, 193-211` 함정 #13) (3) **m4_v20 s_all_mix mixed sampling + sub trigger** (`curriculum.py:112-127`, `train.py:91-177`) | ⚠⚠ **사후 보존·완화만 있고 root cause (rehearsal·distillation·EWC) 없음** | 위 참조 |

---

## 3. Literature gap 매트릭스

| 기법 | Literature source | RL_robot 적용 여부 | Gap |
|---|---|---|---|
| **Replay buffer 50-50 mix** | Rolnick 2019 CLEAR (arXiv:1811.11682 Table 1 line 239) + Continual World Reservoir 변형 | ❌ 미적용 (SAC.load buffer 미로드) | **Tier 1 권장 — 가장 작은 변경** |
| **Stage 별 entropy floor** | (literature 1차 source 약함, RL_robot 경험 기반) | ✅ 적용 완료 `curriculum.py:50-128` | 없음 — literature 권장 1.2 는 redundant |
| **Mixed parallel sampling** | Rudin 2022 game-based curriculum (arXiv:2109.11978 line 199-227) | ✅ s_all_mix 적용 `curriculum.py:112-127` | 없음 — 단 첫 학습부터 mixed 적용 비교 가능 |
| **Best checkpoint 보존** | (curriculum 일반 wisdom, literature 1차 source 약함) | ✅ 적용 `curriculum.py:280` | 없음 — 단 사후 보존이라 root cause 미해결 |
| **Sub-task reach_rate 추적** | (m4_v20 자체 design) | ✅ 적용 `train.py:91-101, 162-177` | 없음 — Continual World forgetting metric 직접 측정 인프라 |
| **BC distillation (이전 stage policy = teacher)** | CLEAR (line 167) + Lee 2020 ANYmal (Science Robotics) + P&C (Schwarz 2018) + Mix&Match (Czarnecki 2018) — 4 source cross-check | ❌ 미적용 | **Tier 3 권장 — robust mechanism** |
| **EWC Fisher penalty** | Kirkpatrick 2017 (PNAS arXiv:1612.00796 eq.3 line 143) + Continual World §EWC (forgetting 0.02 — line 454) | ❌ 미적용 | **Tier 3 권장 — 단 forward transfer -0.17 negative 위험** |
| **PackNet per-stage mask** | Mallya 2018 + Continual World §PackNet (forgetting 0.00, forward transfer +0.19 — Continual World 11 baseline 중 1위) | ❌ 미적용 | Tier 3 — architecture 변경 부담 큼 |
| **Adam optimizer state reset** | (literature 1차 source 약함) | ❌ 미적용 | Tier 2 — A/B test 후보 |
| **LR decay stage 별** | (literature 1차 source 약함, EWC 와 mechanism 유사) | ❌ 미적용 | Tier 2 — A/B test 후보 |
| **Reverse curriculum start state** | Florensa 2017 (arXiv:1707.05300 §3 line 68-71) | ❌ 미적용 | Tier 3 — RL_robot 의 reset 함수 변경 필요 |
| **Progressive Networks (column per stage)** | Rusu 2016 (arXiv:1606.04671) | ❌ 미적용 | Narvekar 2020 §5 limitation 명시 — 6 stage column 폭증 → 부적합 |

**Literature baseline 정량**: Continual World §SAC fine-tuning forgetting = **0.73 [0.72, 0.75]** (line 451). 즉 사용자가 겪는 forgetting 은 SAC + fine-tuning 의 well-documented limitation 으로 정상 결과. mitigation 필수.

---

## 4. 권장사항 — Tier 별

### Tier 1: 코드 변경 최소 — 즉시 시도

#### 1.1 Replay buffer rehearsal (50-50 mixing) ★★★

**현 gap**: `curriculum.py:240-247` 의 `SAC.load()` 는 weight 만 로드. SB3 의 `model.replay_buffer` 는 빈 상태로 새 stage 시작 → 이전 stage transition (s, a, r, s') 완전 소실. SAC 의 off-policy 본질상 buffer = 학습 분포 → forgetting 직결.

**변경**:
```python
# curriculum.py 의 stage loop (line 217-) 안에 추가:
prev_buffer_path = None
if prev_model_path is not None:
    prev_buffer_path = prev_model_path.parent / "replay_buffer.pkl"

# SAC.load 직후 (line 242 다음):
if prev_buffer_path is not None and prev_buffer_path.exists():
    model.load_replay_buffer(str(prev_buffer_path))
    print(f"[curriculum] {prev_buffer_path} replay buffer 로드 (rehearsal)")

# stage 종료 시 (line 309 직후):
model.save_replay_buffer(str(run_dir / "replay_buffer"))
```

SB3 SAC buffer 는 기본 FIFO ring (`buffer_size=200_000`) — 새 sample 이 자동으로 old 를 밀어내므로 **50-50 점진 mixing 효과**. 더 적극적이라면 custom replay buffer 로 50% 확률 prev_buffer 추출.

**Source**: Rolnick 2019 CLEAR Table 1 — "50-50 mix 31.40 vs no replay 28.66 vs 100% replay 31.09" (line 239-242). "50-50 represents a good tradeoff" (line 310-315). Continual World §Reservoir SAC 결과 — buffer-based forgetting 완화 약 0.10 (vs fine-tune 0.73).

**Effort**: 1 시간 / **Risk**: 낮음 (저장 file 추가만) / **Expected effect**: forgetting metric 0.73 → ~0.30 추정 (CLEAR + Reservoir 중간).

#### 1.2 (이미 구현됨) Stage 별 entropy floor

`curriculum.py:50-128 STAGES` 의 `ent_floor` 필드 + `train.py:241-290 EntCoefFloorCallback`. Literature 의 권장과 일치. **추가 작업 없음**.

#### 1.3 (이미 구현됨) Mixed parallel sampling (m4_v20)

`curriculum.py:112-127 s_all_mix` + `train.py:91-177 sub trigger`. Rudin 2022 game-based curriculum 의 직접 구현. **단 비교 가치 있는 ablation**: s3a-d 자체를 처음부터 mixed sampling 으로 변경 (sequential → parallel) — m4_v20 가 "회복 stage" 인 한 forgetting 발생 후 recovery cost 가 큼.

---

### Tier 2: 새 hyperparameter 추가

#### 2.1 Adam optimizer state reset (옵션) ★★

**현 gap**: `SAC.load(prev_model_path, env=env)` 가 actor·critic 의 Adam optimizer state (m, v moment) 도 이어받음 (`curriculum.py:240-247`). 새 stage 의 다른 reward landscape 에서 stale moment 가 misleading gradient.

**변경**: `curriculum.py` 의 STAGES list 에 `reset_optimizer: bool` field 추가 + load 후 옵션적으로 optimizer 재생성:
```python
if stage.get("reset_optimizer", False):
    import torch.optim as optim
    model.actor.optimizer = optim.Adam(model.actor.parameters(), lr=model.learning_rate)
    model.critic.optimizer = optim.Adam(model.critic.parameters(), lr=model.learning_rate)
```

**Source**: literature 1차 source 약함 (Adam reset 은 일반 fine-tuning wisdom). Continual World 도 명시 X. **A/B test 권장**.

**Effort**: 30 분 / **Risk**: 중간 (학습 instability 가능) / **Expected effect**: 불확실, ablation 필요.

#### 2.2 LR decay stage 별 ★

**변경**: STAGES list 에 `learning_rate` field 추가, stage 후반 (s3c-d) 에서 lr=1e-4 로 낮춤. 메커니즘: weight 변화 폭 제한 → 이전 stage 의 weight 에서 멀어지지 않음 → forgetting 약화 (EWC 와 mechanism 유사).

**Source**: literature 1차 source 약함 — EWC 가 더 정확한 mechanism.

**Effort**: 30 분 / **Risk**: 낮음 / **Expected effect**: forgetting 부분 완화, 단 stage 통과 속도 ↓ trade-off.

---

### Tier 3: 구조 변경 — 코드 추가 多

#### 3.1 BC distillation loss (이전 stage = teacher) ★★★

**메커니즘**: stage k 종료 모델 = `pi_teacher` (frozen). Stage k+1 SAC actor loss 에 추가:
```
L_total = L_SAC_actor + λ_BC · E_s~D [ KL( pi_teacher(·|s) || pi_student(·|s) ) ]
```
D = current replay buffer + prev_buffer (Tier 1.1 와 결합).

**Source 4 source cross-check**:
- Rolnick 2019 CLEAR `L_policy-cloning` (arXiv:1811.11682 line 167)
- Lee 2020 ANYmal teacher-student (arXiv:2010.11251 line 351) — Hutter lab, robotics 사례
- Schwarz 2018 P&C compress phase (arXiv:1805.06370 line 169-171)
- Czarnecki 2018 Mix & Match (arXiv:1806.01780 line 167)

**Hyperparameter**: λ_BC ≈ 0.1~1.0 (start small, anneal up).

**수정 위치**: `sim/train.py:486` 부근 + `model.policy.actor.train()` step 에 custom callback 또는 SAC subclass.

**Effort**: 4~6 시간 (SB3 SAC subclass + actor loss override) / **Risk**: 중간 (teacher pi 의 분포 외 state 에서 KL 계산 시 noise) / **Expected effect**: forgetting metric → ~0.15 추정 (CLEAR Atari 결과 외삽).

#### 3.2 EWC Fisher penalty (SAC actor) ★★

**메커니즘**: stage k 종료 시 actor net 의 Fisher matrix 추정 (action log prob gradient² batch 평균, 100 sample). Stage k+1 actor loss 에 `λ_EWC · Σ F_i (θ_i − θ_i^k)²` 추가.

**Source**: Kirkpatrick 2017 eq.3 (arXiv:1612.00796 line 143) + Continual World §EWC HP (forgetting **0.02**, forward transfer **-0.17** — line 454).

**Hyperparameter**: λ_EWC ≈ 400 (Kirkpatrick Atari), 100 samples Fisher. Continual World 별도 tuning 필요. Critic 에는 적용 X (Continual World §line 847 "CL algorithms only for the policy").

**수정 위치**: `sim/train.py:486` + SAC subclass actor loss.

**⚠ Continual World 결과 EWC forward transfer = negative -0.17** — forgetting 막지만 새 stage 학습 느려질 위험. **λ_EWC 작게 시작** 권장.

**Effort**: 6~8 시간 (Fisher 추정 + actor loss override) / **Risk**: 중간 / **Expected effect**: forgetting 강하게 mitigate, 단 stage 통과 속도 ↓.

#### 3.3 PackNet per-stage mask (★ — 구현 부담 가장 큼)

**메커니즘**: stage k 종료 후 actor net weight 의 50% 를 mask freeze. Stage k+1 은 unfreeze 부분만 학습. 6 stage 처리 가능 (50% pruning × 6 = 2^6 = 64 partition 으로 capacity 충분).

**Source**: Mallya 2018 (arXiv:1711.05769 §3 line 135-165) + Continual World §PackNet — **avg perf 0.80 (1위), forgetting 0.00, forward transfer +0.19** (line 457) — Continual World 11 baseline 중 유일하게 forward transfer + 와 forgetting ≈0 동시 달성.

**수정 위치**: `sim/train.py` + SAC MlpPolicy 의 custom mask layer (각 Linear weight 에 binary mask 부착).

**Effort**: 2~3 일 / **Risk**: 높음 (architecture 변경 + per-stage fine-tune iteration 필요) / **Expected effect**: 가장 강한 mitigation 단 구현 비용도 큼.

#### 3.4 (이미 구현됨) Mixed-stage parallel sampling — `curriculum.py:112-127 s_all_mix`

#### 3.5 Reverse curriculum start state (Florensa 2017) — 인접 사례

사용자 case 의 stage 별 task 가 init position 분포가 아니라 theta_max·success_radius 변경이라 직접 mapping X. 단 stage 별 goal sampling 을 이전 stage 성공 trajectory 도착 영역 근처에서 점진 확장하는 변형 가능. **현 우선순위 낮음**.

---

## 5. 검증 protocol

권장사항 적용 시 forgetting metric 측정 방법:

### 5.1 Continual World forgetting metric (직접 적용)

`forgetting_i = max_t P_i(t) − P_i(T)` (Wołczyk 2021 eq.2).

RL_robot 의 sub trigger 메커니즘 (`train.py:91-101, 132-177`) 이 이미 sub_id 별 reach_rate 추적 → 측정 인프라 절반 존재. 다음 추가 필요:
- s_all_mix stage 동안 sub_id 별 max reach_rate 기록 (peak)
- s_all_mix 종료 시점 sub_id 별 reach_rate (final)
- forgetting_sub = max − final

### 5.2 A/B 비교 protocol

각 권장사항 (Tier 1.1, 2.1, 2.2, 3.1, 3.2) 별로:
1. Baseline: 현재 `m4_vXX` 그대로 6-stage 통과
2. 권장사항 적용: 동일 seed (≥3 seed) 로 6-stage 통과
3. s_all_mix 시작 시점 sub_id 별 reach_rate 비교 → baseline 의 forgetting 정량
4. s_all_mix 종료 시점 비교 → 권장사항의 mitigation 효과

**비용**: 1 seed × 6 stage ≈ 8 시간 (사용자 보고 기준). Tier 1.1 부터 시작 권장 — effort 1 시간, 효과 큼.

### 5.3 추가 metric

- **Forward transfer** (Wołczyk 2021): 권장사항 적용 stage 의 학습 step 수 (threshold 도달까지) vs baseline. EWC 처럼 negative forward transfer 위험 검출.
- **det eval 안정성** (`train.py:107-114` 함정 #13 차단): det eval cooldown 동안 reach_rate drift 측정. forgetting 의 부산물.

---

## 6. 우선순위 결정 (사용자 immediate action)

| Tier | 권장 | Effort | Expected forgetting↓ | Risk | 권장도 |
|---|---|---|---|---|---|
| 1.1 | Replay buffer 50-50 | 1h | 大 (외삽 추정 0.73→~0.30, **신뢰도 낮음** — §6.1 caveat) | 낮음 | **★★★ 즉시** |
| 2.1 | Adam reset | 30m | 불확실 | 중 | ★ A/B |
| 2.2 | LR decay | 30m | 小~中 | 낮음 | ★★ Tier 1.1 와 같이 |
| 3.1 | BC distillation | 4-6h | 大 (외삽 추정 0.73→~0.15, **신뢰도 낮음** — §6.1 caveat) | 중 | ★★ Tier 1.1 후 |
| 3.2 | EWC | 6-8h | 大 (CW 직접 0.02), 단 forward transfer 위험 | 중 | ★ Tier 3.1 와 비교 |
| 3.3 | PackNet | 2-3d | 最 (CW 직접 0.00 + forward +0.19) | 高 | ☆ 마지막 |

### 6.1 Expected forgetting 추정 신뢰도 caveat (§6 critic 지적)

- **Tier 3.2 (EWC = 0.02), 3.3 (PackNet = 0.00)** 의 forgetting 수치는 Continual World §Table 1 (Wołczyk 2021) **직접 측정값** — SAC backbone + Meta-World CW10/CW20 환경. 사용자 case 와 algorithm 1축 일치, task semantics 다름 (manipulation ≠ BCF fish).
- **Tier 1.1 (~0.30), 3.1 (~0.15)** 의 수치는 **단일 source 외삽 추정**:
  - Tier 1.1: CLEAR (Rolnick 2019) 의 IMPALA 50-50 mixing 결과를 SAC off-policy replay 가족 공통 가정으로 transfer. **algorithm-family 1축 일치만**.
  - Tier 3.1: CLEAR BC + Lee 2020 ANYmal student loss + P&C compress + M&M KL distillation 의 mechanism cross-check 강하나, forgetting metric **정량 측정값은 SAC backbone 에서 직접 측정된 source 없음**.
- **권장**: Tier 1.1 적용 후 §5.2 A/B 측정으로 실제 forgetting↓ 수치 확인. literature 추정 ≠ 사용자 case 직접 검증. 추정값은 우선순위 결정 참고용으로만.
- **Tier 1.1 의 mechanism transferability 가정**: "SAC 의 replay buffer 도 IMPALA 처럼 50-50 mixing 에서 forgetting 감소" 가 핵심 가정. Continual World §SAC Reservoir 결과가 부분 지지하나 50-50 ratio 자체는 IMPALA 결과 외삽.

**추천 순서**: Tier 1.1 (replay buffer rehearsal) 단독 적용 → A/B 측정 → 효과 부족 시 Tier 3.1 (BC distillation) 추가 → 그래도 부족 시 Tier 3.2 (EWC — Continual World 직접 검증 source 강함).

---

## 7. Literature link (자매 문서)

본 보고서의 모든 권장사항 source 는 `~/research/fish_rl/literature_curriculum_forgetting_fish_rl.md` §본 표·인접 사례 §에서 paper 본문 fetch + arxiv PDF 저장 확인 완료.

- 본 표 1편 (Continual World ≥2축 매칭) + 인접 사례 11편 (1축 매칭) + 미검증 7편
- Lab 다양성: 6 그룹 (DeepMind 42% ≤ 60% 임계 충족)
- 직접 fetch 12편 / citation only 7편 (분리 명시)
- §3 cross-check: Tier 1.1 (CLEAR + Continual World), Tier 3.1 (CLEAR + Lee + P&C + M&M = 4), Tier 3.2 (Kirkpatrick + Continual World), Tier 3.3 (Mallya + Continual World)
- §3 cross-check 약함 명시: Tier 1.2 (entropy floor), Tier 2.1·2.2 (Adam reset / LR decay)
