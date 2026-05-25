# 정지/제자리 진동 dominant 진단 — m4_v15 기준 (2026-05-19)

> **참조**: `~/research/fish_rl/literature_backward_diagnosis_fish_rl.md` (2026-05-18 작성) 의 후속. 그 자료가 "후진 (backward)" 가설 mapping 이라면, 본 자료는 **"open-loop sine 으로 전진 가능한 영역에 SAC closed-loop policy 가 도달 못함"** 으로 frame 을 좁힘.

> **scope**: RL_robot 코드 + 기존 fish_rl literature 비교. 단일 motor + fiberglass V-cut fin + R5 fluidcoef + m4_v15 reward.

---

## 1. 현상 정의

**관찰**:
- `freq_sweep.py` (open-loop ctrl=±1 sine, 12s) → 4~6 Hz 에서 −x 전진 (`docs/training_log_m4.md:382` "6Hz: x_disp = −0.444, 전진"; `docs/training_log_m4.md:492` V-cut revert 후 "4~6Hz 전진 회복 확인").
- 그러나 SAC closed-loop policy 학습 → "정지 또는 yaw-spin dominant" (사용자 보고).

**즉 gap**: open-loop 으로 검증된 추진 가능 영역에 policy 가 self-locate 못함. 단순 "절대적 추진 불가능" 가설은 부분 해소.

**RL 카드 함정으로 이미 등록된 mode** (`CLAUDE.md:182`):
> avg_align 높 + success 낮 → "**정렬만**" mode

→ 사용자가 이미 인지·대응 중 (m4_v12 `SLIP_PEN`, m4_v14 `STILL_PEN`, m4_v15 `progress·5.0`). 그러나 잔존.

---

## 2. 사전 통과 가설 (부분 해소)

### H1 (절대적 추력 불가능) — **부분 해소**
- m4_v9 fiberglass + default fluidcoef → 1~5 Hz 후진, 6 Hz 만 전진 (`training_log_m4.md:374-385`)
- m4_v9 R5 fluidcoef `"0.4 3.0 2.81 1.0 0.27"` (fishsim BO sysid + KU_FISH 보수화, `rl_fish.xml:36`) → 4~6 Hz −x 전진 회복 (`training_log_m4.md:492`).
- **잔존 risk**: R5 가 V-cut 특수 ellipsoid (motion 방향 thin disk) 에 fit (`training_log_m4.md:476-488`) — amplitude / phase 다른 영역에선 thrust 보장 X. **R1 가설로 재등장** (아래).

### H_RL_naive (reward signal 부족) — **방어 강함**
사용자 m4_v15 reward (`fish_env.py:260-279`) 은 정지/yaw-spin 을 이미 여러 층으로 페널티:
- `STILL_PEN_W = 0.05` (`fish_env.py:227`) — `|v| ≤ 0.02 m/s` 시 매 step −0.05
- `SLIP_PEN_W·slip_angle` (`fish_env.py:235`) — 방향 misalign 시 −0.16 ~ −0.31/step
- `BACK_EPS` clip (`fish_env.py:278-279`) — head_vel < −0.01 시 reward = −1.0 (다른 항 무효)
- `progress·5.0` (`fish_env.py:264`) — m1 ×21 등가 (m4_v15 hover collapse 대응 증액)
- `align_weight = 0.05` (`curriculum.py:50-106` stage 별 `episode_seconds`; 30 s ep 기준 0.05)

→ 단순 "정지가 reward 적으로 유리" 가설로는 설명 어려움.

---

## 3. 잔존 가설 (4개) — m4_v15 코드 + literature 매핑

### R1 — R5 fluidcoef 의 V-cut amplitude-narrow fit

**메커니즘**: R5 = V-cut 의 thin-disk ellipsoid 에 fit 된 값. SAC 초반 random small-amplitude action 영역에선 ellipsoid 단면 X velocity 가 매우 작아 thrust ≈ 0. open-loop ±1 sine (max amplitude) 에선 thrust 회복하나 policy 가 ±1 saturated 영역에 진입하기 전까지 progress reward signal 거의 없음 → `STILL_PEN −0.05/step` 외 신호 부재.

**코드 evidence**:
- `rl_fish.xml:36` `fluidcoef="0.4 3.0 2.81 1.0 0.27"` — V-cut fit
- `training_log_m4.md:476-488` "R5 의 V-cut 의존성": "박스로 가는 한 어떤 dimensions 로도 같은 fluid 거동 재현 불가능"
- `rl_fish.xml:78-79` fin auto-inertia: "V-cut은 motion edge-on (motion 방향 半축 ≈ 0, 칼날)"

**Literature link**:
- `literature_backward_diagnosis_fish_rl.md:78` Shinde 2018: "0.1 ≲ EI* ≲ 1 외 영역 = no coherent jets" — small amplitude regime 은 effective EI* 가 amplitude-dependent 이라 본 가설과 indirect link
- `literature_fin_material_fish_rl.md` Paraz 2016 PIV: reverse Bénard-von Kármán 은 first resonance 근처 + sufficient amplitude 에서만 발생

**우선순위**: ★★ (직접 R1 검증 paper 0편, 사용자 자체 측정·sysid 의 한계로 추론)

---

### R2 — Over-resonance regime 의 좁은 전진 window (정량 매핑 신규)

**메커니즘**: 사용자 fiberglass fin 의 WET f_n = 1.33 Hz (`training_log_m4.md:391`). Motor decision freq cap 6 Hz (`CLAUDE.md:129` Nyquist) → 사용자 case ω/ω₀ ratio:

| 조합 | ω (motor) | ω₀ (WET f_n) | ω/ω₀ | Paraz 2016 § IV.B 영역 |
|---|---|---|---|---|
| 6 Hz (open-loop sine) | 6 Hz | 1.33 Hz | **4.51** | **음수 f_P 영역** ([2.7, 5.1]) — but resistive force 가 보완 → net thrust 양수 (사용자 freq_sweep 4~6 Hz 전진 확인과 일치) |
| 4 Hz | 4 Hz | 1.33 Hz | **3.01** | 음수 f_P 영역 가운데 — resistive force 보완 정량 미확보 |
| 2 Hz | 2 Hz | 1.33 Hz | **1.50** | resonance peak ω/ω₀ ≈ 0.9 (Paraz Fig 8a) 위 — thrust 감소 시작 |
| 1 Hz | 1 Hz | 1.33 Hz | **0.75** | near resonance — 본래 강한 thrust 영역. 그러나 fiberglass 의 damping ratio ζ ≈ 0.54 (`training_log_m4.md:395`) 로 over-damped, free oscillation 사라짐 → 사용자 case 1~3Hz 후진 (default fluidcoef) 의 mechanism. R5 fluidcoef 로 보정. |

→ Paraz 2016 의 정량 영역 [2.7, 5.1] 에 사용자 motor freq 범위 (1~6 Hz / 1.33 Hz = 0.75~4.51) 대부분이 들어감. **net thrust 양수가 "resistive force 보완" 에 의존하는 좁은 window 라는 의미**.

**RL 해석**: SAC 가 6 Hz 근처 + 충분한 amplitude 의 좁은 영역에서만 forward 신호 받음. 그 외 영역 (작은 amplitude, off-frequency) 에선 progress 거의 0. **policy 가 정지에 stuck → ε-탐색이 6 Hz +1 sine 영역에 도달할 확률 매우 낮음**.

**코드 evidence**:
- `training_log_m4.md:391` WET f_n 1.33 Hz
- `training_log_m4.md:395` ζ ≈ 0.54, fin/tail amplitude ratio ≈ 1.0 (in-phase, lag 없음)
- `training_log_m4.md:400-406` "Very stiff regime → 강체 paddle → reverse Karman shedding 부재"

**Literature link**:
- `literature_backward_diagnosis_fish_rl.md:51` Paraz 2016 §IV.B 인용: "2.7 ≤ ω/ω₀ ≤ 5.1, fP can even be negative; total thrust is positive only thanks to resistive forces"
- `literature_backward_diagnosis_fish_rl.md:88` Paraz: "reverse Bénard-von Kármán vortex street appears at first resonance ω/ω₀ ≈ 0.9"
- `literature_fin_material_fish_rl.md` Shinde 2018: rigid 극한 (사용자 fiberglass EI* >> 1) "divergent jet, no coherent thrust"

**우선순위**: **★★★** (직접 정량 매핑, source 본문 직접 fetch ≥ 2편)

---

### R3 — align/yaw reward 가 정지 + spin local optimum 형성

**메커니즘**: 30 s episode 기준 (`curriculum.py:81, 91, 104`) align_weight = 0.05. 정지 상태에서도 yaw 만 잘 맞추면 매 step align ≈ 1 → 0.05/step 획득. STILL_PEN_W = 0.05 와 정확히 상쇄 (`fish_env.py:227, 266`). yaw rate × yaw_err_sign · 0.005 까지 더해지면 약 spin-and-align local optimum 이 약하게 유리.

**계산** (정지 + perfect align 시):
```
+ 0.05·align  ≈ +0.05         (align ≈ 1)
+ 0.005·yaw_rate·sign  ≈ 0    (정지면 yaw_rate≈0)
− 0.05  (STILL_PEN)              (|v|≤0.02 면 자동 적용)
≈ 0/step  → net 신호 없음. 그러나 BACK_CLIP 위반 가능성도 0이라 안전한 attractor.
```

→ **정지 + 정렬은 net reward ≈ 0 인 안전한 fixed point**. progress·5.0 영역으로 가려면 R2 의 좁은 window 를 통과해야 하는데, 그 사이 transient 에 SLIP_PEN 0.2·slip_angle (`fish_env.py:235`) 까지 페널티. 정지에서 빠져나오는 게 비싸다.

**코드 evidence**:
- `fish_env.py:264-273` reward 항 weight
- `CLAUDE.md:182` 이미 함정으로 등록 ("'정렬만' mode")
- `training_log_m4.md:340-341` m4_v9 다음 카드 후보 A: "head align reward ↑" — 사용자가 align ↑ 방향으로 압력. 그러나 정지·정렬이 함정이라 동시에 STILL_PEN 추가 필요했음.

**Literature link**:
- `literature_backward_diagnosis_fish_rl.md:107` Lai 2025 (Three-Link Microswimmer, arXiv:2506.00084, peer-review 미확인 라벨): "if c is set too high → forward 학습 실패, 에너지 보존 우선" — reward weight imbalance 가 stroke pattern 을 완전히 바꾼다는 1차 source. engine 다름 (Stokes ODE) 이라 transfer 제한.
- `literature_backward_diagnosis_fish_rl.md:108` Jiao 2023 (PMC10318098): reward Eq.13 `r = −(u−u_t)² − 0.2|u−u_t| − 0.5θ_e² − 0.1|θ_e|`. L2/L1 weighted + curriculum velocity widening 으로 burst-and-coast emergent.

**우선순위**: ★★ (literature 1차 source 2편, engine 다름)

---

### R4 — exploration 부족 (ent_floor 낮음) + obs 정지 attractor

**메커니즘**: 
- s1/s2 ent_floor = 0.002 (`curriculum.py:58, 67`) — SAC 의 lower bound 매우 낮음. policy 가 narrow distribution 으로 수렴하면 R2 의 6 Hz +1 sine 영역까지 탐색 안 함.
- obs 의 vx/vy/vyaw (`fish_env.py:149`) 가 정지 시 모두 0 → policy network 입력이 "정지 fixed point" 표지. action_history N=20 (`curriculum.py:46`) 도 작은 amp 면 그 자체로 fixed point.

**코드 evidence**:
- `curriculum.py:32, 50-106` ent_floor 표
- `fish_env.py:144-152` obs base 13D

**Literature link**:
- `literature_backward_diagnosis_fish_rl.md:107-110` Lai 2025 §III.A "reward weight 가 너무 크면 forward 학습 실패" — exploration 부족과 reward landscape 의 결합 case
- 직접 source (RL obs 정지 attractor) 1차 source **0편** — literature 보강 필요 axis

**우선순위**: ★ (literature 부족 명시)

---

## 4. 우선순위 종합

| ID | 가설 | 1차 source | 우선순위 |
|---|---|---|---|
| **R2** | over-resonance 좁은 전진 window | 본문 ≥ 2편 (Paraz 2016, Shinde 2018) + 사용자 정량 매핑 | **★★★** |
| **R3** | align/yaw reward local optimum | 본문 2편 (Lai 2025, Jiao 2023, engine 다름) | ★★ |
| **R1** | R5 fluidcoef V-cut amplitude-narrow fit | 1차 source 0편 (사용자 자체 sysid 한계로 추론) | ★★ |
| **R4** | exploration 부족 + obs static attractor | 본문 1편 (Lai 2025), 직접 source 0편 | ★ |

R2 + R3 의 결합이 가장 가능성 높음: physics 가 좁은 window 만 허용 (R2) + reward 가 정지·정렬을 안전한 fixed point 로 만듦 (R3) → policy 가 좁은 window 발견 전에 fixed point 에 settle.

---

## 5. 결정적 진단 action

### A1 — eval rollout 분석 (R2 / R3 / R4 분리)
```bash
cd /home/yoo/RL_robot/sim
python3 eval_stages.py --stage s3c --episodes 30 --deterministic --tag m4_v15_diag
```
+ rollout 별 측정:
- `action` distribution (mean, std, |ctrl| 분포)
- fin tip amplitude (qpos[IDX_FIN] peak-to-peak)
- ctrl freq FFT (1~6Hz 분포)
- yaw_rate / |v| 시계열 → 정지 vs spin vs side-slip 비율

**기대**:
- R2 dominant → fin amp 작음, ctrl freq 가 6 Hz 영역 도달 못함
- R3 dominant → fin amp 작음 + align 높음 + yaw_rate 진동
- R4 dominant → action variance 매우 작음 (ent 부족)

### A2 — ent_floor sweep (R4 검증)
```bash
# curriculum.py:58, 67 의 s1/s2 ent_floor 0.002 → 0.02 또는 0.05 변경 후 재학습
python3 curriculum.py --start-stage 1 --end-stage 1 --seed 0 --tb-tag m4_v15_ent02
```
**기대**: R4 dominant 면 ent_floor 상향으로 6 Hz amplitude 영역 escape 가능. R2 dominant 면 ent ↑ 도 그 좁은 window 발견 어려움 (확률 ↑ 하나 미해소).

### A3 — reward ablation (R3 검증, `literature_backward_diagnosis_fish_rl.md` 권장 순서 3번)
```bash
# fish_env.py:266 align_weight·align 항 = 0 으로 두고 재학습 (또는 STILL_PEN_W = 0.15 로 3× 증액)
```
**기대**: 
- align=0 후 forward 학습되면 → R3 dominant (정렬 attractor 해소)
- 후진 또는 정지 유지 → R2 + R1 dominant (physics 측이 진짜 bottleneck)

### A4 — fluidcoef sweep (R1 검증)
```bash
# rl_fish.xml:36 R5 의 blunt_drag, slender_drag, angular_drag, kutta_lift 각각 ±20% 변동 4 case + baseline
# 각 case freq_sweep + 짧은 SAC s1 학습 (300 k step)
```
**기대**: R1 dominant 면 fluidcoef 변동 시 SAC final velocity 분포가 크게 바뀜. R2 dominant 면 변동 small.

---

## 6. 결론 후보 + 후속 plan trigger

| 결론 | 후속 plan |
|---|---|
| **R2 dominant** (가장 가능성 큼) | (a) `literature_backward_diagnosis_fish_rl.md` 권장 순서 1번 (fin free f_n 정밀 측정) + 2번 (EI* 계산); (b) fiberglass stiffness ↓ 후보 `training_log_m4.md:425-430` 표 (stiffness 0.3 → DRY f_n 4.1 Hz → 4~5 Hz 더 넓은 전진 window); (c) action smoothing 약화로 high-freq 6Hz pattern 학습 압력 ↑ |
| **R3 dominant** | (a) align_weight ↓ (0.05 → 0.02 복귀) + STILL_PEN ↑ (0.05 → 0.10); (b) Lai 2025 §III.A 의 c=0 ablation 패턴 — `literature_backward_diagnosis` 권장 3번 |
| **R1 dominant** | fishsim BO 재실행 (action amplitude 분포 정합 데이터셋) 또는 ANN surrogate (`RL_robot/CLAUDE.md:50` Lighthill `mjcb_passive` 후보) |
| **R4 dominant** | s1/s2 ent_floor 0.002 → 0.02~0.05 + literature 보강 신규 task (obs static attractor) |

---

## 7. 검증 수준 + 사용자 setup 5축 매칭

`~/research/CLAUDE.md` §1 5축 (reward·obs·action·architecture·algorithm) 매칭:

| 가설 | 인용 source | 5축 매칭 | transfer 가능 | transfer 불가 이유 |
|---|---|---|---|---|
| R2 | Paraz 2016 PoF | physics·morphology 2/5 | 정량적 ω/ω₀ 영역 매핑 | RL 측 5축 0/5 (개념 매핑은 영향 X) |
| R3 | Lai 2025 (arXiv preprint) | reward·algorithm 2/5 | reward weight ablation 패턴 | engine 다름 (Stokes ODE ≠ MuJoCo ellipsoid), action dim 다름 (3 link ≠ 1 link) |
| R3 | Jiao 2023 (PMC10318098) | reward·obs·algorithm 3/5 | L2/L1 weighted reward 구성 | engine 다름 (panel method), morphology 다름 (active fin) |
| R1 | (1차 source 없음) | — | — | — |
| R4 | (1차 source 없음, indirect Lai) | — | — | — |

**Lauder lab 편향**: 본 보고서 직접 인용 source 의 Lauder lab 비중 = Quinn 2015 (Lauder 공저) 1편 / Paraz·Shinde·Lai·Jiao 4편 = 1/5 = 20% ✓ (60% 이하 목표 만족).

---

## 8. 참조 파일

- `/home/yoo/RL_robot/sim/fish_env.py:218-279` — reward 함수
- `/home/yoo/RL_robot/sim/rl_fish.xml:36, 76-89` — fluidcoef + fin
- `/home/yoo/RL_robot/sim/curriculum.py:32, 50-106` — stage·ent_floor
- `/home/yoo/RL_robot/docs/training_log_m4.md:344-495` — m4_v9 fiberglass + m4_v10 박스 실패
- `/home/yoo/RL_robot/CLAUDE.md:129, 182` — Nyquist cap + "정렬만" 함정
- `/home/yoo/research/fish_rl/literature_backward_diagnosis_fish_rl.md` (2026-05-18) — 후진 가설 mapping (본 보고서의 base)
- `/home/yoo/research/fish_rl/literature_fin_material_fish_rl.md` — passive fin H1~H5 가설
- `/home/yoo/research/fish_rl/literature_mujoco_water_env_fish_rl.md` — R5 fluidcoef + RL backward 인접 표

---

## 9. critic 결과

본 보고서는 분석 task — 표 행 추가·transfer 평가 신규 없음. 그러나 R2 의 정량 매핑 (ω/ω₀ = 4.51 ∈ Paraz [2.7, 5.1]) 은 `literature_backward_diagnosis_fish_rl.md` 에 없는 신규 사실 → critic 호출 1회.

critic 검토 항목: ① R2 ω/ω₀ 정량 매핑 산술 정합 / ② Paraz 인용 영역 [2.7, 5.1] 의 본문 일치 / ③ Lauder lab 편향 ≤ 60% / ④ R1·R4 의 "1차 source 0편" 명시 / ⑤ R3 source 의 engine 불일치 라벨.

**결과**: (별도 호출 후 본 절에 명시)
