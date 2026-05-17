# 학습 로그 — m4 환경 (frame_skip 42 + V2 spec actuator, 2026-05~)

mode3-planar 브랜치 안에서 환경만 분리. `m4_` prefix로 v22~v33 (model3 환경)과 구분.
브랜치는 분리 X (mode3-planar 그대로).

> mode3-planar 환경의 v1~v33 history (model3 환경) → [`docs/training_log.md`](training_log.md).
> s3d_90 라인 → [`docs/s3d_90_line.md`](s3d_90_line.md).

## 작성 규칙

- m4 환경에서의 학습 history (표·매트릭스·산문·통찰·RL 카드 함정·다음 후보) 모두 본 파일에 작성.
- 산문 진단: `## 주요 진단 (산문)` 끝에 `### m4_vN-X (한 줄 제목)` 새 절 추가.
- 종결·돌파 카드면 본 파일의 그룹화 표·변경점 표·Stage 진행 비교 표 갱신.
- 새 RL 카드 함정 발견 시 mode3 history(`docs/training_log.md`)의 `## RL 카드 함정` 번호 이어서 추가 (함정은 model 무관).
- CLAUDE.md는 코드/모델/규칙 변경 시만 갱신 (학습 결과 변경에 따른 갱신 없음).

---

## 환경 변경 (mode3-planar 환경 → m4 환경)

- `sim/fish_env.py:52` `frame_skip: int = 42` (이전 10) — Nyquist 6Hz cap, ETH Verma/Novati "action every Tp/2" 패턴.
- `sim/rl_fish.xml` actuator V2 spec: `tail_joint damping=0.087 armature=1.4e-4`, `actuator forcerange=±2.2`. BL4260 V2 + 24V + 6.8:1 평기어 spec 직접 반영.
- 외부 reference: `~/research/fish_rl/literature_timestep_fish_rl.md`.
- 자세한 spec: CLAUDE.md `### Tail motor spec` subsection.

## 환경 변경 효과 (검증 완료)

| 측정 | 결과 |
|---|---|
| T1 (정책 매 step alternate) | max ctrl freq **5.91Hz** ✓ (Nyquist 6Hz cap) |
| 6Hz sine wet (default fluidcoef) | tail amp 36.8°, xdisp -1.77m (이전 baseline 94%) |
| 6Hz 위 명령 (8/10Hz) | aliasing → 정책 표현 자체 불가 |
| fin amp wet 6Hz | 41° (추진 source 보존) |
| 1~5Hz 정상 추진 | baseline 92~98% 유지 |

---

## m4_v1 환경 추가 변경 (2026-05-16)

dry/wet 차이 보정 + reward scale 정합 + ep_sec 보정.

### 변경 내역

- `sim/rl_fish.xml` line 4: integrator `implicit → implicitfast` (fishsim 정합, MuJoCo 공식 "동일 정확도 + 빠른 step")
- `sim/rl_fish.xml` `<default geom>`: `fluidcoef="0.4 3.0 2.81 1.0 0.27"` 추가 (fishsim Bayesian opt sysid `[0.4, 7.79, 2.81, 3.84, 0.27]` 차용 + KU_FISH 형상 보수화 — slender drag 12× ↑, Kutta lift default 유지)
- `sim/curriculum.py` STAGES: ep_sec s1/s2 10s → 20s, s3a~d 30s → 60s (m4 추진 속도 ~50% ↓·yaw rate 1/3.6 ↓ 보정)
- `sim/curriculum.py` STAGES: max_steps 모든 stage 1M 통일 (안전장치, CurriculumStopCallback이 90% 졸업 시 조기 종료)
- `sim/fish_env.py:190` reward `progress·15 → progress·3.6` (dt 0.084s 보정, m1 dt 0.02s 대비 step당 progress 4.2× ↑를 weight 비례 축소 — literature 권고)
- `sim/eval_stages.py`: STAGES ep_sec sync (20s/60s)
- `CLAUDE.md`: `## Curriculum 학습` 표 max_steps·ep_sec sync + footnote

외부 reference:
- `~/research/fish_rl/literature_mujoco_water_env_fish_rl.md`: fishsim fluidcoef + MuJoCo 공식 fluid model
- `~/research/fish_rl/literature_sim2real_drag_fish_rl.md`: drag coeff sysid + DR 6 패턴
- `~/research/fish_rl/literature_timestep_fish_rl.md`: progress weight dt 비례 축소

### 변경 효과 (측정)

| 측정 | 변경 전 (default fluidcoef) | 변경 후 (m4_v1) | 평가 |
|---|---|---|---|
| WET 6Hz tail amp (p2p) | 36.8° | 34.6° | -6% (motor가 fluid 압도 — PD 강함) |
| WET 6Hz xdisp | -1.77m / 12s | -0.77m / 30s | 평균 vel 50% ↓ (drag ↑로 추진 손실, 부작용) |
| **dry/wet xdisp 비율** | **10×** | **300×** | **사용자 의도 "유의미한 차이" ✓** |
| yaw rate (sine + bias) | 4.6°/s | 1.27°/s | 1/3.6 ↓ — ep_sec 60s × 1.27 = 76° 회전 여유 |
| Stage 1 도달 (단순 sine) | 모든 freq | **1Hz도 ep 20s 안에** ✓ | ep_sec 보정 효과 |
| **정책 ctrl freq** (m4_v1 학습 후) | — | **5.88~5.95Hz peak, ≤6Hz 100%** ✓ | Nyquist cap 완벽 준수 |
| tail qpos freq (inner mj_step, 500Hz 샘플링) | — | ≤6Hz 99% + ZOH harmonics 1% (17.9·29.8Hz) | 정책 step input의 자연 부산물 (실모터 PWM 동일) |

---

## 학습 결과 요약

### m4_v1 (2026-05-16) — Stage 1 학습

**3 seed × Stage 1 (s1_forward) 학습 결과**:

| seed | 졸업 step | 학습 시간 | TB stochastic 100ep | det eval 100ep |
|---|---|---|---|---|
| 0 | 30k | 4분 09초 | 100% | 100/100 |
| 1 | 30k | 4분 09초 | 100% | 100/100 |
| 2 | 85k | 7분 23초 | 100% | 100/100 |

학습 명령: `python3 curriculum.py --start-stage 1 --end-stage 1 --seed {0,1,2} --tb-tag m4_v1_seed{0,1,2} --runs-subdir m4_v1_seed{0,1,2} --no-viewer` (3 seed 병렬 background).

**6 stage deterministic eval (post-Stage 1)**:

| Stage | seed 0 | seed 1 | seed 2 | **mean ± σ** |
|---|---|---|---|---|
| s1_forward | 100% | 100% | 100% | **100.0% ± 0.0%** ✓ (학습 stage) |
| s2_anchor | 100% | 100% | 100% | **100.0% ± 0.0%** ✓ (bonus — sr 0.04 정밀 자연 만족) |
| s3a_arc15 | 62% | 60% | 61% | 61.0% ± 0.8% |
| s3b_arc30 | 32% | 30% | 33% | 31.7% ± 1.2% |
| s3c_arc60 | 12% | 11% | 13% | 12.0% ± 0.8% |
| s3d_arc90 | 6% | 7% | 7% | 6.7% ± 0.5% |

### m4_v1 (2026-05-16) — Stage 3a fine-tune

init = `runs/m4_v1_seed{N}/s1_forward/model.zip` (Stage 1 학습 직후, s2 자연 100%이라 skip).

**3 seed × Stage 3a (s3a_arc15) 학습 결과**:

| seed | 졸업 step | 학습 시간 | TB stochastic 100ep | det eval 100ep |
|---|---|---|---|---|
| 0 | 285k | 32분 | 100% | 100/100 |
| 1 | 375k | 37분 | 100% | 98/100 |
| 2 | 75k | 10.6분 | 94% | 99/100 |
| **mean** | **245k** | **26.5분** | — | **99.0 ± 0.8%** ✓ |

학습 명령: `python3 curriculum.py --start-stage 3 --end-stage 3 --seed {0,1,2} --tb-tag m4_v1_seed{0,1,2} --runs-subdir m4_v1_seed{0,1,2} --init-from runs/m4_v1_seed{0,1,2}/s1_forward/model.zip --no-viewer` (3 seed 병렬 background).

**6 stage deterministic eval (post-Stage 3a)**:

| Stage | seed 0 | seed 1 | seed 2 | **mean ± σ** | s1 정책 대비 |
|---|---|---|---|---|---|
| s1_forward | 100% | 100% | 100% | **100.0% ± 0.0%** ✓ | 100% 유지 (forgetting X) |
| s2_anchor | 100% | 100% | 100% | **100.0% ± 0.0%** ✓ | 100% 유지 |
| **s3a_arc15** | **100%** | **98%** | **99%** | **99.0% ± 0.8%** ✓ **졸업** | 61.0% → **+38%p** |
| s3b_arc30 | 55% | 57% | 55% | 55.7% ± 0.9% | 31.7% → **+24%p** (일반화 효과) |
| s3c_arc60 | 29% | 27% | 28% | 28.0% ± 0.8% | 12.0% → **+16%p** |
| s3d_arc90 | 16% | 17% | 17% | 16.7% ± 0.5% | 6.7% → **+10%p** |

---

## GitHub Release 인덱스 (m4)

(없음 — 다음 stage 학습·release 시 추가)

---

## 주요 진단 (산문)

### m4_v1 (Stage 1 졸업·s2 자연 만족)

m4_v1 첫 학습. 환경 변경 (fluidcoef + integrator + ep_sec + progress·3.6) 적용 후 Stage 1 학습 trigger.

**핵심 결과**:
- 3 seed 모두 **30k~85k step에 졸업** (det 100% × 3). 이전 model3 s1 ~100초 대비 wall-clock 약 4분 — frame_skip 42로 step당 시간 4.2×라 비슷한 시간.
- **s2도 자연 100%** — Stage 1 학습 정책이 sr 0.04 정밀도까지 자동 만족 (eval 최종 distance ~0.04m). 사실상 s1·s2 동시 졸업.
- catastrophic forgetting 없음 (s1만 학습 → s1·s2 모두 100% 보존).
- σ = 0% — m4 환경 매우 안정 (seed 간 차이 거의 없음).

**dry/wet 보정 효과**:
- 사용자가 fluidcoef 변경의 dry/wet 차이를 "유의미"하다 평가. xdisp 비율 10× → 300×로 fluid의 영향이 명확히 분리됨.
- 부작용: WET 추진 속도 50% ↓ — ep_sec 20s 보정으로 Stage 1·2 도달 가능 영역 회복.
- 추진력 손실의 본질은 fluidcoef를 `<default geom>`에 적용해 base_link도 drag (slender 12×) 받기 때문. 추후 tail/fin geom에만 적용 분리 검토 후보.

**Stage 3 회전 능력 ⚠**:
- yaw rate 1.27°/s × ep_sec 60s = 76° 회전 여유 → s3c (±60°)·s3d (±90°) 마진 작음.
- 직진 정책의 자연 발생률 (s3a 61%, s3b 32%, s3c 12%, s3d 7%)은 random target 분포에서 회전 없이 도달 가능한 부분만.
- Stage 3 학습 시 회전 능력 학습 필요. 학습 안 풀리면 fluidcoef angular drag (2.81) 재튜닝 또는 motor 약화 (forcerange ↓) 후보.

**다음 카드 후보**:
- **m4_v2** Stage 2 학습 — s2가 이미 100%이므로 init 정책 그대로 사용해 Stage 3a 학습 직행 가능 (사용자 결정)
- **m4_v2-A** Stage 3a 학습 — init = m4_v1_seedN s1_forward, sr 0.08, ±15°. 회전 능력 학습 첫 점.
- **m4_v1-A** progress weight ablation — 15 그대로 vs 3.6 학습 차이 (학습 속도·sample efficiency)
- **fluidcoef 분리** — `<default>` 빼고 tail/fin geom에만 적용 → 추진 속도 회복

### m4_v1 (2026-05-16) — Stage 3b fine-tune

init = `runs/m4_v1_seed{N}/s3a_arc15/model.zip` (s3a 졸업 직후).

**3 seed × Stage 3b (s3b_arc30) 학습 결과**:

| seed | 졸업 step | 학습 시간 | TB stochastic 100ep | det eval 100ep |
|---|---|---|---|---|
| 0 | 250k | 36분 | 93% | 98/100 |
| 1 | 470k | 49분 | 94% | 94/100 |
| 2 | 230k | 34분 | 99% | 99/100 |
| **mean** | **317k** | **40분** | — | **97.0 ± 2.2%** ✓ |

**6 stage deterministic eval (post-Stage 3b)**:

| Stage | seed 0 | seed 1 | seed 2 | **mean ± σ** | s3a 정책 대비 |
|---|---|---|---|---|---|
| s1_forward | 100% | 100% | 100% | **100.0% ± 0.0%** ✓ | 유지 |
| s2_anchor | 100% | 100% | 100% | **100.0% ± 0.0%** ✓ | 유지 |
| s3a_arc15 | 100% | 100% | 100% | **100.0% ± 0.0%** ✓ | 99.0% → +1%p (더 안정) |
| **s3b_arc30** | **98%** | **94%** | **99%** | **97.0% ± 2.2%** ✓ **졸업** | 55.7% → **+41%p** |
| s3c_arc60 | 60% | 50% | 55% | 55.0% ± 4.1% | 28.0% → **+27%p** (일반화) |
| s3d_arc90 | 41% | 37% | 38% | 38.7% ± 1.7% | 16.7% → **+22%p** (일반화) |

---

### m4_v1 (Stage 3a 졸업·회전 일반화)

m4_v1 s1 정책 init → s3a fine-tune (3 seed 병렬, max_steps 1M cap).

**핵심 결과**:
- 3 seed 모두 **75k~375k step에 졸업** (mean 245k, det 98~100%). seed 2는 outlier로 10.6분 만에 졸업.
- 졸업 step 분산 큼 (75k vs 375k, 5× 차이) — seed 운으로 학습 trajectory 다양.
- **forgetting 없음** — s1·s2 100% 유지.
- **σ ≤ 0.9%p** — m4 환경 안정성 재확인.

**회전 능력 일반화** ⭐:
- s3a 학습이 더 큰 회전 stage로 자연 transfer
  - s3b (±30°): 31.7% → **55.7%** (+24%p)
  - s3c (±60°): 12.0% → **28.0%** (+16%p)
  - s3d (±90°): 6.7% → **16.7%** (+10%p)
- 직진 정책만으로는 s3b 31%만 가능했던 게 ±15° 회전 학습으로 ±30°·±60°·±90° 모두 향상.
- 단 s3b 55%는 아직 90% 미달 → 직접 s3b 학습 필요. s3c·s3d는 더 큰 회전 learning 필요.

**V1 정책 freq 측정 (학습 전 검증)**:
- 정책 ctrl 신호 5.88~5.95Hz peak, **≤6Hz 100%** ✓ (Nyquist 6Hz cap 완벽 준수)
- tail qpos (inner mj_step, 500Hz 샘플링): ≤6Hz **99%** + ZOH harmonics 1% (17.9Hz·29.8Hz)
- harmonic은 정책 step input (0.084s마다 ZOH)의 자연 부산물 — 실모터 PWM·step input도 동일 spectrum. sim2real 영향 X.

**다음 카드 후보**:
- **m4_v1 Stage 3b** 학습 — init = s3a model, ±30° 목표. s3b 55%에서 90% 졸업 시도.
- **m4_v1 Stage 3c** 직행 — s3b 자연 향상이라 skip 가능 여부 (사용자 결정)
- **fluidcoef 분리** — base_link drag 부작용 제거 → 추진 속도 회복 (회전 학습 더 쉬울 수 있음)

### m4_v1 (Stage 3b 졸업·v33 천장 돌파·큰 일반화)

m4_v1 s3a 졸업 모델 init → s3b fine-tune (3 seed 병렬, ±30°).

**핵심 결과**:
- 3 seed 모두 **230k~470k step에 졸업** (mean 317k, det 94~99%). v33 s3b 졸업 step 654k 대비 절반.
- **v33 s3b 천장 (80~86%) 자연 돌파** — m4 환경 변경 (frame_skip 42 + V2 spec + fluidcoef + progress·3.6 + ep_sec 60s)이 천장 해소. v33은 multi-axis 누적(yaw reward·progress·15·sign-aware 등)으로 86%까지 만든 카드였는데 m4_v1은 첫 시도에 97%.
- **forgetting 없음** — s1·s2·s3a 모두 100%.
- **σ 2.2%p** — v33 (5.4~11.4%p) 대비 매우 안정.

**v33 천장 돌파 가설**:
1. `frame_skip 42` (Nyquist 6Hz cap): 정책이 비현실적 high-freq pattern 학습 불가 → 더 robust한 swimming policy
2. `V2 spec` (damping 0.087, armature 1.4e-4, forcerange ±2.2): motor 본연 한계 sim에 반영 → sim2real 정합
3. `fluidcoef` 강화: fluid 영향 의미 있게 학습 신호에 전달
4. `progress·3.6` (dt 보정): reward scale 통일로 SAC critic Q 안정
5. `ep_sec 60s` (이전 30s × 2): 정책이 회전 + 도달까지 충분한 학습 시간

**큰 일반화 효과** ⭐⭐:
- s3c (±60°): 28.0% → **55.0%** (+27%p) — s3a 학습 시(+16%p)보다 큰 transfer
- s3d (±90°): 16.7% → **38.7%** (+22%p) — s3a 학습 시(+10%p)보다 큰 transfer
- 즉 ±30° 학습이 ±60°·±90°로 더 효율적으로 일반화 (curriculum learning 효과 누적)

**다음 카드 후보**:
- **m4_v1 Stage 3c** 학습 — init = s3b model, ±60° 목표. baseline 55%에서 90% 졸업 시도.
- **m4_v1 Stage 3d 직행** — s3c 일반화 효과 확인 후 결정. s3b로도 s3d 38%까지 향상.
- **fluidcoef 분리** — 추진력 회복으로 s3c·s3d 더 쉬워질 수 있음 (단 현재 진행 잘 됨이라 우선순위 ↓)

---

## m4_v3~v8 — B mode 제거 reward sweep (2026-05-17, 4 cycle)

### 사용자 본질 정의
- **F mode** = carangiform sweep 시간 비대칭 (한 cycle 안 +sweep/−sweep 시간 다름, power/recovery stroke)
- **D mode** = 매 step ±1 toggle (5.95Hz Nyquist cap wagging) — 직진에 최적
- **B mode** = sine + DC offset (mean ctrl ≠ 0, tail 한쪽 편향) — 추진 X + 회전 비효율 → **제거 대상**

### Reward shaping sweep (`sim/fish_env.py`)

기존 m4_v2 baseline에 cycle별로 누적:

| version | 변경 | s3a ±15° 결과 (3 seed × ±15° 6 runs) |
|---|---|---|
| m4_v3 | DC_PEN_W=0.05 (`window_mean²`, 12 step 윈도우) | F 2/6 (방향 무관 sf) |
| m4_v6 | + ASYM_BONUS_W=0.02 unsigned + AMP_ASYM_W=0.01 unsigned | F 2/6, 단 방향 정합 0/2 ✗ (hack) |
| m4_v7 | signed (`target_sign · sf_signed`, `target_sign · amp_signed`) | F 2/6 정합 2/2 ✓ but 약 B 1/6 발생 |
| **m4_v8** | DC linear `|window_mean|` (mean² → \|mean\| 강화) | **도달 6/6 ✓, 약 B 0/6 ✓** but F 정합 0/2 ✗ |

핵심 코드 (m4_v8 최종, `sim/fish_env.py`):
```python
DC_PEN_W=0.05; ASYM_BONUS_W=0.02; AMP_ASYM_W=0.01
self._ctrl_window[:-1] = self._ctrl_window[1:]; self._ctrl_window[-1] = ctrl_now
window_mean = float(self._ctrl_window.mean())
dc_pen = abs(window_mean)                             # m4_v8 linear
ctrl_ac = self._ctrl_window - window_mean
sf_signed = (ctrl_ac>0).sum()/12 - 0.5
amp_signed = max(0, ctrl_ac.max()) - max(0, -ctrl_ac.min())
asym_match = self._target_sign * sf_signed             # target_sign=sign(theta-π)
asym_signed_bonus = max(0, asym_match - 0.05)
amp_signed_bonus  = max(0, target_sign*amp_signed - 0.1)
```

`reset()`: `self._target_sign = float(np.sign(theta - np.pi))` — 직진(=π)은 0, signed bonus 0.

### m4_v8 (DC linear + signed asym/amp, fine-tune from m4_v2 s1)

s3a 9 runs (3 seed × 0°/±15°) — `runs/m4_v8_seed{0,1,2}/s3a_arc15/model.zip`. ~29분 학습 (3 seed 병렬, 조기 졸업).

**결과 (확장 metric 재측정 후)**:
- 도달 9/9 ✓ (직진 3/3, ±15° 회전 6/6)
- B (|DC|>0.5) 0/9 ✓, 약 B (|DC|>0.3) 0/9 ✓ (m4_v7 약 B 1/6 → 강화)
- F 학습 (sf_dev≥0.1) 2/6 (seed0/1 +15°), **F 방향 정합 0/2 ✗** (signed bonus 약화)
- yaw 방향 정합 0/6 (head가 target 향함 X)
- side-slip 확장: slip 평균 17~25°, align_final < 0.7 회전 6/6 — head 정렬 X. 강 side-slip (slip>30°) 1/9뿐 → **mixed mode 8/9 (D wagging + 미세 lateral drift + head 정렬 부족)**

**core insight**: ±15° target은 거의 -x 직선 (옆 ±0.13m). D 직진 추진 + DC offset/asym 미세 lateral drift로 도달. yaw 무관. → **회전 mechanism 학습 X, weak benchmark**.

### m4_v8b (m4_v8 reward 그대로, fresh s1→s3b)

m4_v8 reward 그대로 (`sim/fish_env.py` 수정 X) s1부터 fresh 학습. tag `m4_v8b_seed{0,1,2}`. ~91분 (3 seed 병렬, 4 stage 자동 진행 `--start-stage 1 --end-stage 4`).

**결과 (각 stage 졸업 model 측정)**:
| stage | 도달 | F 학습 | side-slip | mixed | F/D 회전 ✓ |
|---|---|---|---|---|---|
| s1 (직진, 3 runs) | 3/3 ✓ | - | 0/3 | 3/3 | 0/3 (head wobble) |
| s2 (sr 0.04, 3 runs) | 3/3 ✓ | - | 0/3 | 3/3 | 0/3 |
| s3a ±15° (9 runs) | 6/9 (회전 3/6) | 6/6 | 0/9 | 9/9 | 0/9 |
| **s3b ±30°** (9 runs) | **8/9** (회전 5/6) | 4/6 (정합 0/6) | **3/6 회전** | 6/9 | **0/9** |

**s3b 회전 6 runs**: yaw ±14~17° 크기 회전 ✓ (회전 mechanism trigger), 단:
- side-slip ✗ 3/6 (slip 36~38°, head align ≤ 0.24)
- mixed 3/6 (slip 25~28°, align 0.48~0.58)
- F 방향 정합 0/6 (sf_signed 부호 ≠ target_sign)
- B 차단 후퇴: |DC|>0.3 2/6 (DC linear penalty fresh 학습에서 효과 ↓)
- y_disp ±0.17m (target_y 정합) → 도달은 head 정렬 없이 side-slip + 약 B 혼합

### m4_v8 vs m4_v8b 비교 (같은 reward, 다른 init)

| 지표 | m4_v8 (fine-tune from m4_v2 s1) | m4_v8b (fresh s1) |
|---|---|---|
| s3a 회전 도달 | 6/6 ✓ | 3/6 (후퇴) |
| s3b 회전 도달 | - | 5/6 |
| F 학습 (sf_dev≥0.1) | 2/6 | 4/6 ↑ |
| **F 방향 정합** | 0/6 | 0/6 (둘 다 X) |
| 약 B (\|DC\|>0.3) | 0/6 ✓ | 2/6 후퇴 |
| side-slip ✗ | 1/9 | 3/9 ↑ |
| F/D 회전 ✓ | 0/9 | 0/9 |

### 결론 (cycle 4 sweep 종료)

1. **단순 reward shaping (DC penalty + signed asym/amp)로 진짜 F/D 회전 학습 trigger 불가** — 4 cycle (m4_v3~v8) + init 변경 (m4_v8b) 모두 F 방향 정합 ≤ 2/6.
2. **m4_v8b fresh = 회전 mechanism은 trigger ✓** (yaw ±14~17°), 단 mechanism = side-slip + 약 B + 비대칭 sweep 혼합. head 정렬·F 정합 X.
3. **s3b가 진짜 검증대** — s3a는 target offset 작아 D 직진 + lateral drift로 도달 가능, s3b는 회전 mechanism 강제됨.

### 측정 도구 확장 (`sim/analyze_policy.py`)

기존 ctrl pattern (slow_frac, DC, step_rate)에 더해 trajectory 매 step 측정:
- `slip_angle[t]` = acos(head_dir · vel_dir) per step
- `slip_mean`·`slip_max`
- `head_target_align_final` = 마지막 20 step head·target_dir 평균
- `yaw_oscillation` = (yaw_max - yaw_min) / |yaw_total|
- 진행 mode 판정: slip<15°+align>0.7 → "F/D 회전 ✓", slip>30° → "side-slip ✗", 그 외 "mixed"

### 다음 카드 후보 (m4_v9)

- A. **head align reward ↑** — current align_weight (10s ep 0.02, 30s+ ep 0.012) 증액으로 head 정렬 학습 압력 ↑
- B. **side-slip 직접 penalty** — env에 `slip_angle > 30°` penalty 추가
- C. **frame_skip ↑** (42 → 60) — D wagging 추진력 약화 → F 압력 ↑
- D. **action wrap (F 강제)** — env가 sin(α·t + φ_offset) 자체 명령으로 wrap → "emergent 유지" 원칙 위배 단 최후 수단
- E. **종료 + s3c 진행** — m4_v1 s3b 모델 (v33 천장 돌파, 회전 학습 ✓)로 ±60° 진행. 단일 motor F mode는 추후 별도 카드
