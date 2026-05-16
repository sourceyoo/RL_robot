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
