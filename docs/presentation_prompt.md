# PPT 발표 자료용 Claude 프롬프트 (처음 발표 — 전체 균형)

이 문서는 KUFIsh_III RL 프로젝트의 발표 자료 작성에 Claude를 활용할 때 컨텍스트로 던질 프롬프트다. 그대로 복사해서 Claude(claude.ai 웹 / Claude Code / 기타)에 붙여넣으면 된다.

> 마지막 갱신: 2026-05-13 (v27 결과 반영)
> 출처 데이터: [`CLAUDE.md`](../CLAUDE.md), [`docs/training_log.md`](training_log.md)

---

# 프로젝트 컨텍스트: KUFIsh_III 단일 관절 물고기 로봇 강화학습

처음 발표하는 자료이므로 **연구 배경부터 향후 계획까지 모든 섹션이 균형 있게** 들어가야 한다. 특정 결과나 학습 카드에 집중되지 않도록 주의.

---

## 1. 연구 배경 및 동기

- **생체모방 수중 로봇**: 물고기의 BCF(Body-Caudal-Fin) 영법은 자연계에서 검증된 효율적 추진 방식. 단 액추에이터·제어 복잡도가 높음.
- **단일 관절 모델 선택 이유**: 액추에이터 최소화로 실물 제작·제어 단순화. **단점은 정밀 회전 제어가 어렵다는 것 — 본 연구의 핵심 도전 과제**.
- **강화학습 선택 이유**: 비선형 유체역학 + 비대칭 꼬리 진동 패턴은 분석적 컨트롤러로 다루기 어려움. RL이 직접 정책을 학습.
- **시뮬레이션 우선 (sim-first)**: 실물 학습은 시간·하드웨어 손상 비용 큼. MuJoCo로 학습 후 sim2real로 이전 계획.

## 2. 실물 하드웨어 (KUFIsh_III)

- 자체 CAD 설계 (`fish_urdf/RL_SIM_MODE_3_description/`) + 3D 프린팅 (PLA)
- **단일 꼬리 관절 (active)**: BL4260 BLDC 모터 (forcerange ±3 Nm), 가동 범위 ±20°
- **수동 지느러미 (passive)**: Ecoflex 00-30 실리콘 (density 1070 kg/m³). 능동 제어 없이 변형만으로 wave-thrust 생성
- **질량 분포**: base 2.43 kg + tail 0.07 kg + fin 0.01 kg
  - base 질량·CoM은 **MATLAB CG.m으로 측정값 검증** (sim2real 정합성 위함)

## 3. 시뮬레이션 환경 (MuJoCo)

- **3DOF planar 운동** (`slide_x` + `slide_y` + `hinge_yaw`): 수면 영법 가정으로 roll·pitch·z 제거.
  - 함정: freejoint(6DOF)로 두면 pitch wobble로 추진 방향이 뒤집힘 → planar로 잠금 필수
- **수중 물리**: density 1000 kg/m³, viscosity 0.001 (실제 물), 중력 비활성. added mass + drag는 `fluidshape="ellipsoid"`로 자동 계산
- **추진 검증** (`freq_sweep.py`): 꼬리 ±1 sine 12초 → 1~6 Hz 모든 주파수에서 머리 방향 전진 (1Hz −0.49 m, 3Hz −2.17 m, 6Hz −3.36 m)
- **회전 능력 진단** (`yaw_test.py`): 
  - 대칭 sine wagging: 0.49°/s (느림)
  - 비대칭 D2 패턴 (75% 음 + 25% 양): **4.6°/s** ⭐
  - → ±90° 회전이 20초 안에 물리적으로 가능. RL이 비대칭 패턴을 발견해야 함.

## 4. 강화학습 알고리즘 (SAC)

- **SAC (Soft Actor-Critic)** — Maximum-Entropy off-policy actor-critic
  - Twin Q-network (overestimation 방지)
  - Entropy 자동 조절 (auto-tuned α) — 탐색·이용 균형
  - off-policy + Replay buffer 200k → 샘플 효율 ↑
- 구현: Stable-Baselines3 `stable_baselines3.SAC`
- 하이퍼파라미터: LR=3e-4, batch=256, γ=0.99, τ=0.005, ent_coef "auto_0.1" (floor 0.003)

## 5. 환경 설정 (Gymnasium)

- **관측 (31D = 11 + 20)**:
  - 물리 상태 9D: tail/fin qpos·qvel, world velocity (vx·vy), yaw_rate, sin·cos(yaw)
  - target 상대 위치 2D (world frame)
  - **action history 최근 20 step** — 비대칭 wagging 패턴 학습에 필수 (대칭 sine만으론 회전 불가)
- **행동 (1D)**: tail motor ctrl ∈ [−1, +1] (실제 ±20°)
- **보상 함수**:
  ```
  reward = progress·10                # 매 step 거리 감소량 × 10
         + (10 if reached else 0)     # 성공 보너스
         - 0.001 · ctrl²              # 제어 비용
         + 0.012 · align              # 정렬 (머리 방향 vs 목표)
         + 0.005 · |yaw_rate|         # 회전 시도 (v21 핵심)
  ```
  - 마지막 `|yaw_rate|` 항이 본 연구의 핵심 발견: **"회전 시도 자체에 보상"으로 "정렬만 하고 추진 안 함" 모드 깨고 회전·추진 시퀀스 학습 강제**

## 6. Curriculum 학습 설계

- 가설: random target 전체 원은 단일 모터에 너무 어려움 → **좌우 회전 폭을 점진 확장**
- 6단계:

| Stage | 목표 | target 범위 | ep 길이 |
|---|---|---|---|
| s1_forward | 직진 | π fixed | 10s |
| s2_anchor | 정밀 추진 | π fixed, r=0.04 m | 10s |
| s3a_arc15 | 작은 회전 | π ± 15° | 30s |
| s3b_arc30 | 중간 회전 | π ± 30° | 30s |
| s3c_arc60 | 큰 회전 | π ± 60° | 30s |
| s3d_arc90 | 최대 회전 | π ± 90° | 30s |

- **졸업 조건**: deterministic eval 100 ep에서 reach_rate ≥ 90% (TB stochastic 진동의 false positive 차단)

## 7. 실험 진행 — 주요 돌파 (v1 ~ v27)

각 카드는 "한 가지 가설을 검증하기 위한 1회 학습". 27개 카드 중 의미 있는 돌파만:

| 시기 | 카드 | 핵심 변경 | 결과 (s3d 또는 핵심 stage) |
|---|---|---|---|
| Baseline | v1 ~ v10 | reward·entropy 다양한 조합 | s3d 13~24% (천장 미돌파) |
| **시간적 표현력** ⭐ | **v11** | action history N=8 → 16 | **s3d 26%** (첫 천장 돌파, +9%p) |
| 표현력 추가 | v17 | N=24 | 정점 30% 일시 돌파 |
| 학습량 가설 | v18 | s3d 1M | reject (학습량만으론 못 깸) |
| **회전 보상** ⭐⭐ | **v21** | `+|yaw_rate|·0.005` 항 추가 | **s3d 29%** (단독 천장 돌파) |
| 학습량 + s3b | v22 | s3b max_steps 1M | TB 90% but deterministic 79% (**TB false positive 함정 발견**) |
| Stochasticity 줄이기 | v25-A | ent_floor linear decay (0.003 → 0) | det 84% (gap 12%p → 8%p) |
| **Multi-seed 검증** ⭐⭐⭐ | **v27** | v25-A × seed 0/1/2 | **s3b det 90/92/90% (mean 90.7%)** — **카드 valid 확정**, single seed 결론 정정 |

## 8. 현재 결과 (2026-05-13)

| Stage | 회전 범위 | reach_rate (deterministic) |
|---|---|---|
| s1·s2 | 직진 | 100% ✓ |
| s3a | ±15° | ~100% ✓ |
| **s3b** | **±30°** | **90.7% ± 1.2%** ✓ ⭐ |
| s3c | ±60° | ~42% ← **다음 미달 stage** |
| s3d | ±90° | ~28% |

**v27 학습 시간**: 3 seed 평균 ~70분 (졸업 step 245k ~ 880k, ×3.6 분산)

## 9. 학습 방법론 (메타 통찰)

v22~v26 single seed 결과(79~84%)가 "카드 천장"으로 잘못 결론났던 것을 v27 multi-seed가 정정한 경험에서 정착:

1. **multi-seed × deterministic eval** — single seed로 카드 천장 판단 금지 (lower outlier 위험)
2. **`--det-check` callback** — TB stochastic 진동(±10%p)의 false positive 차단
3. **`eval_stages.py` 6 stage forgetting 점검** — fine-tune이 이전 stage 망가뜨리는지 확인
4. **`model_best.zip`** — 학습 후반 catastrophic drift 우회 (peak 시점 자동 저장)

→ 이 4가지가 정착된 카드 평가 표준.

## 10. 향후 계획

1. **선결**: v27 best 모델 6 stage 평가 → catastrophic forgetting 확인 (~20분 × 3 seed)
2. **v28** (다음 카드): s3c 본격 학습 — multi-seed × 1M × det-check (예상 ~3시간)
3. **v29 이후**: s3d 학습 (s3c 졸업 후)
4. **장기 목표**: 
   - full circle target 학습 (현재는 좌우만, 사용자 결정으로 제외 중)
   - **sim2real 이전** — 실물 KUFIsh_III에서 학습 정책 검증

## 11. 알아둘 점 (해결된 함정들)

- **6DOF freejoint** → pitch wobble로 추진 방향 뒤집힘 (3DOF planar로 해결)
- **곱셈 보상 (reward × align)** → mode collapse (가산식 정착)
- **단일 축 카드 한계** (N / entropy / reward 가중치 각각 단독으로는 천장 못 깸) → yaw reward (v21)가 본질적 돌파
- **single seed로 카드 평가** → multi-seed 표준 (v27 정정 사례)

---

# 요청

이 컨텍스트를 바탕으로 PPT 발표 자료 작성 부탁드립니다. **처음 발표하는 것이라 모든 섹션이 균형 있게 들어가야 합니다** — 특정 결과나 학습 카드(예: v27)에 집중되지 않도록.

## 발표 정보
- **분량**: 약 10~15분 / 슬라이드 12~15장
- **청중**: RL·로보틱스 기본 지식 있는 연구실 동료
- **언어**: 한국어

## 작성 요청

다음 슬라이드 구조로 **각 1장씩 균등 배분**해 주세요:

1. **표지** — 프로젝트명·발표자·날짜
2. **연구 배경 및 동기** (위 1번 섹션)
3. **실물 하드웨어 KUFIsh_III** (위 2번)
4. **시뮬레이션 환경** (위 3번)
5. **RL 알고리즘 SAC** (위 4번)
6. **환경 설정 — 관측·행동·보상** (위 5번)
7. **Curriculum 학습 설계** (위 6번)
8. **실험 진행 — 주요 돌파 (v11 N=16, v21 yaw reward)** (위 7번 일부)
9. **실험 진행 — TB false positive 함정과 v27 multi-seed** (위 7번 일부)
10. **현재 결과** (위 8번)
11. **학습 방법론 정착 (메타 통찰)** (위 9번)
12. **향후 계획** (위 10번)
13. **마무리 · Q&A**

각 슬라이드마다:
- **제목** (간결한 한국어)
- **핵심 bullet 3~5개**
- **시각화 제안** (다이어그램·표·그래프·실물 사진·플롯 등 — 자료 형식 명시)

어려운 용어(BCF, mode catalysis, ent_floor, stochastic vs deterministic 등)는 처음 등장할 때 1줄 풀이를 본문에 포함해 주세요. 청중이 발표 중 막히지 않도록.
