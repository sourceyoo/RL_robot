# 학습 로그 — main flow v1 ~ v28+

CLAUDE.md에서 분리한 학습 history 전체 — 결과 표·매트릭스·산문 진단·통찰·RL 카드 함정·결정적 변경 회고·다음 카드 후보. CLAUDE.md는 규칙·사실·코드 작업 가이드만 유지 (모델·물리 함정 #1~#6 포함).

> **s3d_90 라인 (구 v22~v30, 분리됨)** → 별도 파일 [`docs/s3d_90_line.md`](s3d_90_line.md). 본 파일은 main flow만 다룸.

## 작성 규칙

- 학습 history (표·매트릭스·산문·통찰·RL 카드 함정·다음 후보) 모두 본 파일에 작성. CLAUDE.md엔 학습 결과 추가하지 않음.
- 산문 진단: `## 주요 진단 (산문) — main flow` 끝에 `### vN-X (한 줄 제목)` 새 절 추가.
- 종결·돌파 카드면 본 파일의 그룹화 표·변경점 표·Stage 진행 비교 표·s3d 결과 표·핵심 통찰·결정적 변경 회고·다음 카드 후보 갱신.
- 새 RL 카드 함정 발견 시 `## RL 카드 함정 (#7~)` 끝에 번호 이어서 추가.
- CLAUDE.md는 코드/모델/규칙 변경 시만 갱신 (학습 결과 변경에 따른 갱신 없음).

## GitHub Release 인덱스 (main flow)

[`models-v1`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v1), [`models-v2`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v2), [`models-v5`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v5) (실패), [`models-v10`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v10), [`models-v12`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v12), [`models-v13`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v13), [`models-v14`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v14), [`models-v17`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v17) (N=24, 정점 30% 천장 일시 돌파), **[`models-v21`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v21)** (yaw reward 천장 단독 돌파) ⭐⭐.

**모델 binary 손실 / 백업** (main flow):

- v11: v12 학습이 sim/runs/s3d_arc90/ 덮어씀 — 영구 손실.
- v16·v19·v20: `/tmp/models-v{16,19,20}-backup/` 로컬 백업.
- v18: v17 s3d만 fine-tune (1M step) — sim/runs/s3d_arc90/이 그 후 덮어써짐, 별도 백업 없음.

**release 없음** (main flow): v3·v4·v6~v9·v11·v15·v16·v18·v19·v20.

---

## 학습 결과 요약 — main flow v1 ~ v29-C

**현재 미달 stage = s3b, 카드 박스 86~87% (90% 임계 미달)**. v22 79 → v25-A 84 → v26-A 80 → v27 85.7 → v28 84.3 → v29-B 87.0 (progress×15) → v29-C 86.0 (σ 11.4 ↑) → **v30-A 87.3 ± 5.7%p (yaw sign-aware, σ 단조 ↑ 추세 깨짐)** ⭐. progress axis는 ×15 sweet spot 확정, sign-aware axis는 **σ 줄임 효과 입증** ✓ but mean stalling. forgetting 없음 (s1·s2·s3a 100% 3 seed). **다음 카드 = v30-A + heading² 축 누적** (v31) 또는 v30-B (multi-seed +4로 σ 5.7%p 진정성 검증).

### 결별 그룹화 (main flow)

| 그룹 | 멤버 | 특징 | s3d 천장 |
|---|---|---|---|
| A. Baseline + 가산식 reward | v1·v2·v3 | `+0.02·align` 가산항 | 24% |
| B. 곱셈 보상 실패 | v5·v6·v7 | mode collapse (머리 반대 / 도망) | 16~20% |
| C. 가산식 복귀 + ent_floor | v8·v9·v10 | 균등/차등. ent_floor 카드 종결 | 13~17% |
| **D. 정책 표현력 확장** ⭐ | **v11** | action history N=8→16. ±90° 천장 첫 돌파 | **26%** |
| E. NN 확장 | v12 | net_arch [256,256,128]. s3d catastrophic | 6% ❌ |
| F~I. reward/NN 카드 | v13~v16 | align ep 비례/fix·reach 5→10·NN 회귀 | 19~21% |
| **J. action history N=24** | **v17** | s3c·d 동시 회복, 정점 30% 일시 돌파 | **25%** |
| K. 학습량 ↑ (N=24) | v18 | 500k → 1M — 학습량 부족 가설 reject | 25% |
| L. N 절충 (N=20) | v19 | s3a 96% best + s3c·d mode collapse | 23% ❌ |
| M. v19 + ent_floor ↑ | v20 | mode collapse 해결 ✓ but reach 후퇴 (정렬 best 0.84) | 19% |
| **N. yaw reward** ⭐⭐ | **v21** | `+YAW_W·\|yaw_rate\|·0.005`. **v11 천장 단독 돌파**. 모든 stage 회복 | **29%** |
| **O. s3b 학습량 ↑** ⭐⭐⭐ | **v22** | s3b max_steps 250k → **1M**, v21 정책 이어받기 (`--start-stage 4`). **s3b TB 90% 졸업** (654k 조기 종료, det 79% — 함정 #13) | 32% (s3b TB 90%, det 79%) |
| P. ent_floor schedule | **v25-A** | s3b `ent_floor 0.003 → 0` linear decay (1M). **stochastic-det gap 12%p → 8%p**로 감소 ✓ but det 84%로 90% 미달 | s3b det **84%** |
| Q. det check + 학습량 추가 ❌ | **v26-A** | v25-A 정책 + det check callback (TB 90% trigger 후 det 50 ep 확인) + 학습 1M 추가. **s3b det 80%로 후퇴** — single seed 결과로 카드 천장 추정 | s3b det **80%** (single) |
| **R. v25-A × multi-seed** ⭐⭐ | **v27** | v25-A 카드 seed 0/1/2 + `--det-check` (50 ep). **학습 졸업 50 ep det 90/92/90% (mean 90.7%)** — 일견 졸업. 그러나 Step 0 후속 100 ep eval로 **90/81/86% (mean 85.7%)** — **50 ep는 sample noise로 운 좋게 측정**. s3b 졸업 가설 partial. forgetting 없음 (s1·s2·s3a 100%) | s3b det **50 ep 90.7%** vs **100 ep 85.7%** ⚠ |
| **S. v25-A × multi-seed × `--det-episodes 100`** ❌ | **v28** | v25-A 카드 seed 0/1/2 + `--det-check` **100 ep**. seed1만 진짜 졸업 (1M 직전 det 90%), seed 0/2는 max_steps 종료 (det 85%/76%). **100 ep eval 86/90/77% (mean 84.3%)** — v27 100 ep와 σ 안. **현 카드 천장 ~84% 확정, H1 reject** | s3b det **84.3 ± 5.4%p** ❌ |
| **T. progress 가중치 ↑** ⭐ | **v29-B** | v25-A 카드 + `progress×10 → ×15`. 3 seed: 2 졸업 (405k/405k det 93%/93%) + 1 outlier (75%). **100ep eval mean 87.0 ± 8.5%p** (+2.7%p, **새 axis 첫 효과**). step 효율 ×1.5. **86~89 박스 진입, 90% 미달 −3%p** | s3b det **87.0 ± 8.5%p** ⚠ |
| **U. progress ×18 (axis 강화)** ❌ | **v29-C** | v29-B 그대로 + `progress ×15 → ×18`. 3 seed: 2 졸업 (190k/195k det 92%/93%) + 1 outlier (73%). **100ep eval mean 86.0 ± 11.4%p** — v29-B와 σ 안. **mean stalling + σ ↑ — axis 강화 한계** ✗. 졸업 step ×2 빨라짐 (190k) but mean ↓. s3a 100→98 미세 후퇴 (작은 회전 정밀도 약화) | s3b det **86.0 ± 11.4%p** ❌ |
| **V. yaw sign-aware (σ 줄임 axis)** ⭐ | **v30-A** | v29-B (progress×15) + `YAW_SIGN_W · yaw_rate · sign(yaw_err) · 0.005` 신규 가산. 3 seed: 1 졸업 (575k det 95%) + 2 중간 (88/80, det 86/85). **100ep eval mean 87.3 ± 5.7%p** — v29-B mean 동급 but **σ 단조 ↑ 추세 깨짐** (11.4 → 5.7) ⭐. outlier seed 깊이 ↓ (v29 75/73 → v30 80/88). mean 90% 미달 -2.7%p but 매트릭스 행3 (86~89/σ≤6) 통과 — 다음 후보 axis 추가 단계 | s3b det **87.3 ± 5.7%p** ⚠ |

### 버전별 변경점 (main flow v1~v28)

v22까지 변경점은 위 그룹화 표 참조. v22 이후 카드 (s3b 천장 시도):

| 변경 | v22 | v25-A | v26-A | v27 | v28 | **v29-B** | **v29-C** | **v30-A** |
|---|---|---|---|---|---|---|---|---|
| 기반 | v21 (yaw·N=20·ent_floor 차등) | v22 | v25-A | v25-A | v25-A | **v25-A + progress ×15** | v29-B + progress ×18 | **v29-B + yaw sign-aware** |
| init | v21 | v22 | v25-A | v25-A | v25-A | v25-A | v25-A | v25-A |
| seed | default | default | default | 0/1/2 | 0/1/2 | 0/1/2 | 0/1/2 | **0/1/2** |
| det check | — | — | 50 ep | 50 ep | 100 ep | 100 ep | 100 ep | 100 ep |
| s3b TB peak | 91% | 92% | 90% | 90/91/94% | 90/90/91% | 91/91/99% | 91/96/96% | 90/98/96% |
| **s3b det (100 ep)** | 79% | 84% | 80% | 85.7%* | 84.3%(86/90/77) | 87.0%(93/75/93) | 86.0%(73/93/92) | **87.3%(80/94/88)** |
| σ | — | — | — | 3.7 | 5.4 | 8.5 | **11.4** ↑ | **5.7** ⬇ ⭐ |
| 졸업 step | 654k F⁺ | 615k F⁺ | 1M (det 미달) | 705k/880k/245k | ~1M/~1M/1M(s1만) | 405k/1M/405k | 1M/195k/190k | **1M/575k/1M** (1 졸업, det 95%) |
| 한 줄 평가 | TB false | gap ↓ ✓ 84% | single 80% | 50ep 운 좋음 | 100ep 천장 ~84% | +2.7%p 첫 axis 효과 | mean stalling + σ↑ | **σ 단조 ↑ 추세 깨짐** ⭐ |

*v27 50ep det 90.7% / Step 0 100ep 85.7%, F⁺ = TB false positive (det 미달)

### Stage 진행 비교 (main flow)

**v22까지는 TB stochastic 기준. 함정 #13 발견 후 v22~v28은 deterministic eval로 보정**:

| Stage | v8 | v10 | v11 | v17 | v19 | v20 | v21 | **v22 (TB/det)** | **v25-A** | **v26-A** | **v27 (50/100ep)** | **v28** | **v29-B (×15)** | **v29-C (×18)** | **v30-A (sign)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s3a (±15°) | 74 | 89 | 67 | 66 | 96* | 63 | **90** ⭐ | 100/100 | 98 | 100 | 100/100 ✓ | 100 ✓ | **100** ✓ | 98 ⚠ | **100** ✓ |
| s3b (±30°) | 70 | 72 | 44 | 62 | 61 | 53 | 69 | **90/79** | 84 | 80 ↓ | 90.7/85.7 ⚠ | 84.3 ± 5.4 ❌ | 87.0 ± 8.5 ⚠ | 86.0 ± 11.4 ❌ | **87.3 ± 5.7** ⚠⭐ |
| s3c (±60°) | **46** | 28 | 38 | 33 | 28 | 21 | 38 | 37/42 | 43 | 42 | —/44.0 | 41.7 | 44.3 | 48.3 | **45.0** |
| s3d (±90°) | 17 | 13 | 26 | 25 | 23 | 19 | 29 | **32**/26 | 30 | 28 | —/30.7 | 29.0 | 31.0 | 32.3 | **31.7** |

*v19 96%는 seed 분산 운. v22~v29-C는 `--start-stage 4`로 s3b만 학습 (init=v25-A). **v29-C 결과**: s3b mean **86.0 ± 11.4%p** — v29-B 87.0과 σ 안 (-1.0%p). **progress axis 강화 한계점**. σ는 v28 5.4 → v29-B 8.5 → v29-C 11.4로 단조 증가 (axis 강화가 분산도 키움). s3a 100→98 미세 후퇴 (작은 회전 정밀도 약화). s3c·s3d는 학습 안 했지만 fine-tune transfer로 +1~4%p. forgetting 없음 (s1·s2 100%, s3a 98).

### s3d (±90°) 결과 — 마지막 100 ep 윈도우 (v11~v22)

| 버전 | reach | avg_align | final_dist | 한 줄 평가 |
|---|---|---|---|---|
| **v11** ⭐ | 26% | +0.58 | 0.57m | N=16 — 천장 첫 돌파 |
| **v17** | 25% | +0.43 | 0.55m | N=24 — s3c·d 동시 회복, 정점 30% 일시 돌파 |
| v18 | 25% | +0.57 | 0.57m | N=24 + 1M — 학습량 부족 가설 reject |
| v19 ❌ | 23% | −0.37 | 0.35m | N=20 단독 — s3c·d mode collapse |
| v20 | 19% | **+0.84** ⭐ | 0.78m | ent_floor ↑ — 정렬 best but 추진 worst |
| **v21** ⭐⭐ | 29% | +0.51 | 0.67m | **+yaw reward — v11 천장 단독 돌파**. 정점 38%, ep 21.8s best |
| **v22** ⭐⭐⭐ | **32%** ⭐ / 28%† | +0.73 | 0.64m | s3b 1M 졸업 정책 → s3d 1M. run1 32%/peak 50%, run2 28%/peak 50%. 정착 카드의 s3b 90% 위 첫 결과 |

†v22 s3d 두 번 run (s3d_arc90_1·_2) — 의미 분석 보류.

---

## 주요 진단 (산문) — main flow

### v8 (mode 정상화)

v5~v7 곱셈 보상 mode collapse → 가산식 복귀 + ent floor 0.005. avg_align 처음 양수(+0.58). s3c 46% (이전 best). 그러나 s3d 17% — fish가 회전+추진 시퀀스 못 만듦 (정책 표현력 부족 가설).

### v9 (ent_floor stage 의존성)

floor 0.005 → 0.002. s3a 74→90% (가설 입증) but s3c 46→29%, s3d 17→13% 후퇴. **단일 floor로 전체 cover 불가**.

### v10 (차등 floor 한계)

stage별 차등 (s1~s3a 0.002 / s3b 0.003 / s3c 0.005 / s3d 0.006). 메커니즘 OK (ent_coef 정확히 floor에 머무름) but s3c 28%·s3d 13%. **이전 stage의 낮은 entropy가 만든 narrow mode를 다음 stage 높은 floor가 못 깸** — entropy 양 ≠ 정책 표현력. **ent_floor 카드 종결**.

### v11 (천장 첫 돌파) ⭐

action history N=16으로 1 wag cycle 커버 (obs 19→27D). **±90° 26% — v1~v10 24% 천장 +9%p 돌파**. ep_seconds 22.7s (효율). 단 작은 회전 후퇴 (s3a 67%·s3b 44%) — N과 stage 복잡도 매칭 trade-off. **진짜 병목이 정책 표현력(시간적 패턴)이었음 입증**.

### v12 (NN 확장 catastrophic ❌)

net_arch [256,256] → [256,256,128]. 작은 회전 회복(s3a 84%·s3b 67%) but **s3d 26→6% catastrophic interference** — 학습 시작 50% (s3c 정책) → 끝 6%. v10 패턴(정렬 best, 추진 worst) 재현. 진단: align 누적 +18 vs reach +5 = **3.6:1로 align dominance** → "정렬만" mode. **차등 N/NN은 SAC.load 호환으로 막힘** (함정 #9).

### v13 (catastrophic 해결, 정렬 부족)

`align_weight = 0.02·10/ep_seconds` (30s → 0.0067). **catastrophic 완전 해결** (학습 우상향, s3d 6→19%) but align +0.19로 정렬 신호 부족. 비율 1.2:1 — 적정값 0.012 (비율 2.2:1) 추정.

### v14 (정렬 회복, 천장 미돌파)

30s ep 직접 fix `0.012`. 정렬 회복 (avg_align +0.19→+0.50, v11 +0.58 근접). catastrophic 안 일어남 (15→20% 우상향). 단 s3d 20% / **s3c 모든 버전 중 worst (20%, v8 best 46% 대비 −26%p)**. 실측 비율 7.4:1로 align dominance 여전.

### v15 (s3c 회복 ✓, 천장 여전)

reach 보너스 5→10. **s3c 20→27% 회복 (가설 검증 ✓)** + s3b 59→62%. 단 s3d 21% — v11 26% 천장 미돌파. **s3d 학습 곡선 V자×2 변동** (50→12→21%, v12 직전 패턴). 비율 3.6:1(=v12) but reach가 sparse라 catastrophic 안 일어남 — **신호 형태(dense vs sparse) > 비율**.

### v16 (NN default 회귀 — 작은 회전 회복 ✓, 천장 미돌파)

net_arch [256,256] 회귀. **가설 입증**: 작은 회전 회복 (s3a 69→80%·s3b 62→70%, v15 대비 +11/+8%p) — NN [256,256,128]이 narrow mode 학습 가속한 부작용 제거. 단 s3c 27→21% (-6%p), s3d 21→19% (-2%p) — NN 회귀가 큰 회전엔 표현력 부족. **s3d 학습 곡선 V자×2 사라짐** (50→27→28→19% 안정 후 점진 후퇴) — V자×2 변동은 NN 확장 + reach=10 조합 부작용이었음. v12 best (s3a 84%·s3b 67%) 못 따라감 — NN 확장이 그만큼 강력한 효과였음을 역으로 입증. **v11 26% 천장은 NN 카드도 못 깸**: reward 4종 + NN 1종 = 5장 모두 미돌파.

### v17 (N=24 — 큰 회전 회복 ✓, 천장 거의 근접) ⭐

action history N=16 → 24 (1.5 wag cycle 커버, obs 27→35D). NN default + reward 카드 모두 v16 그대로. 가설(시간적 표현력 추가 확장)에 대응. **결과**:

- ✅ **s3c·s3d 동시 회복**: s3c 21→33% (+12%p, v8 46% 빼면 best), s3d 19→25% (+6%p, v11 26%에 -1%p 근접).
- ✅ **s3d 정점 30%로 v11 26% 천장 일시 돌파** (270~310k 구간) — **동일 카드 조합에서 N 축만으로 천장 가능성 입증**. 단 후반 25%로 안정 (학습량 부족? max_steps 500k).
- ✅ **ep_seconds 23.1s — v11 22.7s 다음 효율** (N=24 표현력이 더 빠른 도달 정책).
- ✅ **s3d 학습 곡선 단조 우상향** (0→14→27→30→25, V자 없음).
- ❌ 작은 회전 후퇴: s3a 80→66% (-14%p), s3b 70→62% (-8%p) — v11 패턴 재현 (N과 stage 복잡도 매칭 trade-off).
- ⚠ avg_align +0.43 (v16 +0.53 대비 -0.10) — 정렬 신호 양보, 큰 회전 reach 우선 정책 분배.

→ **N 축이 천장 카드의 가장 인과 명확한 축임 재입증**. v11이 N=8→16으로 24%→26% 돌파했고, v17이 N=16→24로 19%→25% (정점 30% 일시 돌파). 다만 학습 안정성·작은 회전 trade-off는 여전. 다음은 **max_steps ↑ 또는 N=20 절충**.

### v18 (v17-B: s3d 1M fine-tune — 학습량 부족 가설 reject)

v17 s3c model에서 init, s3d만 max_steps 500k → 1M. 다른 변수 모두 v17과 동일.

- ❌ **학습량 가설 reject**: end 25% (v17 동일). 정점 30% 3회 도달(22.9k·312k·944k)하나 100ep window 안정화 실패.
- ✅ **정렬 회복**: avg_align +0.43 → +0.57 (v11 +0.58 동급). v17의 "정렬 양보, 큰 회전 우선" mode가 학습량 ↑로 양쪽 강화.
- ✅ ep_rew_mean 6.6 → 8.2 — 정렬 누적 reward 증가.
- ❌ 22.9k에 정점 30% 도달 (v17 312k와 매우 다른 시점) — early peak 후 진동, **학습량과 무관한 본질적 진동**.

→ **단일 모터 + N=24 + 현 reward 카드의 이론적 천장 ≈ 25~30%, 평균 25%**. v11 26% 천장이 카드 변경 없이 깨질 가능성 낮음. 진단: 정책이 mode 전환(정렬↔추진)을 못 하고 25% 부근에서 진동 — `yaw_test.py` 4.6°/s × 20s = 92°라 마진 작아 반복 안정 어려움. 다음은 **N=20 절충(v17-C) 또는 ent_floor 강화(v17-D) 또는 더 근본적 카드(HER, custom yaw reward)**.

### v19 (v17-C: N=20 절충 — s3a 96% best ⭐ + s3c·d mode collapse ❌)

N=24 → 20 (1.25 wag cycle, obs 35→31D). N=16(v11 s3a 67%)·N=24(v17 s3a 66%) 사이 절충 가설.

- ⭐ **s3a 96% (170k 조기 종료)** — v10 89% 동급 best 능가, **모든 버전 중 best**. N=20이 작은 회전 학습엔 강력.
- ❌ **s3c·d mode collapse**: avg_align s3a +0.58 → s3b +0.10 → s3c **−0.17** → s3d **−0.37** 단조 감소. ep_rew_mean −2.5 (음수).
- final_dist 0.35m (< 시작 0.5m) — 도달은 일부 하지만 **머리 반대 방향**. 정책이 "거꾸로 가면서 가까워지면 prog 양수" mode 발견 → align 음수 + prog 양수 + reach 일부에 갇힘.
- v17(N=24) align +0.43과 비교 — 같은 카드에서 N 4 차이로 **mode 양→음 반전**. **N=20이 작은 회전 학습 가속하면서 narrow mode 잔재가 큰 회전 entropy floor(0.005/0.006)로도 못 깨짐**.

→ **N=20은 단독 카드로 부적합** — s3a 96% best는 의미 크지만 큰 회전에서 v5/v7 함정 #8 패턴 재현 (mode collapse). 다음은 **v19-D (s3c/d ent_floor ↑로 mode collapse 막기)** 또는 **N=20 + reward 변경 (align dominance 약화로 collapse 막기)**, 또는 **v11(N=16)로 회귀**.

### v20 (v19 + ent_floor s3c·d 강화 — mode collapse 해결 ✓ + reach 후퇴 ❌)

s3c 0.005 → 0.008, s3d 0.006 → 0.010. 다른 변수 v19 그대로.

- ✅ **mode collapse 완전 해결**: avg_align s3d −0.37 → **+0.84** (모든 버전 best, v10 +0.61 능가). 가설(narrow mode 잔재 + entropy 부족) 정확히 입증.
- ✅ ent_coef 정확히 floor에 머무름 (0.008/0.010).
- ❌ **reach 큰 후퇴**: s3a 96→63% (-33%p, but v19 96%는 seed 분산이라 의미 약함), s3c 28→21%, s3d 23→19%.
- ❌ **final_dist 0.78m** (시작 0.5m보다 멀어짐) — v10 "정렬 best, 추진 worst" 패턴 더 심하게 재현. ep_rew_mean 3.4 (v17 6.6의 절반).
- 진단: ent_floor 0.010이 entropy 강제하나 SAC actor가 그 entropy를 *추진* 학습엔 못 쓰고 *정렬* 정확도에만 사용. **이전 stage(s3a 0.002 narrow mode)가 만든 정책 분포가 entropy↑로도 못 깨짐 — v10 함정 재발견**.

→ **ent_floor 카드 한계 입증**: collapse는 막지만 reach 천장 못 깸. **N·entropy·reward 단일 축 카드 모두 한계 도달**. 다음은 근본 카드 (HER / Custom yaw reward) 또는 N=16(v11) 안전 회귀.

### v21 (yaw reward — v11 천장 단독 돌파) ⭐⭐

`+YAW_W·|yaw_rate|` (YAW_W=0.005). v20 진단 직접 대응 — 회전 시도 자체에 인센티브로 "정렬만 mode" 깨고 회전·추진 시퀀스 학습 강제.

- ⭐⭐ **v11 26% 천장 단독 돌파**: s3d end **29%** (+3%p), 정점 **38%** (모든 v best, v17 30% 능가).
- ⭐ 모든 stage 동시 회복: s3a 90% (159k 조기) · s3b 69% (모든 v best) · s3c 38% · s3d 29%.
- ⭐ **ep_seconds 21.8s** — 모든 v best.
- ✅ 추진 회복: final_dist 0.78m → 0.67m. avg_align +0.51 양수 안정.
- ✅ align 단조 우상향: s3d −0.13 → 0.27 → 0.38 → 0.52 → 0.58 → 0.57 → 0.51 (end).

→ **천장 돌파의 본질 = "회전 시도 자체에 보상"**.
- N (시간적 표현력) — 정점 30% 가능
- entropy (mode 전환) — collapse 해결
- **yaw reward (회전·추진 시퀀스) — 천장 돌파 ⭐**

s3b/c init 음수 align(-0.98, -0.40)에서 학습할수록 양수 회복 — yaw reward 받으려 회전만 하다가 align reward가 옳은 방향 강제. **두 신호의 균형이 핵심**.

### v22 (s3b 학습량 ↑ — s3b 90% 졸업, 학습량 부족 가설 입증) ⭐⭐⭐

s3b `max_steps` 250k → **1M**. 다른 변수 v21 그대로 (N=20·ent_floor 차등·yaw `|·|`·0.005). `--start-stage 4`로 v21 정책 이어받아 s3b부터 진행. 코드 주석상 `v31-A` (s3d_90 line v22~v30과 번호 충돌 회피 표기, main flow 트래킹은 v22).

**가설 검증 결과 — 학습량 부족 가설 입증**:

- ⭐⭐⭐ **s3b end 90%, peak 91% (654k 조기 종료)**: v21 단일 seed 69%, 강제 진행이었던 게 **카드 한계가 아니라 학습량 부족** 입증. 1M 안에 90% 임계 도달.
- s3c end 37% (peak 50%, 348k): v21 38%와 동급 — 카드 변경 없으므로 변동 없음.
- s3d end 32% / peak 50% (996k): v21 29% +3%p. s3d_arc90 디렉토리에 두 번 run (`_1` 32% / `_2` 28%, 둘 다 peak 50%) — 의미 분석 보류.

**핵심 통찰**:

- **`max_steps`가 곧 카드 한계 진단의 함정** — 250k에 강제 진행되면 "카드 한계로 69%"로 오해하기 쉬움. 학습량 충분(1M)을 먼저 줘야 카드 효과/한계 판단 가능.
- s3b·s3c·s3d 모두 같은 의심 가능 — s3c도 학습량 부족일 수 있음. 다음 카드(s3c 1M)에서 검증.
- s3d 32% (peak 50%)은 s3b 90% 졸업 정책 위에서 측정한 첫 결과. s3d_90 line의 v22~v30(s3c 미달 상태의 s3d fine-tune)과는 base가 다르므로 직접 비교 어려움 — s3c 90% 달성 후 s3d 다시 평가가 fair.

→ **누적 진행**: v21에서 s3b 69%로 강제 진행이 천장처럼 보였으나, 학습량만 늘려도 90% 졸업. **단계 졸업의 본질적 병목 = 학습량**, 카드 변경 전에 학습량부터 충분히 줘야 한다는 교훈.

### v22 후속 진단 — TB false positive 발견 (함정 #13) ⚠

v22 졸업 직후 `eval_stages.py` (사용자 후속 요청으로 새로 작성, 100 ep deterministic) 6 stage 측정 결과:

| Stage | v22 TB end | v22 deterministic |
|---|---|---|
| s1·s2·s3a | 100% | **100%** |
| **s3b** | **90% (peak 91%)** | **79%** ⚠ |
| s3c | 37% (peak 50%) | 42% |
| s3d | 32% (peak 50%) | 26% |

**v22 s3b "90% 졸업"이 false positive**. 원인 두 갈래:

- (a) `CurriculumStopCallback`이 TB stochastic 91% 시점에 trigger (마지막 100 measurement mean 85.9%, min 79%, max 91% — 4 측정점 정점 시점 운).
- (b) **stochastic-deterministic gap 12%p**: SAC random action의 entropy 기여가 정책 효과의 일부를 담당 → deterministic eval (mean action만)에서 약함.

**대응**:
- CLAUDE.md 함정 #13 추가 (TB 졸업 신호 false positive).
- 졸업 확정은 항상 `eval_stages.py` deterministic eval. TB는 학습 중 신호일 뿐.
- 다음 카드(v25-A, v26-A)는 이 gap을 직접 공격하는 방향.

### v25-A (ent_floor schedule — stochastic-det gap 12%p → 8%p)

**카드**: s3b `ent_floor 0.003 → 0` linear decay (0~1M step). 가설: 학습 후반에 `log_ent_coef`를 0으로 압박해 정책의 stochasticity를 점점 줄이면, 학습 종료 시점에 deterministic action ≈ stochastic action 수렴. 즉 stochastic-det gap을 학습 메커니즘으로 직접 좁힘. 다른 변수 v22 그대로.

**결과**:

| Stage | v22 det | v25-A det | Δ |
|---|---|---|---|
| s1·s2 | 100/100 | 100/100 | 0/0 |
| s3a | 100 | 98 | −2 (noise) |
| **s3b** | **79** | **84** | **+5** ⭐ (gap 12%p → 8%p) |
| s3c | 42 | 43 | +1 |
| s3d | 26 | 30 | +4 |

- 615k step에서 TB callback 92% trigger 조기 종료. final ≡ best (모델 동일).
- ✓ **가설 부분 성공**: gap 좁혀짐 (TB 92% → det 84%, gap 8%p), 방향성 옳음.
- ❌ **90% 미달**: schedule만으로 부족. v26-A에서 det check + 학습량 추가로 검증 시도.
- forgetting 없음 (s1·s2·s3a 100/100/98).

### v26-A (det check + 학습량 1M 추가 — 카드 천장 입증) ❌

**카드**: v25-A 카드 그대로 + (1) train.py `CurriculumStopCallback`에 `det_env_fn` 추가 — TB 90% trigger 후 deterministic eval 50 ep 자동 실행, det < 90%면 cooldown 100k step 후 재평가. (2) v25-A 모델을 init으로 추가 1M 학습. 가설: v25-A가 615k에서 stochastic 92% trigger로 조기 종료된 것이 한계라면, det check + 학습량 추가로 진짜 det 90% 도달 가능.

**det check trigger 기록** (학습 중):

```
trigger 1 (step ~180k): TB 92% → det 82% (41/50) ✗ — cooldown until 280k
trigger 2 (step ~880k): TB 90% → det 72% (36/50) ✗ — cooldown until 980k (후퇴)
trigger 3 (step 1000k): TB 90% → det 82% (41/50) ✗ — cooldown until 1100k
```

1M 완주, det 모든 시점 미달 → 진짜 조기 종료 안 됨.

**6 stage deterministic eval 결과**:

| Stage | v22 | v25-A | **v26-A final** | v26-A best |
|---|---|---|---|---|
| s1·s2·s3a | 100/100/100 | 100/100/98 | **100/100/100** | 100/100/100 |
| **s3b** | **79** | **84** ⭐ | **80** ↓ | **71** ↓↓ |
| s3c | 42 | 43 | 42 | 38 |
| s3d | 26 | 30 | 28 | 25 |

- **best (TB peak)가 final보다 더 나쁨** (71% < 80%): best는 TB stochastic 92% peak 시점에 저장 → 그 시점 det는 71%. TB-det gap이 학습 중 21%p까지 벌어진 사례.
- v22 79% → v25-A 84% → v26-A 80% — **s3b det 80~84% 진동**. 학습량 1M 추가로 정책 안정화 ❌.

**핵심 통찰**:

- ❌ **카드 자체 천장 입증**: 현 카드(yaw `|·|`0.005·N=20·ent_floor schedule)의 s3b 본질적 천장 ~80~84%. 학습량/ent_floor/det check 모두 깨지 못함.
- ⚠ **best 모델의 함정**: TB stochastic peak가 deterministic 기준 최악일 수 있음. 함정 #13 + best metric 의존성 둘 다 문제.
- 단일 축 카드(ent_floor/det check)는 함정 #9 (v12~v16 단일 축 카드 한계) 패턴 재현. **새 축 필요** (reward 방향성 / N / multi-seed).

→ **누적 진행**: v22 졸업이 false positive였고, v25-A·v26-A 두 카드(schedule + det check)로도 s3b 80~84% 천장 못 깸. v22~v26 4개 카드의 단일 seed 결과인 만큼, 다음 카드는 **multi-seed**로 분산 확인 또는 새 reward 축(yaw sign-aware 등) 시도.

### v27 (v25-A × 3 seed — s3b 졸업 본격 확정, 카드 천장 가설 reject) ⭐⭐⭐

**카드**: v25-A 카드 그대로 (ent_floor `0.003 → 0` linear decay, yaw `|·|`·0.005·N=20). init = v25-A 모델 (`sim/runs/v25/s3b_arc30/model.zip`), `--start-stage 4 --end-stage 4` (s3b만), `--det-check` 활성, seed=0/1/2 3 run. 명령:

```bash
python3 sim/curriculum.py --start-stage 4 --end-stage 4 --no-viewer --det-check \
  --seed N --tb-tag model3_v27_seedN --runs-subdir v27_seedN \
  --init-from sim/runs/v25/s3b_arc30/model.zip
```

**가설**: v22 79% / v25-A 84% / v26-A 80% — single seed 결과로 "현 카드 본질 천장 80~84%" 결론. multi-seed로 분리:
- 가설 1 (카드 한계): 3 seed 모두 78~85% 수렴 → 진짜 천장 → 새 reward 축 (yaw sign-aware / N) 필요.
- 가설 2 (seed 분산): 70~92% 분산 → multi-seed가 정착 방법론, best seed 90%+ 가능.
- 가설 3 (카드 valid): 모두 90%+ → v22~v26 single seed 결과는 lower outlier들이었음.

**결과 — 가설 3 입증 ⭐⭐⭐**:

| seed | 졸업 step | TB stochastic | **det eval** | 학습 시간 | 패턴 |
|---|---|---|---|---|---|
| 0 | 705k | 90% | **90.0%** (45/50) | 82분 | 218k에서 TB peak 94% → 후퇴 → 705k에서 진짜 졸업 |
| 1 | 880k | 91% | **92.0%** (46/50) | 92분 | 547k부터 안정 80~88% → 880k에서 trigger |
| 2 | 245k | 94% | **90.0%** (45/50) | 36분 | **145k에서 TB 90% false positive 차단** (det 84%) → cooldown 후 245k 진짜 졸업 |
| **mean** | 610k | 91.7% | **90.7% ± 1.2%** | 70분 | — |

- ⭐⭐⭐ **3 seed 모두 det 90%+ — s3b 졸업 본격 확정**.
- v22~v26 (single seed 79~84%)는 lower outlier들이었음. **카드(v25-A ent_floor schedule + yaw `|·|`·0.005·N=20·1M)는 valid**.
- 졸업 step 245k~880k (×3.6 분산) — seed에 따라 학습 속도 큰 차이. max_steps는 충분히 길게(1M+) 잡아야.

**v27 seed2 함정 #13 실증** (det check 효과):

```
step 145k: TB stochastic 90% (best 갱신) → det eval 12/50 = 84% ❌ false positive
          → 학습 계속, cooldown until 245k
step 245k: TB stochastic 94% → det eval 45/50 = 90% ✓ 진짜 졸업
```

det check 없었다면 145k에서 잘못 졸업 (v22 패턴 재발). **`--det-check` 인프라 효과 본격 입증** — 새 학습은 default 활성 권장.

**핵심 통찰 (메타)**:

1. **single seed로 카드 평가 절대 금지** — s3d_90 라인에서 발견된 교훈이 main flow에도 동일하게 적용. v22~v26 4개 카드 single seed 결과(79/84/80)는 모두 같은 카드의 lower tail이었음.
2. **s3b 졸업의 본질** = ent_floor schedule (stochastic-det gap 좁힘) + 충분 학습량(1M) + multi-seed + det check.
3. **카드 평가 표준 = multi-seed × deterministic eval × det check + 6 stage forgetting 점검** (4가지 모두 필수).

**미해결 (다음 작업)**:

- ⚠ **6 stage deterministic eval 미실시** — v27 best 모델 3 seed (특히 seed1 92%) 각각 `eval_stages.py` 실행 필요. s1·s2·s3a forgetting 여부 확인 후 다음 카드 (s3c) 진행.

→ **누적 진행**: v22~v26 4개 카드의 single seed 결론(s3b 천장 80~84%) 정정. **s3b는 v25-A 카드 + multi-seed로 50 ep 졸업 임계 통과 (det 90.7%)**. 단 Step 0 후속 100 ep eval에서 sample noise 발견 — 후속 진단 참조.

### Step 0 (v27 후속 — 100 ep eval로 sample noise 발견) ⚠

**작업**: v27 best 모델 3 seed (seed1 det 92% best, seed0 90%, seed2 90%) 각각 `eval_stages.py` 100 ep로 6 stage 측정 — catastrophic forgetting 점검 + 졸업 진짜 확정.

**6 stage deterministic eval 결과 (100 ep × 3 seed)**:

| Stage | seed1 | seed0 | seed2 | mean |
|---|---|---|---|---|
| s1_forward | 100% | 100% | 100% | 100% ✓ |
| s2_anchor | 100% | 100% | 100% | 100% ✓ |
| s3a_arc15 | 100% | 100% | 100% | 100% ✓ |
| **s3b_arc30** | **90%** | **81%** | **86%** | **85.7% ± 3.7%p** ⚠ |
| s3c_arc60 | 43% | 44% | 45% | 44.0% |
| s3d_arc90 | 32% | 29% | 31% | 30.7% |

**핵심 발견**:

1. ✅ **catastrophic forgetting 없음** — 3 seed 모두 s1·s2·s3a 100/100/100%. s3b fine-tune이 이전 stage 망가뜨리지 않음. mixed sampling/rehearsal 새 축 카드 불필요.

2. ⚠ **50 ep det 졸업 측정 vs 100 ep eval 5%p gap**:
   - 학습 졸업 시 50 ep det: 90 / 92 / 90% (mean 90.7%)
   - Step 0 100 ep eval: 90 / 81 / 86% (mean **85.7%**)
   - seed1만 정확히 90%, seed0는 −9%p 빠짐, seed2는 −4%p
   - **50 ep는 sample noise로 운 좋은 측정** — 100 ep로 분산 평균화하면 진짜 분포가 낮음

3. s3c·s3d는 v22~v26-A 측정값(42·28%)과 거의 동일 — v27이 s3b만 fine-tune이라 다른 stage는 init(v25-A)에서 변경 없음.

**메타-결론**: 

- **졸업 확정 표준 = 100 ep `eval_stages.py`**. 50 ep det check은 학습 trigger엔 적합하나 졸업 확정 측정으론 부족.
- **`--det-episodes` 50 → 100으로 강화 필요** (함정 #16, CLAUDE.md 갱신).
- v27 결론 "s3b 졸업 본격 확정"은 "50 ep 임계 통과 but 100 ep 기준 85.7% 부족"으로 정정.

**다음 카드**: **v28 (s3b 안정화, --det-episodes 100)** — 학습 중 100 ep det check로 진짜 90% 도달까지 학습 지속.

### v28 (v25-A 카드 × 3 seed × `--det-episodes 100` — 카드 천장 84% 확정) ❌

**카드**: v25-A 카드 그대로 (ent_floor `0.003 → 0` linear decay, yaw `|·|`·0.005, N=20). init = v25-A 모델 (v27과 동일). seed 0/1/2 × `--det-check --det-episodes 100`. `--start-stage 4 --end-stage 4` (s3b만), max_steps 1M. 명령:

```bash
python3 curriculum.py --no-viewer --start-stage 4 --end-stage 4 \
  --seed N --tb-tag model3_v28_seedN --runs-subdir v28_seedN \
  --init-from sim/runs/v25/s3b_arc30/model.zip \
  --det-check --det-episodes 100
```

**가설**:
- H1 (메인): 100 ep det check로 진짜 90% 도달까지 학습 지속 → mean 90~92% 달성
- H2 (reject 시): 카드 진짜 천장 86~89% → 새 축 필요

**학습 중 det check 결과** (3 seed 합산, false positive 차단 실증):

| trigger | TB stochastic | det 100 ep | 결과 |
|---|---|---|---|
| 1차 | 90% | 83% | ✗ cooldown 705k |
| 2차 | 90% | 82% | ✗ cooldown 805k |
| 3차 | 91% | 69% | ✗ cooldown 835k (gap 22%p) |
| 4차 | 90% | 90% | **✓ seed1 진짜 졸업** (~1.0M 도달 직전) |
| 5차 | 91% | 81% | ✗ cooldown 940k |
| 6차 | 92% | 85% | ✗ cooldown 1.1M (1M 직전 false) |
| 7차 | 90% | 69% | ✗ cooldown 980k |
| 8차 | 91% | 76% | ✗ cooldown 1.095M |

- **seed 1**: 진짜 졸업으로 조기 종료 (~1M 직전, TB 90% → det 100/100)
- **seed 0**: 1M max_steps 종료 (final det check 85%로 미달)
- **seed 2**: 1M max_steps 종료 (final det check 76%로 미달)

**6 stage deterministic eval 결과 (100 ep × 3 seed final 모델)**:

| Stage | seed0 | seed1 | seed2 | **mean** | v27 mean (참고) |
|---|---|---|---|---|---|
| s1_forward | 100 | 100 | 100 | **100** ✓ | 100 |
| s2_anchor | 100 | 100 | 100 | **100** ✓ | 100 |
| s3a_arc15 | 100 | 100 | 100 | **100** ✓ | 100 |
| **s3b_arc30** | **86** | **90** | **77** | **84.3 ± 5.4%p** ⚠ | 85.7 |
| s3c_arc60 | 40 | 43 | 42 | 41.7 | 44.0 |
| s3d_arc90 | 29 | 32 | 26 | 29.0 | 30.7 |

**핵심 발견**:

1. ❌ **H1 reject — s3b 100 ep mean 84.3%** (90% 임계까지 -5.7%p). v27 85.7%과 σ 범위 (-1.4%p 차이로 통계 noise). **현 카드 본질 천장 = ~84%로 확정**.

2. ⚠ **개별 seed 분산 큼 (σ 5.4%p)**:
   - seed1 90%: 진짜 졸업 정책 → eval에서도 정확히 임계 통과
   - seed0 86%: 1M 종료, det check 85% 결과와 일관
   - seed2 77%: 1M 종료, det check 76% 결과와 일관
   - **seed별 학습 운에 따라 σ 큼** — 1 seed만 졸업 임계 통과, 나머지 2 seed는 분명히 미달

3. ⚠ **s3b 84%대 천장의 일관성**:
   - v22 (79%) → v25-A (84%) → v26-A (80%) → v27 100ep (85.7%) → **v28 (84.3%)**
   - 카드 변형 (ent_floor schedule / det check / multi-seed / 100 ep) 모두 80~86% 박스 안.
   - **v27의 50 ep 90.7% upper outlier 확정 — sample noise 정확한 측정값**.

4. ⭐ **`--det-episodes 100` 효과 실증** — TB stochastic-det gap이 22%p까지 벌어지는 경우도 잡아냄 (3차 trigger). 50 ep로는 운 좋게 trigger 가능했으나 100 ep는 진짜 90% 도달만 허용.

5. ✓ **catastrophic forgetting 없음** — 3 seed 모두 s1·s2·s3a 100%, s3c·s3d도 v25-A init 그대로 (-2.3%p / -1.7%p, σ 안). v25-A → v28 1M 추가 학습이 다른 stage 손상 없음.

**메타-결론**:

- **현 카드 (yaw `|·|`·0.005·N=20·ent_floor schedule·NN [256,256])의 진짜 천장 = s3b ~84%**. 단일 변경(학습량 / det check / multi-seed / 100 ep) 모두 천장 못 깸.
- **다음 카드 = 새 축 필수** (단순 학습량 증가나 측정 강화론 임계 90% 미돌파).
- 새 축 후보: yaw sign-aware reward / N 변경 / progress 가중치 ↑ / NN 확장 ([256,256,128]) / heading error term 추가 등.

**다음 카드**: **v29 (s3b 새 축)** — v29-B (progress 가중치 ↑, ×10 → ×15) 우선 진행 중.

### v29-B (progress×15 — s3b 첫 axis 효과 +2.7%p) ⭐

**카드**: v25-A 카드 그대로 + `reward progress · 10 → · 15`. init=v25-A, seed 0/1/2, `--det-check --det-episodes 100`, `--start-stage 4 --end-stage 4`.

**가설**: v22~v28 5개 카드(학습량/ent_floor schedule/det check/multi-seed/100 ep) 모두 80~86% 박스 → 단일 변경 한계 → **새 axis (progress 가중치 ↑) 필요**. 추진 신호 강화로 "정렬만 mode" 잔재 제거.

**졸업 기록**:

| seed | 졸업 | step | TB peak | 학습 중 det | det check trigger 수 |
|---|---|---|---|---|---|
| 0 | ✓ 진짜 졸업 | 405k | 96% | 80/88/89/81/**93** | 5회 (4 false + 1 진짜) |
| 1 | ❌ 1M max_steps | — | 92% | 74/81/**85** | 3회 (모두 false, mode lock 40k) |
| 2 | ✓ 진짜 졸업 | 405k | 99% | 79/89/**93** | 3회 (2 false + 1 진짜) |

- ⭐⭐ **2/3 진짜 졸업** — v22 이후 단일 카드 첫 multi-seed 졸업
- seed0/2 둘 다 정확히 405k에 졸업 (인상적 일관성)
- TB-det gap: v22 12%p / v25-A 8%p / v29-B seed2 **6%p** (TB 99% → det 93%)
- seed1만 학습 40k에 TB 91% 일찍 mode lock → 1M까지 못 깨고 final 75%

**100 ep eval 결과** (sim/runs/v29_seedN/s3b_arc30/model.zip):

| Stage | seed0 | seed1 | seed2 | mean | v28 mean | Δ |
|---|---|---|---|---|---|---|
| s1·s2·s3a | 100/100/100 | 100/100/100 | 100/100/100 | **100/100/100** ✓ | 100 | 0 |
| **s3b** | **93** | **75** | **93** | **87.0 ± 8.5%p** ⚠ | 84.3 | **+2.7** ⭐ |
| s3c | 50 | 35 | 48 | 44.3 | 41.7 | +2.6 |
| s3d | 35 | 25 | 33 | 31.0 | 29.0 | +2.0 |

**핵심 발견**:

1. ⭐ **s3b 단조 증가 추세 +2.7%p** — v22~v28 5개 카드 박스(79~86)에서 **첫 의미 있는 돌파**. progress axis 가설 입증.
2. ⚠ **σ 5.4 → 8.5%p ↑** — progress 강화가 분산도 키움. seed1 outlier가 mean을 끌어내림.
3. **ep_seconds 단축**: v28 5.6~8.3s → v29-B 4.1~4.9s (졸업 seed). 추진 신호 강화 효과 직접 확인.
4. **catastrophic forgetting 없음** — 3 seed 모두 s1·s2·s3a 100% 유지.
5. **transfer 효과 (positive)**: s3c·s3d도 +2~3%p 작은 개선 — s3b axis 강화가 위 stage에도 도움.
6. ❌ **90% 임계 -3%p 미달** — H1 (mean ≥ 90%) reject. seed0/2 단독은 93% 졸업, seed1 75%로 mean 87%.

→ **카드 평가 매트릭스 (training_log L574)**: 87.0% → "86~89% 박스" → **"progress 더 강하게 (×18~20) 또는 yaw sign-aware 조합"**. v29-C 진행.

### v29-C (progress×18 — axis 강화 한계 확정) ❌

**카드**: v29-B 그대로 + `progress · 15 → · 18`. 다른 변수 모두 v29-B/v25-A와 동일 (init=v25-A, seed 0/1/2, det check 100 ep).

**가설**:
- H1 (main): axis 더 강화로 s3b mean ≥ 90% 도달
- H2 (reject): 정렬 약화로 75~85%로 후퇴
- H3 (partial): σ ↑, 0/2 95%+ but 1 outlier로 86~89% 박스

**졸업 기록**:

| seed | 졸업 | step | TB peak | 학습 중 det |
|---|---|---|---|---|
| 0 | ❌ 1M 종료 | — | 91% (665k) | 89/85/**73** (후퇴) |
| 1 | ✓ 진짜 졸업 | **195k** ⭐ | 96% | 80/**93** |
| 2 | ✓ 진짜 졸업 | **190k** ⭐ | 96% | 74/84/**92** |

- ⭐ **졸업 step 190~195k** — v29-B 405k 대비 **×2 효율**. progress×18로 axis 더 강하니 학습 더 빠름.
- seed0는 665k에 TB 91% best 도달 후 det 89→85→73으로 후퇴 → 1M 종료 (mode lock 깊어짐).

**100 ep eval 결과** (sim/runs/v29c_seedN/s3b_arc30/model.zip):

| Stage | seed0 | seed1 | seed2 | mean | v29-B mean | Δ |
|---|---|---|---|---|---|---|
| s1·s2 | 100/100 | 100/100 | 100/100 | **100/100** ✓ | 100/100 | 0/0 |
| **s3a** | **94** | 100 | 100 | **98.0** ⚠ | 100 | **-2.0** |
| **s3b** | **73** | **93** | **92** | **86.0 ± 11.4%p** | 87.0 | **-1.0 (σ 안)** |
| s3c | 42 | 50 | 53 | 48.3 | 44.3 | +4.0 |
| s3d | 27 | 32 | 38 | 32.3 | 31.0 | +1.3 |

**핵심 발견**:

1. ❌ **H1 reject** — s3b mean 86.0% (-1.0%p, σ 안에 stalling). v29-B와 동일 86~89 박스 → **progress axis 강화 한계점 확정**.
2. ⚠ **H3 적중 (σ 강화)** — σ v28 5.4 → v29-B 8.5 → v29-C **11.4%p** 단조 증가. axis 강화가 분산도 키움.
3. ⭐ **졸업 step 효율 ×2** — 190~195k 졸업 (v29-B 405k 대비). axis 강화가 학습 속도엔 명확한 도움.
4. ⚠ **s3a 미세 후퇴 100 → 98** (seed0 94%) — progress×18이 너무 강해서 작은 회전 정밀도 약화. s3a 단조 100% 유지 못한 첫 사례 (v22 이후).
5. **outlier 더 깊어짐** — v29-B seed1 75% → v29-C seed0 **73%**. progress 강화가 mode lock도 깊게 만듦.
6. **s3c +4%p**: 흥미롭게도 s3c 평균이 v29-B 44.3 → v29-C 48.3로 더 올라감 (s3b 학습이 s3c에 transfer).

**메타-결론**:

- **progress axis 강화의 effective range**: ×10 (v22~v28 84%) → ×15 (v29-B 87%) → ×18 (v29-C 86%). **×15가 sweet spot**, ×18은 oversaturation.
- **단일 axis 한계 입증** — progress만으로는 86~89 박스 못 넘음 (s3b 90% 미달).
- **다음 카드 = 새 axis (yaw sign-aware / heading squared) 또는 multi-seed 5~7로 분산 vs 본질 판정** 필수.

### v30-A (yaw sign-aware — σ 줄임 axis 효과 입증) ⭐

**카드**: v29-B (progress×15) 그대로 + `YAW_SIGN_W · yaw_rate · sign(yaw_err) · 0.005` 신규 가산. sign(yaw_err)은 head_dir × rel 2D cross product. 다른 변수 모두 v29-B와 동일 (init=v25-A, seed 0/1/2, det check 100 ep, progress·15).

**가정**: 단일 tail motor 물리상 절대적 제자리 회전 불가능 → yaw|·| 보상 그대로 보존. sign 항은 *회전 방향* 정확도 강제 (target 왼쪽이면 왼쪽 회전만 +).

**가설**:
- H1 (main): outlier seed mode lock 해소 → mean ≥ 90% AND σ ≤ 6%p
- H2 (reject): 반대 방향 회전 페널티가 mode catalysis 약화 → 후퇴
- H3 (partial): mean 86~89% but σ ↓ → 다음 axis 추가 단계

**졸업 기록**:

| seed | 졸업 | step | TB peak | 학습 중 det | final |
|---|---|---|---|---|---|
| 0 | ❌ 1M 종료 | — | 90% (385k) | 86% | det 86% |
| 1 | ✓ 진짜 졸업 | **575k** ⭐ | 98% | 85/86/85/**95** | **det 95%** (사상 최고) |
| 2 | ❌ 1M 종료 | — | 96% (890k) | 80/84/87/80/83/82/85 | det 85% |

**100 ep eval 결과** (sim/runs/v30a_seedN/s3b_arc30/model.zip):

| Stage | seed0 | seed1 | seed2 | mean | v29-B | Δ |
|---|---|---|---|---|---|---|
| s1·s2·s3a | 100/100/100 | 100/100/100 | 100/100/100 | **100** ✓ | 100 | 0 |
| **s3b** | **80** | **94** | **88** | **87.3 ± 5.7%p** | 87.0 ± 8.5 | mean +0.3 (오차 안), σ **-2.8** ⬇ |
| s3c | 43 | 49 | 43 | 45.0 | 44.3 | +0.7 |
| s3d | 30 | 33 | 32 | 31.7 | 31.0 | +0.7 |

**핵심 발견** ⭐⭐:

1. ⭐⭐ **σ 단조 ↑ 추세 깨짐** — v28 5.4 → v29-B 8.5 → v29-C **11.4** → v30-A **5.7** ⬇. v22 이후 처음으로 σ 감소. **outlier seed mode lock 해소 axis 본격 발견**.
2. **mean 효과 미미** (+0.3 vs v29-B) — H3 적중 (mean stalling but σ ↓).
3. **outlier 깊이 ↓**: v29-B 75 / v29-C 73 → **v30-A 80**. v22 이후 lower outlier 가장 높음.
4. **seed1 det 95%** — v22 이후 단일 seed det 최고. mode lock 완전 해소 가능성 입증.
5. **forgetting 없음**: 3 seed s1·s2·s3a 100% 유지.
6. **transfer 약함**: s3c +0.7, s3d +0.7로 v29-B 대비 거의 동일. axis 추가가 fine-tune transfer엔 영향 없음.

**ep_s + align 양상** (seed별):
- seed0: ep_s 7.55s, align +0.82 — "정렬 best + reach 80% 낮음" (v20 패턴 부분 재현, but reach 양호)
- seed1: ep_s 3.76s, align +0.69 — "이동+정렬 균형" 졸업 mode
- seed2: ep_s 5.34s, align +0.70 — 중간 양상

**메타-결론**:

- ⭐⭐ **σ 줄임 axis 발견** — 매트릭스 σ ≤ 6%p 기준 통과 (5.7%p). v30-A axis는 카드 정착 가치 ✓.
- **mean 천장 ~87%** — sign-aware 단독으로 90% 미돌파. 다음 axis 누적 필요.
- **누적 진행**: v29-B/C에서 progress sweet spot ×15 확정, v30-A에서 σ 줄임 axis 추가. **다음 = v30-A + heading² (v31)** 또는 v30-A + multi-seed +4 (σ 5.7%p 진정성 검증).

---

## 핵심 통찰 (main flow)

- ✅ **Stage 1·2** 60k step에 100% (모든 버전).
- ⭐ **N 축이 천장 카드의 가장 인과 명확한 축** — v11 (+9%p) → v17 (정점 30%).
- ❗ **단일 축 카드(N/ent/reward) 모두 v11 26% 천장 못 깸**. 천장 돌파 = **yaw reward (v21)**.
- ⭐⭐ **yaw reward (v21)**: 회전 시도 자체에 +보상 → "정렬만 mode" 깨고 mode catalysis trigger. v11 천장 단독 돌파.
- ⭐⭐⭐ **s3b 학습량 부족 (v22)**: v21까지 s3b max_steps 250k가 짧았던 것. 250k → 1M으로 **s3b TB 90% 달성** (654k 조기 종료). 다른 stage도 같은 의심 — `max_steps`는 카드 한계 진단 전에 학습량부터 충분히 줘야 한다는 교훈.
- ⚠ **v22 졸업의 false positive (함정 #13)**: TB 90%는 stochastic 진동 정점 운. deterministic eval은 79% — **s3b 90% 졸업 무효**. 졸업 확정은 항상 `eval_stages.py` deterministic.
- ⚠ **stochastic-det gap 본질 (v25-A)**: `ent_floor 0.003 → 0` linear decay로 학습 후반 정책을 deterministic하게 압박 → gap 12%p → 8%p로 감소. 단 det 84%로 90% 미달 (single seed).
- ❌ **v26-A single seed 80%**: v25-A 정책 + det check + 1M 추가 학습 = s3b det 80% (single). v22 79% → v25-A 84% → v26-A 80% 진동을 카드 천장으로 추정했으나 **v27이 single seed artifact임을 반박**.
- ⭐⭐ **v27 multi-seed로 s3b 졸업 임계 통과 (50 ep 기준)**: 같은 v25-A 카드 3 seed → 학습 졸업 시 det **90 / 92 / 90%** (mean 90.7 ± 1.2). v22/v25-A/v26-A 79~84% single seed 결론을 부분 반박. 졸업 step 245~880k (×3.6 분산, seed 운).
- ⚠ **Step 0 100 ep eval — 50 ep는 sample noise로 운 좋은 측정**: 같은 모델 100 ep로 재측정 → 90/81/86% (mean 85.7%, 5%p gap). **seed1 만 정확히 90%**, seed0 81%·seed2 86%로 임계 미달. **50 ep det check는 표본 적어 학습 중 졸업 trigger에는 적합하나 진짜 졸업 확정 측정으로는 부족**. 함정 #16 추가.
- ✓ **forgetting 없음 (Step 0 + v28)**: 3 seed 모두 s1·s2·s3a 100/100/100% — s3b fine-tune이 이전 stage 망가뜨리지 않음. mixed sampling/rehearsal 새 축 카드 불필요.
- ⭐ **det check 인프라 실증 (v27 seed2 + v28 다수)**: 145k에서 TB 90% trigger → det 84% (false positive) → cooldown after 245k에서 TB 94% / det 90% (진짜 졸업). v28도 8회 trigger 중 1회만 진짜 졸업 (TB stochastic-det gap이 22%p까지 벌어진 사례 차단). **`--det-check` 옵션 default 활성 권장**.
- ❌ **v28 (100 ep det check) — 카드 천장 ~84% 확정**: v25-A 카드 그대로 + `--det-episodes 50 → 100` 변경. 3 seed 100 ep eval mean **84.3% (86/90/77)** — v27 85.7%과 σ 안. v22 79% → v25-A 84% → v26-A 80% → v27 100ep 85.7% → v28 84.3% — 카드 변형 5개 모두 80~86% 박스. **단순 학습량/측정 강화론 90% 미돌파, 새 축 필수**.
- ⭐ **v29-B (progress×15) — s3b 첫 axis 효과 +2.7%p**: v25-A 카드 + `progress · 10 → · 15`. 3 seed 100 ep eval mean **87.0% (93/75/93)** — **새 axis 가설 입증** (v22~v28 박스 첫 돌파). 2/3 진짜 졸업 (405k det 93%/93%), seed1만 outlier 75% (40k mode lock). ep_seconds 5.6~8.3s → 4.1~4.9s 단축 (추진 신호 강화 직접 확인). σ 5.4 → 8.5%p (분산 ↑). 90% 임계 -3%p 미달.
- ❌ **v29-C (progress×18) — axis 강화 한계 확정**: v29-B 카드 + `progress · 15 → · 18`. 3 seed 100 ep eval mean **86.0% (73/93/92)** — v29-B 87.0과 σ 안 (-1%p, **stalling**). σ 8.5 → 11.4%p 단조 증가. ⭐ **졸업 step ×2 빨라짐 (190~195k vs v29-B 405k)** but mean 효과 0. s3a 100→98 미세 후퇴 (seed0 94%) — 작은 회전 정밀도 약화. **progress axis effective range ×10~15, ×18은 oversaturation**. 다음 = 새 axis (yaw sign-aware / heading squared) 또는 multi-seed 5~7.
- ⭐⭐ **v30-A (yaw sign-aware) — σ 줄임 axis 효과 입증**: v29-B + `YAW_SIGN_W · yaw_rate · sign(yaw_err) · 0.005` 신규 가산. 3 seed eval mean **87.3 ± 5.7%p (80/94/88)** — mean v29-B 동급 but **σ 단조 ↑ 추세 깨짐** (5.4 → 8.5 → 11.4 → **5.7** ⬇). seed1 진짜 졸업 575k det 95%(v22 이후 최고). outlier seed 깊이 ↓ (v29 75/73 → 80). 가정 "제자리 회전 물리적 불가능 → yaw|·| 보존" 하에 sign 추가 axis가 mode lock 해소 메커니즘. 매트릭스 행3 (86~89/≤6) 통과. mean 천장 ~87% — 다음 axis 누적 필요.

---

## 핵심 yaw_test.py 발견

| 측정 | yaw rate |
|---|---|
| 학습된 v3 정책 | 0.94°/s |
| 대칭 sine | 0.49°/s |
| **D2 (75% 음 + 25% 양 비대칭)** | **4.60°/s** ✓ |

→ ±90° 회전이 20s 안에 물리적 가능 (D2 × 20s = 92°). RL이 비대칭 패턴 발견 못 함이 본질적 병목 — v11 N=16에서 부분 해결.

물리 상한 ~4.6°/s 기준: s3b ±30°는 yaw 여유 큼 — 즉 s3b 84% 천장은 yaw 능력 한계가 아닌 **reward 신호/정책 표현력 한계**.

---

## 가장 결정적이었던 변경 (회고)

1. **3DOF planar** — pitch wobble 제거 (부호 뒤집기 본질)
2. **MATLAB CG.m 값** — sim2real 정합성
3. **MODE_3 fin + Ecoflex 1070** — wave-thrust
4. **action history N=16 (v11)** — ±90° 천장 첫 돌파
5. **yaw_rate 보상 (v21)** ⭐⭐ — v11 천장 단독 돌파 (s3d 29%, 정점 38%)
6. **v22 s3b 학습량 1M** ⭐⭐ — 250k가 부족이었음 입증 (단 TB false positive였음)
7. **v25-A ent_floor schedule + v27 multi-seed** ⭐⭐ — stochastic-det gap 좁힘 + multi-seed로 s3b 50 ep 졸업 임계 통과 (90.7%). v22~v26 single seed 결론 부분 정정.
8. **Step 0 100 ep eval (v27 후속)** — 50 ep det check sample noise 발견 (5%p gap). 졸업 확정 표준 = 100 ep eval 본격 확립. 함정 #16.
9. **v28 (100 ep det check × multi-seed)** ❌ — 카드 천장 ~84% 확정. v22~v28 5개 카드 80~86% 박스. 단순 측정 강화로 임계 못 깬 것 입증 → 새 축 카드 필요. 함정 #17.
10. **v29-B (progress×15 — s3b 첫 axis 효과)** ⭐ — v22~v28 박스 첫 돌파 +2.7%p. 2/3 진짜 졸업 (det 93%/93%, 405k). progress axis 가설 입증. but 90% 임계 -3%p 미달 (seed1 75% outlier).
11. **v29-C (progress×18 — axis 강화 한계)** ❌ — mean 86% stalling, σ 11.4%p ↑. 졸업 step ×2 효율 (190k) but mean 변화 없음. **progress axis effective range ×10~15** 입증 → 함정 #18.
12. **v30-A (yaw sign-aware — σ 줄임 axis 발견)** ⭐⭐ — σ 단조 ↑ 추세 깨짐 (11.4 → 5.7). mean v29-B 동급(87.3) but outlier 해소. seed1 det 95% v22 이후 최고. axis 정착 ✓ but mean 천장 ~87% → v31 (sign + heading² 누적) 또는 multi-seed 검증 필요.

---

## RL 카드 함정 (#7~)

물리·모델 함정 (#1~#6) → CLAUDE.md 참조. 본 섹션은 RL 카드 시행착오 회고.

7. **곱셈 보상 (v5/v7)**: mode collapse (머리 반대·도망). 가산식이 정착.
8. **차등 N / 차등 NN**: SAC.load weight transfer 불가 (obs Box mismatch 또는 layer random init). **curriculum init_from 무력화**.
9. **단일 축 reward/NN 카드 (v12~v16)**: align ratio·reach bonus·NN 확장/회귀 — 모두 v11 26% 천장 미돌파. 단독 카드 한계.
10. **N=24 + 학습량 ↑ (v18)**: 1M도 end 25% — 학습량 부족 가설 reject. N=24 본질적 진동.
11. **N=20 단독 (v19)**: s3a 96% best but s3c·d mode collapse (align −0.37, final_dist 0.35m). narrow mode 매개.
12. **ent_floor 강화 (v20)**: collapse 해결 ✓ but reach 후퇴 ❌. entropy ↑이 정렬에만 활용.
13. **TB 졸업 callback false positive** ⭐⭐⭐: `CurriculumStopCallback`은 random eval window(±10%p 진동)에서 진동 정점 시점에 trigger 가능. 진짜 정책 실력보다 1차 졸업 신호가 ↑ 나옴. **v22 s3b 사례**: TB end 90.0%/peak 91% → deterministic eval 79% (마지막 100 measurement mean 85.9%, min 79%, max 91%). callback이 4 측정점 91% 시점에 trigger되어 false 졸업. → **졸업 확정은 항상 `eval_stages.py` deterministic eval**. TB는 학습 중 신호일 뿐. **v26-A에서 det check callback (train.py `CurriculumStopCallback.det_env_fn`) 추가** — TB 90% trigger 후 det 50 ep로 진짜 확인 → false positive 차단. v26-A는 1M까지 det 모두 미달(82·72·82%)로 학습 완주, 카드 천장 입증.
14. **v25-A·v26-A 단일 seed 80~84% 결론은 lower outlier (v27이 반박)**: ent_floor `0.003 → 0` linear decay가 stochastic-det gap 12%p → 8%p로 좁힘 ✓. v22 79% → v25-A 84% → v26-A 80% single seed 결과로 "현 카드 본질 천장 80~84%" 결론 → **v27 multi-seed (3 seed mean 90.7%)로 카드 valid 입증**. **single seed로 카드 천장 판단 X — 최소 3 seed**. s3d_90 라인 교훈이 main flow에도 동일하게 적용 (함정 #14 패턴).
15. **TB false positive 차단 = `--det-check` callback (v27 seed2 실증)**: v22 함정 #13 차단 인프라. v27 seed2가 145k에서 TB 90% trigger → det 84% ❌ → cooldown 100k → 245k에서 진짜 졸업. det check 없었다면 145k에서 잘못 졸업했을 것. **새 학습은 default `--det-check` 활성**.
16. **`--det-episodes` 50은 학습 trigger엔 OK but 졸업 확정 측정으로 부족** (v27 Step 0 발견): v27 학습 중 50 ep det check 90/92/90% (mean 90.7) 통과 → 졸업 → 100 ep eval로 재측정하니 90/81/86% (mean 85.7%, **5%p gap**). 50 ep는 sample 적어 운 좋은 분포에서 trigger 가능. **졸업 확정은 항상 `eval_stages.py` 100 ep** (50 ep는 학습 중 cheap check). **다음 학습부터 `--det-episodes 100` 권장**.
17. **단순 측정 강화로 카드 천장 못 깸 (v28 입증)**: `--det-episodes 50 → 100`만 변경한 v28도 100 ep mean **84.3%** (v27 85.7%과 σ 안). 학습 중 100 ep det check는 false positive (TB-det gap 22%p까지 벌어짐) 잡아내는 데 효과적이나, 카드 본질 천장은 그대로. v22 79% → v25-A 84% → v26-A 80% → v27 85.7% → v28 84.3% — **단일 변경(학습량/측정/seed)으로 80~86% 박스 못 벗어남**. **s3b 90% 임계 돌파 = 새 reward 축 (yaw sign-aware / progress 강화 / N 변경) 필수**.
18. **progress axis 강화의 effective range (v29-B vs v29-C)** ⚠: `progress · 10 → · 15`로 v29-B mean +2.7%p (87.0%) — 새 axis 첫 효과. but `· 15 → · 18`로 v29-C mean **86.0%로 stalling** + σ 11.4%p ↑. ⭐ **졸업 step ×2 빨라짐** (190k vs 405k) 하지만 mean 효과 0. s3a 100→98 미세 후퇴 (작은 회전 정밀도 약화). **단일 axis 강화는 sweet spot(×15)까지만 효과, 그 위는 oversaturation**. **다음 카드 = 새 axis (yaw sign-aware / heading squared / N 변경) 또는 multi-seed 5~7로 outlier vs 본질 판정 필수**. ✗ 단일 axis 더 밀어도 의미 없음.
19. **σ가 카드 평가의 mean과 동등 핵심 (v30-A 입증)** ⭐⭐: v22~v29-C σ 단조 ↑ (5.4 → 8.5 → 11.4) 추세를 v30-A (yaw sign-aware)가 **5.7%p로 깨짐**. mean은 v29-B 동급(87.3)이지만 outlier seed 깊이 ↓ (v29 75/73 → v30 80) + seed1 det 95% (v22 이후 최고). 카드 평가 시 mean ≥ 90% 못 봐도 **σ ↓ + outlier 양상 ↑이면 카드 정착 가치 있음** (다음 axis 누적의 기반). 매트릭스 행3 (86~89/≤6) 통과는 reject 아니라 "다음 후보 axis 추가" 의미. 단일 카드로 90% 못 깰 때 다축 누적이 본질.
20. **m4_cpg 좌/우 parallel 진단의 2대 함정 (부호·메트릭)** ⚠⚠ (2026-06-08, 부호 2회·메트릭 1회 혼동 후 정리): ① **부호 규약 고정** — `+offset = 좌회전`(target 아래/−y, θ=π+off), `−offset = 우회전`(target 위/+y, θ=π−off). 모든 diagnostics(`offset_sign_sweep`·`left_parallel_reachable` 등)가 이미 이 부호로 일관 — plot 스크립트에서 임의로 뒤집지 말 것 (그렇게 해서 "D-H1 부호 버그"라 오판한 전례). ② **parallel 판정은 course(진행방향)·target 기준** — `head·target`·`turn_ratio`·head-기준 `slip`·도달 여부는 작은 각(s3a 12°)에서 머리가 −x여도 `head·tgt≈0.98`이라 **게걸음을 parallel로 오분류**. D-H1 `is_parallel`(turn_ratio≥0.4 & head-slip≤0.6)이 게걸음(좌 4Hz+0.75: **course·tgt −0.23, 머리-진행각 111°**)을 "parallel 1위"로 잘못 선정. **올바른 판정 = `course·target`(진행이 target 향함, parallel이면 ≥0.85) + 머리-진행각(≤20°)**. open-loop constant 실측: 우 −0.75 course 0.92·각 15° vs 좌 +0.75 course 0.72·각 33° → 비대칭. mirror "86% 도달"도 *도달*일 뿐 parallel 아님(같은 함정). ③ **그러나 "우=parallel"도 틀림 — 정책은 우조차 sideslip** (2026-06-08 추가, s3b 6카드/seed 정책 측정): 우(ts+1) course **0.72~0.78**·머리-진행각 **31~39°**·parallel(course≥0.85&각≤20°) **0/20 전부**, 좌(ts−1) 0.37~0.52. open-loop 우 0.92는 *이상치*이고 **학습 정책은 우도 머리 −x 고정 + 대각선 sideslip으로 도달** (곡선 선회 아님, `images/policy_right_v16_v20.png`서 v16/s0=v20/s0 동일). v16/s0가 수치상 최선(각 31°)이나 질적으론 같은 sideslip. **우>좌는 정도 차이지 '우=parallel'이 아님 — 진짜 parallel 곡선 선회는 아직 어떤 정책도 미달성**. 진단 도구: `/tmp/dissect_parallel.py`·`/tmp/policy_course.py`·`images/left_right_course.png`·`images/policy_right_v16_v20.png`.

**s3d_90 라인 함정** → [`docs/s3d_90_line.md`](s3d_90_line.md) (YAW_W 변형 / single seed baseline / end metric drift artifact 등).

→ **정착 조합 = MODE_3 + Ecoflex passive fin + 3DOF planar + v21 카드 (N=20·ent_floor 차등·yaw \|·\|·0.005)**.

---

## 카드 평가 방법론

(s3d_90 라인 + main flow v27~v28에서 본격 확립)

- **multi-seed × deterministic eval** (single seed로 카드 천장 판단 금지, 최소 3 seed)
- **`--det-check --det-episodes 100` callback** (TB false positive 차단)
- **`eval_stages.py` 6 stage × 100 ep** (졸업 확정 + catastrophic forgetting 점검)
- **`model_best.zip` 인프라** (peak 시점 보존)

### v22~v28 인프라 추가 사항 (정착됨)

- **train.py `CurriculumStopCallback.det_env_fn`** (`--det-check`, `--det-episodes`, `--det-cooldown-steps`) — TB 90% trigger 후 deterministic eval 자동 검증. 함정 #13 차단. v27 seed2 + v28에서 효과 실증. **default 활성 권장**.
- **curriculum.py `--init-from`** — start-stage init 모델 path 명시. v26-A에서 추가.
- **curriculum.py `--seed --tb-tag --runs-subdir --end-stage`** — multi-seed + single stage 학습 인프라. v27에서 활용.
- **eval_stages.py** — 모델 1개 6 stage deterministic eval (~20분 CPU). 졸업 확정·forgetting 점검 필수.

---

## 다음 카드 후보 상세

**현재 우선순위 = s3b 새 reward 축 또는 multi-seed 확장**. v22~v29-C 7개 카드 80~87% 박스 (v29-B/C가 첫 axis 효과 ✓ but mean stalling). forgetting 없음 ✓ → s3b에서만 작용하면 됨.

### v30 후보 (v29-C 결과 매트릭스 적용: "86~89 박스 → 새 axis 또는 multi-seed")

**v29-B/C 결론**: progress axis effective range ×10~×15 (sweet spot 87%). ×18 oversaturation. **단일 axis 강화는 끝, 다음 = 새 axis 또는 분산 판정**.

**v30 후보 A (yaw sign-aware) ⭐ 추천 1**:
- 현재: `reward += YAW_W · |yaw_rate| · 0.005` (회전 시도 자체에 +)
- v30-A: `reward += YAW_W · yaw_rate · sign(target_yaw_error) · 0.005` (target 방향만 +)
- 가설: progress×15 보존(추진 신호) + sign-aware로 정렬 정확도 강제. 두 축 동시 작용.
- s3d_90 라인 reject 이력 (구 v24-A/v27/v28) but s3b 기준 + progress×15 조합은 재검증 가치.
- 비용: ~3시간 (3 seed × 1M).

**v30 후보 B (multi-seed 5~7로 outlier vs 본질 판정) ⭐ 추천 2**:
- v29-B 카드 그대로 + seed 3/4/5/6 추가 → 총 7 seed.
- 가설: v29-B 87.0% / v29-C 86.0% σ 8.5~11.4%p가 random sampling인지 카드 본질 분산인지 판정. seed별 운에 따라 75~93% 분산이 카드 random이면 mean 더 높아질 가능성.
- 비용: ~3.7시간 (4 seed 추가 × 56분).

**v30 후보 C (heading error squared)**:
- v30-C: `reward += -K · yaw_error²` (정렬 정확도 직접 강제 axis)
- 가설: progress 강화로 추진은 OK, 이제 정렬 정확도 직접 가산 → 임계 90% 도달.
- 비용: 3시간.

**v30 후보 D (N=20 → 16/24)**:
- v11 N=16 / v17 N=24 / v19~ N=20.
- ⚠ SAC.load obs Box mismatch → init_from 무력화 (함정 #8). s1부터 새 학습 필요.
- 비용: 2~3시간/seed.

### 우선순위 (v30-A 완료 후 갱신)

**v30-A 결과** (이미 진행): mean 87.3% / σ 5.7%p — 매트릭스 행3 (86~89/≤6) 통과. **σ 줄임 axis 발견 ✓, mean 천장 ~87% → 다음 axis 누적**.

전제: 단일 tail motor 물리상 절대적 제자리 회전 불가능 → yaw \|·\| 보상 + sign-aware 보존. mean 끌어올릴 추가 axis 필요.

1. **v31-A (v30-A + heading²)** ⭐ — 축 누적 카드. v30-A 그대로 + `-K · yaw_err²` 신규 가산. sign이 σ 줄였으니 heading²이 mean 끌어올릴지 검증. progress×15 + yaw\|·\| + sign + heading² 4축.
2. **v30-B (v30-A × seed +4 = 총 7 seed)** — v30-A σ 5.7%p가 random sampling 운인지 카드 본질 안정성인지 판정. seed 3/4/5/6 추가.
3. v30-D (N 변경) — 축 누적 후 추가 검토. SAC.load 무력화 비용 감수, s1부터 새 학습.

(v30-C heading² 단독은 v31-A로 흡수 — v30-A axis 위에 누적이 더 효율적. v30-α "yaw·forward 결합"은 motivation 무효화로 제거.)

### v30 결과 평가 매트릭스 (공통)

| mean | σ | 판정 → 다음 |
|---|---|---|
| ≥ 90% | ≤ 6%p | ✓ s3b 안정 졸업 → v31 (s3c) |
| ≥ 90% | > 6%p | ⚠ mean 통과 but 분산 ↑ → v30-B (multi-seed +4) 분산 검증 |
| 86~89% | ≤ 6%p | ⚠ 부분 개선 + 안정성 ✓ → 다음 후보로 axis 추가 |
| 86~89% | > 6%p | ❌ mean·σ 둘 다 미흡 → axis reject, 다음 후보 |
| < 86% | any | ❌ axis reject |

⚠ σ 기준 = v28 baseline 5.4%p + 여유. 단조 ↑ 추세 (v28 5.4 → v29-B 8.5 → v29-C 11.4) 카드는 reject 신호.

### 우선순위 (장기): s3c·s3d (s3b 90% 졸업 후)

v31 — s3c 1M + s3b 졸업 axis + multi-seed + 100 ep. v32 — s3d 같은 방법론.

### s3 외 인프라/메타 카드 (낮은 우선순위)

- **HER (Hindsight Experience Replay)** — env Dict obs 큰 변경. 강력하나 구현 비용 큼.
- **fin actuator 추가** — 단일 모터 한계 자체를 풂. (사용자 명시 제외)
- **ANN surrogate (Lighthill / Zhong)** — fluid model 한계 우회. 실물 motion capture 필요.

---

> **m4 환경 (frame_skip 42 + V2 spec actuator, 2026-05~)** → 별도 파일 [`docs/training_log_m4.md`](training_log_m4.md). 본 파일은 mode3-planar 환경의 v1~v33 history만.
