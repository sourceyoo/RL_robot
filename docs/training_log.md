# 학습 로그 — main flow v1 ~ v21

CLAUDE.md 본문에서 분리한 산문 진단 + 릴리스 인덱스. 표·매트릭스·함정·다음 후보는 CLAUDE.md에 유지.

> **s3d_90 라인 (구 v22~v30, 분리됨)** → 별도 파일 [`docs/s3d_90_line.md`](s3d_90_line.md). 본 파일은 main flow만 다룸.

## 작성 규칙

- 본 파일: `## 주요 진단 (산문)` 섹션 끝에 `### vN-X (한 줄 제목)` 새 절 추가. 가설·결과·진단·다음 방향 자유 산문.
- CLAUDE.md: 표 갱신만 (그룹화·변경점·Stage 비교·s3d 100 ep). 종결·돌파 카드면 통찰/함정/매트릭스 한 줄도 갱신.
- 카드 후보가 바뀌면 본 파일 `## 다음 카드 후보 상세` 섹션도 같이 수정.
- CLAUDE.md char 수 30k 임계 모니터링.

## GitHub Release 인덱스 (main flow)

[`models-v1`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v1), [`models-v2`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v2), [`models-v5`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v5) (실패), [`models-v10`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v10), [`models-v12`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v12), [`models-v13`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v13), [`models-v14`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v14), [`models-v17`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v17) (N=24, 정점 30% 천장 일시 돌파), **[`models-v21`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v21)** (yaw reward 천장 단독 돌파) ⭐⭐.

**모델 binary 손실 / 백업** (main flow):

- v11: v12 학습이 sim/runs/s3d_arc90/ 덮어씀 — 영구 손실.
- v16·v19·v20: `/tmp/models-v{16,19,20}-backup/` 로컬 백업.
- v18: v17 s3d만 fine-tune (1M step) — sim/runs/s3d_arc90/이 그 후 덮어써짐, 별도 백업 없음.

**release 없음** (main flow): v3·v4·v6~v9·v11·v15·v16·v18·v19·v20.

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

## 다음 카드 후보 상세

**현재 우선순위 = s3b 안정화** (Step 0 100 ep eval로 mean 85.7% 미달 발견, 50 ep 졸업 측정 90.7%과 5%p gap). forgetting 없음 ✓.

### 우선순위 1: v28 — s3b 안정화 (`--det-episodes 100`)

1. **v28 — s3b 안정화** ⭐⭐⭐
   - init: v25-A 모델 (`sim/runs/v25/s3b_arc30/model.zip`) — v27과 깨끗한 비교
   - 카드 (ent_floor schedule, yaw `|·|`·0.005, N=20, NN [256,256]): v25-A 그대로
   - **본질 변경**: `--det-episodes 50 → 100` ⭐
   - `--det-check` 활성, s3b max_steps 1M
   - seed 0/1/2 multi-seed
   - 학습 후 `eval_stages.py` × 3 seed 100 ep 필수
   - 예상 시간: ~6시간 (sequential) 또는 ~2시간 (병렬)
   - 가설 1: 학습 중 100 ep 검증으로 진짜 90%까지 학습 지속 → mean 90~92% 달성
   - 가설 2: 1M 안 100 ep 90% 도달 불가 → 카드 진짜 천장 86~89%, 새 축 필요

### 우선순위 2: v28 결과 따라 다음 카드

| v28 결과 | 판정 → 다음 카드 |
|---|---|
| 100 ep mean ≥ 90% | ✓ s3b 졸업 확정 → v29 s3c |
| 86~89% | ⚠ 카드 한계 → 새 축 (yaw sign-aware / N 변경 / progress 강화) |
| < 86% | ❌ 후퇴 → 카드 변경 |

### 우선순위 3: s3c·s3d (s3b 100 ep 졸업 후)

v29 — s3c 1M + det-episodes 100 + multi-seed. v30 — s3d 같은 방법론.

### 새 축 카드 (현재 필요 없음 — v27이 reject)

v22~v26 single seed 결과로 "s3b 천장 80~84%" 결론에 기반한 새 축 후보들 (yaw sign-aware / N 변경 / progress 가중치 등)은 v27 multi-seed로 카드 valid 입증되어 **현재 진행 불필요**. s3c·s3d에서 카드 한계 입증되면 그때 검토.

### s3 외 인프라/메타 카드 (낮은 우선순위)

2. **HER (Hindsight Experience Replay)** — env Dict obs 큰 변경. 강력하나 구현 비용 큼.
3. **fin actuator 추가** — 단일 모터 한계 자체를 풂. (사용자 명시 제외)
4. **ANN surrogate (Lighthill / Zhong)** — fluid model 한계 우회. 실물 motion capture 필요.

### v22~v27 인프라 추가 사항 (정착됨)

- **train.py `CurriculumStopCallback.det_env_fn`** (`--det-check`, `--det-episodes`, `--det-cooldown-steps`) — TB 90% trigger 후 deterministic eval 자동 검증. 함정 #13 차단. v27 seed2에서 효과 실증. **default 활성 권장**.
- **curriculum.py `--init-from`** — start-stage init 모델 path 명시. v26-A에서 추가.
- **curriculum.py `--seed --tb-tag --runs-subdir --end-stage`** — multi-seed + single stage 학습 인프라. v27에서 활용.
- **eval_stages.py** — 모델 1개 6 stage deterministic eval (~20분 CPU). 졸업 확정·forgetting 점검 필수.
