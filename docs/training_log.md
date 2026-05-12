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


## 다음 카드 후보 상세

**현재 우선순위 = s3c 90% 달성** (main flow에서 가장 낮은 미달 stage). 카드 평가는 항상 "가장 낮은 미달 stage" 위에서 — s3d 카드는 s3c 졸업 후에야 본질적 의미.

### 우선순위 1: s3c (현 미달, v22 단일 seed 37%)

1. **새 v23 — s3c 본격 학습** ⭐⭐⭐ — `s3c max_steps 350k → 1M` + v22 s3b 졸업 정책 이어받기 (`--start-stage 5`) + 1 seed. v22가 s3b에서 입증한 학습량 부족 가설을 s3c에 그대로 적용. ~55분.
   - 가설: v22까지 s3c max_steps 350k가 짧았던 것. 1M 주면 s3b와 같이 90%+ 가능?
   - 90% 달성 → 다음 s3d 카드
   - 천장 50~70% → s3c 카드 한계로 입증, 새 v24~ 카드 (yaw reward 강화 / ent_floor / N 조정)
2. **새 v24~ — s3c 카드 한계 입증 시** — yaw reward 강화 / ent_floor / N 조정 등 s3c 위주 매핑.

### 우선순위 2: s3c 졸업 후 s3d (v22 32%, peak 50%)

3. **s3d 본격 학습** — max_steps 500k → 1M (이미 v22에서 1M 돌렸지만 s3c 90% 졸업 정책에서 시작하면 다시 평가 필요). 과거 s3d 카드 분석 [`docs/s3d_90_line.md`](s3d_90_line.md) 참조 (s3b·s3c 미달 상태 부수적 정보).

### s3 외 인프라/메타 카드 (낮은 우선순위)

4. **학습 후반 안정화** — ent_floor 후반 ↓ / LR decay / max_steps ↓. catastrophic drift 자체 늦추는 축. best metric ≈ end metric 목표. s3b·c·d 공통 적용 가능.
5. **HER (Hindsight Experience Replay)** — env Dict obs 큰 변경. 강력하나 구현 비용 큼.
6. **N=24 + yaw `|·|`·0.005 multi-seed** — v17 표현력 + v21 catalysis. N 변경은 s1부터 새 학습 (함정 #8, ~9시간).
7. **fin actuator 추가** — 단일 모터 한계 자체를 풂. (사용자 명시 제외)
8. **ANN surrogate (Lighthill / Zhong)** — fluid model 한계 우회. 실물 motion capture 필요.
