# 학습 로그 — v1 ~ v28 상세 진단

CLAUDE.md 본문에서 분리한 산문 진단 + 릴리스 인덱스. 표·매트릭스·함정·다음 후보는 CLAUDE.md에 유지.

## 작성 규칙 (v26~ 새 결과 추가 시)

- 본 파일: `## 주요 진단 (산문)` 섹션 끝에 `### vN-X (한 줄 제목)` 새 절 추가. 가설·결과·진단·다음 방향 자유 산문.
- CLAUDE.md: 표 갱신만 (그룹화·변경점·Stage 비교·s3d 100 ep). 종결·돌파 카드면 통찰/함정/매트릭스 한 줄도 갱신.
- 카드 후보가 바뀌면 본 파일 `## 다음 카드 후보 상세` 섹션도 같이 수정.
- CLAUDE.md char 수 40k 임계 모니터링. 다가오면 종결 카드 산문을 본 파일로 더 옮긴다.

## GitHub Release 인덱스

[`models-v1`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v1), [`models-v2`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v2), [`models-v5`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v5) (실패), [`models-v10`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v10), [`models-v12`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v12), [`models-v13`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v13), [`models-v17`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v17) (N=24, 정점 30% 천장 일시 돌파), [`models-v21`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v21) (yaw reward 천장 단독 돌파) ⭐, [`models-v22`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v22) (v21 + 1M, **v30으로 best metric 평균 근접 입증**) ⭐⭐, [`models-v23-v28`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v23-v28) (yaw reward 변형 6종), [`models-v29`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v29) (v22 multi-seed end), [`models-v30`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v30) (**v22 + `model_best.zip` 3-seed, 진짜 천장 best 31.2±4.0%, end는 catastrophic drift artifact**) ⭐⭐⭐.

**모델 binary 손실 / 백업**:
- v11: v12 학습이 sim/runs/s3d_arc90/ 덮어씀 — 영구 손실.
- v23·v24: model.zip 손실 (s3d_arc90/이 그 후 덮임). **tb_logs는 [`models-v23-v28`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v23-v28) release 포함**.
- v25·v26·v27·v28: 완전 백업 — [`models-v23-v28`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v23-v28) release + `/tmp/models-v{25,26,27,28}/` 로컬.
- v29 (seed=0/1/2): 완전 백업 — [`models-v29`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v29) release + `/tmp/models-v29/` 로컬.
- v30 (seed=0/1/2 + `model_best.zip`): 완전 백업 — [`models-v30`](https://github.com/sourceyoo/RL_robot/releases/tag/models-v30) release + `/tmp/models-v30/` 로컬.
- v16·v19·v20: `/tmp/models-v{16,19,20}-backup/` 로컬 백업.
- v18: v17 s3d만 fine-tune (1M step) — sim/runs/s3d_arc90/이 그 후 v19~v28에 의해 덮어써짐, 별도 백업 없음.

**release 없음**: v3·v4·v6~v9·v11·v14~v16·v18~v20.

---

## 주요 진단 (산문)

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

### v22 (v21 + 1M fine-tune — 천장 +6%p 추가 돌파) ⭐⭐⭐

max_steps 500k → 1M, 다른 변수 v21 그대로.

- ⭐⭐⭐ **천장 +6%p 추가 돌파**: end 29% → **32%** (v11 26% 대비 +6%p).
- ⭐⭐ **peak 50%** (6.4k init step, 모든 v best).
- ⭐ align +0.51 → +0.73 회복.
- ⭐ ep_seconds 21.2s — 모든 v best.
- ✅ 마지막 10% 26~30% 안정 (900k 28% / 929k 29% / 955k 28% / 994k 30% / end 32%). v21 정점 후 단기 후퇴 패턴 사라짐.

**v18 vs v22 핵심 인사이트**:
- v18 (N=24, no yaw, 1M): 50→27→23→22→20→19 (점진 후퇴, end 25%) ❌
- v22 (N=20, yaw, 1M): 25→18→13(저점)→16→21→**30→32 (저점 후 우상향, end 32%)** ⭐⭐
- → **yaw reward 있으면 학습량 ↑가 정점→평균 안정화에 효과**. yaw reward = mode 전환 catalysis, 학습량 ↑가 본격 활용 가능.

→ **누적 천장 진전**: v11 26% → v17 정점 30% → v21 29% (단독 돌파) → **v22 32% (peak 50%)**.

### v23-A (YAW_W 강화 — 정점 ↑ but 평균 ❌)

YAW_W 0.005 → **0.007** (1.4배). v22 카드 그대로(N=20·ent_floor 차등·1M·s3c init).

- ⭐ **정점 41% (window 100, 모든 v best)**: 625k~650k 약 25k step 동안 38~41% 유지.
- ❌ **end 23%** (v22 32% −9%p): 700k 부근 instability event (actor_loss 0.7→−2.5, ent_coef 0.010→0.018 spike) 후 reach 41→34→27→23 단조 후퇴.
- ❌ 마지막 10% 21~23% 후퇴 (v22 26~30% 안정과 정반대) — 940k부터 평균 catastrophic.
- final_dist 0.764m (v20 0.78 근접) — 추진 양보, **v20 "정렬만" mode 부분 재현**.
- ep 23.8s (v22 +2.6s) — 효율 후퇴.

→ **YAW_W 1.4배 trade-off: 정점 +3%p / 평균 −9%p / 진동 ↑**. 회전 신호 dominance가 진동 trigger — v5/v7 곱셈 보상 mode collapse와 다른 mechanism이지만 비슷한 결과. **v22 YAW_W=0.005가 균형점**.

### v24-A (signed yaw — 진동 해결 ✓ but 천장 ◯)

`+YAW_W·yaw_rate·sign(d(align)/d(yaw))` (옳은 방향 +, 틀린 −). YAW_W=0.005, 다른 카드 v22. v23-A 진동 진단(`|yaw_rate|` 좌우 흔들기 trigger) 직접 검증.

- ✅ **진동 해결**: 700k actor_loss spike 사라짐. ent_coef spike도 없음 (시작 직후 0.017→0.010 floor 빠른 수렴 후 평탄).
- ✅ 마지막 10% 26~28% 안정.
- ✅ **final_dist 0.599m — 모든 v best ⭐** (옳은 방향 추진 시퀀스 학습).
- ✅ 학습 곡선 단조 우상향: ep_rew_mean −19→+3, actor_loss 2.1→0.8 단조 감소, V자 있으나 spike 없음.
- ❌ **천장 돌파 못 함**: end 28% (v22 32% −4%p), peak 35% at 405k (v23-A peak 41% −6%p).
- ❌ avg_align +0.615 (v22 +0.73 −0.115) — signed reward가 회전 시도 보수화 → 정렬 학습 약화.

→ **역설 — `|yaw_rate|`의 "좌우 모두 보상"이 사실 학습 dynamic에 도움**: v22의 좌우 무차별 보상이 회전 빈도 ↑ → mode 전환 catalysis(정렬↔추진) 강화 → 천장 돌파. signed yaw는 "옳은 방향만"으로 제한 → 진동 ↓ but 학습 dynamic ↓. **천장 돌파 = 회전 신호 양 + 진동 회피 모두 필요**.

### v25-A (signed + YAW_W 0.007 — spike 없음 but 천장 catastrophic ❌)

v24-A signed × v23-A YAW_W 0.007 결합. 가설(signed면 weight ↑해도 진동 없음) 검증.

- ✅ **spike 없음 (진단 부분 입증)**: actor_loss V자×3 있으나 v23-A −2.5 폭락 없음.
- ⭐ peak 33% (520~535k) — v24-A 35%와 동급.
- ❌ **W자 점진 후퇴 — 모든 v 중 worst end**: 23→25.6(90k peak)→13(225k 저점#1)→17→21(310k)→30(490k)→**33(520~535k peak)** ⭐→28(580k)→20(690k 저점#2)→27(755~820k 회복)→18(890k)→**12 (970k 모든 v 후반 최저)** ❌→16(end). 530k peak 후 470k 동안 −21%p 점진 catastrophic.
- ❌ avg_align +0.558, final_dist 0.775m, ep 25.5s (모든 v worst).

→ **v23-A 700k 즉시 폭락이 v25-A에서는 470k 점진 붕괴로 변형**. signed × 강한 weight는 spike는 막아도 학습 dynamic 천천히 망침. 가설 부분 입증(spike 없음) but 천장 돌파 실패(weight 0.007이 signed 보수성 깨뜨려 mode drift). **v22 \|·\|0.005 sweet spot 재확인** — yaw reward 두 축(신호 형태·강도)에서 v22 기본값이 최적.

### v26 (v22 reproduce — baseline의 seed 분산 입증) ◯

v22 best 카드 그대로 (N=20·ent_floor 차등·`+|·|·0.005`·1M·s3c init), seed만 SAC default. 가설: v22 32%/peak 50%이 카드 효과인지 seed 운인지 검증. v23-A~v25-A가 모두 v22 reject였는데, v22 자체 재현 안 되면 baseline 자체가 흔들리는 것.

**결과 — v22 baseline의 seed 분산 입증**:

- ❌ **end 26%** (v22 32% **−6%p**, v11·v17·v18 동급) — 동일 카드로 v22 32% 재현 못 함.
- ◯ peak 36% (425~430k window 100, v22 ~38% 안정 정점 −2%p 근접). v22 peak 50%은 6.4k init step의 매우 짧은 구간이라 안정 정점 비교에선 v26 36% 부근이 fair.
- 학습 곡선 V자×2: 28(345k)→**36(425~430k peak)** ⭐→23(540k 저점#1)→27(845k)→23(870k 저점#2)→**29(945~985k 안정)**→26(end).
- 마지막 10% 26~29% 안정 (v22 26~30% 동급). spike 없음 — actor_loss 0.18 (v23-A −2.5 패턴 회피), ent_coef 0.010 floor 안정. **카드 dynamic은 valid**, 단순히 도달 횟수만 ↓.
- avg_align +0.687 (v22 +0.73 −0.04, v24-A +0.615 능가), final_dist 0.717m (v22 0.64 +0.08), ep 22.4s (v22 21.2 +1.2).

**핵심 통찰**:
- **v22 32%/peak 50%은 단일 seed 운 의존**. 카드 자체는 v11·v17·v18 25~26% 수준에서 ±3~6%p 분산.
- v23-A·v24-A·v25-A 평가를 단일 v22 32% 기준으로 한 게 too strict — 실제로는 v22~v26 평균 ~29%, ±3%p 분산이 baseline. **v23-A −9%p, v24-A −4%p, v25-A −13%p가 노이즈 vs 진짜 후퇴인지 재해석 필요**: v24-A 28%은 baseline 노이즈 안일 가능성 (reject가 아닐 수도), v25-A 16%은 명확한 catastrophic.
- 카드 효과 측정은 **single seed로는 불가** — 다음부터 v22 평가 시 multi-seed 권장.

**다음 후보**:
- **v27 soft signed yaw (`tanh`)** — signed/|·| trade-off의 중간 카드. 권장 우선순위 ↑.
- **v27 multi-seed v22** — 같은 카드 3~5 seed로 분산 본격 측정. v22~v26 ±3%p가 진짜 noise level인지 검증.
- **v27 N=24 + yaw `|·|`·0.005** — v17 표현력 + v22 mode catalysis (s1부터 새 학습).

### v27 (soft signed yaw TANH_K=2 — K 너무 컸음, V자×3 ❌)

`+YAW_W·yaw_rate·tanh(yaw_dir × TANH_K)`, YAW_W=0.005, **TANH_K=2.0**. 가설: v22 \|·\| (좌우 무차별 → mode catalysis ↑, 노이즈)과 v24-A signed (sign 점프 → 안정 but 보수화) 사이 중간. 양 극단(|yaw_dir| > 1)에서 signed에 근접, 0 부근에서 부드러움.

**결과 — K=2가 너무 큼**:

- ❌ **end 24%** (v26 26% −2%p, v22 32% −8%p, baseline noise 안). v22~v26 ±6%p 평균 ~29% 기준으로 보면 −5%p — 노이즈 안에서 약간 낮은 쪽.
- ❌ **peak ~24% (1차 peak 265k)** — v22 ~38%, v26 36%, v24-A 35% 모두 큰 폭 미달. **천장 돌파 X**.
- 학습 곡선 V자×3: 17(70k 시작)→**24(265k 1차 peak)** ⭐→22(340k)→15(415k)→**11(440k 저점#1)** ❌→18(590k)→23(740k 회복 정점)→20(815k)→15(890k 저점#2)→**12(950~960k 저점#3)**→24(end).
- ✅ spike 없음 (actor_loss 0.07~0.3 안정, ent_coef 0.010 floor 정상) — v24-A 패턴과 같음.
- avg_align +0.698 (v26 +0.687 동급), final_dist 0.753m (v22 0.64 +0.11, v25-A 0.775 근접 — 추진 후퇴), ep 23.9s (v25-A 25.5s 다음 worst).

**핵심 진단 — TANH_K=2의 산술적 의미**:

- `tanh(±2) ≈ ±0.964` — **signed의 96.4% 신호**. 0.964 vs 1.000 차이는 dynamic 학습에 무시할 수준.
- `tanh(±1) ≈ ±0.762`, `tanh(±0.5) ≈ ±0.462` — 더 작은 K는 \|·\| 쪽으로 이동.
- yaw_dir의 일반적 범위가 ±1 안이라는 점 고려하면, K=2일 때 yaw_dir=0.5에서도 tanh(1) ≈ 0.762로 sign 변형 효과 약함.
- **결론: K=2는 사실상 v24-A signed 재현**. 의도한 "\|·\|/signed 중간" 효과 안 나옴.

**v24-A vs v27 비교**:
- v24-A: end 28%, peak 35%, V자×2, 마지막 10% 26~28% 안정.
- v27 (K=2): end 24%, peak ~24%, V자×3, 마지막 ~13~24% 진동.
- 신호는 거의 같은데 v27이 더 안 좋음 — seed 분산 효과로 추정 (v24-A v27 모두 baseline ±6%p noise 안에 있다고 가정 가능).

**다음 후보**:
- **v28 soft signed K=1.0 또는 0.5** — 진짜 중간 카드 시도. K=0.5라면 tanh(±0.5) ≈ ±0.46로 \|·\| 쪽 가까움.
- **v28 multi-seed v22** — v23~v27 reject 결론이 모두 v22 단일 32% 기준이라 baseline 신뢰도가 모든 결론에 영향. multi-seed로 본격 측정 권장.

### v28 (soft signed yaw TANH_K=0.5 — K↓이 \|·\|를 안 만듦 ❌)

`+YAW_W·yaw_rate·tanh(yaw_dir × 0.5)`. v27 K=2 (signed 96.4%) reject 후 K↓으로 \|·\| 쪽 이동 시도. 가설: tanh(±0.5) ≈ ±0.46이면 sign 점프 ↓, 0 부근 부드러움 ↑로 진짜 중간 카드 가능.

**결과 — 가설 reject, soft signed 카드 축 자체가 잘못된 것 입증**:

- ❌ **end 18%** (v22 32% **−14%p**, v22~v26 평균 29% **−11%p**, 명확한 reject 영역). 모든 v 중 v25-A 16% 다음 worst.
- 학습 곡선 N자: 19(70k)→26(135k)→**29(140k 1차 peak)** ⭐→27(165k)→20(240k)→17(265k)→**11(290k 저점#1)** ❌→17(315k)→18~20(340~415k)→21(465k)→26(490k)→28(515k)→**29(540k 2차 peak)** ⭐→26(565k)→23~24(590~690k)→21(715k)→19(740~765k)→16~17(790~815k 저점#2)→23(840k 회복)→20~22(865~975k 평탄)→18(end).
- ✅ spike 없음 (actor_loss 0.7~0.8 평탄, ent_coef 0.010 floor 안정). 학습 dynamic은 정상이나 평균값 자체가 낮음.
- avg_align +0.689 (v26 동급), final_dist 0.699m (v22 0.64 +0.06), **ep 25.0s (v25-A 25.5s 다음 worst tied)**, ep_rew 2.23 (v26 8.0 대비 ↓ — 누적 보상 부진).

**핵심 진단 — soft signed K 조정의 산술적 한계**:

```
yaw_dir=0.5, yaw_rate=10/s 시점 신호 강도:
- v22 |·|·0.005     = 0.005 × 10 × 1.0     = 0.050  (부호 무시, 강함)
- v24-A signed·0.005 = 0.005 × 10 × 1.0     = 0.050  (정렬 부호, 강함)
- v27 tanh(K=2)      = 0.005 × 10 × 0.762   = 0.038  (정렬 부호, 76%)
- v28 tanh(K=0.5)    = 0.005 × 10 × 0.245   = 0.012  (정렬 부호, 24%)
```

- **K↓이 \|·\|를 안 만듦**: K 조정은 **신호 강도(magnitude)** 만 조정. \|·\|와 signed의 차이는 **부호 정보(direction)**, 이건 K로 못 조정.
- v22 \|·\|와 v28 tanh(K=0.5)는 신호 강도 4배 차이, 부호 정보 차이도 그대로. **결과적으로 v24-A signed (강함) → v28 (약함) 약화 버전 = v20 패턴 부분 재현** (정렬 +0.689 OK, 추진 부진 final_dist 0.699m).
- 진짜 \|·\|/signed 중간을 만들려면: `α·|yaw_rate| + (1-α)·yaw_rate·sign(yaw_dir)` 같이 두 신호를 가중합해야 함 (K 조정으로는 불가).

**카드 축 매트릭스 재정리 (yaw reward 신호 형태·강도)**:

| 카드 | 신호 형태 | 강도 (\|·\| 대비) | end |
|---|---|---|---|
| v22 \|·\|·0.005 | 부호 없음 | 100% | **32%** (single seed) |
| v23-A \|·\|·0.007 | 부호 없음 | 140% | 23% ❌ spike |
| v26 \|·\|·0.005 reproduce | 부호 없음 | 100% | 26% (baseline ~29%) |
| v24-A signed·0.005 | 정렬 부호 | 100% | 28% ◯ |
| v25-A signed·0.007 | 정렬 부호 | 140% | 16% ❌ catastrophic |
| v27 tanh(K=2)·0.005 | 정렬 부호 | 96% | 24% ❌ |
| **v28 tanh(K=0.5)·0.005** | 정렬 부호 | **24%** | **18%** ❌ |

→ **부호 없음(\|·\|) 카드만 baseline 이상, 부호 있음 카드는 모두 미달**. v22 \|·\| 0.005가 unique sweet spot 재확인.

**결론**:
- **yaw reward 신호 형태(signed/\|·\|/tanh 변형) 카드 완전 종결** — 모든 변형이 v22 못 넘음.
- v22 \|·\|의 좌우 무차별 보상 = mode catalysis의 본질, 다른 형태로는 대체 불가.
- 다음은 **multi-seed v22 평가**로 모든 reject 결론의 baseline 신뢰도부터 확립. v23~v28 평가가 모두 v22 단일 32%(single seed) 기준이라 결론 자체가 흔들림.

**다음 후보**:
- **v29 multi-seed v22** — 3~5 seed로 baseline 평균·분산 본격 측정 (~3시간).
- **v29 N=24 + yaw \|·\|·0.005** — v17 표현력 + v22 catalysis 결합 (s1부터 ~3시간).

### v29-A (v22 multi-seed — baseline 신뢰도 본격 확립) ⭐⭐

**카드**: v22 카드 그대로 (N=20·ent_floor 차등·yaw `|·|`·0.005·1M·s3c init), seed=0/1/2 3-run. fish_env.py를 v22 yaw reward로 복원, curriculum.py에 `--seed` `--tb-tag` `--runs-subdir` 인자 추가하여 동일 s3c model.zip에서 fine-tune. v22 release의 s3c model.zip은 현재 `sim/runs/s3c_arc60/model.zip`과 md5 동일 (d6d9ca97...) 확인됐고 — 즉 v23~v28까지 모든 카드가 같은 s3c init을 사용해 왔음.

**가설**: v23~v28 reject 결론(v25-A 16% / v27 24% / v28 18% 등)이 모두 v22 단일 32%/peak 50% 기준이라 baseline 신뢰도가 모든 카드 평가의 토대. v26 reproduce에서 end 26% (−6%p) 나오면서 단일 seed 운 의존이 의심됐고, v22~v26 평균 ~29%(±3%p)이 fair baseline이라는 잠정 결론까지 갔음. v29는 이 결론을 본격 검증.

**결과 (s3d 1M, 마지막 100 ep window)**:

| seed | end | peak (구간) | avg_align | final_dist | ep_sec | ep_rew | 패턴 |
|---|---|---|---|---|---|---|---|
| 0 | **16%** | 27% (900k) | 0.530 | 0.760m | 25.3s | 4.70 | 후반 catastrophic drift (peak→end −11%p) |
| 1 | **14%** | 22% (830k) | 0.646 | 0.805m | 26.1s | 4.46 | 후반 catastrophic drift (peak→end −8%p) |
| 2 | **21%** | 25% (865~970k) | 0.723 | 0.771m | 24.0s | 7.43 | 후반 안정 (W자, end peak 근처) |
| **mean** | **17.0%** | **24.7%** | **0.633** | **0.779m** | **25.1s** | **5.53** | 3 seed 중 어느 것도 v22 32%·peak 50% 근접 못함 |
| std | 2.94 | 2.05 | 0.080 | 0.019 | 0.87 | 1.35 | — |

**v22+v26+v29 5 seed 통합 baseline**: end = {32, 26, 16, 14, 21}, mean = **21.8%**, std = **6.86** (sample std). v22 32%은 평균 + 1.48σ, v26 26%은 평균 + 0.61σ, v29 0/1/2은 평균 ± 1σ 안. **v22 단일값이 학습 곡선 분포의 분명한 outlier**.

**v22 카드 100 ep window 분포 (이론)**:
- 진짜 평균 = ~22% (sample mean)
- 1σ 범위 = 15~29%
- 95% CI = 8~36% (±2σ)
- v22 32% = +1.48σ → 일반적인 학습 noise의 우측 outlier 영역. 카드 효과(yaw reward+1M)는 v21 평균 29% → v22~v29 5 seed 평균 22%로 오히려 약간 후퇴한 셈 (1M fine-tune이 평균을 안 올림).
- v22 peak 50% (6.4k init) = 단일 episode의 peak이지 평균이 아님 → seed에 따라 매우 다름. v29 seed0~2은 peak 22~27% (mean 24.7%, v22 50%과는 stagnation 가능성).

**v23~v28 reject 결론 재평가**:

| 카드 | end | baseline +xσ | 재평가 |
|---|---|---|---|
| v23-A | 23% (peak 41%) | end −0.20σ / **peak +2.79σ** | peak 효과 통계 유의. end는 noise 안 |
| v24-A | 28% (peak 35%) | end +0.90σ / peak +1.50σ / **final_dist 0.599m = mean−3.05σ** | end·peak은 noise but **final_dist는 강력한 카드 효과** |
| v25-A | 16% (peak 33%) | end −0.85σ (970k 12%은 −1.43σ) | noise 안. 카드 무효 결정적 증거 부족 |
| v26 | 26% | end +0.61σ | noise 안 |
| v27 | 24% (peak ~24%) | end +0.32σ | noise 안. 카드 무효 결정적 증거 부족 |
| v28 | 18% (peak 29%) | end −0.55σ | noise 안. 카드 무효 결정적 증거 부족 |

→ **단일 seed × 단일 metric으로 카드 비교는 신뢰도 낮음**. v23-A peak 41% (+2.8σ)와 v24-A final_dist 0.599m (−3.0σ best)만 baseline 분포에서 통계 유의한 outlier — 두 카드는 **본격 multi-seed 재검증 가치**.

**왜 v22 32%이 outlier인가**:

- SB3 SAC seed=None일 때 매번 다른 seed (system entropy). v22 학습 당시 우연히 매우 좋은 seed에 걸림.
- s3d 1M fine-tune은 long horizon이라 학습 dynamic이 stochastic하게 발산/수렴 — 같은 init·하이퍼파라미터도 seed에 따라 ±10%p 분산.
- yaw reward + ent_floor + 1M의 조합이 "긴 학습 시간 × 진동 가능성" 둘 다 키움 → 분산 ↑.
- v22의 학습 곡선(25→18→13(저점)→16→17→저점 후 우상향 → 30~32 end)도 후반 우상향 패턴 — v29 seed0/1은 peak 후 drift, seed2만 후반 안정. 4 seed 중 1개꼴로 v22 패턴 재현 가능.

**진단의 메타-결론 (카드 평가 방법론)**:

- **카드 비교는 최소 3 seed 평균으로**. 단일 seed 결과만으론 16~32% noise 안의 차이를 판별 못함.
- **통계 유의 카드 효과 판정 기준**: 평균 ± 2σ (baseline 16~32%) 밖. 이번 통계로:
  - **유의 효과 +**: v23-A peak 41%, v24-A final_dist 0.599m
  - **유의 효과 −**: 없음 (v25-A 970k 12% 단일 episode도 noise 안)
- v22 단일값(32%) 기준 reject한 카드 ≠ 정말 reject. 사실 v22+v26 시점 결론도 신뢰도 낮았음.

**다음 권장 카드**:

1. **v30 v23-A·v24-A 3-seed 재평가** — 통계 유의했던 카드들의 진짜 효과 확정. 각 3 seed = 6시간.
2. **v30 N=24 + yaw `|·|`·0.005 multi-seed** — v17 표현력 카드, 단 N 변경은 s1부터 새 학습 ~9시간 (3 seed).
3. **v30 v22 + 2M fine-tune multi-seed** — 학습량 ↑이 분산 줄이는지 (v18 패턴 위험).

**검증 사항**: v29 seed0/1의 catastrophic drift 패턴이 카드 본질인지 (즉 1M fine-tune이 단일 seed 모두 후반 발산 위험을 갖는지). 만약 그렇다면 학습량 ↓ (700k 정도) + multi-seed 가 더 안정적 카드일 수 있음.

### v30 (v22 카드 + `model_best.zip` 인프라 + 3 seed) ⭐⭐⭐

**카드**: v22 카드 그대로 (N=20·ent_floor 차등·yaw `|·|`·0.005·1M·s3c init) + **`model_best.zip` 자동 저장** (각 step에서 reach_rate 100ep window가 갱신될 때마다 model 저장). seed=0/1/2, 동일 s3c init.

**가설**: v29에서 발견된 late catastrophic drift (peak→end −10~19%p, seed0/1) 자체는 카드의 본질일 수 있음. 그러나 학습 중간 peak에 도달한 정책 자체는 v22 32%/peak 50%에 가까울 가능성. `model_best.zip` 보존으로 end metric의 noise를 우회하고 카드의 진짜 천장을 본격 측정.

**결과 (s3d 1M, end-100ep + best-100ep)**:

| seed | end (1M) | **best** | best step | end avg_align | end final_dist | 패턴 |
|---|---|---|---|---|---|---|
| 0 | 16% | **30%** | @880k | 0.530 | 0.760m | peak→end −14%p (120k 후퇴) |
| 1 | 14% | **33%** ⭐ | @520k | 0.646 | 0.805m | peak→end −19%p (480k 점진 W자) |
| 2 | 21% | **25%** | @865k | 0.723 | 0.771m | peak→end −4%p (안정) |
| **mean** | **17.0%** | **29.3%** | — | 0.633 | 0.779m | catastrophic drift |
| std | 2.94 | 3.30 | — | 0.080 | 0.019 | — |

**v22+v26+v30 5 seed 종합** (best metric):

| seed | best |
|---|---|
| v22 | 32% (end도 32%, single peak burst) |
| v26 | 36% (@425~430k, end는 26%) |
| v30 s0 | 30% (@880k) |
| v30 s1 | 33% (@520k) |
| v30 s2 | 25% (@865k) |
| **mean** | **31.2%** |
| **std** | **4.04** |

→ **v22 32% = mean − 0.20σ** (평균에 매우 가까움). v29 결론의 "+1.4σ outlier"는 end metric 기준이라 metric 선택의 artifact였음.

**핵심 발견**:

1. **v22 카드 진짜 천장 = ~30% (best metric mean)**, std 4%p 정도. 단일 seed로도 ~25~33% 범위에 들어옴 — 카드 효과의 본질적 변동.

2. **end metric (17 ± 2.9, 5 seed 22 ± 7%p)은 catastrophic drift의 영향** — 같은 카드에서 peak 도달 후 학습이 계속되면서 정책이 발산하는데, 발산 정도와 timing이 seed에 따라 크게 다름. seed1은 480k에 걸쳐 점진 W자 후퇴, seed2는 후반 안정, seed0는 마지막 120k drift.

3. **v22 32%은 운 좋게 best 시점 근처에서 학습이 끝남** — v22 학습 곡선(25→18→13→16→17→우상향→30~32 end)을 보면 후반 우상향 → 1M에 30~32% 도달, 즉 catastrophic drift trigger 안 됨. v30 s0와 같은 패턴인데 s0는 880k peak 후 drift, v22는 1M까지 drift 안 됨.

4. **카드 평가 표준 = best metric**. v23~v28 reject 결론도 best metric으로 재평가 시 결론 바뀔 가능성:
   - v23-A peak 41% (window 100, 모든 v best) — 이미 best metric으로 +2.8σ outlier 입증, v22 카드 천장 30%보다 11%p 위. **카드 효과 확실, multi-seed 재검증으로 분산 측정 필요**.
   - v24-A peak 35% — v22 best 평균 30%보다 5%p 위, 1σ 정도. multi-seed로 확실 검증.
   - v25-A peak 33% — v22 best 분포 안. 카드 무효 가능성 높음 (단 W자 catastrophic drift는 카드 특성으로 유지).
   - v27 peak 24% / v28 peak 29% — v22 best 평균보다 낮거나 동급. 카드 무효 가능성.

5. **best metric의 한계**: best는 학습 중 어느 시점이든 1번 도달하면 보존됨 → 카드 효과 측정에 유리하지만, **실제 deploy 시 어떤 정책을 쓸지는 별도 문제**. end metric은 "1M 학습 후 deploy할 정책"의 reach.

**메타-결론 (v29 vs v30 차이)**:

- v29: end metric × 3 seed → baseline 17±2.9% / 5 seed 종합 22±7%p (v22 32% outlier)
- v30: best metric × 3 seed → baseline 29.3±3.3% / 5 seed 종합 31.2±4.0% (v22 32% 평균 근접)
- **v29 결론은 metric 선택의 artifact였음**. 진짜 카드 효과 = best metric.
- 단 end metric이 의미 없는 건 아님 — **catastrophic drift trigger 여부**는 카드의 안정성·sim2real 신뢰도를 판단하는 중요 신호. drift 자체를 막는 카드 (학습 후반 안정화) 가 다음 후보로 부상.

**다음 권장**:

1. **v31 v23-A·v24-A 3-seed best-save** — best metric으로 카드 효과 본격 확정. v22 best 평균 30%와 비교, +2σ 이상이면 진짜 천장 돌파 카드.
2. **v31 학습 후반 안정화 카드** — ent_floor 후반 ↓, LR decay, max_steps ↓ (peak 직후 종료) 등. end metric ≈ best metric으로 만드는 게 목표 (catastrophic drift 자체를 늦춤).
3. **v31 N=24 + yaw 멀티-seed best-save** — v17 큰 회전 표현력 + v22 catalysis 결합. N 변경은 s1부터 새 학습 (~9시간).

---

## 다음 카드 후보 상세

1. **v31 v23-A·v24-A 3-seed best-save** ⭐⭐ — v22 best 평균 30%과 비교, 통계 유의 카드 효과 확정. ~6시간 (3+3 seed).
2. **v31 학습 후반 안정화 카드** — ent_floor 후반 ↓ / LR decay / max_steps ↓. forgetting 자체를 늦추는 축. best metric ≈ end metric 목표.
3. **v31 N=24 + yaw `|·|`·0.005 multi-seed best-save** — v17 큰 회전 표현력 + v22 mode catalysis. N 변경은 s1부터 (함정 #9, ~9시간).
4. **v31 v22 + s3d 2M fine-tune multi-seed** — 학습량 2배 추가 안정화 (v18 패턴 위험).
4. **HER (Hindsight Experience Replay)** — env Dict obs 큰 변경. 가장 강력하나 구현 비용 큼.
5. **fin actuator 추가** — 단일 모터 한계 자체를 풂. (사용자 명시 제외)
6. **ANN surrogate (Lighthill / Zhong 2026)** — fluid model 한계 우회. 실물 motion capture 필요.
