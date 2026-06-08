# m4_cpg 좌/우 선회 진단 (working doc, 2026-06-08)

> 목적: "좌/우 선회를 게걸음/앞지름이 아닌 진짜 parallel로" 목표를 두고 진단한 여정·확정 사실·**반복된 오류**·미확정 사항을 한곳에. 같은 함정에 또 빠지지 않기 위함. 처방 history는 [training_log_m4.md](training_log_m4.md), 함정 #20은 [training_log.md](training_log.md).

## 목표 (사용자 정의)
- 회전 stage(s3a~)에서 좌회전이 **target을 −x로 앞질러간 후 sideslip으로 되돌아오는** 비효율 궤적 → **v16/s0 우회전처럼 "target 향해 곧장" 가게**.
- 궁극은 추진+정렬 parallel([[feedback_move_while_aligning]]), 게걸음·제자리회전 X.

## ★ 확정 사실 (검증됨)

1. **부호 규약**: `+offset = 좌회전`(target 아래/−y, θ=π+off), `−offset = 우회전`(target 위/+y, θ=π−off). 모든 diagnostics 일관. plot에서 임의로 뒤집지 말 것.

2. **parallel 판정자 = course(진행방향)·target + 머리-진행각**. `head·target`·`turn_ratio`·head-기준 slip·도달 여부는 **작은 각(s3a 12°·s3b 16~22°)에서 머리가 −x여도 `head·tgt≈0.96~0.98`이라 게걸음/직진을 parallel로 오분류**. (`/tmp/dissect_parallel.py`)

3. **정책은 좌·우 모두 sideslip** (s3b 6카드/seed): 우 course 0.72~0.78·머리-진행각 31~39°·parallel(course≥0.85&각≤20°) **0/20 전부**, 좌 0.37~0.52. "우=parallel"은 틀림, 우>좌는 정도 차이. (`/tmp/policy_course.py`, `images/policy_right_v16_v20.png`)

4. **fin 비대칭 = 순수 물리 "위쪽(+y) 드리프트"** (offset=0 대칭 sine, 모든 freq/amp/seed에서 y_drift +0.003~0.015). 작음(추진의 2~3%). 우(target 위)엔 순풍, 좌(target 아래)엔 역풍. (`/tmp/drift_physics.py`)

5. **앞지름 원인 = progress의 머리방향 게이트**: `progress = dist_reduction × clip(cos(이동,머리),0,1)^10` (fish_env.py:343-346). 좌에서 −x 직진이 full progress → y(target 아래) 안 내림 → target_x 지나침. 좌 정책 t120 y(+0.011)가 순수 물리 드리프트(+0.013)와 거의 같음 = 정책이 좌에서 횡 방향 노력 거의 안 함. (`/tmp/overshoot_diag.py`, `/tmp/lr_overshoot_compare.py`)

6. **선회 초입 reward 골짜기 = align 항**: offset 0→0.15(선회 첫발)에서 total reward −16.9→−27.0 하락. 주범 = `align 항(0.30·align·turn_ratio)`이 +0.01→**−6.42** 폭락 (머리 틀어 turn_ratio↑인데 정렬 전이라 align<0 → 음수 처벌) + lat_pen −5.6. progress·par는 오히려 +2.1 개선. (`/tmp/term_decomp.py`, `/tmp/turn_gradient.py`)

7. **★ 곡선 선회는 물리적으로 거의 불가능** (핵심 결론): constant offset 최소 선회 반경 **≥1.5m** (target 거리 0.5m의 3배+). 작은 offset=−x 직진, 큰 offset(0.9)=추진 죽어 멈춤, 누구도 원을 못 그림. **target 도달의 유일한 물리적 길 = 직진 추진 + 약간의 sideslip**. → "게걸음 vs parallel"은 사실 물리 제약. parallel 강제 = 도달 불가능한 요구(v15 도달률 60% 붕괴가 이걸로 설명). (`/tmp/curvature.py`, `images/curvature_s3b.png`)

## ⚠ 반복된 내 오류 (다음 세션 주의)
- **부호 혼동 2회**: plot 스크립트에서 +offset↔−offset 뒤집어 "D-H1 부호 버그"라 오판(실제 D-H1 정상). → 부호는 사실 #1 고정.
- **메트릭 함정 2회**: "좌 parallel 도달 4개"(turn_ratio 기준, 실제 게걸음), "oracle course 0.85·각 10°"(실제 −x 직진) — 둘 다 작은 각 방향cos 함정(#2). **trajectory를 안 보고 숫자만 믿으면 또 속음.** 사용자가 매번 trajectory 보자고 해서 잡힘.
- **oracle 실패**: P제어 oracle이 선회 못 하고 −x 직진하다 멈춤(`images/oracle_s3b.png`). "0.2 반경 맴돎"은 오독.
- 교훈: **숫자(course/turn_ratio) 보고 전 반드시 trajectory(head+course 화살표) 확인.** 작은 각에선 방향 cos 무력.

8. **★ 좌/우 비대칭 = 물리 아니라 학습 편향** (2026-06-08 확정): constant action 좌(+offset)/우(−offset) 대칭 sweep(s3b)에서 v_td·course·최근접·궤적 모두 거의 대칭 (좌 best 0.110 ≈ 우 best 0.121). `+offset`=아래 횡력, `−offset`=위 횡력이 거울상 (`images/lr_asym_constant.png`: 좌 아래로·우 위로 같은 대각선). fin 위드리프트(#4)는 너무 작아 offset 효과에 묻힘 → "역풍이 좌 실패 주원인"은 과대평가였음(정정). **v15 우 도달/좌 실패(도달 0)는 정책이 좌에서 +offset을 안 쓰는 학습 편향** (mirror 테스트 "offset 반전 시 좌 86% 도달"과 정합). constant는 좌·우 둘 다 곡률 부족 도달 X(0.11)지만 closed-loop 학습이면 도달(우가 증거) → **좌도 학습만 시키면 우와 동일 곧장 sideslip 도달 가능**. (`/tmp/lr_asym.py`, `/tmp/v15_check.py`, `images/v15_left_right.png`)

## 목표·처방 방향 (2026-06-08 확정)
- **목표 = v15/seed0 우회전의 곧장 대각선 sideslip(course 0.80)을 좌우 균등하게.** 곡선 선회(parallel) 아님 — 곡률 한계(#7) 수용.
- **처방 = reward 재설계가 아니라 좌 학습 유도** (물리 대칭이므로): mirror augmentation(우 성공 경험 offset 반전으로 좌 주입, mirror 86% 근거) / 탐색 강화 / 좌 방향 보상 비대칭. v15 lat_pen=6이 좌를 죽인 것도 재검토.

## ★ 자발적 발견을 막는 요인 (2026-06-08, 처방은 "답 주입(mirror) X, 정책 스스로 발견" — [[feedback_emergent_over_injection]])
사용자 측정 종합: 정책은 좌 **방향(+offset)은 이미 스스로 탐색함** (v15 97%·v20 76%, `/tmp/explore_offset.py`). 못 하는 건 "방향 찾기"가 아니라 "곧장 패턴 정착". 이를 막는 요인:

**보상 레벨 (처방 대상)**:
1. **보상이 곧장을 안 가리킴** (높음): reward_probe 곧장 4.79 ≈ 앞지름 4.78 (동점) → 곧장 발견해도 정착 동기 0.
2. **lat_pen이 sideslip을 처벌** (높음): 곡선선회만 면제인데 그게 물리 불가(#7) → **유일한 물리적 도달법(곧장 sideslip)이 처벌받는 모순**. 우는 위드리프트 순풍으로 견디나 좌는 역풍+lat_pen에 막힘(v15 좌 도달 0). lat_pen 도입 의도(게걸음 차단)가 곡률 불가 현실에서 역효과.
3. **align 항 골짜기** (중, noisy): term_decomp 좌 선회 초입 align +0.01→−6.42. 머리 트는 과도기(turn_ratio↑·정렬 전 align<0)가 음수 처벌.

**물리 레벨 (수용, 목표를 여기 맞춤)**:
4. 곡률 한계(#7): 곧장 sideslip이 목표지 곡선선회 아님.
5. fin 위드리프트 역풍(좌, #4): 작음(2~3%)이나 좌가 약간 hard.

**미확정**: 탐색이 곧장 시변 패턴 공간 닿는지 / curriculum init 우편향.

→ **핵심 = 보상 1·2 정비** (곧장 우대 + sideslip 처벌 완화)로 정책이 스스로 곧장 발견하게. 답 주입 X.

## 미확정
- **시변(비대칭 꼬리질) 선회**: constant는 불가(#7)지만 시변은 별도. 단 현 action(freq/amp/offset)에서 비대칭 수단은 offset(DC bias)뿐이고 약함. timevarying_reachable에서 s3b 30° 2-phase 도달 0이었음 → 기대 낮음.

## 처방 함의 (현 시점)
- **parallel 강제(align 항 등)는 물리적으로 도달 불가능한 걸 요구** → 도달률 붕괴 위험. 폐기 후보.
- **현실 목표 = "앞지름 없는 효율적 sideslip"** (v16/s0 우처럼 곧장). 사용자 첫 목표가 물리적으로 옳았음.
- 후보: progress를 target방향 진행 보상으로(머리게이트 완화) — 좌가 곧장 대각선 sideslip. 게걸음 과함은 lat_pen이 cap. **단 reward 처방 전 곡률 한계상 "곧장 sideslip"이 목표지 parallel이 아님을 못박기.**
- 처방은 학습 trigger → plan + critic + 사용자 "학습 시작" 후 실행.

## ★ 결과 (2026-06-09, m4_cpg_v21) — 예상 #5 틀림: 탐색 강화만으로 좌/우 곧장 달성
- **reward 미변경**으로 좌회전 게걸음이 곧장 sideslip으로 정착. 처방은 위 "보상 1·2 정비"가 아니라 **탐색 강화(`ent_floor 0.008→0.004` decay) + 졸업 기준 strict화**(course≥0.70 & 머리각≤40°)였음.
- 좌(−y) 검증: s3a strict 98%(course 0.789·머리각 26.2°), s3b strict 100%(0.774·28.9°). v15 우 수준 달성, σ 작음. 이전 v19 좌 게걸음(course 0.52·49°) 완전 해소.
- **정정**: 사실 #5("progress 머리방향 게이트가 곧장을 막아 reward 처방 필요")는 **충분조건이 아니었음** — reward를 그대로 둬도 탐색이 곧장 패턴 공간에 닿으면 정책이 스스로 발견·정착([[feedback_emergent_over_injection]]과 정합). 미확정 항목 "탐색이 곧장 시변 패턴 공간 닿는지" = **닿음(확인)**.
- 상세 표·추세는 [training_log_m4.md](training_log_m4.md) m4_cpg_v21. 단일 seed → multi-seed 재현 후속.

## 진단 도구 (재현용, `/tmp/` 일회용 — 필요시 sim/diagnostics/로 이전)
| 스크립트 | 측정 |
|---|---|
| dissect_parallel.py | course·머리-진행각으로 parallel vs sideslip |
| policy_course.py | 카드/seed별 정책 우/좌 course (s3b) |
| drift_physics.py | 순수 물리 y 드리프트 (offset=0) |
| overshoot_diag.py / lr_overshoot_compare.py | 좌/우 앞지름 시간축 |
| turn_gradient.py / term_decomp.py | 선회 강도별 reward + 항 분해 |
| oracle_turn.py / oracle_v2.py | oracle 선회 컨트롤러 (실패) |
| curvature.py | 최소 선회 반경 (#7 핵심) |
