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
| 6Hz sine wet | tail amp 36.8°, xdisp -1.77m (이전 baseline 94%) |
| 6Hz 위 명령 (8/10Hz) | aliasing → 정책 표현 자체 불가 |
| fin amp wet 6Hz | 41° (추진 source 보존) |
| 1~5Hz 정상 추진 | baseline 92~98% 유지 |

## m4_v1 학습 시작 plan

- **단일 stage 학습** (CLAUDE.md §2: `--start-stage N --end-stage N`).
- prefix `m4_v1_seedN`. 학습 명령 예시:
  ```bash
  python3 curriculum.py --start-stage 1 --end-stage 1 --seed 0 \
      --tb-tag m4_v1_seed0 --runs-subdir m4_v1_seed0 --no-viewer
  ```
- 학습 전·후 6 stage deterministic eval 필수 (`sim/eval_stages.py`).
- 학습 trigger 직전 critic 호출 (CLAUDE.md §5).
- frame_skip 42로 episode step 수 1/4 (10s ep → 119 step). max_steps 예산 영향 점검 필요.

## GitHub Release 인덱스 (m4)

(없음 — 학습 시작 후 추가)

## 학습 결과 요약

(없음 — 학습 시작 후 추가)

## 주요 진단 (산문)

(없음 — 학습 시작 후 추가)
