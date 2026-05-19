# `runs_archive/` — 옛 학습 산출물 보존

현재 활성 학습은 [`../runs/r5_back05_seed0/`](../runs/r5_back05_seed0). 이 디렉토리는 옛 학습의 `model.zip`·`SAC_1/`·log를 보존만 함 (curriculum chain에서 참조되지 않음).

## 분류

| 디렉토리 | 시대 | 내용 | 개수 |
|---|---|---|---|
| `v4/`, `v8/` | 매우 옛 (m1~m3 초기) | runs_v{4,8}_backup이었던 백업 | 6, 6 |
| `m1_m3_stages/` | 구 환경 (frame_skip 4) | s1_forward·s2_anchor·s3*·s3_head_arc 등 stage 이름만으로 정리 | 8 |
| `m1_m3_multi_seed/` | 구 환경 multi-seed | v24~v33 (single+multi-seed) | 29 |
| `m4_v1_v8b/` | m4 환경 (frame_skip 42) v1~v8b 시리즈 | 9 버전 × 3 seed | 27 |
| `mode3_planar/` | 옛 학습 시도 (3DOF planar 도입기) | SAC_1 + model.zip | 2 |
| `smoke/` | smoke 테스트 산출물 | smoke, smoke2, smoke_v30 | 3 |

## 관련 로그

옛 학습의 `*.log` (tee 출력)는 [`../logs/`](../logs/)에 분리:
- `../logs/m1_m3/` — v2~v33 curriculum·eval·multi-seed (63개)
- `../logs/m4_v8/` — m4 v8·v8b (6개)
