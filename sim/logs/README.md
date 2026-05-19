# Curriculum 학습 로그 archive

옛 학습 로그 보존. 학습 history·통찰은 [`../../docs/training_log.md`](../../docs/training_log.md) 및 [`../../docs/training_log_m4.md`](../../docs/training_log_m4.md) 참조.

## 디렉토리

| 디렉토리 | 시대 | 내용 |
|---|---|---|
| `m1_m3/` | v2 ~ v33 (구 환경, frame_skip 4) | 단일 학습 + multi-seed + eval 로그 (42개) |
| `m4_v8/` | m4 환경 v8·v8b (frame_skip 42, V2 spec 초기) | 6개, v10 이전 시도 |

## 현재 활성 학습 로그 (m4_v10, V-cut + R5 + backward penalty)

별도 위치: `../runs/<tag>/`. 예:
- `../runs/r5_back05_seed0/s3a_train.log` ~ `s3c_train.log` (stage별 학습)
- `../runs/r5_back05_seed0/s3b_eval.log`, `s3c_eval.log` (deterministic 6-stage eval)

이 archive는 참조용이고, 신규 학습은 `runs/<tag>/`에 직접 저장됨.
