# `images/` — 시각화 산출물

학습·진단 과정에서 생성된 이미지를 분류 보관. 원본 생성 스크립트와 산출 경로 명시.

## 디렉토리 구조

```
images/
├── ellipsoid/                          # MuJoCo fluidshape="ellipsoid" 시각화 (mjVIS_INERTIA)
│   ├── vcut/         (9 PNG)          # 현재 정착 fin (fiberglass 0.88mm V-cut)
│   ├── box1_disk/    (6 PNG)          # 박스 후보 1 (10.5×10.0 cm) — 실패
│   └── box2_cigar/   (6 PNG)          # 박스 후보 2 (15.2×6.9 cm) — 실패
└── trajectories/                       # 정책 deterministic rollout 경로
    └── r5_back05_seed0_s3c_arc60/  (6 PNG)  # 현재 최종 모델 (m4_v10)
```

## `ellipsoid/`

각 형태별 7~9뷰 (all_iso, all_top, fin_iso, fin_top, fin_side, fin_front, mesh_only). vcut엔 base_iso, tail_iso 추가.

- **출처**: `sim/experiments/box_fin/ellipsoid_viz/render_ellipsoid.py`
- **재생성**: `python3 sim/experiments/box_fin/ellipsoid_viz/render_ellipsoid.py` (기본 출력 `~/Pictures/fish_ellipsoid/`, 직접 옮기거나 스크립트 OUT 경로 수정)
- **인사이트**: vcut motion 방향 半축 ≈ 0 (칼날) → 전진. 박스는 motion 방향 半축 큼 (face-on) → 후진. 자세한 분석 `sim/experiments/box_fin/README.md`.

## `trajectories/`

각 stage 분포에서 deterministic rollout 20 episode의 fish (x, y) 경로 plot. 시작점 ★ (0,0), target ×, success_radius 점선 원, 초록=성공, 빨강=실패, ● 마지막 위치.

- **출처**: `sim/diagnostics/plot_trajectories.py`
- **재생성**: `python3 sim/diagnostics/plot_trajectories.py <model.zip>` (기본 출력 `<model_dir>/trajectories_<stage>.png`, 옮겨야 git 추적됨)
- **현재 모델**: `sim/runs/r5_back05_seed0/s3c_arc60/model.zip` (V-cut + R5 + backward penalty)
- **6 stage 사후 det eval**: s1=100%, s2=100%, s3a=100%, s3b=100%, s3c=100%, s3d=94%

## 향후 모델 추가 시

새 학습 산출 trajectory plot은 `trajectories/<runs_subdir>_<stage_tag>/` 형태로 분류해서 옮김. 모델별 비교 용이.
