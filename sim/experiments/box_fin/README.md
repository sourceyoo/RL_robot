# Box fin 실험 (실패 기록)

V-cut mesh → 면적 등가 박스 fin 교체 시도. **모든 후보 전 freq 후진** — V-cut으로 revert. 자세한 진단은 [`../../../docs/training_log_m4.md`](../../../docs/training_log_m4.md) "박스 fin 시도 (실패, 2026-05-19)" 섹션.

이 디렉토리는 박스 시도를 **재현 가능한 형태로 보존** (학습 환경 `sim/rl_fish.xml`엔 영향 없음).

## 디렉토리 구조

```
sim/experiments/box_fin/
├── README.md
├── xml/                              # 모델 데이터 (sim/rl_fish.xml 파생)
│   ├── build_xmls.py                 # 박스 xml 2개 생성 스크립트
│   ├── rl_fish_box1_disk.xml         # 박스 후보 1 (10.5 × 10.0 cm)
│   └── rl_fish_box2_cigar.xml        # 박스 후보 2 (15.2 ×  6.9 cm)
├── ellipsoid_viz/                    # fluid ellipsoid 시각화
│   └── render_ellipsoid.py           # → ~/Pictures/fish_ellipsoid/
└── freq_sweep/                       # 추진성 측정
    ├── run.py                        # V-cut + box1 + box2 비교
    └── results/
        └── freq_sweep_*.log
```

## 사용법

```bash
cd sim/experiments/box_fin

# 1. xml 재생성 (sim/rl_fish.xml 갱신 후 반영할 때)
python3 xml/build_xmls.py

# 2. ellipsoid 시각화 (~/Pictures/fish_ellipsoid/ 에 저장)
python3 ellipsoid_viz/render_ellipsoid.py

# 3. freq sweep 비교 (freq_sweep/results/에 log 저장)
python3 freq_sweep/run.py
```

## 박스 후보 사양

| 후보 | chord × span | thickness | 면적 | 의도 |
|---|---|---|---|---|
| box1_disk | 10.5 × 10.0 cm | 0.88 mm | 105 cm² | chord ≈ span 정사각 평판 |
| box2_cigar | 15.2 × 6.9 cm | 0.88 mm | 105 cm² | chord > span 길쭉한 평판 |

면적은 V-cut 실 면적(105 cm²)과 등가. thickness는 fiberglass 0.88mm 유지.

박스 leading edge가 fin hinge(=fin_1 body 원점)에 오도록 박스 center를 `pos="chord/2 0 0"`로 +x 이동. (초기 시도엔 박스가 base 안으로 박혀있던 버그 있었음, 정렬 후에도 후진 결과는 동일)

## Fluid ellipsoid semi-axes 비교 (mm, principal frame)

| config | a (motion-x) | b (lateral-y) | c (span-z) | 형태 |
|---|---|---|---|---|
| V-cut | **0.6** | 82.3 | 89.2 | thin disk, motion 방향 edge-on (≈ 칼날) |
| box1_disk | 67.8 | 0.6 | 64.5 | flat round disk, motion 방향 **face-on** |
| box2_cigar | 98.1 | 0.6 | 44.5 | flat elongated blade, motion 방향 face-on (가장 큼) |

## Freq sweep 결과 요약

| config | 1~3 Hz | 4~6 Hz | 결론 |
|---|---|---|---|
| V-cut | 후진 | **전진 ✓** | 4~6Hz 정상 BCF |
| box1_disk | 후진 | 후진 (큼) | 전 freq 후진 |
| box2_cigar | 후진 | 후진 | 전 freq 후진 |

## 핵심 인사이트

- **외형(visual geom) ≠ fluid가 보는 ellipsoid**. MuJoCo `fluidshape="ellipsoid"`는 inertia tensor에서 등가 ellipsoid를 만들기 때문.
- 박스의 uniform mass 분포가 motion 방향(chord)으로도 mass를 분포시켜 → ellipsoid motion 방향 半축이 큼 → **forward drag**가 thrust보다 커서 후진.
- V-cut mesh는 vertex가 motion 방향 0.88mm 안에만 모여 → motion 방향 inertia ≈ 0 → ellipsoid motion 방향 半축 ≈ 0 → **forward drag 거의 0** → 4~6Hz 전진.
- **R5 fluidcoef는 V-cut 형상에 fit된 값**. 박스로 가려면 fishsim BO sysid 재실행 필요.

## 결론

박스 채택 불가. V-cut + R5가 현재 정착. 향후 박스 재시도하려면:
1. fishsim BO 재실행 (자원 큼)
2. 또는 fluid model 자체 개선 (Lighthill `mjcb_passive` / ANN surrogate)

## 산출 이미지

`~/Pictures/fish_ellipsoid/{vcut,box1_disk,box2_cigar}/` — 각 형태별 7뷰 (all_iso, all_top, fin_iso, fin_top, fin_side, fin_front, mesh_only). V-cut엔 base_iso, tail_iso 추가.
