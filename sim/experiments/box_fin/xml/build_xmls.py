"""sim/rl_fish.xml 기반 + fin geom만 box로 교체한 xml을 생성.

box 후보 2종 (면적 105 cm² 등가, thickness 0.88mm fiberglass):
  box1_disk : chord 10.5 × span 10.0 cm
  box2_cigar: chord 15.2 × span  6.9 cm

leading edge가 fin hinge에 정렬되도록 박스 center를 +x로 chord/2 이동.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent  # sim/experiments/box_fin/xml/
SIM_XML = HERE.parents[2] / "rl_fish.xml"  # sim/rl_fish.xml

CANDIDATES = {
    "box1_disk":  (0.105, 0.00088, 0.100),
    "box2_cigar": (0.152, 0.00088, 0.069),
}

src = SIM_XML.read_text()

# meshdir 상대 경로 보정 (sim/experiments/box_fin/xml/ 기준 — 4단계 위)
src = src.replace(
    'meshdir="../fish_urdf/RL_SIM_MODE_3_description/meshes/"',
    'meshdir="../../../../fish_urdf/RL_SIM_MODE_3_description/meshes/"',
)

fin_pat = re.compile(
    r'<geom\s+pos="-?[\d.]+\s+[\d.]+\s+-?[\d.]+"\s*\n\s*type="mesh"\s+mesh="fin_1"\s*\n\s*mass="0\.024"\s+material="fiberglass"/>',
    re.MULTILINE,
)
assert fin_pat.search(src), "fin_1 mesh geom 매칭 실패"

for name, (c, t, s) in CANDIDATES.items():
    box_geom = (
        f'<geom pos="{c/2:.5f} 0 0" type="box" '
        f'size="{c/2:.5f} {t/2:.5f} {s/2:.5f}" '
        f'mass="0.024" material="fiberglass"/>'
    )
    out_xml = fin_pat.sub(box_geom, src)
    out_path = HERE / f"rl_fish_{name}.xml"
    out_path.write_text(out_xml)
    print(f"wrote {out_path.relative_to(HERE.parents[3])}")
