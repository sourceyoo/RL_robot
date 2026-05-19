"""V-cut · box1_disk · box2_cigar fluid ellipsoid 시각화.

저장: ~/Pictures/fish_ellipsoid/{vcut,box1_disk,box2_cigar}/
"""

from pathlib import Path
import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent  # sim/experiments/box_fin/ellipsoid_viz/
SIM_XML = HERE.parents[2] / "rl_fish.xml"  # sim/rl_fish.xml
XML_DIR = HERE.parent / "xml"  # 박스 xml 디렉토리
OUT_ROOT = Path.home() / "Pictures" / "fish_ellipsoid"

# V-cut은 sim/rl_fish.xml을 임시로 절대 meshdir로 치환해 사용
def vcut_tmp_path():
    src = SIM_XML.read_text().replace(
        'meshdir="../fish_urdf/RL_SIM_MODE_3_description/meshes/"',
        f'meshdir="{HERE.parents[3]}/fish_urdf/RL_SIM_MODE_3_description/meshes/"',
    )
    tmp = HERE / ".rl_fish_vcut_abs.xml"  # 점 prefix로 hidden
    tmp.write_text(src)
    return tmp

CONFIGS = {
    "vcut":       vcut_tmp_path(),
    "box1_disk":  XML_DIR / "rl_fish_box1_disk.xml",
    "box2_cigar": XML_DIR / "rl_fish_box2_cigar.xml",
}

VIEWS = [
    # (name, body_lookat, distance, azimuth, elevation, ellipsoid_on)
    ("all_iso",    None,    0.9,  -135, -25, True),
    ("all_top",    None,    0.9,   90,  -89, True),
    ("fin_iso",    "fin_1", 0.22, -135, -25, True),
    ("fin_top",    "fin_1", 0.20,  90,  -89, True),
    ("fin_side",   "fin_1", 0.22,  90,   0,  True),
    ("fin_front",  "fin_1", 0.18,   0,   0,  True),
    ("mesh_only",  None,    0.9,  -135, -25, False),
]

def ellipsoid_axes(model, body):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    m = model.body_mass[bid]
    Ix, Iy, Iz = model.body_inertia[bid]
    k = 5.0 / (2.0 * m)
    return tuple(np.sqrt(max(v, 0)) for v in (
        k * (-Ix + Iy + Iz), k * (Ix - Iy + Iz), k * (Ix + Iy - Iz),
    ))

def render(model, data, out_path, body, dist, az, el, ell_on):
    opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(opt)
    if ell_on:
        opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = True
        opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if body is None:
        cam.lookat[:] = [0.0, 0.0, 0.0]
    else:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        cam.lookat[:] = data.xpos[bid]
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    with mujoco.Renderer(model, height=480, width=640) as r:
        r.update_scene(data, camera=cam, scene_option=opt)
        Image.fromarray(r.render()).save(out_path)
    print(f"  {out_path.relative_to(OUT_ROOT)}")

print("=== fin_1 ellipsoid semi-axes [mm] ===")
print(f"{'config':>12} | {'a (motion-x)':>13} {'b (lat-y)':>10} {'c (span-z)':>11}")

for cfg_name, xml_path in CONFIGS.items():
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    a, b, c = ellipsoid_axes(model, "fin_1")
    print(f"{cfg_name:>12} | {a*1000:13.1f} {b*1000:10.1f} {c*1000:11.1f}")

    out_dir = OUT_ROOT / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"-- {cfg_name} --")
    for vname, body, dist, az, el, ell_on in VIEWS:
        if vname == "mesh_only" and cfg_name != "vcut":
            continue
        render(model, data, out_dir / f"{vname}.png", body, dist, az, el, ell_on)

# V-cut: base / tail zoom 추가
print("-- vcut extras --")
model = mujoco.MjModel.from_xml_path(str(CONFIGS["vcut"]))
data = mujoco.MjData(model); mujoco.mj_forward(model, data)
print("=== body별 ellipsoid semi-axes [mm] (vcut) ===")
print(f"{'body':>10} | {'a':>6} {'b':>6} {'c':>6}")
for b in ["base_link", "tail_link", "fin_1"]:
    a_, b_, c_ = ellipsoid_axes(model, b)
    print(f"{b:>10} | {a_*1000:6.1f} {b_*1000:6.1f} {c_*1000:6.1f}")
render(model, data, OUT_ROOT / "vcut" / "base_iso.png", "base_link", 0.45, -135, -25, True)
render(model, data, OUT_ROOT / "vcut" / "tail_iso.png", "tail_link", 0.18, -135, -25, True)

# 임시 V-cut xml 정리
(HERE / ".rl_fish_vcut_abs.xml").unlink(missing_ok=True)
