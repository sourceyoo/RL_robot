"""D6: D4 96% 감소의 진짜 원인 검증.

D5 에서 tail_link mesh_quat = (R90, P-0.8, Y81.8°) 의 8.2° offset 관찰.
가설: 이 frame 회전이 fluid drag 좌우 비대칭의 원인.
의심: mesh aspect 거의 1:1 (D5: x_range 36.6mm vs y_range 36.3mm)
      → PCA eigenvalue degenerate → 8.2° 가 numerical noise 가능성.

Step A: raw STL covariance eigenvalue 비율 → PCA stability
Step B: case 별 ctrl_mirror ablation
  case 0  baseline (D4 와 동일)
  case 1  tail_link geom 에 quat 명시 (mesh_quat inverse) → fluid frame body axis 정렬
  case 2  fin_1 inertial pos.y = 0 강제
  case 3  case 1 + case 2 동시
case 1 단독으로 비대칭 큰 폭 감소 → frame 회전 = 원인 확정.
case 1 후 잔존 + case 2 에서 감소 → fin CoM 비대칭도 source.
case 1·2 모두 효과 X → 8.2° 는 hyperparametric noise, 다른 원인.
"""
import os
import struct
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fish_env import DEFAULT_XML

STL_DIR = Path(__file__).resolve().parent.parent.parent / "fish_urdf" / "RL_SIM_MODE_3_description" / "meshes"


def parse_stl_binary(path):
    """STL binary → unique vertex coordinates (m, 단위 변환 X)."""
    with open(path, "rb") as f:
        f.read(80)
        n_faces = struct.unpack("<I", f.read(4))[0]
        verts = np.empty((n_faces * 3, 3), dtype=np.float64)
        for i in range(n_faces):
            f.read(12)  # normal vector skip
            for j in range(3):
                verts[i * 3 + j] = struct.unpack("<fff", f.read(12))
            f.read(2)  # attribute byte count
    return verts


def pca_eigenvalues(verts):
    """vertex cov eigenvalue + eigenvector (centered). 큰 순 정렬."""
    centered = verts - verts.mean(axis=0)
    cov = (centered.T @ centered) / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]


def quat_inverse(q):
    """w,x,y,z conjugate (unit quat 가정)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_to_str(q):
    return " ".join(f"{v:.6f}" for v in q)


def load_xml_with_mods(case_name, mods):
    """원본 xml 읽어 case-specific 수정 후 임시 파일로 저장 → load."""
    with open(DEFAULT_XML, "r", encoding="utf-8") as f:
        xml = f.read()
    for find, replace in mods:
        if find not in xml:
            raise RuntimeError(f"[{case_name}] '{find[:60]}...' not in xml")
        xml = xml.replace(find, replace)
    parent = Path(DEFAULT_XML).parent
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{case_name}.xml", dir=parent, delete=False, encoding="utf-8"
    ) as f:
        f.write(xml)
        tmp_path = f.name
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
        return model
    finally:
        os.unlink(tmp_path)


def ctrl_mirror_sim(model, sim_seconds=20.0, freq=4.0, amp=0.5, offset=0.3):
    """ctrl(t) vs -ctrl(t) sim. (y_pos, yaw) 끝값 + max 비대칭 반환."""
    dt = model.opt.timestep
    n_steps = int(sim_seconds / dt)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tail")

    def run(sign):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        ys, yaws = [], []
        for k in range(n_steps):
            t = k * dt
            data.ctrl[actuator_id] = sign * (amp * np.sin(2 * np.pi * freq * t) + offset)
            mujoco.mj_step(model, data)
            ys.append(data.qpos[1])
            yaws.append(data.qpos[2])
        return np.array(ys), np.array(yaws)

    y_pos, yaw_pos = run(+1)
    y_neg, yaw_neg = run(-1)
    # mirror symmetric system: y_neg(t) ≈ -y_pos(t), yaw_neg(t) ≈ -yaw_pos(t)
    y_asym = np.abs(y_pos + y_neg)
    yaw_asym = np.abs(yaw_pos + yaw_neg)
    return {
        "y_end_pos": y_pos[-1], "y_end_neg": y_neg[-1],
        "yaw_end_pos": yaw_pos[-1], "yaw_end_neg": yaw_neg[-1],
        "y_asym_max": y_asym.max(), "yaw_asym_max": yaw_asym.max(),
        "y_asym_end": y_asym[-1], "yaw_asym_end": yaw_asym[-1],
    }


# ============================================================
# Step A: raw STL PCA stability
# ============================================================
print("=" * 64)
print("Step A: raw STL covariance eigenvalue (PCA stability)")
print("=" * 64)

for name in ["base_link", "tail_link_1_1", "fin_1"]:
    stl = STL_DIR / f"{name}.stl"
    verts_raw = parse_stl_binary(stl)
    # mujoco scale 적용 (xml: base 0.001, tail 0.001, fin 0.001 x 0.0011 y x 0.001 z)
    if name == "fin_1":
        verts = verts_raw * np.array([0.001, 0.0011, 0.001])
    else:
        verts = verts_raw * 0.001
    eigvals, eigvecs = pca_eigenvalues(verts)
    print(f"\n[{name}]  n_verts={len(verts)}")
    print(f"  eigenvalues (sorted desc): {eigvals[0]:.4e}  {eigvals[1]:.4e}  {eigvals[2]:.4e}")
    print(f"  ratio λ1/λ2 = {eigvals[0]/eigvals[1]:.3f}")
    print(f"  ratio λ2/λ3 = {eigvals[1]/eigvals[2]:.3f}  ← degenerate 신호 (1에 가까울수록 PCA frame unstable)")
    # eigenvector 의 첫 두 축이 raw STL frame 의 어떤 축과 정렬되는지
    for i, axis_name in enumerate(["1st (long)", "2nd (mid)", "3rd (short)"]):
        v = eigvecs[:, i]
        # 가장 큰 component
        j = np.argmax(np.abs(v))
        sign = "+" if v[j] > 0 else "-"
        print(f"    {axis_name}: ({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})  → mostly {sign}{'xyz'[j]}")

# ============================================================
# Step B: ablation
# ============================================================
print("\n" + "=" * 64)
print("Step B: ctrl_mirror ablation (sim 20s, freq 4Hz, amp 0.5, offset 0.3)")
print("=" * 64)

# baseline mesh_quat from D5 (tail_link) — geom 의 quat 명시 시 cancel 위해 inverse 사용
# D5 출력: tail_link mesh_quat = (0.5312, 0.5376, 0.4590, 0.4669)
tail_mesh_quat = np.array([0.5312, 0.5376, 0.4590, 0.4669])
tail_mesh_quat_inv = quat_inverse(tail_mesh_quat)
print(f"\ntail_link mesh_quat (D5): {quat_to_str(tail_mesh_quat)}")
print(f"  → inverse (geom quat 명시값): {quat_to_str(tail_mesh_quat_inv)}")

# 각 case 의 xml 수정 패턴 (find → replace)
TAIL_GEOM_FIND = '<geom pos="-0.144817 0 0.03565" type="mesh" mesh="tail_link"/>'
TAIL_GEOM_QUAT_CANCEL = f'<geom pos="-0.144817 0 0.03565" type="mesh" mesh="tail_link" quat="{quat_to_str(tail_mesh_quat_inv)}"/>'

FIN_INERTIAL_FIND_PREFIX = '<body name="fin_1"'
# fin_1 은 inertial 명시 없으면 자동 PCA — D5 의 ipos.y=0.000385 가 자동값.
# case 2 는 fin_1 inertial 명시 + pos.y=0 강제. D5 의 mass/diag 값 사용.
FIN_BODY_FIND = '<body name="fin_1" pos="0.057 0 -0.011375">\n          <joint name="fin_joint" type="hinge" axis="0 0 -1"'
FIN_BODY_REPLACE = '''<body name="fin_1" pos="0.057 0 -0.011375">
          <inertial pos="0.074486 0 0.006001" mass="0.024"
                    diaginertia="7.0713e-05 3.8197e-05 3.2519e-05"/>
          <joint name="fin_joint" type="hinge" axis="0 0 -1"'''

cases = {
    "case_0_baseline":        [],
    "case_1_tail_quat":       [(TAIL_GEOM_FIND, TAIL_GEOM_QUAT_CANCEL)],
    "case_2_fin_ipos_y0":     [(FIN_BODY_FIND, FIN_BODY_REPLACE)],
    "case_3_tail_and_fin":    [(TAIL_GEOM_FIND, TAIL_GEOM_QUAT_CANCEL), (FIN_BODY_FIND, FIN_BODY_REPLACE)],
}

results = {}
for name, mods in cases.items():
    model = load_xml_with_mods(name, mods)
    r = ctrl_mirror_sim(model)
    results[name] = r
    print(f"\n[{name}]")
    print(f"  y(+ctrl) end={r['y_end_pos']:+.5f}   y(-ctrl) end={r['y_end_neg']:+.5f}")
    print(f"  yaw(+) end ={r['yaw_end_pos']:+.5f}  yaw(-) end ={r['yaw_end_neg']:+.5f}")
    print(f"  |y+y_neg| max={r['y_asym_max']:.5f}  end={r['y_asym_end']:.5f}")
    print(f"  |yaw+yaw_neg| max={r['yaw_asym_max']:.5f}  end={r['yaw_asym_end']:.5f}")

# summary
print("\n" + "=" * 64)
print("Summary: baseline 대비 변화율")
print("=" * 64)
b = results["case_0_baseline"]
print(f"  baseline  y_asym_max={b['y_asym_max']:.5f}  yaw_asym_max={b['yaw_asym_max']:.5f}")
for name in ["case_1_tail_quat", "case_2_fin_ipos_y0", "case_3_tail_and_fin"]:
    r = results[name]
    dy = (r["y_asym_max"] - b["y_asym_max"]) / max(b["y_asym_max"], 1e-9) * 100
    dyaw = (r["yaw_asym_max"] - b["yaw_asym_max"]) / max(b["yaw_asym_max"], 1e-9) * 100
    print(f"  {name:>26}  y {dy:+6.1f}%   yaw {dyaw:+6.1f}%")
