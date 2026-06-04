"""D5: tail_link fluid ellipsoid 정밀 진단.

D4 결과로 tail_link 의 fluid drag 가 y 비대칭의 주범 확정 (96% 절대 감소).
어떤 비대칭이 fluid ellipsoid 에 들어가는지 정량:

1. mesh PCA frame vertex 분포 (y-mirror NN dist + bin histogram)
2. mesh asset 의 PCA 변환 (mesh_pos, mesh_quat) — STL → PCA frame
3. body inertia (mass, ipos, iquat, diagonal) — fluid ellipsoid axes 결정
4. geom orientation (geom_quat, geom_pos, geom_aabb) — world frame ellipsoid 정합
5. ctrl_mirror sim 에서 yaw 발산 곡선 (어느 phase 에서 비대칭 누적되는지)

세부 후보:
- mesh_quat 이 단위 quat 아님 → STL 자체가 body frame 대비 회전
- body_iquat 이 단위 quat 아님 → PCA principal axis 가 body axis 와 다름
- ipos.y != 0 → CoM 좌우 편향
- mesh y-axis vert 분포가 비대칭 → mesh 형상 자체 좌우 비대칭
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fish_env import DEFAULT_XML


def quat_to_euler_deg(q):
    """w,x,y,z → roll, pitch, yaw (deg)."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch_sin = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(pitch_sin, -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def y_mirror_stats(verts):
    """vertex (n,3) 의 y=0 mirror NN distance."""
    n = len(verts)
    mirrored = verts.copy()
    mirrored[:, 1] *= -1
    dists = np.empty(n)
    chunk = 500
    for i in range(0, n, chunk):
        q = mirrored[i:i + chunk]
        d = np.linalg.norm(q[:, None, :] - verts[None, :, :], axis=2)
        dists[i:i + chunk] = d.min(axis=1)
    return dists


def main():
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)

    # ===== 모든 mesh 의 PCA 변환 + body inertia 비교 (참고용) =====
    print("=== mesh PCA 변환 (STL → mujoco mesh frame) ===")
    print(f"{'mesh':<12} {'mesh_pos':<28} {'mesh_quat (w,x,y,z)':<32} {'euler (R,P,Y°)'}")
    for mi, name in enumerate(["base_link", "tail_link", "fin_1"]):
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, name)
        mpos = model.mesh_pos[mid]
        mquat = model.mesh_quat[mid]
        e = quat_to_euler_deg(mquat)
        print(f"{name:<12} ({mpos[0]:+.5f},{mpos[1]:+.5f},{mpos[2]:+.5f})  "
              f"({mquat[0]:+.4f},{mquat[1]:+.4f},{mquat[2]:+.4f},{mquat[3]:+.4f})  "
              f"({e[0]:+.1f},{e[1]:+.1f},{e[2]:+.1f})")

    print("\n=== body inertia (mass, ipos, iquat, diag) ===")
    for name in ["base_link", "tail_link", "fin_1"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        m = model.body_mass[bid]
        ipos = model.body_ipos[bid]
        iquat = model.body_iquat[bid]
        diag = model.body_inertia[bid]
        e = quat_to_euler_deg(iquat)
        is_aligned = np.allclose(iquat, [1, 0, 0, 0], atol=1e-6)
        print(f"  {name:<12} mass={m:.5f}  ipos=({ipos[0]:+.6f},{ipos[1]:+.6f},{ipos[2]:+.6f})")
        print(f"  {' ':<12} iquat=({iquat[0]:+.4f},{iquat[1]:+.4f},{iquat[2]:+.4f},{iquat[3]:+.4f}) "
              f"euler(R,P,Y°)=({e[0]:+.1f},{e[1]:+.1f},{e[2]:+.1f})  "
              f"{'(축 정렬 ✓)' if is_aligned else '(회전됨 ✗)'}")
        print(f"  {' ':<12} diag(Ixx,Iyy,Izz)=({diag[0]:.4e},{diag[1]:.4e},{diag[2]:.4e})")

    print("\n=== mesh PCA frame 의 vertex 분포 ===")
    print("mesh_vert 는 mujoco PCA frame 으로 정렬된 좌표. 좌우(y) 비대칭이면 fluid ellipsoid 도 비대칭.")
    print(f"{'mesh':<12} {'n_verts':>8} {'x range (m)':>22} {'y range (m)':>22} {'z range (m)':>22} {'y mean':>10}")
    for name in ["base_link", "tail_link", "fin_1"]:
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, name)
        start = model.mesh_vertadr[mid]
        n = model.mesh_vertnum[mid]
        verts = np.array(model.mesh_vert[start:start + n])
        xr = (verts[:, 0].min(), verts[:, 0].max())
        yr = (verts[:, 1].min(), verts[:, 1].max())
        zr = (verts[:, 2].min(), verts[:, 2].max())
        ym = verts[:, 1].mean()
        print(f"{name:<12} {n:>8d} [{xr[0]:+.4f},{xr[1]:+.4f}] "
              f"[{yr[0]:+.4f},{yr[1]:+.4f}] [{zr[0]:+.4f},{zr[1]:+.4f}] {ym:+10.6f}")

    print("\n=== tail_link mesh y-mirror NN distance 정밀 ===")
    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "tail_link")
    start = model.mesh_vertadr[mid]
    n = model.mesh_vertnum[mid]
    verts = np.array(model.mesh_vert[start:start + n])
    dists = y_mirror_stats(verts)
    print(f"  n={n}, max_d={dists.max():.6f}m, mean_d={dists.mean():.6f}m, "
          f"median={np.median(dists):.6f}m")
    print(f"  >1mm: {(dists > 1e-3).sum()} verts ({100*(dists > 1e-3).sum()/n:.1f}%)")
    print(f"  >5mm: {(dists > 5e-3).sum()} verts ({100*(dists > 5e-3).sum()/n:.1f}%)")
    # +y vs -y bin
    print(f"  vertex 분포 by y (+y / -y):")
    for thresh in [0, 1e-4, 1e-3, 5e-3]:
        plus = (verts[:, 1] > thresh).sum()
        minus = (verts[:, 1] < -thresh).sum()
        near = ((verts[:, 1] >= -thresh) & (verts[:, 1] <= thresh)).sum()
        print(f"    |y|>{thresh*1000:.1f}mm:  +y={plus:5d}  -y={minus:5d}  |y|≤={near:5d}")

    print("\n=== geom orientation + bounding box ===")
    print("mujoco fluid ellipsoid 는 body inertia + iquat 으로 결정. geom_quat 는 추가 회전.")
    for gi in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if name in ("ground", "target"):
            continue
        bid = model.geom_bodyid[gi]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        gpos = model.geom_pos[gi]
        gquat = model.geom_quat[gi]
        gsize = model.geom_size[gi]
        rbound = model.geom_rbound[gi]
        # geom_aabb (6,) per geom: center(3) + half_size(3) in geom frame
        aabb = model.geom_aabb[gi]
        e = quat_to_euler_deg(gquat)
        is_aligned = np.allclose(gquat, [1, 0, 0, 0], atol=1e-6)
        print(f"  geom[{gi}] body={bname or '?'}")
        print(f"    gpos=({gpos[0]:+.5f},{gpos[1]:+.5f},{gpos[2]:+.5f})  "
              f"gquat=({gquat[0]:+.4f},{gquat[1]:+.4f},{gquat[2]:+.4f},{gquat[3]:+.4f})  "
              f"{'(정렬 ✓)' if is_aligned else '(회전 ✗ R,P,Y='+str(e.round(1).tolist())+')'}")
        print(f"    aabb center=({aabb[0]:+.5f},{aabb[1]:+.5f},{aabb[2]:+.5f}) "
              f"half_size=({aabb[3]:.5f},{aabb[4]:.5f},{aabb[5]:.5f})  rbound={rbound:.5f}")

    # ===== Test: mesh inertia 로부터 fluid ellipsoid axes 역산 =====
    print("\n=== fluid ellipsoid 의 effective axes (mass+diag 로부터 역산) ===")
    print("uniform density solid ellipsoid: Ixx=m(b²+c²)/5, Iyy=m(a²+c²)/5, Izz=m(a²+b²)/5")
    print("→ a²=(5/2m)(-Ixx+Iyy+Izz), b²=(5/2m)(Ixx-Iyy+Izz), c²=(5/2m)(Ixx+Iyy-Izz)")
    print("(a,b,c) = body inertia frame 의 ellipsoid semi-axes")
    for name in ["base_link", "tail_link", "fin_1"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        m = model.body_mass[bid]
        Ixx, Iyy, Izz = model.body_inertia[bid]
        a2 = (5 / (2 * m)) * (-Ixx + Iyy + Izz)
        b2 = (5 / (2 * m)) * (Ixx - Iyy + Izz)
        c2 = (5 / (2 * m)) * (Ixx + Iyy - Izz)
        a = np.sqrt(max(a2, 0))
        b = np.sqrt(max(b2, 0))
        c = np.sqrt(max(c2, 0))
        print(f"  {name:<12} a={a*1000:.2f}mm  b={b*1000:.2f}mm  c={c*1000:.2f}mm "
              f"(inertia frame 축 순서)")


if __name__ == "__main__":
    main()
