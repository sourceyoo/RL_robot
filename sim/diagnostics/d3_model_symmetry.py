"""D3: 좌우 비대칭의 근원 추적.

D2 에서 phase 무관 steady-state 비대칭 확인 → xml/mesh 자체 비대칭이 원인.

검사:
1. Mesh vertex y=0 plane mirror 대칭성 — STL 파일 자체 좌우 대칭 여부.
2. Body 자동 적분 inertia (mass + iquat) — fin_1 의 PCA 회전 정량.
3. Geom orientation (geom_quat) — fluid ellipsoid 방향.
4. Ctrl mirror sim — ctrl(t) vs -ctrl(t) 두 sim 비교. 완벽 대칭이면 y(t)=-y(t).
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fish_env import DEFAULT_XML


def mesh_y_asymmetry(model, mesh_name):
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    start = model.mesh_vertadr[mesh_id]
    n = model.mesh_vertnum[mesh_id]
    verts = np.array(model.mesh_vert[start:start + n])  # (n, 3)
    mirrored = verts.copy()
    mirrored[:, 1] *= -1
    # numpy broadcast NN — chunked to limit memory
    chunk = 500
    dists = np.empty(n)
    for i in range(0, n, chunk):
        q = mirrored[i:i + chunk]
        d = np.linalg.norm(q[:, None, :] - verts[None, :, :], axis=2)
        dists[i:i + chunk] = d.min(axis=1)
    return {
        "n": n,
        "y_range": (float(verts[:, 1].min()), float(verts[:, 1].max())),
        "y_mean": float(verts[:, 1].mean()),
        "max_d": float(dists.max()),
        "mean_d": float(dists.mean()),
    }


def run_ctrl(ctrl_fn, sim_seconds=20.0):
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    data = mujoco.MjData(model)
    n_steps = int(sim_seconds / model.opt.timestep)
    ys, yaws = [], []
    for i in range(n_steps):
        t = i * model.opt.timestep
        data.ctrl[0] = float(ctrl_fn(t))
        mujoco.mj_step(model, data)
        ys.append(float(data.qpos[1]))
        yaws.append(float(data.qpos[2]))
    return np.array(ys), np.array(yaws)


def main():
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)

    print("=== Test 1: Mesh STL y=0 mirror 대칭성 ===")
    print("max_d / mean_d = mirror vertex 의 원본 mesh 최근접 거리. 0 이면 완벽 좌우 대칭.")
    print(f"{'mesh':<12} {'n_verts':>8} {'y∈[min,max]':>22} {'y_mean':>10} {'max_d (m)':>10} {'mean_d (m)':>11}")
    for name in ["base_link", "tail_link", "fin_1"]:
        r = mesh_y_asymmetry(model, name)
        print(f"{name:<12} {r['n']:>8d} [{r['y_range'][0]:+.5f}, {r['y_range'][1]:+.5f}] "
              f"{r['y_mean']:+10.6f} {r['max_d']:>10.6f} {r['mean_d']:>11.6f}")

    print("\n=== Test 2: Body 자동 적분 inertia (mass + diag + iquat) ===")
    for bi in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi)
        if name in ("world",):
            continue
        diag = model.body_inertia[bi]
        iquat = model.body_iquat[bi]
        ipos = model.body_ipos[bi]
        mass = model.body_mass[bi]
        # iquat 가 단위 quaternion (1, 0, 0, 0) = 회전 없음 = 좌우 대칭
        is_aligned = np.allclose(iquat, [1, 0, 0, 0], atol=1e-6)
        print(f"  {name:<12}: mass={mass:.4f}, ipos={ipos}, diag={diag}")
        print(f"  {' ':<12}  iquat={iquat}  {'(축 정렬 ✓)' if is_aligned else '(회전됨 ✗)'}")

    print("\n=== Test 3: Geom orientation / size (fluid ellipsoid) ===")
    for gi in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if name in ("ground", "target"):
            continue
        quat = model.geom_quat[gi]
        size = model.geom_size[gi]
        bid = model.geom_bodyid[gi]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        is_aligned = np.allclose(quat, [1, 0, 0, 0], atol=1e-6)
        print(f"  geom_{name or gi}({bname}): quat={quat} "
              f"{'(축 정렬 ✓)' if is_aligned else '(회전됨 ✗)'}")

    print("\n=== Test 4: Ctrl mirror sim — ctrl(t) vs -ctrl(t) ===")
    print("완벽 좌우 대칭 model 이면 y_A(t) = -y_B(t), yaw_A(t) = -yaw_B(t).")
    ctrl_A = lambda t: 0.5 * np.sin(2 * np.pi * 4 * t) + 0.3
    ctrl_B = lambda t: -(0.5 * np.sin(2 * np.pi * 4 * t) + 0.3)
    yA, yawA = run_ctrl(ctrl_A, 20.0)
    yB, yawB = run_ctrl(ctrl_B, 20.0)
    y_err = yA + yB
    yaw_err = yawA + yawB
    print(f"  baseline |y|_max  = {abs(yA).max():.4f}m, |yaw|_max = {np.degrees(abs(yawA)).max():.2f}°")
    print(f"  y mirror error:   max={abs(y_err).max():.5f}m   mean={abs(y_err).mean():.5f}m")
    print(f"  yaw mirror error: max={np.degrees(abs(yaw_err)).max():.3f}°  mean={np.degrees(abs(yaw_err)).mean():.3f}°")
    # relative error
    rel_y = abs(y_err).max() / (abs(yA).max() + 1e-9) * 100
    rel_yaw = abs(yaw_err).max() / (abs(yawA).max() + 1e-9) * 100
    print(f"  상대 mirror error: y={rel_y:.1f}%, yaw={rel_yaw:.1f}%  (0% = 완벽 대칭)")


if __name__ == "__main__":
    main()
