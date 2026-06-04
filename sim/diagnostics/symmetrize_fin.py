"""fin_1.stl 좌우 대칭화 — D3 진단 후 처방.

STL binary 직접 파싱 (scipy/trimesh 호환 회피).
알고리즘: triangle centroid y >= -eps 인 것 keep + y-mirror copy 추가
         (winding 보존 위해 v1/v2 swap). 결과 = +y 쪽 mesh + 그 mirror.

결정적 진단 (2026-05-27): STL raw 의 x 축이 폭축 (span 152mm), 그 축으로
mirror NN distance 평균 31.6mm = 20.7% 비대칭. y (두께 0.8mm)·z (길이 165mm)
는 완벽 대칭. → mirror plane = x = midpoint.

사용:
  python3 diagnostics/symmetrize_fin.py
"""
import struct
import sys
from pathlib import Path

import numpy as np

MESH_DIR = Path("/home/yoo/RL_robot/fish_urdf/RL_SIM_MODE_3_description/meshes")
IN_PATH = MESH_DIR / "fin_1.stl"
BACKUP_PATH = MESH_DIR / "fin_1_orig.stl"
OUT_PATH = MESH_DIR / "fin_1_sym.stl"

TRI_DTYPE = np.dtype([
    ('normal', '<f4', 3),
    ('v0', '<f4', 3),
    ('v1', '<f4', 3),
    ('v2', '<f4', 3),
    ('attr', '<u2'),
])


def load_stl(path):
    with open(path, 'rb') as f:
        header = f.read(80)
        n_tris = struct.unpack('<I', f.read(4))[0]
        tris = np.fromfile(f, dtype=TRI_DTYPE, count=n_tris)
    return header, tris


def save_stl(path, header, tris):
    with open(path, 'wb') as f:
        f.write(header)
        f.write(struct.pack('<I', len(tris)))
        tris.tofile(f)


def report_x_stats(tris, label):
    all_x = np.concatenate([tris['v0'][:, 0], tris['v1'][:, 0], tris['v2'][:, 0]])
    cent_x = (tris['v0'][:, 0] + tris['v1'][:, 0] + tris['v2'][:, 0]) / 3
    print(f"  [{label}] tris={len(tris)}, vert_x∈[{all_x.min():+.4f},{all_x.max():+.4f}], "
          f"x_mean={all_x.mean():+.5f}, cent_x∈[{cent_x.min():+.4f},{cent_x.max():+.4f}]")


def symmetrize(tris, eps=1e-2):
    # midpoint = (max+min)/2 of all vertex x
    all_x = np.concatenate([tris['v0'][:, 0], tris['v1'][:, 0], tris['v2'][:, 0]])
    mid = (all_x.max() + all_x.min()) / 2.0
    print(f"  mirror plane: x = {mid:.4f}")

    cent_x = (tris['v0'][:, 0] + tris['v1'][:, 0] + tris['v2'][:, 0]) / 3
    pos_mask = cent_x > mid + eps     # +x half
    zero_mask = np.abs(cent_x - mid) <= eps  # midplane 근처 (한 번만 keep)

    pos_tris = tris[pos_mask].copy()
    zero_tris = tris[zero_mask].copy()

    # mirror copy of +x half: x → 2*mid - x, winding reverse (v1/v2 swap).
    # normal 은 v1/v2 swap 으로 자동 부호 반전되니까 별도 처리 X (좌표만 reflect 하면
    # winding 반대로 normal 음수). 단, STL 의 명시 normal 도 reflect 해야 일관 — 안 그러면
    # rendering 만 영향 (fluid 적분은 vertex 만 사용).
    mir = pos_tris.copy()
    for k in ['v0', 'v1', 'v2']:
        mir[k][:, 0] = 2 * mid - mir[k][:, 0]
    mir['normal'][:, 0] *= -1
    tmp = mir['v1'].copy()
    mir['v1'] = mir['v2']
    mir['v2'] = tmp

    out = np.concatenate([pos_tris, mir, zero_tris])
    return out


def main():
    if not IN_PATH.exists():
        sys.exit(f"입력 없음: {IN_PATH}")

    # backup
    if not BACKUP_PATH.exists():
        import shutil
        shutil.copy2(IN_PATH, BACKUP_PATH)
        print(f"backup: {BACKUP_PATH}")
    else:
        print(f"backup 이미 존재: {BACKUP_PATH} (덮어쓰지 않음)")

    # 원본 STL 다시 로드 (backup 본). 기존 잘못된 fin_1_sym.stl 덮어쓰기 전.
    header, tris = load_stl(BACKUP_PATH if BACKUP_PATH.exists() else IN_PATH)
    print("\n=== 입력 fin_1.stl (원본) ===")
    report_x_stats(tris, "원본")

    sym_tris = symmetrize(tris)
    print("\n=== 출력 (좌우 대칭화, x mirror) ===")
    report_x_stats(sym_tris, "sym")

    save_stl(OUT_PATH, header, sym_tris)
    print(f"\n저장: {OUT_PATH}")

    # 검증: x mean 이 midpoint 와 일치 + 좌우 대칭 분포
    all_x_sym = np.concatenate([sym_tris['v0'][:, 0], sym_tris['v1'][:, 0], sym_tris['v2'][:, 0]])
    mid = (all_x_sym.max() + all_x_sym.min()) / 2
    print(f"\n검증: vertex x_mean={all_x_sym.mean():+.6f}, midpoint={mid:.4f} "
          f"(mean ≈ midpoint 이면 대칭)")
    print(f"     |x>mid|={(all_x_sym > mid + 1e-6).sum()}, |x<mid|={(all_x_sym < mid - 1e-6).sum()}, "
          f"|x≈mid|={(np.abs(all_x_sym - mid) <= 1e-6).sum()}")


if __name__ == "__main__":
    main()
