"""V-cut · box1_disk · box2_cigar freq sweep 비교.

각 config의 추진 방향을 측정. 결과는 results/에 timestamp log 저장.
"""

from datetime import datetime
from pathlib import Path
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent  # sim/experiments/box_fin/freq_sweep/
SIM_XML = HERE.parents[2] / "rl_fish.xml"  # sim/rl_fish.xml
XML_DIR = HERE.parent / "xml"

def vcut_tmp_path():
    src = SIM_XML.read_text().replace(
        'meshdir="../fish_urdf/RL_SIM_MODE_3_description/meshes/"',
        f'meshdir="{HERE.parents[3]}/fish_urdf/RL_SIM_MODE_3_description/meshes/"',
    )
    tmp = HERE / ".rl_fish_vcut_abs.xml"
    tmp.write_text(src)
    return tmp

CONFIGS = {
    "vcut (reference)": vcut_tmp_path(),
    "box1_disk":        XML_DIR / "rl_fish_box1_disk.xml",
    "box2_cigar":       XML_DIR / "rl_fish_box2_cigar.xml",
}

FREQS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
DURATION = 15.0
SETTLE = 3.0
X, Y, YAW = 0, 1, 2

lines = []
def out(s):
    print(s); lines.append(s)

out(f"# freq sweep — {datetime.now().isoformat(timespec='seconds')}")
out(f"# ctrl = sin(2π f t), {DURATION}s, settle {SETTLE}s. world −x = 머리(전진).")

for cfg_name, xml_path in CONFIGS.items():
    out(f"\n=== {cfg_name} ===")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    out(f"{'f[Hz]':>5} | {'x_disp':>10} {'y_disp':>10} {'yaw[°]':>10} | 방향")
    out("-" * 60)
    for f in FREQS:
        data = mujoco.MjData(model)
        x_pos, y_pos, yaws = [], [], []
        for _ in range(int(DURATION / model.opt.timestep)):
            t = data.time
            data.ctrl[0] = np.sin(2 * np.pi * f * t)
            mujoco.mj_step(model, data)
            if t >= SETTLE:
                x_pos.append(data.qpos[X])
                y_pos.append(data.qpos[Y])
                yaws.append(data.qpos[YAW])
        x_d = x_pos[-1] - x_pos[0]
        y_d = y_pos[-1] - y_pos[0]
        yaw_d = np.degrees(yaws[-1] - yaws[0])
        if x_d < -0.01:    verdict = "← 머리쪽(전진) ✓"
        elif x_d > 0.01:   verdict = "→ 꼬리쪽(후진)"
        else:              verdict = "  거의 정지"
        out(f"{f:5.2f} | {x_d:+10.4f} {y_d:+10.4f} {yaw_d:+10.2f} | {verdict}")

# log 저장
log_path = HERE / "results" / f"freq_sweep_{datetime.now().strftime('%Y-%m-%d')}.log"
log_path.parent.mkdir(exist_ok=True)
log_path.write_text("\n".join(lines) + "\n")
print(f"\nsaved {log_path.relative_to(HERE.parents[3])}")

(HERE / ".rl_fish_vcut_abs.xml").unlink(missing_ok=True)
