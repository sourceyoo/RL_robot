"""rl_fish.xml 모델 이미지 렌더링 — fin_1 대칭화 후 시각 확인.

OUT_DIR = /home/yoo/RL_robot/images/fin_sym/
  top_view.png   — fixed_top camera (위에서, 좌우 대칭 확인용)
  side_view.png  — x-axis 측면
  iso_view.png   — 45° 사선
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

XML = "/home/yoo/RL_robot/sim/rl_fish.xml"
OUT_DIR = Path("/home/yoo/RL_robot/images/fin_sym")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render(model, data, camera, path, w=640, h=480):
    renderer = mujoco.Renderer(model, width=w, height=h)
    renderer.update_scene(data, camera=camera)
    img = renderer.render()
    Image.fromarray(img).save(path)
    print(f"  saved: {path}")


def main():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Top view (fixed cam from xml)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_top")
    render(model, data, cam, OUT_DIR / "top_view.png")

    # Side view (custom free cam)
    cam2 = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam2)
    cam2.lookat[:] = [0.0, 0.0, 0.0]
    cam2.distance = 0.6
    cam2.azimuth = 90
    cam2.elevation = 0
    render(model, data, cam2, OUT_DIR / "side_view.png")

    # Iso view
    cam3 = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam3)
    cam3.lookat[:] = [0.0, 0.0, 0.0]
    cam3.distance = 0.6
    cam3.azimuth = 45
    cam3.elevation = -30
    render(model, data, cam3, OUT_DIR / "iso_view.png")

    # Close-up on fin (where the change matters)
    cam4 = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam4)
    cam4.lookat[:] = [0.25, 0.0, 0.0]
    cam4.distance = 0.25
    cam4.azimuth = 90
    cam4.elevation = -85
    render(model, data, cam4, OUT_DIR / "fin_topdown.png")


if __name__ == "__main__":
    main()
