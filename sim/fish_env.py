"""물고기 swim-to-target 강화학습 환경 (3DOF planar 모델용).

qpos layout (5): [root_x, root_y, root_yaw, tail_joint, fin_joint]
qvel layout (5): [vx, vy, vyaw, vtail, vfin]
부호 규약: world -x = 머리 방향(전진).

관측 (13 + action_history_n 차원):
  0 tail_qpos       1 tail_qvel       2 fin_qpos        3 fin_qvel
  4 world vx        5 world vy        6 yaw_rate
  7 sin(yaw)        8 cos(yaw)
  9 target_rel_x    10 target_rel_y   (world frame)
  11 |yaw_err|      12 distance       (v32-C: derived obs 명시)
  13~ 최근 N step의 sf_asym 이력 (action_history_n>0일 때)

행동 (1차원): sf_asym ∈ [-1, 1].
  m4_v17: 정책은 더 이상 motor ctrl 직접 X. 4Hz sine carrier 강제 (amp 고정 1.0).
  정책 출력 = 시간 비대칭 ratio.
    sf_asym = 0  → 대칭 sine (직진)
    sf_asym > 0  → +쪽 sweep slow, −쪽 sweep fast (한쪽 회전)
    sf_asym < 0  → 반대 회전
  ctrl 합성 (frame_skip 내부 매 mj_step):
    ratio = 0.5 + sf_asym · 0.3  (∈ [0.2, 0.8])
    phase = (data.time · 4Hz) mod 1
    p = phase/ratio · 0.5            if phase < ratio
        0.5 + (phase−ratio)/(1−ratio) · 0.5  otherwise
    ctrl = sin(2π · p)
  의도: hover mode 원천 차단 (정책이 "꼬리 안 흔들기" 선택 불가).
        amp 자유도 제거 = 추진 보장. 정책은 회전만 결정.

보상 (m4_v19): progress·5.0 + reach·10 + align_weight·align
      + ALIGN_VEL_W·align_vel
      + YAW_SIGN_W·yaw_rate·sign(yaw_err)
      − TIME_PEN_W·distance·current_time
      − BACK_PEN_W·(−cos(head,v))·|v|  (cos<0이고 |v|>0.05일 때만)

  - align_weight: 10s ep 0.02, 그 외(20s·60s) 0.05. align ∈ [-1, +1] (cos(머리, 목표)).
  - ALIGN_VEL_W=0.05: align_vel ∈ [-1, +1] (cos(velocity, 목표)).
    fluid drift cancel 학습 압력. v_norm>0.02 + rel_norm>1e-6 deadzone.
  - BACK_PEN_W=5.0: cos(head,v)<0(slip>90°)면 후진. cos·|v| 비례 강 페널티.
    deadzone |v|>0.05: tail wag 측면진동·정지 면제.
    m4_v18c s3c·d에서 정책이 cycle 변조로 effective 저주파 발현해 후진으로 target
    진입하는 mode 발견 (slip_μ 100°). memory feedback_no_backward 규칙 적용.

  m4_v17 제거된 항 (4Hz amp 고정 후 의미 사라짐):
    - ACTION_DIFF_W (ctrl smooth는 sine carrier로 자동 보장)
    - DC_PEN_W (B mode 차단 — 4Hz sine은 mean=0 보장)
    - ASYM_BONUS_W (sf 비대칭은 action으로 직접 결정)
    - STILL_PEN_W (amp 고정 1.0이라 정지 불가)
    - SLIP_PEN_W (amp 고정 + carrier 강제로 sideslip cheat 메커니즘 사라짐.
                   회전 시 자연 sideslip은 단일 motor 물리 부산물이라 제거)
    - BACK clip (sf±1.0 sustained 검증은 전진이지만, cycle 변조 mode는
                 검증 못 잡음 → m4_v19에서 reward 페널티로 직접 차단)
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

DEFAULT_XML = str(Path(__file__).parent / "rl_fish.xml")

# qpos/qvel 인덱스 (3DOF planar + tail + fin)
IDX_X, IDX_Y, IDX_YAW, IDX_TAIL, IDX_FIN = 0, 1, 2, 3, 4

# m4_v17: tail wagging carrier 주파수 (Hz). 4Hz = 실모터 dry 5.88Hz의 68% (5Hz 85%보다 보수적).
# 추진 속도 ~0.043 m/s (12s 변위 -0.52m, freq_sweep 측정). overshoot·미세조정 trade-off.
WAVE_FREQ_HZ = 4.0
# sf_asym ratio 범위: 0.5 ± SF_RANGE → [0.2, 0.8] 안전 영역.
SF_RANGE = 0.3


class FishSwimEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        frame_skip: int = 125,
        episode_seconds: float = 10.0,
        target_radius: float = 0.5,
        success_radius: float = 0.08,
        target_theta_range: tuple[float, float] = (-np.pi, np.pi),
        target_theta_offset_range: tuple[float, float] | None = None,
        action_history_n: int = 0,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.max_steps = int(episode_seconds / self.dt)
        self.target_radius = target_radius
        self.success_radius = success_radius
        self.target_theta_range = target_theta_range
        self.target_theta_offset_range = target_theta_offset_range
        # m4_v14: align_weight 0.012 → 0.05 (회전 incentive 강화, annular 분포 학습용).
        self.align_weight = 0.02 if episode_seconds <= 10.0 else 0.05
        self.action_history_n = max(0, int(action_history_n))
        self._action_history = np.zeros(self.action_history_n, dtype=np.float32)
        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None

        self._target_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target"
        )
        self._torso_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "torso"
        )
        if self._target_geom_id < 0 or self._torso_site_id < 0:
            raise RuntimeError("target geom 또는 torso site를 모델에서 찾지 못했습니다.")

        if self.model.nq != 5 or self.model.nv != 5:
            raise RuntimeError(
                f"기대 nq=nv=5 (3DOF planar + tail + fin), 실제 nq={self.model.nq} nv={self.model.nv}"
            )
        if self.model.nu != 1:
            raise RuntimeError(f"기대 nu=1 (단일 motor), 실제 nu={self.model.nu}")

        obs_dim = 13 + self.action_history_n
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # m4_v17: action 의미 = sf_asym (motor ctrl 직접 X).
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._step_count = 0
        self._prev_distance = 0.0

    def _torso_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._torso_site_id].copy()

    def _target_pos(self) -> np.ndarray:
        return self.model.geom_pos[self._target_geom_id].copy()

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self._target_pos() - self._torso_pos()))

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        yaw = float(qpos[IDX_YAW])
        rel = self._target_pos() - self._torso_pos()
        rel_xy = rel[:2]
        rel_norm = float(np.linalg.norm(rel_xy))
        head_dir = np.array([-np.cos(yaw), np.sin(yaw)])
        align = float(np.dot(head_dir, rel_xy / rel_norm)) if rel_norm > 1e-6 else 0.0
        abs_yaw_err = float(np.arccos(np.clip(align, -1.0, 1.0)))
        distance = float(np.linalg.norm(rel))
        base = np.array([
            qpos[IDX_TAIL], qvel[IDX_TAIL],
            qpos[IDX_FIN],  qvel[IDX_FIN],
            qvel[IDX_X],    qvel[IDX_Y],   qvel[IDX_YAW],
            np.sin(yaw),    np.cos(yaw),
            rel[0],         rel[1],
            abs_yaw_err,    distance,
        ], dtype=np.float32)
        if self.action_history_n > 0:
            return np.concatenate([base, self._action_history])
        return base

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = self.np_random
        if self.target_theta_offset_range is not None:
            mn, mx = self.target_theta_offset_range
            offset = rng.uniform(mn, mx) if mn != mx else mn
            sign = 1.0 if rng.uniform() < 0.5 else -1.0
            theta = np.pi + sign * offset
        else:
            lo, hi = self.target_theta_range
            theta = rng.uniform(lo, hi) if lo != hi else lo
        target = np.array([self.target_radius * np.cos(theta),
                           self.target_radius * np.sin(theta),
                           0.0])
        self.model.geom_pos[self._target_geom_id] = target

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._prev_distance = self._distance_to_target()
        if self.action_history_n > 0:
            self._action_history = np.zeros(self.action_history_n, dtype=np.float32)
        return self._get_obs(), {}

    def _synth_ctrl(self, sf_asym: float) -> float:
        """4Hz sine carrier with time-asymmetric ratio.
        sf_asym ∈ [-1, 1] → ratio ∈ [0.2, 0.8].
        한 cycle (1/4초) 안에서 +쪽 sweep 시간 = ratio·T, −쪽 sweep 시간 = (1−ratio)·T.
        """
        ratio = 0.5 + sf_asym * SF_RANGE
        phase = (self.data.time * WAVE_FREQ_HZ) % 1.0
        if phase < ratio:
            p = (phase / ratio) * 0.5
        else:
            p = 0.5 + ((phase - ratio) / (1.0 - ratio)) * 0.5
        return float(np.sin(2.0 * np.pi * p))

    def step(self, action: np.ndarray):
        clipped = np.clip(action, -1.0, 1.0)
        sf_asym = float(clipped[0])
        # m4_v17: frame_skip 내부에서 매 mj_step마다 5Hz sine ctrl 합성·갱신.
        for _ in range(self.frame_skip):
            self.data.ctrl[0] = self._synth_ctrl(sf_asym)
            mujoco.mj_step(self.model, self.data)
        if self.action_history_n > 0:
            self._action_history[:-1] = self._action_history[1:]
            self._action_history[-1] = sf_asym

        distance = self._distance_to_target()
        progress = self._prev_distance - distance
        reached = distance < self.success_radius

        yaw = float(self.data.qpos[IDX_YAW])
        head_dir = np.array([-np.cos(yaw), np.sin(yaw)])
        rel = self._target_pos()[:2] - self._torso_pos()[:2]
        rel_norm = float(np.linalg.norm(rel))
        align = float(np.dot(head_dir, rel / rel_norm)) if rel_norm > 1e-6 else 0.0
        yaw_err_sign = float(np.sign(head_dir[0] * rel[1] - head_dir[1] * rel[0])) if rel_norm > 1e-6 else 0.0

        YAW_SIGN_W = 0.005
        ALIGN_VEL_W = 0.05
        TIME_PEN_W = 1e-5
        BACK_PEN_W = 5.0
        yaw_rate = float(self.data.qvel[IDX_YAW])
        current_time = self._step_count * self.dt

        # m4_v18: velocity align — fluid drift cancel 학습 압력.
        v_world = np.array([float(self.data.qvel[IDX_X]), float(self.data.qvel[IDX_Y])])
        v_norm = float(np.linalg.norm(v_world))
        if v_norm > 0.02 and rel_norm > 1e-6:
            align_vel = float(np.dot(v_world / v_norm, rel / rel_norm))
        else:
            align_vel = 0.0

        # m4_v19: 후진 강 페널티. cos(head, v) < 0 (slip > 90°)이면 후진.
        # deadzone |v| > 0.05: tail wag 측면진동·정지 면제.
        back_pen = 0.0
        if v_norm > 0.05:
            cos_hv = float(np.dot(head_dir, v_world / v_norm))
            if cos_hv < 0.0:
                back_pen = BACK_PEN_W * (-cos_hv) * v_norm

        reward = (
            float(progress * 5.0)
            + (10.0 if reached else 0.0)
            + self.align_weight * align
            + ALIGN_VEL_W * align_vel
            + YAW_SIGN_W * yaw_rate * yaw_err_sign
            - TIME_PEN_W * distance * current_time
            - back_pen
        )

        # slip 측정 (reward 영향 X, eval sideslip 통계용). v_world·v_norm 위에서 계산.
        if v_norm > 0.02:
            cos_slip = float(np.clip(np.dot(head_dir, v_world / v_norm), -1.0, 1.0))
            slip_deg = float(np.degrees(np.arccos(cos_slip)))
        else:
            slip_deg = 0.0

        self._prev_distance = distance
        self._step_count += 1

        terminated = bool(reached)
        truncated = self._step_count >= self.max_steps

        info = {"distance": distance, "reached": reached, "align": align, "slip_deg": slip_deg}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
        self._renderer.update_scene(self.data, camera="tracking_top")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
