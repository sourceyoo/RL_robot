"""물고기 swim-to-target 강화학습 환경 (3DOF planar 모델용).

qpos layout (5): [root_x, root_y, root_yaw, tail_joint, fin_joint]
qvel layout (5): [vx, vy, vyaw, vtail, vfin]
부호 규약: world -x = 머리 방향(전진).

관측 (11 + action_history_n 차원):
  0 tail_qpos       1 tail_qvel       2 fin_qpos        3 fin_qvel
  4 world vx        5 world vy        6 yaw_rate
  7 sin(yaw)        8 cos(yaw)
  9 target_rel_x    10 target_rel_y   (world frame)
  11~ 최근 N step의 ctrl (action_history_n>0일 때)
       action history는 정책이 비대칭 wagging 패턴(time-asymmetric ctrl)을
       발견하는 데 필요. yaw_test.py 진단으로 비대칭 패턴이 yaw 회전의 핵심임 확인.

행동 (1차원): tail motor ctrl ∈ [-1, 1].

보상 (v14): progress·10 + reached_bonus - ctrl_cost + ALIGN_W·align.
  - v8 가산식 + v14 ALIGN_W 직접 fix: 10s ep는 0.02 그대로, 30s ep는 0.012 직접 박음.
    - s1·s2 (10s ep): 0.02 (align 누적 +5 ≈ reach +5, 1:1).
    - s3a~d (30s ep): 0.012 — align 누적 ≈ 1500×0.012×0.6 = +11 vs reach +5 (2.2:1 비율).
  - v12 catastrophic 진단: ep 30s에서 align 누적 +18 vs reach +5 = 3.6:1로 reach 신호 묻힘.
  - v13 (`0.02·10/episode_seconds` 비례)는 30s에서 0.0067 → 비율 1.2:1로 catastrophic 해결했으나
    정렬 신호 부족(align +0.19)으로 천장 미돌파. v12·v13 양 극단의 산술 중간 0.012가 적정값.
  - 곱셈 보상(v5/v7)은 mode collapse — 가산식 + 비율 튜닝이 정착.
  - align ∈ [-1, +1] (cos(머리, 목표)).
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


class FishSwimEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        frame_skip: int = 10,
        episode_seconds: float = 10.0,
        target_radius: float = 0.5,
        success_radius: float = 0.08,
        target_theta_range: tuple[float, float] = (-np.pi, np.pi),
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
        self.target_theta_range = target_theta_range  # curriculum용 목표 각도 범위
        # v14: align_weight 직접 fix. 10s ep는 0.02, 30s ep는 0.012.
        # v12 (0.020, 비율 3.6:1) → catastrophic / v13 (0.0067, 비율 1.2:1) → 정렬 부족.
        # 산술 중간 0.012 (비율 2.2:1)로 reach 신호 살리면서 정렬 학습도 유지.
        self.align_weight = 0.02 if episode_seconds <= 10.0 else 0.012
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

        # 모델 정합성 체크 — 5DOF planar 가정
        if self.model.nq != 5 or self.model.nv != 5:
            raise RuntimeError(
                f"기대 nq=nv=5 (3DOF planar + tail + fin), 실제 nq={self.model.nq} nv={self.model.nv}"
            )
        if self.model.nu != 1:
            raise RuntimeError(f"기대 nu=1 (단일 motor), 실제 nu={self.model.nu}")

        obs_dim = 11 + self.action_history_n
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
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
        base = np.array([
            qpos[IDX_TAIL], qvel[IDX_TAIL],
            qpos[IDX_FIN],  qvel[IDX_FIN],
            qvel[IDX_X],    qvel[IDX_Y],   qvel[IDX_YAW],
            np.sin(yaw),    np.cos(yaw),
            rel[0],         rel[1],
        ], dtype=np.float32)
        if self.action_history_n > 0:
            return np.concatenate([base, self._action_history])
        return base

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 목표를 매 에피소드마다 random하게 — target_theta_range 안에서 균등 분포.
        # curriculum 단계별로 좁혀가며 학습 가능 (예: (π,π) 직진만, (π/2, 3π/2) 머리쪽만).
        rng = self.np_random
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

    def step(self, action: np.ndarray):
        clipped = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = clipped
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        # action history shift (oldest 버리고 최신 추가)
        if self.action_history_n > 0:
            self._action_history[:-1] = self._action_history[1:]
            self._action_history[-1] = float(clipped[0])

        distance = self._distance_to_target()
        progress = self._prev_distance - distance
        ctrl_cost = 0.001 * float(np.square(self.data.ctrl).sum())
        reached = distance < self.success_radius

        # Yaw alignment: 머리 방향(yaw=0이면 world -x)이 목표를 얼마나 향하는가.
        # head_dir = R(-yaw)·(-1,0) = (-cos(yaw), sin(yaw))  (axis (0,0,-1) 보정 반영)
        yaw = float(self.data.qpos[IDX_YAW])
        head_dir = np.array([-np.cos(yaw), np.sin(yaw)])
        rel = self._target_pos()[:2] - self._torso_pos()[:2]
        rel_norm = float(np.linalg.norm(rel))
        align = float(np.dot(head_dir, rel / rel_norm)) if rel_norm > 1e-6 else 0.0

        # v15: reach 보너스 5 → 10. align_weight (10s 0.02 / 30s 0.012)는 v14 그대로.
        # v14의 align dominance(7.4:1) 완화 위해 reach 절대값만 2배 강화.
        # v21: yaw 변화 자체 보상 (`+YAW_W·|yaw_rate|`, 0.005) — 회전 시도 인센티브로
        # "정렬만 mode" 깨고 mode catalysis. v22: v21 + 1M fine-tune (peak 50%).
        # v23-A·v24-A·v25-A·v27·v28 모두 yaw reward 형태 변형 카드 reject — v22 |·|0.005
        # 가 sweet spot. v26 reproduce로 v22 32% baseline은 seed 운 입증 (end 26%,
        # v22~v26 평균 ~29% / ±6%p 분산).
        # v29: v22 카드 그대로 multi-seed (3 seed) 평가 — baseline 신뢰도 본격 확립.
        # yaw reward = +YAW_W·|yaw_rate| (v22 그대로, v25-A: 0.007 → 0.005 복귀).
        YAW_W = 0.005
        yaw_rate = float(self.data.qvel[IDX_YAW])
        reward = (
            float(progress * 10.0)
            + (10.0 if reached else 0.0)
            - ctrl_cost
            + self.align_weight * align
            + YAW_W * abs(yaw_rate)
        )

        self._prev_distance = distance
        self._step_count += 1

        terminated = bool(reached)
        truncated = self._step_count >= self.max_steps

        info = {"distance": distance, "reached": reached, "align": align}
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
