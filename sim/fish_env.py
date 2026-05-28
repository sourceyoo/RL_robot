"""물고기 swim-to-target 강화학습 환경 (3DOF planar 모델용).

qpos layout (5): [root_x, root_y, root_yaw, tail_joint, fin_joint]
qvel layout (5): [vx, vy, vyaw, vtail, vfin]
부호 규약: world -x = 머리 방향(전진).

관측 (13 + action_history_n·3 차원):
  0 tail_qpos       1 tail_qvel       2 fin_qpos        3 fin_qvel
  4 world vx        5 world vy        6 yaw_rate
  7 sin(yaw)        8 cos(yaw)
  9 target_rel_x    10 target_rel_y   (world frame)
  11 |yaw_err|      12 distance       (v32-C: derived obs 명시)
  13~ 최근 N step의 normalized action 이력 (freq, amp, offset × N step, flatten)

target 분포 (m4_v20): sub_distributions 인자로 mixed sampling 지원.
  list[dict] — 각 dict {"sub_id", "kind"("theta"|"offset"), "range"(mn,mx), "weight"}.
  reset마다 weight 비례 sub 선택 → 해당 분포에서 theta sample → info["sub_id"] 전달.
  callback이 sub별 reach_rate 측정해 sub trigger(모든 sub ≥ threshold일 때만 졸업).
  forgetting + 평균 inflated 동시 차단. None이면 기존 단일 분포 logic.

행동 (3차원, m4_cpg_v1): CPG (Central Pattern Generator) 3 요소.
  action = [freq_norm, amp_norm, offset_norm] ∈ [-1, +1]³
  매핑 (step 안 inline):
    freq   = 4 + (a0+1)/2 · 2   ∈ [4, 6] Hz   (fiberglass fin 전진 영역)
    amp    = (a1+1)/2           ∈ [0, 1]      (full range; hover cheat은 reward로 차단)
    offset = a2                  ∈ [-1, +1]    (= tail joint ±20° 평균 bias)
  ctrl 합성 (frame_skip 내부 매 mj_step):
    ctrl(t) = clip(amp · sin(2π · phase) + offset, -1, 1)
    phase ← (phase + freq · timestep) mod 1   (oscillator state 누적)
  의도: CPG 4요소 중 하드웨어에서 의미 있는 3요소 (frequency·amplitude·offset) 를
        모두 RL action 으로 노출. duty cycle 비대칭(sf_asym, m4_v17~v20) 은 offset 으로
        회전 ctrl 역할 대체되어 제거. phase 누적으로 freq 변경 시 ctrl 연속.

보상 (m4_cpg_v1, dt-normalized): step-cumulative 항에 dt 곱 (시간 적분 의미, fs 변경 robust).
  cumulative W 는 m4_v20 시간당 강도 정확 유지를 위해 모두 4x 보정 (= 1/dt_v20 = 1/0.25, 수학적 정확).
  progress·5.0 + reach·10                                                  # event 항 (dt 곱 X)
  + (align_weight·align + ALIGN_VEL_W·align_vel + YAW_SIGN_W·yaw_rate·sign(yaw_err)) · dt   (align·align_vel 둘 다 |v|>0.02 deadzone)
  − TIME_PEN_W·distance·current_time · dt
  − BACK_PEN_W·(−cos(head,v))·|v| · dt   (cos<0이고 |v|>0.05일 때만)
  − SMOOTH_W·||action_t − action_{t-1}||² · dt

  - align_weight: 10s ep 0.12, 그 외(20s·60s) 0.30 (m4_cpg_v2: m4_cpg_v1 의 0.08/0.20 에서 1.5x).
    align = cos(머리, 목표). ALIGN_VEL_W 동일 가중치였던 m4_cpg_v1 에서 sideslip mode 학습 →
    머리 정렬을 ALIGN_VEL 대비 1.5x 우선 → sideslip 억제 의도.
    m4_cpg_v3 s3c·d 의 amp→0 + yaw-only mode collapse 차단 (m4_cpg_v4): |v|>0.02 deadzone.
    align_vel·back_pen 과 동일 임계로 통일 (정지 시 dense 보상 모두 0 → "정렬+정지" trap 제거).
  - ALIGN_VEL_W=0.20 (m4_v20 0.05 × 4): align_vel = cos(v, 목표). fluid drift cancel 압력. deadzone v_norm>0.02.
  - YAW_SIGN_W=0.020 (m4_v20 0.005 × 4): 목표 방향 yaw rate incentive.
  - TIME_PEN_W=4e-5 (m4_v20 1e-5 × 4): distance·time 누적 페널티.
  - BACK_PEN_W=40.0 (m4_v20 5.0 × 4 × 2): 보정 + 사용자 강화 2x → 시간당 m4_v20 의 정확 2x.
    cos(head,v)<0(slip>90°)면 후진. cos·|v| 비례. deadzone |v|>0.05: tail wag·정지 면제.
    m4_v18c s3c·d에서 cycle 변조 저주파 후진 cheat 발견. memory feedback_no_backward.
  - SMOOTH_W=0.05 (m4_cpg_v1 신규): action 변화 페널티. ETH ANYmal 패턴.
    부드러운 정책 출력 유도 → CPG sine 깨짐 차단·motor jerk ↓·sim2real 격차 ↓.

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

import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

DEFAULT_XML = str(Path(__file__).parent / "rl_fish.xml")

# qpos/qvel 인덱스 (3DOF planar + tail + fin)
IDX_X, IDX_Y, IDX_YAW, IDX_TAIL, IDX_FIN = 0, 1, 2, 3, 4

# m4_cpg_v1: CPG action 3D 범위.
# freq: fiberglass fin (정착 조합) 전진 영역. 옛 Ecoflex 는 1~6Hz 전 영역 전진했으나 현 모델
# 은 0.5~3Hz 후진 (ellipsoid 인공물). 실모터 dry 5.88Hz spec 의 68~100% 영역.
# amp: full range — hover cheat (amp≈0) 위험은 reward time_pen·BACK_PEN 으로 간접 차단.
# offset: tail joint ctrlrange ±1 (= ±20°) 안 평균 bias.
FREQ_MIN, FREQ_MAX = 4.0, 6.0
AMP_MIN, AMP_MAX = 0.0, 1.0
OFFSET_MIN, OFFSET_MAX = -1.0, 1.0


class FishSwimEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        frame_skip: int = 42,
        episode_seconds: float = 10.0,
        target_radius: float = 0.5,
        success_radius: float = 0.08,
        target_theta_range: tuple[float, float] = (-np.pi, np.pi),
        target_theta_offset_range: tuple[float, float] | None = None,
        sub_distributions: list[dict] | None = None,
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
        # m4_v20: mixed sampling — sub_distributions = [{"sub_id", "kind"("theta"|"offset"), "range", "weight"}, ...]
        # 있으면 매 reset마다 weight에 따라 sub 선택 후 그 분포에서 target sample.
        # info["sub_id"]로 callback이 sub별 reach_rate 측정 (sub trigger).
        self.sub_distributions = sub_distributions
        self._current_sub_id: str | None = None
        if sub_distributions is not None:
            weights = np.array([s.get("weight", 1.0) for s in sub_distributions], dtype=np.float64)
            self._sub_probs = weights / weights.sum()
        else:
            self._sub_probs = None
        # m4_v14: align_weight 0.012 → 0.05 (회전 incentive 강화, annular 분포 학습용).
        # m4_cpg_v1: dt-norm 정확 보정 4x (= 1/dt_v20 = 1/0.25, m4_v20 시간당 강도 정확 유지).
        # 10s: 0.02 → 0.08. 그 외: 0.05 → 0.20.
        # m4_cpg_v2: sideslip 차단 — align_weight 1.5x (ALIGN_VEL_W 0.20 대비 머리 정렬 우선).
        # 10s: 0.08 → 0.12. 그 외: 0.20 → 0.30. critic 우려 (scale 불균형) 반영 — 2x 대신 1.5x.
        self.align_weight = 0.12 if episode_seconds <= 10.0 else 0.30
        self.action_history_n = max(0, int(action_history_n))
        # m4_cpg_v1: action 3D 라 history 도 (N, 3). obs 에는 flatten 해서 concat.
        self._action_history = np.zeros((self.action_history_n, 3), dtype=np.float32)
        self._cpg_phase = 0.0
        # m4_cpg_v1: action smoothness reward 계산용 (직전 step action).
        self._prev_action = np.zeros(3, dtype=np.float32)
        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None
        # view_policy.py가 launch_passive viewer를 여기에 붙이면, step 내부 mj_step
        # 루프마다 sync + realtime sleep을 하여 4Hz stroboscopic effect 없이 sine wave가 보임.
        # 학습 시엔 None이라 분기 안 탐.
        self._viewer = None

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

        obs_dim = 13 + self.action_history_n * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # m4_cpg_v1: action 3D = [freq_norm, amp_norm, offset_norm].
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
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
            return np.concatenate([base, self._action_history.flatten()])
        return base

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        rng = self.np_random
        if self.sub_distributions is not None:
            # m4_v20 mixed sampling: weight 비례로 sub 선택 후 그 분포에서 sample.
            idx = int(rng.choice(len(self.sub_distributions), p=self._sub_probs))
            sub = self.sub_distributions[idx]
            self._current_sub_id = sub["sub_id"]
            kind = sub["kind"]
            mn, mx = sub["range"]
            if kind == "theta":
                theta = rng.uniform(mn, mx) if mn != mx else mn
            elif kind == "offset":
                offset = rng.uniform(mn, mx) if mn != mx else mn
                sign = 1.0 if rng.uniform() < 0.5 else -1.0
                theta = np.pi + sign * offset
            else:
                raise ValueError(f"unknown sub kind: {kind!r} (expected 'theta' or 'offset')")
        elif self.target_theta_offset_range is not None:
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
            self._action_history = np.zeros((self.action_history_n, 3), dtype=np.float32)
        self._cpg_phase = 0.0
        self._prev_action = np.zeros(3, dtype=np.float32)
        return self._get_obs(), {}

    def _synth_ctrl(self, freq: float, amp: float, offset: float) -> float:
        """CPG: ctrl(t) = clip(amp·sin(2π·phase) + offset, -1, 1).
        phase 는 env state 로 누적 (oscillator continuity) — freq 가 step 간 변해도 ctrl 연속.
        """
        ctrl = amp * np.sin(2.0 * np.pi * self._cpg_phase) + offset
        self._cpg_phase = (self._cpg_phase + freq * self.model.opt.timestep) % 1.0
        return float(np.clip(ctrl, -1.0, 1.0))

    def step(self, action: np.ndarray):
        clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
        # m4_cpg_v1: action 3D → 물리 단위 매핑.
        freq = FREQ_MIN + (float(clipped[0]) + 1.0) * 0.5 * (FREQ_MAX - FREQ_MIN)
        amp = AMP_MIN + (float(clipped[1]) + 1.0) * 0.5 * (AMP_MAX - AMP_MIN)
        offset = float(clipped[2])
        # frame_skip 내부에서 매 mj_step마다 CPG ctrl 합성·phase 누적.
        timestep = self.model.opt.timestep
        for _ in range(self.frame_skip):
            self.data.ctrl[0] = self._synth_ctrl(freq, amp, offset)
            mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                self._viewer.sync()
                time.sleep(timestep)
        if self.action_history_n > 0:
            self._action_history[:-1] = self._action_history[1:]
            self._action_history[-1] = clipped

        distance = self._distance_to_target()
        progress = self._prev_distance - distance
        reached = distance < self.success_radius

        yaw = float(self.data.qpos[IDX_YAW])
        head_dir = np.array([-np.cos(yaw), np.sin(yaw)])
        rel = self._target_pos()[:2] - self._torso_pos()[:2]
        rel_norm = float(np.linalg.norm(rel))
        align = float(np.dot(head_dir, rel / rel_norm)) if rel_norm > 1e-6 else 0.0
        yaw_err_sign = float(np.sign(head_dir[0] * rel[1] - head_dir[1] * rel[0])) if rel_norm > 1e-6 else 0.0

        # m4_cpg_v1: cumulative W 모두 dt-norm 정확 보정 4x (= 1/dt_v20 = 1/0.25, 시간당 강도 정확 유지).
        YAW_SIGN_W = 0.020   # m4_v20 0.005 × 4
        ALIGN_VEL_W = 0.20   # m4_v20 0.05 × 4
        TIME_PEN_W = 4e-5    # m4_v20 1e-5 × 4
        BACK_PEN_W = 40.0    # m4_v20 5.0 × 4 × 2 (강화 2x 포함). 시간당 m4_v20 의 정확히 2x
        SMOOTH_W = 0.05      # m4_cpg_v1 신규 (action smoothness penalty)
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

        # m4_cpg_v1: action smoothness — 정책 출력 변화 페널티 (ETH ANYmal 패턴).
        # 3D action diff² 합. sine carrier 깨짐 차단·motor jerk ↓.
        action_diff_sq = float(np.sum((clipped - self._prev_action) ** 2))
        smooth_pen = SMOOTH_W * action_diff_sq

        # m4_cpg_v1: dt-normalized reward. fs 변경에 robust (시간 적분 의미).
        # progress·reach·time_pen 은 이미 dt 효과 내재.
        reward = (
            float(progress * 5.0)
            + (10.0 if reached else 0.0)
            + (self.align_weight * align * (1.0 if v_norm > 0.02 else 0.0)) * self.dt
            + (ALIGN_VEL_W * align_vel) * self.dt
            + (YAW_SIGN_W * yaw_rate * yaw_err_sign) * self.dt
            - TIME_PEN_W * distance * current_time * self.dt
            - back_pen * self.dt
            - smooth_pen * self.dt
        )

        # slip 측정 (reward 영향 X, eval sideslip 통계용). v_world·v_norm 위에서 계산.
        if v_norm > 0.02:
            cos_slip = float(np.clip(np.dot(head_dir, v_world / v_norm), -1.0, 1.0))
            slip_deg = float(np.degrees(np.arccos(cos_slip)))
        else:
            slip_deg = 0.0

        self._prev_distance = distance
        self._prev_action = clipped.copy()
        self._step_count += 1

        terminated = bool(reached)
        truncated = self._step_count >= self.max_steps

        info = {"distance": distance, "reached": reached, "align": align, "slip_deg": slip_deg}
        if self._current_sub_id is not None:
            info["sub_id"] = self._current_sub_id
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
