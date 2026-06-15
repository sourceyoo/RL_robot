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
  progress·5.0 + reach·10·reach_gate + PAR_W·max(head_forward,0)·max(align,0)·turn_ratio   # event/변위 항 (dt 곱 X)
    progress = dist_reduction · clip(cos_motion,0,1)^P  (m4_cpg_v9: 정렬-전진 게이팅. cos_motion=이동·머리/|이동|,
      게걸음=옆이동→cos_motion 작아 보상 급감, 순수선회≈1 full, 후퇴는 dist_red 그대로 차단. P=10 (v10, v9의 5에서 ↑))
    reach_gate = TURN_FLOOR + (1−TURN_FLOOR)·clip((yaw−yaw0)·turn_sign/offset, 0,1)  (m4_cpg_v11 B-1:
      도달 보너스 선회비율 게이팅. 게걸음(머리 안 돎)→FLOOR(0.4)만, 선회(머리 target쪽 회전)→full. s1(offset≈0)→full)
    PAR_W·max(head_forward,0)·max(align,0)·turn_ratio (m4_cpg_v15): parallel mode 직접 보상. 복합 게이트.
      계단식(head_forward≈0)→0, 게걸음(turn_ratio≈0)→0, 과회전(align↓)→감액, 정확→full. PAR_W=12.0.
      (v13 turn_ratio 단독=과회전 / v14 align 단독=작은각 게걸음 → 반대 약점이라 곱으로 보완)
  + (align_weight·align·turn_ratio + YAW_SIGN_W·yaw_rate·sign(yaw_err)) · dt   (align |v|>0.02 deadzone)
    (m4_cpg_v13: align 항도 turn_ratio 게이팅 — 게걸음(turn_ratio≈0)의 align 후원 제거, 선회 정렬은 보존)
  − TIME_PEN_W·distance·current_time · dt
  − BACK_PEN_W·(−cos(head,v))·|v| · dt   (cos<0이고 |v|>0.05일 때만)
  − SMOOTH_W·||action_t − action_{t-1}||² · dt
  − LAT_AVG_W·|v_lat_avg| · dt   (m4_cpg_v5: 3 step cycle 평균 sideslip, m4_cpg_v16: LAT_AVG_W=4.0)
      (m4_cpg_v15: turn_ratio 게이팅 제거=항상 켬. v12 게이팅은 turn_ratio↑서 꺼져 "돌며 옆미끄러짐" 허용 → 제거)

  - align_weight: 10s ep 0.12, 그 외(20s·60s) 0.30 (m4_cpg_v2: m4_cpg_v1 의 0.08/0.20 에서 1.5x).
    align = cos(머리, 목표).
    m4_cpg_v3 s3c·d 의 amp→0 + yaw-only mode collapse 차단 (m4_cpg_v4): |v|>0.02 deadzone.
    back_pen 과 동일 임계로 통일 (정지 시 dense 보상 모두 0 → "정렬+정지" trap 제거).
  - m4_cpg_v7: ALIGN_VEL_W(align_vel) 제거. progress min(head_forward) 가 머리방향 추진 역할 대체.
  - YAW_SIGN_W=0.020 (m4_v20 0.005 × 4): 목표 방향 yaw rate incentive.
  - TIME_PEN_W=4e-5 (m4_v20 1e-5 × 4): distance·time 누적 페널티.
  - BACK_PEN_W=40.0 (m4_v20 5.0 × 4 × 2): 보정 + 사용자 강화 2x → 시간당 m4_v20 의 정확 2x.
    cos(head,v)<0(slip>90°)면 후진. cos·|v| 비례. deadzone |v|>0.05: tail wag·정지 면제.
    m4_v18c s3c·d에서 cycle 변조 저주파 후진 cheat 발견. memory feedback_no_backward.
  - SMOOTH_W=0.05 (m4_cpg_v1 신규): action 변화 페널티. ETH ANYmal 패턴.
    부드러운 정책 출력 유도 → CPG sine 깨짐 차단·motor jerk ↓·sim2real 격차 ↓.
  - LAT_AVG_W=3.0 (m4_cpg_v5 신규): cycle-평균 |v_lat| 페널티. 3 step (≈1 CPG cycle) 평균이라
    propulsion side force (좌우 진동, 평균 0) 면제 / 진짜 sideslip (한쪽 미끄러짐) 만 cost.
    s1 propulsion 단독 시 pen·dt ≈ 7.6e-4 (progress 5e-3 대비 미미), s3c sideslip 시 ≈ 7.6e-3.
  - turn_ratio = clip((yaw−yaw0)·turn_sign/offset, 0,1) (m4_cpg_v11 B-1): 머리가 target쪽으로 돈 비율.
    reach_gate(종단, FLOOR=0.4)·lat_pen 게이트(과정, 1−turn_ratio) 공용 신호. 게걸음(0)·선회(1) 직접 구분.
    m4_cpg_v12: v8 ALIGN_GATE 폐기 — align은 작은 각(s3b 30°)서 게걸음도 높아 식별 실패(게걸음 0.69 > 선회 0.40).

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

# m4_cpg_v5: cycle-averaged lateral velocity penalty 용 buffer window.
# 5Hz CPG → 0.2s/cycle → dt=0.084s → ~2.4 step → 3 step window (= 0.252s ≈ 1 cycle).
# propulsion side force (좌우 ±진동) 는 cycle 평균 ≈ 0 → 자동 면제.
# 진짜 sideslip (한쪽 미끄러짐) 은 cycle 평균 유지 → 페널티.
CYCLE_STEPS = 3


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
        # m4_cpg_v24: sub_distributions에 sub별 "radius" 있으면 reset마다 override, 없으면 base 복원.
        self._base_target_radius = target_radius
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
        # m4_cpg_v2: sideslip 차단 — align_weight 1.5x (머리 정렬 우선). (ALIGN_VEL_W 는 m4_cpg_v7 에서 제거)
        # 10s: 0.08 → 0.12. 그 외: 0.20 → 0.30. critic 우려 (scale 불균형) 반영 — 2x 대신 1.5x.
        self.align_weight = 0.12 if episode_seconds <= 10.0 else 0.30
        self.action_history_n = max(0, int(action_history_n))
        # m4_cpg_v1: action 3D 라 history 도 (N, 3). obs 에는 flatten 해서 concat.
        self._action_history = np.zeros((self.action_history_n, 3), dtype=np.float32)
        self._cpg_phase = 0.0
        # m4_cpg_v1: action smoothness reward 계산용 (직전 step action).
        self._prev_action = np.zeros(3, dtype=np.float32)
        # m4_cpg_v5: cycle-평균 lateral velocity buffer (sideslip 페널티용).
        self._v_lat_buffer = np.zeros(CYCLE_STEPS, dtype=np.float32)
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
        self._prev_pos = np.zeros(2)
        # m4_cpg_v11 (B-1): reach 선회비율 게이팅용 — ep 시작 yaw·target offset·target쪽 회전부호.
        self._yaw0 = 0.0
        self._target_offset = 0.0
        self._turn_sign = 0.0

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
        head_dir = np.array([-np.cos(yaw), -np.sin(yaw)])
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
            # m4_cpg_v24: sub별 거리 (s3c sub만 0.7m). radius 없는 sub는 base(0.5) 유지.
            self.target_radius = sub.get("radius", self._base_target_radius)
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
        self._prev_pos = self._torso_pos()[:2].copy()
        if self.action_history_n > 0:
            self._action_history = np.zeros((self.action_history_n, 3), dtype=np.float32)
        self._cpg_phase = 0.0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._v_lat_buffer[:] = 0.0
        # m4_cpg_v21: ep 동안 course(진행·target)·머리각 누적 → 종료 시 strict 졸업 판정용.
        self._course_buf = []
        self._headang_buf = []
        # m4_cpg_v11 (B-1): ep 시작 머리방향 기준 target offset·회전부호 고정 (reach 게이팅용).
        yaw0 = float(self.data.qpos[IDX_YAW])
        head_dir0 = np.array([-np.cos(yaw0), -np.sin(yaw0)])
        rel0 = self._target_pos()[:2] - self._torso_pos()[:2]
        rel0_norm = float(np.linalg.norm(rel0))
        self._yaw0 = yaw0
        if rel0_norm > 1e-6:
            align0 = float(np.dot(head_dir0, rel0 / rel0_norm))
            self._target_offset = float(np.arccos(np.clip(align0, -1.0, 1.0)))
            # head_dir=[-cos,-sin](실제 mesh 코) 매핑상 cross 부호가 곧 "target쪽 yaw 변화>0" 부호. 직접 사용.
            self._turn_sign = float(np.sign(head_dir0[0] * rel0[1] - head_dir0[1] * rel0[0]))
        else:
            self._target_offset = 0.0
            self._turn_sign = 0.0
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

        torso_xy = self._torso_pos()[:2]
        distance = self._distance_to_target()
        reached = distance < self.success_radius

        yaw = float(self.data.qpos[IDX_YAW])
        head_dir = np.array([-np.cos(yaw), -np.sin(yaw)])
        rel = self._target_pos()[:2] - torso_xy
        rel_norm = float(np.linalg.norm(rel))
        align = float(np.dot(head_dir, rel / rel_norm)) if rel_norm > 1e-6 else 0.0
        yaw_err_sign = float(np.sign(head_dir[0] * rel[1] - head_dir[1] * rel[0])) if rel_norm > 1e-6 else 0.0

        # m4_cpg_v9: 정렬-전진 게이팅. cos_motion = 이동방향이 머리방향과 정렬된 정도(이동·머리/|이동|).
        # 전진(dist_red>0)은 cos_motion^P 가중 → 게걸음(옆이동=cos_motion 작음) 보상 급감, 순수선회(≈1) full.
        # 호버링(delta≈0→dist_red≈0)은 progress≈0 이라 trap 없음(페널티 아님). 후퇴(dist_red<0)는 gate 안 곱해 그대로 차단.
        # v7 min 은 s3b(cos30°=0.87) 둔감해 게걸음 13%만 깎였음 → P 거듭제곱으로 날카롭게.
        # m4_cpg_v10: P=5(v9) 에서 게걸음이 cos_motion 0.917 로 게이트 회피(선회비율 0.26 정체) → P↑로 분리 강화.
        #   P=10: 게걸음 0.917^10=0.42 (보상 급감) vs 순수선회 0.99^10=0.90 (유지). 물리 한계면 도달률↓로 식별.
        PROGRESS_P = 10
        delta = torso_xy - self._prev_pos
        delta_norm = float(np.linalg.norm(delta))
        head_forward = float(np.dot(delta, head_dir))
        cos_motion = head_forward / (delta_norm + 1e-8)
        dist_reduction = self._prev_distance - distance
        motion_gate = float(np.clip(cos_motion, 0.0, 1.0)) ** PROGRESS_P
        progress = dist_reduction * motion_gate if dist_reduction > 0 else dist_reduction

        # m4_cpg_v1: cumulative W 모두 dt-norm 정확 보정 4x (= 1/dt_v20 = 1/0.25, 시간당 강도 정확 유지).
        YAW_SIGN_W = 0.020   # m4_v20 0.005 × 4
        TIME_PEN_W = 4e-5    # m4_v20 1e-5 × 4
        BACK_PEN_W = 40.0    # m4_v20 5.0 × 4 × 2 (강화 2x 포함). 시간당 m4_v20 의 정확히 2x
        SMOOTH_W = 0.05      # m4_cpg_v1 신규 (action smoothness penalty)
        yaw_rate = float(self.data.qvel[IDX_YAW])
        current_time = self._step_count * self.dt

        # m4_cpg_v7: align_vel(head-velocity 정렬) 제거 — progress min(head전진) 이 머리방향 추진 역할 대체.
        # v_world/v_norm 은 back_pen·lat_pen·slip 통계가 사용하므로 유지.
        v_world = np.array([float(self.data.qvel[IDX_X]), float(self.data.qvel[IDX_Y])])
        v_norm = float(np.linalg.norm(v_world))

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

        # m4_cpg_v11 (B-1): 머리가 target쪽으로 돈 비율 turn_ratio — reach·lat_pen 공용 게이트.
        # (yaw−yaw0)·turn_sign / offset. 게걸음(yaw≈yaw0)→0, 선회(yaw≈offset)→1. clip[0,1](역/과회전 차단).
        # head_dir=[-cos,-sin](실제 mesh 코). turn_sign은 cross 직접 사용(reset). offset≈0(s1 직진)→1. head wag 1.3°라 안정.
        if self._target_offset > 1e-3:
            turn_ratio = float(np.clip((yaw - self._yaw0) * self._turn_sign / self._target_offset, 0.0, 1.0))
        else:
            turn_ratio = 1.0
        # reach 게이팅(B-1): 게걸음 도달 감액·선회 full. FLOOR=0.4 → 게걸음도 4.0>호버링(0). reach 도달시만.
        TURN_FLOOR = 0.4
        reach_gate = TURN_FLOOR + (1.0 - TURN_FLOOR) * turn_ratio

        # m4_cpg_v15: parallel mode 직접 보상 — 복합 게이트 head_forward·align·turn_ratio.
        # 계단식(head_forward≈0)→0, 게걸음(turn_ratio≈0)→0, 과회전(align↓)→감액, 정확(셋 다 1)→full.
        # turn_ratio(v13)·align(v14)는 반대 약점(과회전↔작은각 게걸음) → 곱으로 상호 보완. landscape: 게걸음 par 94%↓.
        PAR_W = 12.0
        par_reward = PAR_W * max(head_forward, 0.0) * max(align, 0.0) * turn_ratio

        # m4_cpg_v5: cycle-평균 lateral velocity = 진짜 sideslip. propulsion 좌우진동 평균≈0 면제, 한쪽 미끄러짐 유지.
        # m4_cpg_v15: turn_ratio 게이팅 제거(항상 켬). v12 게이팅은 turn_ratio↑서 lat_pen 꺼져
        # "머리 돌리며 옆 미끄러짐"(s3b sideslip 35°) 허용 → 제거. cycle평균이라 꼬리질 진동은 여전히 면제(순 sideslip만 벌).
        # m4_cpg_v16: LAT_AVG_W 6→4. v15(6)은 s3a parallel 완성했으나 큰 각(s3b) 선회 과벌→과소회전·도달률 60%. 강도 완화.
        # 2026-06-08 v15 롤백: v16~v20(lat4·stop_pen·ent_floor 상향) 모두 좌회전 미해결 → v15(LAT_AVG_W=6)로 환원.
        LAT_AVG_W = 6.0
        perp = np.array([-head_dir[1], head_dir[0]])
        v_lat_now = float(np.dot(v_world, perp))
        self._v_lat_buffer[:-1] = self._v_lat_buffer[1:]
        self._v_lat_buffer[-1] = v_lat_now
        v_lat_avg = float(np.mean(self._v_lat_buffer))
        lat_pen = LAT_AVG_W * abs(v_lat_avg)

        # m4_cpg_v1: dt-normalized reward. fs 변경에 robust (시간 적분 의미).
        # progress·reach·time_pen 은 이미 dt 효과 내재.
        reward = (
            float(progress * 5.0)
            + (10.0 * reach_gate if reached else 0.0)
            + par_reward
            + (self.align_weight * align * turn_ratio * (1.0 if v_norm > 0.02 else 0.0)) * self.dt
            + (YAW_SIGN_W * yaw_rate * yaw_err_sign) * self.dt
            - TIME_PEN_W * distance * current_time * self.dt
            - back_pen * self.dt
            - smooth_pen * self.dt
            - lat_pen * self.dt
        )

        # slip 측정 (reward 영향 X, eval sideslip 통계용). v_world·v_norm 위에서 계산.
        if v_norm > 0.02:
            cos_slip = float(np.clip(np.dot(head_dir, v_world / v_norm), -1.0, 1.0))
            slip_deg = float(np.degrees(np.arccos(cos_slip)))
            # m4_cpg_v21: 졸업 strict 판정용 course(진행·target cos)·머리각(=slip_deg) 누적. reward 영향 X.
            if rel_norm > 1e-6:
                self._course_buf.append(float(np.dot(v_world / v_norm, rel / rel_norm)))
                self._headang_buf.append(slip_deg)
        else:
            slip_deg = 0.0

        self._prev_distance = distance
        self._prev_pos = torso_xy
        self._prev_action = clipped.copy()
        self._step_count += 1

        terminated = bool(reached)
        truncated = self._step_count >= self.max_steps

        # m4_cpg_v21: ep 평균 course·머리각으로 strict 성공 판정 (졸업 게이트용 — reached/reward/termination 불변).
        # 게걸음(course 낮음·머리각 큼) 도달을 졸업에서 거름. 캘리브레이션(v15 우):
        # course≥0.70 (우 s3b 0.80 통과·좌 게걸음 0.52 탈락), 머리각≤40° (우 s3b 30±4° 통과·게걸음 49° 탈락).
        course_avg = float(np.mean(self._course_buf)) if self._course_buf else 0.0
        head_angle_avg = float(np.mean(self._headang_buf)) if self._headang_buf else 90.0
        success_strict = bool(reached and course_avg >= 0.70 and head_angle_avg <= 40.0)
        info = {"distance": distance, "reached": reached, "align": align, "slip_deg": slip_deg,
                "course_avg": course_avg, "head_angle_avg": head_angle_avg,
                "success_strict": success_strict}
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
