"""
Multi-agent + static obstacles (cones) MetaDrive script + FAILURE METRICS LOGGING

Adds:
- Per-step logging for ego + a chosen "lead" vehicle (defaults to first lane_offset==0 vehicle)
- Computes failure-centric metrics (no-crash metrics):
    * min_TTC, min_THW, min_gap_m
    * min_accel (most negative), frac_hard_brake
    * max_abs_jerk, frac_high_jerk
    * frac_stopped, max_stop_streak_s
    * osc_index (std(v)/mean(v))
    * num_panic_brakes (count of events a<-4 while v>10)
- Stores results in `metrics_dict` (printed at end; you can also save to json)

UPDATED LOGGING (only metrics/logging changed):
- Logs collisions from info: crash / crash_vehicle / crash_object (best-effort across builds)
- Adds episode-level collision summary in metrics_dict:
    * any_collision, collision_step, collision_vehicle, collision_object
- Adds min_euclid_dist_to_any_object per-step and episode min:
    * min_euclid_dist_m
  (helps reconcile "collision happened but headway metrics look fine")

NOTE:
- Uses lane coordinates for headway/gap on curves: ego_lane.local_coordinates()
- TTC uses along-lane headway and relative speed in lane direction
"""

import math
import json
import os
import numpy as np
import imageio.v2 as imageio
from gymnasium import spaces

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.vehicle.base_vehicle import BaseVehicle

try:
    from metadrive.component.vehicle.vehicle_type import DefaultVehicle
except Exception:
    DefaultVehicle = BaseVehicle

from metadrive.utils.doc_utils import generate_gif

# PID import (your build had this path)
try:
    from metadrive.component.vehicle.PID_controller import PIDController
except Exception:
    from metadrive.utils.pid import PIDController


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ----------------- Utils -----------------
def _clip(x, lo=-1.0, hi=1.0):
    return float(np.clip(x, lo, hi))


def wrap_to_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


def save_snapshots(frames, snapshot_steps, out_dir="snapshots", prefix="scenario"):
    os.makedirs(out_dir, exist_ok=True)
    H, W = frames[0].shape[:2]
    saved = []
    for t in sorted(set(snapshot_steps)):
        if 0 <= t < len(frames):
            img = frames[t]
            path = os.path.join(out_dir, f"{prefix}_t{t:04d}.png")
            imageio.imwrite(path, img)
            saved.append(path)
    return saved

# ----------------- Failure metrics helpers -----------------
def compute_failure_metrics(step_log, dt=0.1,
                            hard_brake_thresh=-3.0,
                            high_jerk_thresh=8.0,
                            stopped_speed_thresh=0.5,
                            panic_brake_accel=-4.0,
                            panic_min_speed=10.0):
    """
    step_log: dict of arrays/lists containing:
      - ego_speed, lead_speed
      - headway_s (lead_s - ego_s) in meters along ego lane (positive means lead ahead)
      - crash, crash_vehicle, crash_object (optional)
      - min_euclid_dist_to_any_object (optional)
    """
    eps = 1e-6
    ego_v = np.asarray(step_log["ego_speed"], dtype=np.float32)
    lead_v = np.asarray(step_log["lead_speed"], dtype=np.float32)
    d = np.asarray(step_log["headway_s"], dtype=np.float32)  # meters along lane

    # Only consider when lead is actually ahead and distance is finite
    valid = np.isfinite(d) & (d > 0)

    # TTC
    rel_v = ego_v - lead_v
    ttc = np.full_like(d, np.inf, dtype=np.float32)
    mask_ttc = valid & (rel_v > 0)
    ttc[mask_ttc] = d[mask_ttc] / np.maximum(rel_v[mask_ttc], eps)
    min_ttc = float(np.min(ttc)) if np.any(np.isfinite(ttc)) else float("inf")

    # THW
    thw = np.full_like(d, np.inf, dtype=np.float32)
    mask_thw = valid & (ego_v > 0)
    thw[mask_thw] = d[mask_thw] / np.maximum(ego_v[mask_thw], eps)
    min_thw = float(np.min(thw)) if np.any(np.isfinite(thw)) else float("inf")

    # Minimum gap (along-lane)
    min_gap_m = float(np.min(d[valid])) if np.any(valid) else float("inf")
    min_time = step_log["time_min_dist"]
    # Acceleration & jerk from ego speed
    a = np.zeros_like(ego_v, dtype=np.float32)
    if len(ego_v) >= 2:
        a[1:] = (ego_v[1:] - ego_v[:-1]) / float(dt)
    j = np.zeros_like(ego_v, dtype=np.float32)
    if len(ego_v) >= 3:
        j[2:] = (a[2:] - a[1:-1]) / float(dt)

    min_accel = float(np.min(a)) if len(a) else float("nan")
    frac_hard_brake = float(np.mean(a < hard_brake_thresh)) if len(a) else 0.0

    max_abs_jerk = float(np.max(np.abs(j))) if len(j) else float("nan")
    frac_high_jerk = float(np.mean(np.abs(j) > high_jerk_thresh)) if len(j) else 0.0

    # Stuck metrics
    stopped = ego_v < stopped_speed_thresh
    frac_stopped = float(np.mean(stopped)) if len(stopped) else 0.0
    max_streak = 0
    cur = 0
    for s in stopped:
        if s:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    max_stop_streak_s = float(max_streak * dt)

    # Oscillation index
    mean_v = float(np.mean(ego_v)) if len(ego_v) else 0.0
    std_v = float(np.std(ego_v)) if len(ego_v) else 0.0
    osc_index = float(std_v / max(mean_v, eps)) if mean_v > 0 else float("nan")

    # Panic brakes: count transitions into panic condition
    panic = (a < panic_brake_accel) & (ego_v > panic_min_speed)
    num_panic_brakes = 0
    if len(panic) >= 2:
        num_panic_brakes = int(np.sum((panic[1:] == True) & (panic[:-1] == False)))

    # Collision summary (from info-derived flags)
    crash_arr = np.asarray(step_log.get("crash", []), dtype=bool)
    crash_vehicle_arr = np.asarray(step_log.get("crash_vehicle", []), dtype=bool)
    crash_object_arr = np.asarray(step_log.get("crash_object", []), dtype=bool)

    any_collision = bool(np.any(crash_arr)) if len(crash_arr) else False
    collision_step = int(np.argmax(crash_arr)) if any_collision else None
    collision_vehicle = bool(np.any(crash_vehicle_arr)) if len(crash_vehicle_arr) else False
    collision_object = bool(np.any(crash_object_arr)) if len(crash_object_arr) else False

    # Min euclidean distance to ANY object (helps interpret lateral/other-object collisions)
    min_euclid_dist_m = float("inf")
    if "min_euclid_dist_to_any_object" in step_log:
        arr = np.asarray(step_log["min_euclid_dist_to_any_object"], dtype=np.float32)
        if arr.size > 0 and np.any(np.isfinite(arr)):
            min_euclid_dist_m = float(np.min(arr[np.isfinite(arr)]))

    return dict(
        min_ttc_s=min_ttc,
        min_thw_s=min_thw,
        min_gap_m=min_gap_m,
        min_accel_mps2=min_accel,
        frac_hard_brake=frac_hard_brake,
        max_abs_jerk_mps3=max_abs_jerk,
        frac_high_jerk=frac_high_jerk,
        frac_stopped=frac_stopped,
        max_stop_streak_s=max_stop_streak_s,
        osc_index=osc_index,
        num_panic_brakes=num_panic_brakes,
        # collision metrics
        any_collision=any_collision,
        collision_step=collision_step,
        collision_vehicle=collision_vehicle,
        collision_object=collision_object,
        # geometric sanity metric
        min_euclid_dist_m=min_euclid_dist_m,
        min_time_dist = min_time,
        n_steps=int(len(ego_v)),
        dt=float(dt),
    )


# ----------------- Steering mixin (lane-follow) -----------------
class _LaneFollowSteeringMixin:
    """Lane-follow steering that works on curves using navigation.current_lane + PID."""
    def _init_lane_follow(self, lookahead_s=6.0):
        self.lookahead_s = float(lookahead_s)
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.35, 0.002, 0.06)

    def _steer_lane_follow(self):
        nav = getattr(self.vehicle, "navigation", None)
        if nav is None or getattr(nav, "current_lane", None) is None:
            return 0.0
        lane = nav.current_lane
        s, t = lane.local_coordinates(self.vehicle.position)
        lane_heading = lane.heading_theta_at(s + self.lookahead_s)
        v_heading = float(getattr(self.vehicle, "heading_theta", 0.0))

        steer = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steer += self.lateral_pid.get_result(-t)
        return _clip(steer)


# ----------------- Policies -----------------

class EgoLaneFollowSpeedPolicy(BasePolicy, _LaneFollowSteeringMixin):
    """
    Lane-follow + speed hold, BUT:
      - Detects "obstacles" near the ego vehicle (cones/static objects/vehicles) in a forward corridor
      - Applies braking with a reaction-time delay (human-ish latency)
      - Optionally holds brake for a short minimum duration once triggered

    Notes:
      - This is best-effort across MetaDrive versions. It tries multiple ways to enumerate nearby objects.
      - You can tune the corridor geometry + thresholds without changing the rest of your script.
    """
    def __init__(
        self,
        control_object,
        random_seed=None,
        target_speed=35.0,
        speed_kp=0.25,
        lookahead_s=6.0,

        # --- obstacle braking parameters ---
        reaction_time_s=0.1,      # delay before braking after hazard is first detected
        dt=0.1,                   # assumed policy tick (set to env.engine.dt if you can)
        corridor_ahead_m=18.0,    # how far ahead to look
        corridor_half_width_m=2.0,# lateral half-width of corridor (meters)
        brake_accel=-1.0,         # accel command when braking (clips to [-1,1])
        brake_hold_s=0.6,         # minimum time to keep braking once triggered
        ttc_threshold_s=1.2,      # trigger if TTC falls below this
        min_detect_speed=1.0,     # ignore hazard braking when nearly stopped
        ignore_object_ids=None,   # optional set of ids to ignore
    ):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self.vehicle = control_object
        self.target_speed = float(target_speed)
        self.speed_kp = float(speed_kp)
        self._init_lane_follow(lookahead_s=lookahead_s)

        self.dt = float(dt)
        self.reaction_steps = int(max(0, round(float(reaction_time_s) / self.dt)))
        self.corridor_ahead_m = float(corridor_ahead_m)
        self.corridor_half_width_m = float(corridor_half_width_m)
        self.brake_accel = float(brake_accel)
        self.brake_hold_steps = int(max(0, round(float(brake_hold_s) / self.dt)))
        self.ttc_threshold_s = float(ttc_threshold_s)
        self.min_detect_speed = float(min_detect_speed)
        self.ignore_object_ids = set(ignore_object_ids or [])

        self._hazard_counter = 0
        self._brake_hold_counter = 0

    def reset(self):
        self._hazard_counter = 0
        self._brake_hold_counter = 0

    def _iter_world_objects(self):
        eng = getattr(self.vehicle, "engine", None)
        if eng is None:
            eng = getattr(getattr(self.vehicle, "env", None), "engine", None)
        if eng is None:
            return []

        objs = []
        try:
            for mgr_name in ("multi_agent_scenario_manager", "scripted_npc_manager", "traffic_manager"):
                mgr = getattr(eng, mgr_name, None)
                if mgr is None:
                    continue
                for o in getattr(mgr, "spawned_objects", {}).values():
                    objs.append(o)
        except Exception:
            pass

        try:
            am = getattr(eng, "agent_manager", None)
            for a in getattr(am, "active_agents", {}).values():
                objs.append(a)
        except Exception:
            pass

        try:
            tm = getattr(eng, "traffic_manager", None)
            for attr in ("vehicles", "traffic_vehicles", "spawned_objects"):
                vv = getattr(tm, attr, None)
                if isinstance(vv, dict):
                    objs.extend(list(vv.values()))
                elif isinstance(vv, (list, tuple)):
                    objs.extend(list(vv))
        except Exception:
            pass

        uniq = []
        seen = set()
        for o in objs:
            if o is None:
                continue
            if id(o) in seen:
                continue
            seen.add(id(o))
            uniq.append(o)
        return uniq

    def _hazard_in_corridor(self):
        nav = getattr(self.vehicle, "navigation", None)
        lane = getattr(nav, "current_lane", None) if nav is not None else None
        if lane is None:
            return False, float("inf"), float("inf")

        ego_pos = np.asarray(getattr(self.vehicle, "position", [0.0, 0.0]), dtype=np.float32)
        ego_s, ego_t = lane.local_coordinates(ego_pos)
        ego_v = float(getattr(self.vehicle, "speed", 0.0))

        if ego_v < self.min_detect_speed:
            return False, float("inf"), float("inf")

        min_ttc = float("inf")
        min_headway = float("inf")
        hazard = False

        for obj in self._iter_world_objects():
            if obj is self.vehicle:
                continue
            oid = getattr(obj, "id", None)
            if oid is not None and oid in self.ignore_object_ids:
                continue

            try:
                pos = np.asarray(getattr(obj, "position", None), dtype=np.float32)
                if pos is None or pos.shape[0] < 2:
                    continue
            except Exception:
                continue

            try:
                s, t = lane.local_coordinates(pos)
            except Exception:
                continue

            headway = float(s - ego_s)
            lat = float(t)

            if headway <= 0.0:
                continue
            if headway > self.corridor_ahead_m:
                continue
            if abs(lat) > self.corridor_half_width_m:
                continue

            obj_v = float(getattr(obj, "speed", 0.0))
            rel_v = ego_v - obj_v

            if rel_v > 0.1:
                ttc = headway / max(rel_v, 1e-6)
                if ttc < min_ttc:
                    min_ttc = ttc

            if headway < min_headway:
                min_headway = headway

            if (min_ttc < self.ttc_threshold_s) or (headway < 6.0):
                hazard = True

        return hazard, min_ttc, min_headway

    def act(self, *args, **kwargs):
        try:
            eng = getattr(self.vehicle, "engine", None)
            if eng is not None and hasattr(eng, "dt"):
                self.dt = float(eng.dt)
        except Exception:
            pass

        steer = self._steer_lane_follow()
        v = float(getattr(self.vehicle, "speed", 0.0))
        accel_cmd = self.speed_kp * (self.target_speed - v)

        if self._brake_hold_counter > 0:
            self._brake_hold_counter -= 1
            return np.array([_clip(steer), _clip(self.brake_accel)], dtype=np.float32)

        hazard, min_ttc, min_headway = self._hazard_in_corridor()

        if hazard:
            self._hazard_counter += 1
        else:
            self._hazard_counter = 0

        if self._hazard_counter >= self.reaction_steps and self.reaction_steps > 0:
            self._hazard_counter = 0
            self._brake_hold_counter = max(1, self.brake_hold_steps)
            return np.array([_clip(steer), _clip(self.brake_accel)], dtype=np.float32)

        if hazard and self.reaction_steps == 0:
            self._brake_hold_counter = max(1, self.brake_hold_steps)
            return np.array([_clip(steer), _clip(self.brake_accel)], dtype=np.float32)

        return np.array([_clip(steer), _clip(accel_cmd)], dtype=np.float32)

    @staticmethod
    def get_input_space():
        return spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

    @staticmethod
    def get_action_space():
        return spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)


class LaneFollowSuddenBrakePolicy(BasePolicy, _LaneFollowSteeringMixin):
    def __init__(
        self, control_object, random_seed=None,
        target_speed=15.0,
        brake_step=80, brake_steps=200, brake_mag=1.0,
        speed_kp=0.10,
        lookahead_s=6.0,
    ):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self.vehicle = control_object
        self.target_speed = float(target_speed)
        self.brake_step = int(brake_step)
        self.brake_steps = int(brake_steps)
        self.brake_mag = float(brake_mag)
        self.speed_kp = float(speed_kp)
        self.t = 0
        self._init_lane_follow(lookahead_s=lookahead_s)

    def reset(self):
        self.t = 0

    def act(self, *args, **kwargs):
        self.t += 1
        steer = self._steer_lane_follow()

        if self.brake_step <= self.t < (self.brake_step + self.brake_steps):
            accel = -abs(self.brake_mag)
        else:
            v = float(getattr(self.vehicle, "speed", 0.0))
            accel = self.speed_kp * (self.target_speed - v)

        return np.array([_clip(steer), _clip(accel)], dtype=np.float32)

    @staticmethod
    def get_input_space():
        return spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

    @staticmethod
    def get_action_space():
        return spaces.Box(low=np.array([-1.0, -1.0]),
                          high=np.array([1.0, 1.0]),
                          dtype=np.float32)


class LaneFollowAccelThenBrakePolicy(BasePolicy, _LaneFollowSteeringMixin):
    def __init__(
        self, control_object, random_seed=None,
        accel_mag=0.7, accel_steps=50,
        cruise_speed=22.0,
        brake_step=120, brake_steps=35, brake_mag=1.0,
        speed_kp=0.14,
        lookahead_s=6.0,
    ):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self.vehicle = control_object
        self.accel_mag = float(accel_mag)
        self.accel_steps = int(accel_steps)
        self.cruise_speed = float(cruise_speed)
        self.brake_step = int(brake_step)
        self.brake_steps = int(brake_steps)
        self.brake_mag = float(brake_mag)
        self.speed_kp = float(speed_kp)
        self.t = 0
        self._init_lane_follow(lookahead_s=lookahead_s)

    def reset(self):
        self.t = 0

    def act(self, *args, **kwargs):
        self.t += 1
        steer = self._steer_lane_follow()

        if self.brake_step <= self.t < (self.brake_step + self.brake_steps):
            accel = -abs(self.brake_mag)
        elif self.t < self.accel_steps:
            accel = abs(self.accel_mag)
        else:
            v = float(getattr(self.vehicle, "speed", 0.0))
            accel = self.speed_kp * (self.cruise_speed - v)

        return np.array([_clip(steer), _clip(accel)], dtype=np.float32)

    @staticmethod
    def get_input_space():
        return spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

    @staticmethod
    def get_action_space():
        return spaces.Box(low=np.array([-1.0, -1.0]),
                          high=np.array([1.0, 1.0]),
                          dtype=np.float32)


class IDMNominalPolicy(BasePolicy):
    def __init__(self, control_object, random_seed=None, target_speed_mps=20.0):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self.vehicle = control_object
        try:
            self.idm = IDMPolicy(control_object, random_seed)
        except TypeError:
            self.idm = IDMPolicy(control_object, random_seed, None)

        if hasattr(self.idm, "target_speed"):
            try:
                self.idm.target_speed = float(target_speed_mps)
            except Exception:
                pass

    def reset(self):
        if hasattr(self.idm, "reset"):
            self.idm.reset()

    def act(self, *args, **kwargs):
        try:
            a = self.idm.act()
        except TypeError:
            a = np.array([0.0, 0.2], dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        if a.shape[0] >= 2:
            return np.array([_clip(a[0]), _clip(a[1])], dtype=np.float32)
        return np.array([0.0, 0.2], dtype=np.float32)

    @staticmethod
    def get_input_space():
        return spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32)

    @staticmethod
    def get_action_space():
        return spaces.Box(low=np.array([-1.0, -1.0]),
                          high=np.array([1.0, 1.0]),
                          dtype=np.float32)


# ----------------- Config you edit -----------------
CFG = {'map': 'CCCCCC', 'traffic_density': 0.22, 'accident_prob': 0.0, 'steps': 360,
       'render': {'screen_size': [700, 900], 'camera_position': [60, -54]},
       'vehicles': [
           {'id': 'lead_passenger_car_hard_brake', 'lane_offset': 0, 'delta_s_m': 12.0, 'delta_t_m': 0.0,
            'direction': 'same', 'policy': 'sudden_brake',
            'params': {'target_speed': 18.0, 'brake_step': 65, 'brake_steps': 120, 'brake_mag': 1.0,
                       'speed_kp': 0.08, 'lookahead_s': 6.0}},
           {'id': 'adjacent_lane_blocker', 'lane_offset': 1, 'delta_s_m': 10.0, 'delta_t_m': 0.0,
            'direction': 'same', 'policy': 'idm_nominal', 'params': {'target_speed_mps': 17.0}}
       ],
       'cones': [{'enabled': False, 'lane_offset': 0, 'start_delta_s_m': 0.0, 'spacing_m': 2.5,
                  'count': 0, 'lateral_m': 0.0}]}


# ----------------- Multi-agent manager -----------------
class MultiAgentScenarioManager(BaseManager):
    def __init__(self,CFG,*args, **kwargs):
        self.CFG = CFG
        super().__init__(*args, **kwargs)
    
    def reset(self):
        super().reset()
        self._spawned = False
        self.agent_ids = []
        self.id_to_spec = {}
        self._cones_spawned = False
        

    def _get_ego(self):
        am = self.engine.agent_manager
        for k in ("default_agent", "agent0", "ego", "egocar0"):
            if k in getattr(am, "active_agents", {}):
                return am.active_agents[k]
        vals = list(getattr(am, "active_agents", {}).values())
        return vals[0] if len(vals) > 0 else None

    def _pick_lane(self, ego, lane_offset: int):
        nav = getattr(ego, "navigation", None)
        if nav is None or getattr(nav, "current_lane", None) is None:
            return None

        cur_lane = nav.current_lane
        ref_lanes = getattr(nav, "current_ref_lanes", None)
        if not ref_lanes:
            return cur_lane

        try:
            ego_lane_id = int(cur_lane.index[-1])
        except Exception:
            ego_lane_id = 0

        lane_num = len(ref_lanes)
        tgt = int(np.clip(ego_lane_id + int(lane_offset), 0, lane_num - 1))
        return ref_lanes[tgt]

    def _get_cone_class(self):
        candidates = []
        try:
            from metadrive.component.static_object.traffic_object import TrafficCone
            candidates.append(TrafficCone)
        except Exception:
            pass
        try:
            from metadrive.component.static_object.traffic_cone import TrafficCone
            candidates.append(TrafficCone)
        except Exception:
            pass
        return candidates[0] if candidates else None

    def _spawn_cone_groups(self):
        ego = self._get_ego()
        if ego is None:
            return

        ConeCls = self._get_cone_class()
        if ConeCls is None:
            print("[WARN] Could not import TrafficCone class in this MetaDrive build. Skipping cones.")
            return

        nav = getattr(ego, "navigation", None)
        if nav is None or getattr(nav, "current_lane", None) is None:
            return

        for g in self.CFG.get("cones", []):
            if not g.get("enabled", True):
                continue

            lane = self._pick_lane(ego, int(g.get("lane_offset", 0)))
            if lane is None:
                continue

            ego_pos = np.asarray(getattr(ego, "position", [0.0, 0.0]), dtype=np.float32)
            s0, _ = lane.local_coordinates(ego_pos)

            start = float(g.get("start_delta_s_m", 8.0))
            spacing = float(g.get("spacing_m", 2.5))
            count = int(g.get("count", 6))
            lateral = float(g.get("lateral_m", 1.2))

            for i in range(count):
                s = s0 + start + i * spacing
                xy = lane.position(s, lateral)
                hdg = lane.heading_theta_at(s)

                self.spawn_object(
                    ConeCls,
                    position=(float(xy[0]), float(xy[1])),
                    heading_theta=float(hdg),
                )

    def _spawn_vehicle_from_spec(self, ego, spec: dict):
        lane = self._pick_lane(ego, int(spec.get("lane_offset", 0)))
        if lane is None:
            return None

        ego_pos = np.asarray(getattr(ego, "position", [0.0, 0.0]), dtype=np.float32)
        s0, _ = lane.local_coordinates(ego_pos)

        s_spawn = s0 + float(spec.get("delta_s_m", 10.0))
        t_spawn = float(spec.get("delta_t_m", 0.0))
        xy = lane.position(s_spawn, t_spawn)
        hdg = lane.heading_theta_at(s_spawn)

        v = self.spawn_object(
            DefaultVehicle,
            vehicle_config=dict(show_navigation_arrow=True),
            position=(float(xy[0]), float(xy[1])),
            heading=float(hdg),
        )
        return v

    def _attach_policy(self, veh, spec: dict):
        pol = str(spec.get("policy", "idm_nominal"))
        params = dict(spec.get("params", {}))

        if pol == "sudden_brake":
            self.add_policy(veh.id, LaneFollowSuddenBrakePolicy, veh, self.generate_seed(), **params)
        elif pol == "accel_then_brake":
            self.add_policy(veh.id, LaneFollowAccelThenBrakePolicy, veh, self.generate_seed(), **params)
        elif pol == "idm_nominal":
            self.add_policy(veh.id, IDMNominalPolicy, veh, self.generate_seed(), **params)
        else:
            raise ValueError(f"Unknown policy: {pol}")

    def _spawn_all(self):
        ego = self._get_ego()
        if ego is None:
            return False

        for spec in self.CFG.get("vehicles", []):
            v = self._spawn_vehicle_from_spec(ego, spec)
            if v is None:
                continue
            self._attach_policy(v, spec)
            self.agent_ids.append(v.id)
            self.id_to_spec[v.id] = spec

        self._spawn_cone_groups()
        self._cones_spawned = True

        self._spawned = True
        return True

    def before_step(self):
        CFG=self.CFG
        if not self._spawned:
            ok = self._spawn_all()
            if not ok:
                return

        for vid in list(self.agent_ids):
            try:
                veh = self.spawned_objects[vid]
                pol = self.get_policy(vid)
                veh.before_step(pol.act())
            except Exception:
                continue

    def after_step(self, *args, **kwargs):
        for obj in self.spawned_objects.values():
            obj.after_step()

import json
import math
import numpy as np

# ============================================================
# NodeNetworkNavigation-safe turn routing (MetaDrive 0.4.3)
# ============================================================
class CustomEnv(MetaDriveEnv):
    def __init__(self,CFG,*args, **kwargs):
        self.CFG = CFG
        super().__init__(*args, **kwargs)

    def setup_engine(self):
        super().setup_engine()
        eng = self.engine
        attached = False
        try:
            eng.update_manager("multi_agent_scenario_manager", MultiAgentScenarioManager(self.CFG))
            attached = True
        except Exception:
            attached = False

        if not attached:
            print("[WARN] Could not add a new manager key; replacing traffic_manager. "
                  "Background traffic_density may not behave as expected.")
            eng.update_manager("traffic_manager", MultiAgentScenarioManager(self.CFG))
def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def _lane_heading_near_end(lane, eps0: float = 2.0, eps1: float = 0.5) -> float:
    p0 = np.array(lane.position(max(0.0, lane.length - eps0), 0.0), dtype=np.float32)
    p1 = np.array(lane.position(max(0.0, lane.length - eps1), 0.0), dtype=np.float32)
    v = p1 - p0
    return float(math.atan2(float(v[1]), float(v[0])))

def _lane_heading_near_start(lane, s0: float = 0.5, s1: float = 2.0) -> float:
    p0 = np.array(lane.position(s0, 0.0), dtype=np.float32)
    p1 = np.array(lane.position(s1, 0.0), dtype=np.float32)
    v = p1 - p0
    return float(math.atan2(float(v[1]), float(v[0])))

def _pick_any_outgoing(graph, node):
    d = graph.get(node, {})
    if not d:
        return None
    try:
        return sorted(list(d.keys()))[0]
    except Exception:
        return list(d.keys())[0]

def _choose_next_node_by_turn(graph, to_node, incoming_heading, turn: str):
    """
    Chooses the outgoing next_node from graph[to_node] according to turn intent.
    turn in {"left","right","straight","uturn"}.
    """
    out_dict = graph.get(to_node, {})
    if not out_dict:
        return None

    candidates = []
    for next_node, lanes in out_dict.items():
        if not lanes:
            continue
        out_heading = _lane_heading_near_start(lanes[0])
        dtheta = _angle_wrap(out_heading - incoming_heading)
        candidates.append((next_node, dtheta))

    if not candidates:
        return None

    t = (turn or "straight").lower()
    if t == "straight":
        return min(candidates, key=lambda x: abs(x[1]))[0]
    if t == "left":
        return max(candidates, key=lambda x: x[1])[0]
    if t == "right":
        return min(candidates, key=lambda x: x[1])[0]
    if t == "uturn":
        return min(candidates, key=lambda x: abs(abs(x[1]) - math.pi))[0]
    return min(candidates, key=lambda x: abs(x[1]))[0]

def init_node_network_checkpoints(env, vehicle, turn: str, min_len: int = 4) -> bool:
    """
    Initializes NodeNetworkNavigation.checkpoints to a SAFE length.
    In MetaDrive 0.4.3, NodeNetworkNavigation may look 2 edges ahead,
    so we prefer >=4 nodes in checkpoints to avoid IndexError.

    Returns True if set, False otherwise.
    """
    if vehicle is None:
        return False

    nav = getattr(vehicle, "navigation", None)
    if nav is None or not hasattr(nav, "checkpoints"):
        return False

    lane = getattr(nav, "current_lane", None)
    if lane is None or not hasattr(lane, "index"):
        return False

    idx = lane.index  # typically (from_node, to_node, lane_id)
    if not (isinstance(idx, (tuple, list)) and len(idx) >= 2):
        return False
    frm, to = idx[0], idx[1]

    m = getattr(env.engine, "current_map", None)
    rn = getattr(m, "road_network", None) if m is not None else None
    g = getattr(rn, "graph", None) if rn is not None else None
    if g is None:
        return False

    incoming_heading = _lane_heading_near_end(lane)

    # next1: chosen by turn at node `to`
    next1 = _choose_next_node_by_turn(g, to, incoming_heading, turn)
    if next1 is None:
        next1 = _pick_any_outgoing(g, to)
        if next1 is None:
            return False

    # next2: any outgoing from next1 (avoid dead-end if possible)
    next2 = _pick_any_outgoing(g, next1)
    if next2 is None:
        # fallback: choose some other outgoing from `to`
        next2 = _pick_any_outgoing(g, to)
        if next2 is None:
            return False

    # next3: optional extra lookahead
    next3 = _pick_any_outgoing(g, next2)

    # Build checkpoint nodes (prefer 4 nodes)
    ckpts = [frm, to, next1, next2]
    if min_len >= 5 and next3 is not None:
        ckpts.append(next3)
    if min_len < 4:
        ckpts = ckpts[:max(3, min_len)]

    nav.checkpoints = ckpts
    return True

def ensure_nav_module_in_cfg(CFG, nav_module: str):
    """
    Best-effort: request NodeNetworkNavigation for ego + scenario vehicles.
    NOTE: This only requests the module; checkpoints still must be initialized.
    """
    try:
        from metadrive.constants import DEFAULT_AGENT
    except Exception:
        DEFAULT_AGENT = "default_agent"

    CFG.setdefault("agent_configs", {})
    CFG["agent_configs"].setdefault(DEFAULT_AGENT, {})
    CFG["agent_configs"][DEFAULT_AGENT].setdefault("vehicle_config", {})
    CFG["agent_configs"][DEFAULT_AGENT]["vehicle_config"]["navigation_module"] = nav_module

    for spec in CFG.get("vehicles", []):
        spec.setdefault("vehicle_config", {})
        if isinstance(spec["vehicle_config"], dict):
            spec["vehicle_config"]["navigation_module"] = nav_module

# ============================================================
# Revised run() with explicit ego_turn / lead_turn routing
# ============================================================
# Optionally append final frame

def run_save_turn(
    CFG,
    scenario_dict,
    idx,
    revise_iter,
    set_policy=None,
    nav_module="NodeNetworkNavigation",
    checkpoint_min_len=4,   # 4 is the safe default for 0.4.3
    snapshot_steps=None,    # int or list[int]
    out_dir="snapshots",
    prefix=None,
    grid_cols=4,
    include_final=False,
):
    """
    Run a scenario (with NodeNetworkNavigation checkpoint warmup/init) and save ONE montage PNG
    composed of frames at `snapshot_steps`.

    - Removes all metric logging + GIF generation.
    - Keeps your warmup + restart-on-crash checkpoint init logic.
    - Captures frames only at snapshot steps (and optionally final).
    """
    import os
    import math
    import numpy as np

    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio

    # ---- normalize snapshot steps ----
    if snapshot_steps is None:
        snapshot_steps = []
    if isinstance(snapshot_steps, (int, np.integer)):
        snapshot_steps = [int(snapshot_steps)]
    snapshot_steps = sorted(set(int(s) for s in snapshot_steps if s is not None))
    snapshot_set = set(snapshot_steps)

    os.makedirs(out_dir, exist_ok=True)

    agent_policy = IDMPolicy if set_policy is None else set_policy

    # Request nav module in config (your helper)
    ensure_nav_module_in_cfg(CFG, nav_module=nav_module)

    env = CustomEnv(
        CFG,
        dict(
            map=CFG["map"],
            agent_policy=agent_policy,
            log_level=50,
            traffic_density=float(CFG["traffic_density"]),
            accident_prob=float(CFG["accident_prob"]),
        ),
    )

    ego_turn = CFG.get("ego_turn", "straight")
    lead_turn = CFG.get("lead_turn", "straight")

    def _get_mgr():
        try:
            return getattr(env.engine, "multi_agent_scenario_manager", None) or env.engine.traffic_manager
        except Exception:
            return None

    def _get_ego():
        try:
            am = env.engine.agent_manager
            for k in ("default_agent", "agent0", "ego", "egocar0"):
                if k in getattr(am, "active_agents", {}):
                    return am.active_agents[k]
            vals = list(getattr(am, "active_agents", {}).values())
            return vals[0] if vals else None
        except Exception:
            return None

    def _get_lead(mgr, lead_name):
        if mgr is None or lead_name is None:
            return None
        try:
            for vid in getattr(mgr, "agent_ids", []):
                spec = getattr(mgr, "id_to_spec", {}).get(vid, {})
                if spec.get("id") == lead_name:
                    return mgr.spawned_objects.get(vid, None)
        except Exception:
            pass
        return None

    # ---- filename prefix ----
    if prefix is None:
        road, sv, cp, wz = scenario_dict
        safe = lambda s: str(s).replace(" ", "_").replace("/", "_").replace("__", "_")
        prefix = f"{safe(CFG.get('map','map'))}_{safe(road)}_{safe(sv)}_{safe(cp)}_{safe(wz)}"

    frames = []
    steps_used = []
    terminated = False
    truncated = False
    last_step = 0
    lead_name = None

    try:
        env.reset()

        # Identify lead id by lane_offset==0 (your convention)
        for spec in CFG.get("vehicles", []):
            if int(spec.get("lane_offset", 999)) == 0:
                lead_name = spec.get("id", None)
                break
        if lead_name is None and CFG.get("vehicles", []):
            lead_name = CFG["vehicles"][0].get("id", None)

        mgr = _get_mgr()
        ego = _get_ego()
        lead = _get_lead(mgr, lead_name)

        # ------------------------------------------------------------
        # Critical: initialize checkpoints BEFORE the main rollout.
        # Warmup step to let nav.current_lane be available.
        # Restart once and retry on AssertionError/IndexError.
        # ------------------------------------------------------------
        def _warmup_and_init_checkpoints():
            nonlocal ego, lead, mgr, terminated, truncated

            obs, rew, terminated, truncated, info = env.step([0.0, 0.0])

            mgr = _get_mgr()
            ego = _get_ego()
            lead = _get_lead(mgr, lead_name)

            ego_ok = init_node_network_checkpoints(env, ego, ego_turn, min_len=checkpoint_min_len)
            lead_ok = init_node_network_checkpoints(env, lead, lead_turn, min_len=checkpoint_min_len)
            return ego_ok, lead_ok

        try:
            _ = _warmup_and_init_checkpoints()
        except (AssertionError, IndexError):
            env.reset()
            mgr = _get_mgr()
            ego = _get_ego()
            lead = _get_lead(mgr, lead_name)
            _ = _warmup_and_init_checkpoints()

        # We already executed one warmup step
        start_step = 1
        steps_total = int(CFG["steps"])

        for step in range(start_step, steps_total):
            last_step = step
            obs, rew, terminated, truncated, info = env.step([0.0, 0.0])

            if step in snapshot_set:
                fr = env.render(
                    mode="topdown",
                    window=False,
                    screen_size=tuple(CFG["render"]["screen_size"]),
                    camera_position=tuple(CFG["render"]["camera_position"]),
                )
                if fr.dtype != np.uint8:
                    fr = np.clip(fr, 0, 255).astype(np.uint8)
                frames.append(fr[:, :, :3])
                steps_used.append(step)

            if terminated or truncated:
                break

        # Optionally append final frame (last executed step)
        if include_final and (last_step not in snapshot_set):
            fr = env.render(
                mode="topdown",
                window=False,
                screen_size=tuple(CFG["render"]["screen_size"]),
                camera_position=tuple(CFG["render"]["camera_position"]),
            )
            if fr.dtype != np.uint8:
                fr = np.clip(fr, 0, 255).astype(np.uint8)
            frames.append(fr[:, :, :3])
            steps_used.append(last_step)

        if len(frames) == 0:
            # fallback: at least one frame
            fr = env.render(
                mode="topdown",
                window=False,
                screen_size=tuple(CFG["render"]["screen_size"]),
                camera_position=tuple(CFG["render"]["camera_position"]),
            )
            if fr.dtype != np.uint8:
                fr = np.clip(fr, 0, 255).astype(np.uint8)
            frames = [fr[:, :, :3]]
            steps_used = [last_step]

        # ---- montage (grid) ----
        H = max(f.shape[0] for f in frames)
        W = max(f.shape[1] for f in frames)

        padded = []
        for f in frames:
            h, w = f.shape[:2]
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            canvas[:h, :w, :] = f
            padded.append(canvas)

        cols = max(1, int(grid_cols))
        rows = int(np.ceil(len(padded) / cols))
        montage = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)

        for i, f in enumerate(padded):
            r = i // cols
            c = i % cols
            montage[r * H:(r + 1) * H, c * W:(c + 1) * W, :] = f

        # include revise_iter in name (like your GIF naming convention)
        path = os.path.join(out_dir, f"R_{idx}_{prefix}_{revise_iter}_MONTAGE_{len(frames)}.png")
        imageio.imwrite(path, montage)
        print(f"[montage] saved {path}")

        return {
            "montage_path": path,
            "steps_used": steps_used,
            "n_frames": len(frames),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "last_step": int(last_step),
            "ego_turn": ego_turn,
            "lead_turn": lead_turn,
            "nav_module": nav_module,
            "checkpoint_min_len": checkpoint_min_len,
            "lead_id": lead_name,
        }

    finally:
        env.close()



def run(
    CFG,
    scenario_dict,
    idx,
    revise_iter,
    set_policy=None,
    nav_module="NodeNetworkNavigation",
    checkpoint_min_len=4,  # 4 is the safe default for 0.4.3
    out_dir=".",
):
    if set_policy is None:
        agent_policy = IDMPolicy
    else:
        agent_policy = set_policy

    # Request nav module in config
    ensure_nav_module_in_cfg(CFG, nav_module=nav_module)

    env = CustomEnv(
        CFG,
        dict(
            map=CFG["map"],
            agent_policy=agent_policy,
            log_level=50,
            traffic_density=float(CFG["traffic_density"]),
            accident_prob=float(CFG["accident_prob"]),
        ),
    )

    ego_turn = CFG['ego_turn']
    lead_turn = CFG['lead_turn']

    step_log = {
        "ego_speed": [],
        "lead_speed": [],
        "headway_s": [],
        "ego_pos": [],
        "lead_pos": [],
        "crash": [],
        "crash_vehicle": [],
        "crash_object": [],
        "arrive_dest": [],
        "time_min_dist": [],
        "min_euclid_dist_to_any_object": [],
    }

    metrics_dict = {}
    frames = []

    def _get_mgr():
        try:
            return getattr(env.engine, "multi_agent_scenario_manager", None) or env.engine.traffic_manager
        except Exception:
            return None

    def _get_ego():
        try:
            am = env.engine.agent_manager
            for k in ("default_agent", "agent0", "ego", "egocar0"):
                if k in getattr(am, "active_agents", {}):
                    return am.active_agents[k]
            vals = list(getattr(am, "active_agents", {}).values())
            return vals[0] if vals else None
        except Exception:
            return None

    def _get_lead(mgr, lead_name):
        if mgr is None or lead_name is None:
            return None
        try:
            for vid in getattr(mgr, "agent_ids", []):
                spec = getattr(mgr, "id_to_spec", {}).get(vid, {})
                if spec.get("id") == lead_name:
                    return mgr.spawned_objects.get(vid, None)
        except Exception:
            pass
        return None

    try:
        env.reset()

        # Identify lead id by lane_offset==0 (your convention)
        lead_name = None
        for spec in CFG.get("vehicles", []):
            if int(spec.get("lane_offset", 999)) == 0:
                lead_name = spec.get("id", None)
                break
        if lead_name is None and CFG.get("vehicles", []):
            lead_name = CFG["vehicles"][0].get("id", None)

        # Grab initial handles (may update later)
        mgr = _get_mgr()
        ego = _get_ego()
        lead = _get_lead(mgr, lead_name)

        # ------------------------------------------------------------
        # Critical: initialize checkpoints BEFORE the main rollout.
        # We may need one warmup step so nav.current_lane is available.
        # NodeNetworkNavigation can crash if it runs with empty ckpts,
        # so we use a "warmup with restart" strategy.
        # ------------------------------------------------------------
        def _warmup_and_init_checkpoints():
            nonlocal ego, lead, mgr

            # 1) warmup step to let current_lane become available
            obs, rew, terminated, truncated, info = env.step([0.0, 0.0])

            # refresh handles after warmup
            mgr = _get_mgr()
            ego = _get_ego()
            lead = _get_lead(mgr, lead_name)

            ego_ok = init_node_network_checkpoints(env, ego, ego_turn, min_len=checkpoint_min_len)
            lead_ok = init_node_network_checkpoints(env, lead, lead_turn, min_len=checkpoint_min_len)

            return ego_ok, lead_ok, terminated, truncated, info

        # Try warmup+init. If it crashes, restart once and retry.
        ego_ck_ok = lead_ck_ok = False
        terminated = truncated = False
        info = {}

        try:
            ego_ck_ok, lead_ck_ok, terminated, truncated, info = _warmup_and_init_checkpoints()
        except (AssertionError, IndexError):
            # restart episode and retry
            env.reset()
            mgr = _get_mgr()
            ego = _get_ego()
            lead = _get_lead(mgr, lead_name)
            ego_ck_ok, lead_ck_ok, terminated, truncated, info = _warmup_and_init_checkpoints()

        metrics_dict.update(
            {
                "ego_turn": ego_turn,
                "lead_turn": lead_turn,
                "nav_module": nav_module,
                "checkpoint_min_len": checkpoint_min_len,
                "ego_checkpoints_init": ego_ck_ok,
                "lead_checkpoints_init": lead_ck_ok,
            }
        )

        # If episode ended during warmup, we’ll still render/log what we can
        start_step = 1  # because we already stepped once for warmup

        # Main rollout
        for step in range(start_step, int(CFG["steps"])):
            obs, rew, terminated, truncated, info = env.step([0.0, 0.0])

            if ego is None:
                ego = _get_ego()
            if mgr is None:
                mgr = _get_mgr()
            if lead is None:
                lead = _get_lead(mgr, lead_name)

            ego_speed = safe_float(getattr(ego, "speed", np.nan)) if ego is not None else np.nan
            lead_speed = safe_float(getattr(lead, "speed", np.nan)) if lead is not None else np.nan
            step_log["ego_speed"].append(ego_speed)
            step_log["lead_speed"].append(lead_speed)

            ego_pos = (
                np.asarray(getattr(ego, "position", [np.nan, np.nan]), dtype=np.float32)
                if ego is not None
                else np.array([np.nan, np.nan], dtype=np.float32)
            )
            lead_pos = (
                np.asarray(getattr(lead, "position", [np.nan, np.nan]), dtype=np.float32)
                if lead is not None
                else np.array([np.nan, np.nan], dtype=np.float32)
            )
            step_log["ego_pos"].append([float(ego_pos[0]), float(ego_pos[1])])
            step_log["lead_pos"].append([float(lead_pos[0]), float(lead_pos[1])])

            crash = bool(info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False))
            step_log["crash"].append(crash)
            step_log["crash_vehicle"].append(bool(info.get("crash_vehicle", False)))
            step_log["crash_object"].append(bool(info.get("crash_object", False)))
            step_log["arrive_dest"].append(bool(info.get("arrive_dest", False)))

            # along-lane headway
            headway = float("inf")
            try:
                nav = getattr(ego, "navigation", None)
                lane = getattr(nav, "current_lane", None) if nav is not None else None
                if lane is not None and lead is not None:
                    s_ego, _ = lane.local_coordinates(ego_pos)
                    s_lead, _ = lane.local_coordinates(lead_pos)
                    headway = float(s_lead - s_ego)
            except Exception:
                headway = float("inf")
            step_log["headway_s"].append(headway)

            # min euclidean distance to any object
            min_dist = float("inf")
            time_min_dist = 0
            try:
                if ego is not None:
                    candidates = []

                    if mgr is not None:
                        try:
                            candidates.extend(list(getattr(mgr, "spawned_objects", {}).values()))
                        except Exception:
                            pass

                    try:
                        am = env.engine.agent_manager
                        candidates.extend(list(getattr(am, "active_agents", {}).values()))
                    except Exception:
                        pass

                    try:
                        tm = getattr(env.engine, "traffic_manager", None)
                        for attr in ("vehicles", "traffic_vehicles", "spawned_objects"):
                            vv = getattr(tm, attr, None)
                            if isinstance(vv, dict):
                                candidates.extend(list(vv.values()))
                            elif isinstance(vv, (list, tuple)):
                                candidates.extend(list(vv))
                    except Exception:
                        pass

                    for obj in candidates:
                        if obj is None or obj is ego:
                            continue
                        try:
                            p = np.asarray(getattr(obj, "position", None), dtype=np.float32)
                            if p is None or p.shape[0] < 2:
                                continue
                            d = float(np.linalg.norm(p - ego_pos))
                            if d < min_dist:
                                min_dist = d
                                time_min_dist = step
                        except Exception:
                            continue
            except Exception:
                min_dist = float("inf")

            step_log["min_euclid_dist_to_any_object"].append(min_dist)
            step_log["time_min_dist"] = time_min_dist

            txt = {
                "t": step,
                "ego_turn": ego_turn,
                "lead_turn": lead_turn,
                "arrive_dest": info.get("arrive_dest", None),
                "lead": lead_name,
                "headway_m": None if not np.isfinite(headway) else round(headway, 2),
                "crash": crash,
                "min_d": None if not np.isfinite(min_dist) else round(min_dist, 2),
            }

            frame = env.render(
                mode="topdown",
                window=False,
                text=txt,
                screen_size=tuple(CFG["render"]["screen_size"]),
                camera_position=tuple(CFG["render"]["camera_position"]),
            )
            frames.append(frame)

            if terminated or truncated:
                break

        dt = 0.1
        try:
            dt = float(getattr(env.engine, "dt", dt))
        except Exception:
            pass

        metrics = compute_failure_metrics(step_log, dt=dt)
        metrics.update(
            dict(
                map=CFG.get("map"),
                traffic_density=float(CFG.get("traffic_density", 0.0)),
                lead_id=lead_name,
                terminated=bool(terminated),
                truncated=bool(truncated),
                ego_turn=ego_turn,
                lead_turn=lead_turn,
                nav_module=nav_module,
                checkpoint_min_len=checkpoint_min_len,
                ego_checkpoints_init=metrics_dict.get("ego_checkpoints_init"),
                lead_checkpoints_init=metrics_dict.get("lead_checkpoints_init"),
            )
        )
        metrics_dict = metrics

        print("\n=== FAILURE METRICS ===")
        for k, v in metrics_dict.items():
            print(f"{k}: {v}")

        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(_jsonable(metrics_dict), f, indent=2)

        filepath = os.path.join(
            out_dir,
            f"{idx}_{scenario_dict[0]}_{scenario_dict[1]}_{scenario_dict[2]}_{scenario_dict[3]}_{revise_iter}.gif",
        )
        if frames:
            generate_gif(frames, gif_name=filepath)
            metrics_dict["gif_path"] = filepath

    finally:
        env.close()

    return metrics_dict