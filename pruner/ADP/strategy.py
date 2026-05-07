"""Online ADP pruning strategy controller.

This module intentionally stays close to the original
`run_libero_eval_prune_v2.py` state-machine logic.
"""

from __future__ import annotations

import numpy as np


class OnlineStrategyController:
    def __init__(self, cfg, quat2axisangle_fn):
        self.cfg = cfg
        self.quat2axisangle_fn = quat2axisangle_fn
        self.curr_state = int(getattr(cfg, "initial_state", 1))
        self.initial_state = int(getattr(cfg, "initial_state", 1))
        self.curr_window_start_frame = None
        self.frame_counter_global = 0
        self.pos_list = []
        self.aa_list = []
        self.ones_streak = (
            1
            if (self.initial_state == 1 and getattr(cfg, "limit_consecutive_pruned_enabled", False))
            else 0
        )

    def _axis_angle_to_matrix(self, aa):
        theta = float(np.linalg.norm(aa))
        if theta < 1e-12:
            return np.eye(3)
        axis = aa / theta
        x, y, z = axis
        k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
        s = np.sin(theta)
        c = np.cos(theta)
        return np.eye(3) + s * k + (1.0 - c) * (k @ k)

    def _relative_rotation_angle(self, r_prev, r_curr):
        r_rel = r_prev.T @ r_curr
        trace = float(np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0))
        ang = float(np.arccos(trace))
        return 0.0 if abs(ang) < 1e-12 else ang

    def observe_step(self, eef_pos, eef_quat):
        aa = self.quat2axisangle_fn(eef_quat)
        if self.curr_window_start_frame is None:
            self.curr_window_start_frame = self.frame_counter_global
        self.pos_list.append(np.array(eef_pos, dtype=np.float64))
        self.aa_list.append(np.array(aa, dtype=np.float64))
        self.frame_counter_global += 1

    def _compute_window_delta(self) -> float:
        positions = np.stack(self.pos_list, axis=0) if len(self.pos_list) > 0 else np.zeros((0, 3))
        axis_angles = np.stack(self.aa_list, axis=0) if len(self.aa_list) > 0 else np.zeros((0, 3))
        del axis_angles
        method = (getattr(self.cfg, "delta_method", "net") or "net").lower()
        if positions.shape[0] <= 1:
            return 0.0
        if method == "net":
            dp = positions[-1] - positions[0]
            return float(np.linalg.norm(dp))
        if method in ("sum", "arc_sum", "hypot"):
            step_pos = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
            step_pos = np.where(step_pos < getattr(self.cfg, "min_delta_pos", 0.0), 0.0, step_pos)
            return float(np.sum(step_pos))
        dp = positions[-1] - positions[0]
        return float(np.linalg.norm(dp))

    def close_window_and_decide_next(self):
        d = self._compute_window_delta()
        next_state = self.curr_state
        dm = (getattr(self.cfg, "decision_method", "avg") or "avg").lower()

        if dm == "avg":
            if not hasattr(self, "_avg_d"):
                self._avg_d = d
            else:
                self._avg_d = 0.5 * self._avg_d + 0.5 * d
            next_state = 1 if (d >= self._avg_d) else 0
        else:
            window = max(3, int(getattr(self.cfg, "adjacent_extrema_window", 3)))
            if not hasattr(self, "_history_d"):
                self._history_d = []
            self._history_d.append(d)
            if len(self._history_d) < window:
                next_state = int(getattr(self.cfg, "initial_state", 0))
            else:
                prev_vals = [float(x) for x in self._history_d[-(window - 1) - 1 : -1]] if len(self._history_d) >= window else []
                if len(prev_vals) == 0:
                    next_state = int(getattr(self.cfg, "initial_state", 0))
                else:
                    up_thr = max(prev_vals)
                    dn_thr = min(prev_vals)
                    if d >= up_thr:
                        next_state = 1
                    elif d < dn_thr:
                        next_state = 0
                    else:
                        if self.cfg.adjacent_last_state:
                            next_state = self.curr_state
                        else:
                            next_state = 1

        ns = int(next_state)
        if bool(getattr(self.cfg, "limit_consecutive_pruned_enabled", False)):
            try:
                streak = int(getattr(self, "ones_streak", 0))
            except Exception:
                streak = 0
            max_ones = int(getattr(self.cfg, "limit_max_consecutive_pruned", 3))
            if ns == 1:
                if streak >= max_ones:
                    ns = 0
                    streak = 0
                else:
                    streak += 1
            else:
                streak = 0
            self.ones_streak = streak

        self.curr_state = ns
        self.pos_list.clear()
        self.aa_list.clear()
        self.curr_window_start_frame = None

    def get_current_state(self) -> int:
        return int(self.curr_state)
