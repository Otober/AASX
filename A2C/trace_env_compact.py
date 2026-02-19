# trace_env_compact.py
from __future__ import annotations
import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional

# 당신의 snapshot_state.py가 제공하는 API에 맞춰 import
# 예) from snapshot_state import TraceSnapshot
from snapshot_state import TraceSnapshot

class EnvInterface:
    obs_dim: int
    n_actions: int
    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]: ...

def _safe_stats(arr):
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    a = np.asarray(arr, dtype=np.float32)
    return float(np.min(a)), float(np.mean(a)), float(np.max(a))

class TraceEnvCompact(EnvInterface):
    """
    15D 관측을 내는 환경:
      obs = [
        op_progress, agv_util, machine_util,               # 3
        min/mean/max sum_pt_on_queue,                      # 3
        min/mean/max processing_remaining,                 # 3
        min/mean/max outq_waiting_count,                   # 3
        agv_count_delivery, agv_count_fetch, agv_count_idle# 3
      ]
    행동공간(n_actions)은 외부에서 결정(예: Nmax 조정 등).
    보상은 reward_fn(prev, cur, action) 콜백으로 주입(기본 0).
    시간전진: 이벤트 타임스탬프 기반(step_mode="event") 또는 고정 간격(step_mode="fixed_dt")
    """
    def __init__(self,
                 snapshot: TraceSnapshot,
                 n_actions: int,
                 step_mode: str = "event",          # "event" or "fixed_dt"
                 fixed_dt: float = 1.0,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None,
                 reward_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any], int], float]] = None):
        self.ss = snapshot
        self.n_actions = int(n_actions)
        self.reward_fn = reward_fn if reward_fn is not None else (lambda prev, cur, a: 0.0)

        # 타임라인 세팅
        times = np.asarray(sorted(self.ss.df["time"].unique()), dtype=np.float64)
        if start_time is None: start_time = float(times[0])
        if end_time   is None: end_time   = float(times[-1])
        self.start_time = float(start_time)
        self.end_time   = float(end_time)
        self.step_mode  = step_mode
        self.fixed_dt   = float(fixed_dt)

        self._times = times[(times >= self.start_time) & (times <= self.end_time)]
        if len(self._times) == 0:
            self._times = np.array([self.start_time, self.end_time], dtype=np.float64)

        # 관측 차원
        self.obs_dim = 15

        # 내부 상태
        self._idx = 0
        self._t   = self.start_time
        self._last = None

    # ---- 공개 API ----
    def reset(self) -> np.ndarray:
        self._idx = 0
        self._t   = self.start_time
        self._last = self.ss.snapshot(self._t)
        return self._build_obs(self._last)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.step_mode == "event":
            self._idx = min(self._idx + 1, len(self._times) - 1)
            self._t = float(self._times[self._idx])
        else:
            self._t = float(min(self._t + self.fixed_dt, self.end_time))
            # 이벤트에 스냅을 맞추고 싶다면:
            # self._idx = int(np.searchsorted(self._times, self._t, side="right") - 1)
            # self._t = float(self._times[self._idx])

        cur = self.ss.snapshot(self._t)
        prev = self._last
        reward = float(self.reward_fn(prev, cur, int(action)))
        self._last = cur

        done = bool(self._t >= self.end_time or self._idx >= len(self._times)-1)
        info = {"t": self._t}
        return self._build_obs(cur), reward, done, info

    # ---- 관측 15D 구성 ----
    def _build_obs(self, snap: Dict[str, Any]) -> np.ndarray:
        feats = []

        # --- 글로벌 3 ---
        # 1) op_progress (0~1): snapshot이 제공한다고 가정. 없으면 간이 추정(처리중 비율).
        op_progress = snap.get("op_progress")
        if op_progress is None:
            # 간단 추정: processing_remaining > 0 인 머신 비율을 진행률의 보수로 보고 1 - 비율
            machines = snap.get("machines", {})
            if machines:
                busy = sum(1 for m in machines.values() if float(m.get("processing_remaining", 0.0)) > 0.0)
                total = max(1, len(machines))
                op_progress = max(0.0, min(1.0, 1.0 - busy/total))
            else:
                op_progress = 0.0
        feats.append(float(op_progress))

        # 2) agv_util (0~1): snapshot 제공 선호. 없으면 delivery/fetch 비율로 추정.
        agv_util = snap.get("agv_util")
        if agv_util is None:
            agvs = snap.get("agvs", {})
            if agvs:
                active = sum(1 for a in agvs.values() if str(a.get("status","idle")) in ("delivery","fetch"))
                agv_util = active / max(1, len(agvs))
            else:
                agv_util = 0.0
        feats.append(float(agv_util))

        # 3) machine_util (0~1): snapshot 제공 선호. 없으면 processing_remaining>0 비율.
        machine_util = snap.get("machine_util")
        machines = snap.get("machines", {})
        if machine_util is None and machines:
            busy = sum(1 for m in machines.values() if float(m.get("processing_remaining", 0.0)) > 0.0)
            machine_util = busy / max(1, len(machines))
        machine_util = float(machine_util if machine_util is not None else 0.0)
        feats.append(machine_util)

        # --- 머신 요약 9 ---
        # 각 머신에서 키가 없을 수도 있으므로 0으로 대체한 뒤 min/mean/max
        def _collect(key: str):
            vals = []
            for m in machines.values():
                vals.append(float(m.get(key, 0.0)))
            return _safe_stats(vals)

        mn, me, mx = _collect("sum_pt_on_queue")
        feats += [mn, me, mx]
        mn, me, mx = _collect("processing_remaining")
        feats += [mn, me, mx]
        mn, me, mx = _collect("outq_waiting_count")
        feats += [mn, me, mx]

        # --- AGV 상태 카운트 3 ---
        agvs = snap.get("agvs", {})
        cnt_delivery = sum(1 for a in agvs.values() if str(a.get("status","idle")) == "delivery")
        cnt_fetch    = sum(1 for a in agvs.values() if str(a.get("status","idle")) == "fetch")
        cnt_idle     = sum(1 for a in agvs.values() if str(a.get("status","idle")) == "idle")
        feats += [float(cnt_delivery), float(cnt_fetch), float(cnt_idle)]

        obs = np.asarray(feats, dtype=np.float32)
        assert obs.shape[0] == 15, f"obs dim mismatch: {obs.shape[0]} vs 15"
        return obs
