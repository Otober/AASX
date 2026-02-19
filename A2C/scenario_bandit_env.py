# scenario_bandit_env.py (updated)
from __future__ import annotations
import os, json
from typing import List, Tuple, Dict, Any, Optional, Sequence
import numpy as np
import pandas as pd

def z(x, mu, sigma):
    return 0.0 if sigma == 0 else (x - mu) / sigma

# --- at top of scenario_bandit_env.py ---
def _get_num(d: dict, *keys, default: float = 0.0) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)


class EnvInterface:
    obs_dim: int
    n_actions: int
    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]: ...

def _safe_stats(arr: Sequence[float]) -> Tuple[float, float, float]:
    if not arr:
        return (0.0, 0.0, 0.0)
    a = np.asarray(arr, dtype=np.float32)
    return float(np.min(a)), float(np.mean(a)), float(np.max(a))

def _parse_release_to_time(release_name: str) -> float:
    # "26_000" -> 26.0  /  "318_000" -> 318.0
    base = release_name.split("_")[0]
    try:
        return float(base)
    except Exception:
        return 0.0

def _infer_15d_from_trace_at_time(trace_csv: str, t_target: float, nearest: str = "nearest") -> np.ndarray:
    """
    trace.csv에서 'time' == t_target 시점(정확히 없으면 근사)으로 15D 관측 생성.
    nearest:
      - "nearest": 가장 가까운 시각으로 스냅샷
      - "floor":   t <= t_target 중 가장 큰 시각(없으면 최소 시각)
      - "ceil":    t >= t_target 중 가장 작은 시각(없으면 최대 시각)
    """
    df = pd.read_csv(trace_csv)
    if "time" not in df.columns:
        raise ValueError(f"[trace] 'time' column missing in {trace_csv}")
    times = np.asarray(sorted(df["time"].unique()), dtype=np.float64)
    if nearest == "nearest":
        idx = int(np.argmin(np.abs(times - t_target)))
    elif nearest == "floor":
        idx = int(np.searchsorted(times, t_target, side="right") - 1)
        idx = max(0, idx)
    elif nearest == "ceil":
        idx = int(np.searchsorted(times, t_target, side="left"))
        idx = min(idx, len(times) - 1)
    else:
        idx = int(np.argmin(np.abs(times - t_target)))
    t_pick = float(times[idx])
    dft = df[df["time"] == t_pick].copy()

    # 머신 요약
    def _col(col):
        return dft[col].values.tolist() if col in dft.columns else []
    sum_pt = [float(x) for x in _col("sum_pt_on_queue") if pd.notna(x)]
    proc_rt = [float(x) for x in _col("processing_remaining") if pd.notna(x)]
    out_q = [float(x) for x in _col("outq_waiting_count") if pd.notna(x)]
    s_min, s_mean, s_max = _safe_stats(sum_pt)
    r_min, r_mean, r_max = _safe_stats(proc_rt)
    o_min, o_mean, o_max = _safe_stats(out_q)

    # AGV 상태 카운트
    cnt_delivery = cnt_fetch = cnt_idle = 0
    if "agv_status" in dft.columns:
        for st in dft["agv_status"].astype(str):
            if st == "delivery": cnt_delivery += 1
            elif st == "fetch":  cnt_fetch    += 1
            elif st == "idle":   cnt_idle     += 1

    # 글로벌 3 (있으면 사용, 없으면 추정)
    def _first_or_none(col):
        return float(dft[col].iloc[0]) if (col in dft.columns and pd.notna(dft[col].iloc[0])) else None
    op_progress = _first_or_none("op_progress")
    agv_util    = _first_or_none("agv_util")
    mach_util   = _first_or_none("machine_util")

    if agv_util is None:
        total_agv = cnt_delivery + cnt_fetch + cnt_idle
        agv_util = (cnt_delivery + cnt_fetch) / total_agv if total_agv > 0 else 0.0
    if mach_util is None:
        if len(proc_rt) > 0:
            busy = sum(1 for v in proc_rt if v > 0.0)
            mach_util = busy / max(1, len(proc_rt))
        else:
            mach_util = 0.0
    if op_progress is None:
        op_progress = max(0.0, min(1.0, 1.0 - mach_util))

    obs = np.asarray([
        float(op_progress), float(agv_util), float(mach_util),    # 3
        s_min, s_mean, s_max,                                     # 3
        r_min, r_mean, r_max,                                     # 3
        o_min, o_mean, o_max,                                     # 3
        float(cnt_delivery), float(cnt_fetch), float(cnt_idle)    # 3
    ], dtype=np.float32)
    assert obs.shape[0] == 15
    return obs

class ScenarioBanditEnv(EnvInterface):
    """
    reset() 시 (scenario, release)를 순회/샘플.
    관측은 항상 scenario/0_000/n0/trace.csv 에서 t = release_time(26, 318, 402 등)의 스냅샷으로 생성.
    step(action): action∈{0..5} → {4,8,12,16,20,24} 매핑, 해당 release 폴더의 objectives.json에서 보상 계산.
    """
    ACTIONS = [4, 8, 12, 16, 20, 24]

    def __init__(self,
                 root_dir: str,
                 scenarios: Optional[List[str]] = None,
                 releases:  Optional[List[str]] = None,   # 예: ["26_000","318_000","402_000"]
                 alpha: float = 0.4,
                 beta: float = 0.3,
                 gamma: float = 0.3,
                 minimize: bool = True,
                 shuffle: bool = False,
                 seed: int = 42,
                 nearest_policy: str = "nearest",  # t 매칭 정책: "nearest" | "floor" | "ceil",
                 reward_scale : float = 1e3,
                 reward_clip: float = 10.0
                 ):
        self.root_dir = os.path.abspath(root_dir)
        self.scenarios = scenarios
        self.releases = releases
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.minimize = bool(minimize)
        self.rng = np.random.RandomState(seed)
        self.nearest_policy = nearest_policy
        self.reward_scale = float(reward_scale)
        self.reward_clip  = float(reward_clip)
        self.n_actions = len(self.ACTIONS)
        self.obs_dim = 15  # 고정

        # 시나리오 자동 탐색 (숫자 폴더)
        if scenarios is None:
            scenarios = sorted([d for d in os.listdir(self.root_dir)
                                if os.path.isdir(os.path.join(self.root_dir, d)) and d.isdigit()],
                               key=lambda x: int(x))
        self.scenarios = [str(s) for s in scenarios]

        # release 자동 탐색: 첫 시나리오 기준으로 하위 폴더 추출 후 0_000 제외
        releases_whitelist = set(releases) if releases else None

        self.releases_map: Dict[str, List[str]] = {}
        for s in self.scenarios:
            base = os.path.join(self.root_dir, s)
            rels = [d for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d)) and d.endswith("_000") and d != "0_000"]
            if releases_whitelist is not None:
                rels = [r for r in rels if r in releases_whitelist]
            rels_sorted = sorted(rels, key=lambda x: int(x.split("_")[0]))
            self.releases_map[s] = rels_sorted

        # --- 전체 (scenario, release) 페어 전개 ---
        self.pairs: List[Tuple[str, str]] = []
        for s in self.scenarios:
            for r in self.releases_map[s]:
                self.pairs.append((s, r))

        if not self.pairs:
            raise RuntimeError("No (scenario, release) pairs found. Check folders/filters.")

        if shuffle:
            self.rng.shuffle(self.pairs)

        self._pair_idx = -1
        self.cur_pair: Optional[Tuple[str, str]] = None

        # 관측 캐시 & 액션 마스크 초기화
        self._obs_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._action_mask = np.ones(len(self.ACTIONS), dtype=np.float32)

    # 경로 유틸
    def _trace_csv_for_obs(self, scenario: str) -> str:
        return os.path.join(self.root_dir, scenario, "0_000", "n0", "trace.csv")

    def _objectives_json_for(self, scenario: str, release: str, nmax: int) -> str:
        return os.path.join(self.root_dir, scenario, release, f"n{nmax}", "objectives.json")

    def reset(self) -> np.ndarray:
        # 다음 (scenario, release)로 전진 (루프)
        self._pair_idx = (self._pair_idx + 1) % len(self.pairs)
        self.cur_pair = self.pairs[self._pair_idx]
        scen, rel = self.cur_pair

        # t_target = release 명에서 숫자 추출(예: "26_000" -> 26.0)
        t_target = _parse_release_to_time(rel)

        key = (scen, rel)
        if key in self._obs_cache:
            return self._obs_cache[key]

        tr = self._trace_csv_for_obs(scen)
        if not os.path.exists(tr):
            raise FileNotFoundError(f"[obs] trace not found: {tr}")

        obs = _infer_15d_from_trace_at_time(tr, t_target, nearest=self.nearest_policy)
        self._obs_cache[key] = obs
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.cur_pair is None:
            raise RuntimeError("call reset() before step()")

        scen, rel = self.cur_pair
        if not (0 <= action < self.n_actions):
            raise ValueError(f"invalid action {action}")
        nmax = self.ACTIONS[action]

        obj_path = self._objectives_json_for(scen, rel, nmax)
        if not os.path.exists(obj_path):
            reward = -1e9 if self.minimize else 0.0
            info = {"error": f"objectives.json not found: {obj_path}",
                    "scenario": scen, "release": rel, "nmax": nmax}
            obs_next = self.reset()
            return obs_next, float(reward), True, info

        with open(obj_path, "r") as f:
            obj = json.load(f)

        M = _get_num(obj, "makespan", "MakeSpan", "M", default=0.0)
        T = _get_num(obj, "agv_travel", "agv_travel_distance", "AGV_travel", default=0.0)
        O = _get_num(obj, "optimization_time", "time_optimization", "Optimization time", default=0.0)
        Mz = z(M, 1058.1, 27.1)
        Tz = z(T, 1792.3, 43.0)
        Oz = z(O, 21.6, 7.2)
        score = self.alpha * Mz + self.beta * Tz + self.gamma * Oz
        # scale down: makespan/여정시간/최적화시간 크기 맞춰 예: 1e6
        scale = 1e6
        reward_raw = (-(score) if self.minimize else score) / self.reward_scale
        reward = float(np.clip(reward_raw, -self.reward_clip, self.reward_clip))

        info = {
            "scenario": scen, "release": rel, "nmax": nmax,
            "makespan": M, "agv_travel": T, "optimization_time": O,
            "score": score, "reward_raw": reward_raw, "reward": reward
        }

        # 단일-스텝 종료 → 다음 (scenario, release)로 넘어가 초기 관측 반환
        obs_next = self.reset()
        return obs_next, float(reward), True, info
