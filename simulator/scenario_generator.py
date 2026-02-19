
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a self-contained scenario from scratch (integer time flavor):
- machines.json           (adds random integer 'available_time' per machine)
- operation_durations.json
- machine_transfer_time.json
- jobs.json
- operations.json
- job_release.json        (release_time as integer seconds)

Schema is kept compatible with your current simulator. New field 'available_time'
is optional; simulators that don't use it can ignore it safely.
"""

import argparse
import json
import random
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import time

# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def dump_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def jid(i: int) -> str: return f"J{i}"
def pid(i: int) -> str: return f"P{i}"
def oid(job_idx: int, k: int) -> str: return f"O{job_idx:02d}{k}"

# ----------------------------
# Config dataclass
# ----------------------------
@dataclass
class GenCfg:
    # high-level
    num_jobs: int = 35
    num_machines: int = 3
    op_types: Optional[List[str]] = None
    min_ops: int = 2
    max_ops: int = 4
    release_min: int = 0
    release_max: int = 300
    # machine initial availability (integer seconds)
    machine_avail_min: int = 0
    machine_avail_max: int = 0
    # distributions (kept from original, used to synthesize dist params;
    # actual sampled times will happen at simulation; we keep params as floats)
    proc_time_mean_range: tuple = (5.0, 40.0)
    proc_time_std_frac: tuple = (0.05, 0.25)
    transfer_mean_range: tuple = (1.0, 10.0)
    # reproducibility
    seed: int = int(time.time())
    due_time_min: int = 0
    due_time_max: int = 0

# ----------------------------
# Core generators
# ----------------------------
def gen_machine_names(k: int) -> List[str]:
    return [f"M{i}" for i in range(1, k+1)]

def gen_machines_json(names: List[str], cfg: GenCfg) -> Dict:
    mi, mx = int(cfg.machine_avail_min), int(cfg.machine_avail_max)
    if mx < mi:
        mx = mi
    return {
        n: {
            "status": "idle",
            "next_available_time": int(random.randint(mi, mx))
        } for n in names
    }


def _rand_distribution(mean_range, std_frac_range):
    mu = int(round(random.uniform(*mean_range)))
    return {"distribution": "normal", "mean": mu, "std": 0.0}

def gen_operation_durations_json(op_types: List[str], machines: List[str], cfg: GenCfg) -> Dict:
    table = {}
    for t in op_types:
        mp = {}
        k = len(machines)
        if k <= 2:
            cand = machines[:]
        else:
            size = random.randint(2, k)
            cand = sorted(random.sample(machines, size))
        for m in cand:
            mp[m] = _rand_distribution(cfg.proc_time_mean_range, cfg.proc_time_std_frac)
        table[t] = mp
    return table

def gen_transfer_time_json(machines: List[str], cfg: GenCfg) -> Dict:
    names = ["GEN"] + machines
    table = {s: {} for s in names}
    # GEN<->Mi
    for m in machines:
        dist = _rand_distribution(cfg.transfer_mean_range, (0.01, 0.05))
        table["GEN"][m] = dist
        table[m]["GEN"] = dist
    # Mi<->Mj
    for i, a in enumerate(machines):
        for b in machines[i+1:]:
            dist = _rand_distribution(cfg.transfer_mean_range, (0.01, 0.1))
            table[a][b] = dist
            table[b][a] = dist
    return table

def sample_op_types_for_job(op_types: List[str], n_ops: int) -> List[str]:
    return [random.choice(op_types) for _ in range(n_ops)]

def gen_jobs_ops_release_json(cfg: GenCfg, op_durations: Dict, due_time_min, due_time_max) -> tuple[list, list, list]:
    op_types = list(op_durations.keys())
    type_to_machines = {t: list(mmap.keys()) for t, mmap in op_durations.items()}

    # --- release_time 배치 생성 ---
    rmin, rmax = int(cfg.release_min), int(cfg.release_max)
    if rmax < rmin:
        rmax = rmin
    batch_size = 10   # 필요시 인자로 조정 가능
    num_batches = (cfg.num_jobs + batch_size - 1) // batch_size

    # 각 배치 release_time 샘플링
    batch_release_times = [int(random.randint(rmin, rmax)) for _ in range(num_batches)]

    # 최소값을 찾아 전체에서 빼줌 → 최소값 배치는 0이 됨
    min_val = min(batch_release_times)
    batch_release_times = [t - min_val for t in batch_release_times]
    # --------------------------------

    jobs, operations, releases = [], [], []
    for i in range(1, cfg.num_jobs + 1):
        job_id, part_id = jid(i), pid(i)
        n_ops = random.randint(cfg.min_ops, cfg.max_ops)
        seq = [random.choice(op_types) for _ in range(n_ops)]
        op_ids = []

        for k, t in enumerate(seq, start=1):
            op_id = oid(i, k)
            op_ids.append(op_id)
            machines = type_to_machines[t]
            operations.append({
                "operation_id": op_id,
                "job_id": job_id,
                "type": t,
                "machines": machines
            })

        jobs.append({
            "job_id": job_id,
            "part_id": part_id,
            "operations": op_ids
        })

        # i번째 job이 속한 배치 인덱스
        batch_idx = (i - 1) // batch_size

        due_time = batch_release_times[batch_idx] + 30 * n_ops +  int(random.randint(due_time_min, due_time_max))

        releases.append({
            "job_id": job_id,
            "release_time": batch_release_times[batch_idx],
            "due_time": due_time
        })

    return jobs, operations, releases


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Full scenario generator (integer times + random machine available_time)")
    ap.add_argument("--out", type=str, default="simulator/scenarios/my_case",
                    help="Output root directory where the scenario folder will be created")
    ap.add_argument("--num_jobs", type=int, default=35)
    ap.add_argument("--min_ops", type=int, default=2)
    ap.add_argument("--max_ops", type=int, default=4)
    ap.add_argument("--num_machines", type=int, default=3)
    ap.add_argument("--op_types", type=str, default="drilling,welding,testing",
                    help="Comma-separated operation type names")
    ap.add_argument("--release_min", type=int, default=0)
    ap.add_argument("--release_max", type=int, default=300)
    ap.add_argument("--machine_avail_min", type=int, default=0,
                    help="Minimum initial available_time for machines (int seconds)")
    ap.add_argument("--machine_avail_max", type=int, default=0,
                    help="Maximum initial available_time for machines (int seconds)")
    ap.add_argument("--seed", type=int, default=42)
    # optional ranges
    ap.add_argument("--proc_mean_min", type=float, default=5.0)
    ap.add_argument("--proc_mean_max", type=float, default=40.0)
    ap.add_argument("--transfer_mean_min", type=float, default=1.0)
    ap.add_argument("--transfer_mean_max", type=float, default=10.0)

    ap.add_argument("--due_time_min", type=int, default=10)
    ap.add_argument("--due_time_max", type=int, default=200,
                    help="If >0, assign random due_time to jobs in [due_time_min, due_time_max]")

    ap.add_argument("--batch_size", type=int, default=10,
                    help="Batch size for job release time sampling (default: 10)")
    args = ap.parse_args()

    cfg = GenCfg(
        num_jobs=args.num_jobs,
        min_ops=args.min_ops,
        max_ops=args.max_ops,
        num_machines=args.num_machines,
        op_types=[s.strip() for s in args.op_types.split(",") if s.strip()],
        release_min=args.release_min,
        release_max=args.release_max,
        machine_avail_min=args.machine_avail_min,
        machine_avail_max=args.machine_avail_max,
        seed=time.time(),
        proc_time_mean_range=(args.proc_mean_min, args.proc_mean_max),
        transfer_mean_range=(args.transfer_mean_min, args.transfer_mean_max),
        due_time_min=args.due_time_min,
        due_time_max=args.due_time_max
    )

    random.seed(cfg.seed)

    # 1) machines.json (with random available_time)
    machine_names = gen_machine_names(cfg.num_machines)
    machines_json = gen_machines_json(machine_names, cfg)

    # 2) operation_durations.json
    if not cfg.op_types:
        cfg.op_types = ["drilling", "welding", "testing"]
    op_durations = gen_operation_durations_json(cfg.op_types, machine_names, cfg)

    # 3) machine_transfer_time.json
    transfer_json = gen_transfer_time_json(machine_names, cfg)

    # 4) jobs.json / operations.json / job_release.json
    jobs_json, operations_json, releases_json = gen_jobs_ops_release_json(cfg, op_durations, due_time_min=args.due_time_min, due_time_max=args.due_time_max)

    # write
    out_root = Path(args.out).resolve()
    scenario_dir = out_root
    ensure_dir(scenario_dir)

    dump_json(scenario_dir / "machines.json", machines_json)
    dump_json(scenario_dir / "operation_durations.json", op_durations)
    dump_json(scenario_dir / "machine_transfer_time.json", transfer_json)
    dump_json(scenario_dir / "jobs.json", jobs_json)
    dump_json(scenario_dir / "operations.json", operations_json)
    dump_json(scenario_dir / "job_release.json", releases_json)

    print(f"[OK] Scenario created at: {scenario_dir}")

if __name__ == "__main__":
    main()
