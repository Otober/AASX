import subprocess
import glob
import os

for i in range(10):
    print(f"=== Iteration {i+1} ===")
    # 1. 시나리오 생성
    subprocess.run([
        "python", "simulator/scenario_generator.py",
        "--num_jobs", "40",
        "--min_ops", "3",
        "--max_ops", "5",
        "--num_machines", "8",
        "--release_min", "0",
        "--release_max", "590",
        "--proc_mean_min", "20",
        "--proc_mean_max", "40",
        "--transfer_mean_min", "5",
        "--transfer_mean_max", "10",
        "--due_time_min", "10",
        "--due_time_max", "200"
    ])

    subprocess.run([
        "python", "-m", "simulator.main",
        "--agv_count", "5",
        "--cnt", str(i+1),
    ])
