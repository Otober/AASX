# train_bandit.py
from scenario_bandit_env import ScenarioBanditEnv
from a2c import A2C

ROOT = "/home/doyoung/Desktop/Work/AASX/results"

if __name__ == "__main__":
    env = ScenarioBanditEnv(
        root_dir=ROOT,
        scenarios=["1", "2","3","4", "5", "6"],                  # ["1","2","3"]로 고정 지정 가능
        releases=None,                   # 기본: 각 시나리오에서 0_000 제외하고 모두 사용
        alpha=1.0, beta=0.001, gamma=0.0,
        minimize=True,
        shuffle=False,
        nearest_policy="nearest"         # "floor" 또는 "ceil"로도 사용 가능
    )

    agent = A2C(env=env, eval_env=env,
                n_steps=1, gamma=0.0, lr=3e-4,
                d_model=64, num_heads=4, hidden_sizes=(128,))
    agent.train(total_steps=10000, log_interval=1000, checkpoint_dir="checkpoints/a2c_bandit")
    agent.evaluate(episodes=10, checkpoint="checkpoints/a2c_bandit")
