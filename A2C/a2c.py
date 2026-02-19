import os
import sys
import math
import time
import argparse
from collections import deque
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # cleaner TF logs


# =========================================================
# Env Interface (replace with your own implementation)
# =========================================================
class EnvInterface:
    """
    최소 요구 인터페이스:
      - reset() -> obs: np.ndarray (shape [obs_dim])
      - step(action: int) -> (obs, reward, done, info)
      - 속성: obs_dim: int, n_actions: int
    """
    obs_dim: int
    n_actions: int

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        raise NotImplementedError


# =========================================================
# Utilities
# =========================================================
def discounted_bootstrap_returns(rewards, dones, last_value, gamma):
    """
    Compute n-step bootstrapped returns G_t backward over a trajectory fragment.

    rewards: [T] float32
    dones:   [T] bool
    last_value: scalar V(s_{T}) for bootstrap
    returns: [T] float32
    """
    T = len(rewards)
    G = np.zeros_like(rewards, dtype=np.float32)
    running = last_value
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running * (1.0 - float(dones[t]))
        G[t] = running
    return G


def entropy_categorical(logits):
    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)
    return -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1))


# =========================================================
# Attention-first Actor-Critic Network
# =========================================================
class AttentionBlock(layers.Layer):
    """
    단순 자기-어텐션 블록:
      - Tokenize scalar features: [B, F] -> [B, F, d_model]
      - MultiHeadAttention (self-attn)
      - Residual + LayerNorm
      - FeedForward (position-wise) + Residual + LayerNorm
    """
    def __init__(self, d_model=64, num_heads=4, ff_mult=2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.embed = layers.Dense(d_model, use_bias=False)  # per-feature projection
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)

        self.ffn = keras.Sequential([
            layers.Dense(d_model * ff_mult, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):  # x: [B, F]
        # [B, F] -> [B, F, d_model]
        tokens = self.embed(tf.expand_dims(x, axis=-1))  # [B, F, 1] -> [B, F, d_model]
        # self-attention: query=key=value=tokens
        attn_out = self.mha(tokens, tokens, tokens)      # [B, F, d_model]
        y = self.ln1(tokens + attn_out)
        ff = self.ffn(y)
        y = self.ln2(y + ff)                             # [B, F, d_model]
        # pool tokens -> [B, d_model]
        pooled = tf.reduce_mean(y, axis=1)
        return pooled


class ActorCritic(keras.Model):
    """
    공유 어텐션 베이스 + 분기 헤드(정책/가치)
    """
    def __init__(self, obs_dim, n_actions,
                 d_model=64, num_heads=4,
                 hidden_sizes=(128,)):
        super().__init__()
        self.input_layer = layers.InputLayer(input_shape=(obs_dim,))
        self.attn = AttentionBlock(d_model=d_model, num_heads=num_heads)

        mlp = []
        for h in hidden_sizes:
            mlp.append(layers.Dense(h, activation="tanh"))
        self.mlp = keras.Sequential(mlp)

        self.policy_logits = layers.Dense(n_actions, activation=None)
        self.value = layers.Dense(1, activation=None)

    def call(self, obs):
        x = tf.convert_to_tensor(obs, dtype=tf.float32)
        x = self.attn(x)           # <-- 첫 레이어를 어텐션 네트워크로
        x = self.mlp(x)
        return self.policy_logits(x), self.value(x)


# =========================================================
# A2C Agent (synchronous)
# =========================================================
class A2C:
    def __init__(self, env: EnvInterface,
                 eval_env: EnvInterface = None,
                 gamma=0.99,
                 n_steps=5,               # rollout length
                 vf_coef=0.5,             # value loss coefficient
                 ent_coef=0.01,           # entropy bonus coefficient
                 lr=3e-4,
                 max_grad_norm=0.5,
                 seed=42,
                 d_model=64,
                 num_heads=4,
                 hidden_sizes=(128,)):
        assert env is not None, "Provide an EnvInterface instance."
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env

        self.gamma = gamma
        self.n_steps = n_steps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.seed = seed

        self.obs_dim = env.obs_dim
        self.n_actions = env.n_actions

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.net = ActorCritic(self.obs_dim, self.n_actions,
                               d_model=d_model, num_heads=num_heads,
                               hidden_sizes=hidden_sizes)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # For logging
        self.global_step = 0
        self.ep_returns = deque(maxlen=100)

    def select_action(self, obs_np):
        obs_tf = tf.convert_to_tensor(obs_np[None, :], dtype=tf.float32)
        logits, value = self.net(obs_tf)
        if hasattr(self.env, "get_action_mask"):
            mask = tf.convert_to_tensor(self.env.get_action_mask()[None, :], dtype=tf.float32)
            neg_inf = tf.constant(-1e9, dtype=logits.dtype)
            logits = tf.where(mask > 0, logits, neg_inf)
        probs = tf.nn.softmax(logits)
        action = tf.random.categorical(tf.math.log(probs), 1)
        return int(action[0, 0].numpy()), float(value[0, 0].numpy())

    def evaluate_value(self, obs_np):
        obs_tf = tf.convert_to_tensor(obs_np[None, :], dtype=tf.float32)
        _, value = self.net(obs_tf)
        return float(value[0, 0].numpy())

    @tf.function
    def train_step(self, obs, actions, returns, advantages):
        """
        One gradient step on a minibatch.
        obs: [B, obs_dim]
        actions: [B] int32
        returns: [B] float32 (bootstrapped targets)
        advantages: [B] float32 (G_t - V(s_t))
        """
        with tf.GradientTape() as tape:
            logits, values = self.net(obs)
            values = tf.squeeze(values, axis=-1)  # [B]

            # Policy loss
            log_probs = tf.nn.log_softmax(logits)
            act_one_hot = tf.one_hot(actions, depth=logits.shape[-1], dtype=tf.float32)
            log_pi_a = tf.reduce_sum(act_one_hot * log_probs, axis=-1)  # [B]
            policy_loss = -tf.reduce_mean(log_pi_a * tf.stop_gradient(advantages))

            # Value loss (MSE)
            value_loss = tf.reduce_mean(tf.square(returns - values))

            # Entropy bonus (maximize entropy => minimize -entropy)
            ent = entropy_categorical(logits)
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent

        grads = tape.gradient(loss, self.net.trainable_variables)
        # Clip global norm
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
        return policy_loss, value_loss, ent, loss

    def rollout_n_steps(self, env: EnvInterface, start_obs):
        """
        Collect a trajectory fragment of length <= n_steps (stops early on terminal).
        Returns: dict with obs, actions, rewards, dones, last_obs, last_value
        """
        obs_list, act_list, rew_list, done_list = [], [], [], []
        obs = start_obs
        ep_return = 0.0
        for _ in range(self.n_steps):
            action, _ = self.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            obs_list.append(obs.copy())
            act_list.append(action)
            rew_list.append(float(reward))
            done_list.append(bool(done))
            ep_return += float(reward)

            obs = next_obs
            self.global_step += 1

            if done:
                obs = env.reset()
                self.ep_returns.append(ep_return)
                ep_return = 0.0

        # Bootstrap from last state
        last_value = 0.0 if (done_list and done_list[-1]) else self.evaluate_value(obs)

        return {
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(act_list, dtype=np.int32),
            "rewards": np.array(rew_list, dtype=np.float32),
            "dones": np.array(done_list, dtype=np.bool_),
            "last_obs": obs,
            "last_value": float(last_value),
        }

    def train(self, total_steps=200_000, batch_updates_per_rollout=1,
              log_interval=1000, checkpoint_dir="checkpoints/a2c_nogym"):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initial reset
        obs = self.env.reset()

        last_log_step = 0
        best_avg = -1e9
        ckpt = tf.train.Checkpoint(model=self.net, optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

        while self.global_step < total_steps:
            # Collect n-step rollout
            traj = self.rollout_n_steps(self.env, obs)
            obs = traj["last_obs"]

            # Compute returns and advantages
            returns = discounted_bootstrap_returns(traj["rewards"], traj["dones"], traj["last_value"], self.gamma)

            obs_tf = tf.convert_to_tensor(traj["obs"], dtype=tf.float32)
            logits, values_tf = self.net(obs_tf)
            values = values_tf.numpy().squeeze(-1).astype(np.float32)
            advantages = returns - values

            if advantages.shape[0] >= 2:
                adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
                advantages = (advantages - adv_mean) / adv_std

            # Train step
            policy_loss, value_loss, ent, total_loss = self.train_step(
                tf.convert_to_tensor(traj["obs"], dtype=tf.float32),
                tf.convert_to_tensor(traj["actions"], dtype=tf.int32),
                tf.convert_to_tensor(returns, dtype=tf.float32),
                tf.convert_to_tensor(advantages, dtype=tf.float32),
            )

            # Logging
            if self.global_step - last_log_step >= log_interval:
                last_log_step = self.global_step
                avg_return = np.mean(self.ep_returns) if self.ep_returns else float("nan")
                print(f"[step {self.global_step:7d}] "
                      f"avg_return={avg_return:7.2f}  "
                      f"loss={float(total_loss):.4f}  "
                      f"pi={float(policy_loss):.4f}  "
                      f"vf={float(value_loss):.4f}  "
                      f"ent={float(ent):.4f}")
                # Save best
                if not np.isnan(avg_return) and avg_return > best_avg:
                    best_avg = avg_return
                    manager.save()

        # Final save
        manager.save()
        print("Training complete. Checkpoints at:", checkpoint_dir)

    def evaluate(self, episodes=10, checkpoint=None, render=False):
        if checkpoint:
            ckpt = tf.train.Checkpoint(model=self.net)
            ckpt.restore(tf.train.latest_checkpoint(checkpoint)).expect_partial()
            print("Loaded checkpoint:", tf.train.latest_checkpoint(checkpoint))

        returns = []
        for ep in range(episodes):
            obs = self.eval_env.reset()
            done = False
            total_r = 0.0
            while not done:
                obs_tf = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
                logits, v = self.net(obs_tf)
                if hasattr(self.eval_env, "get_action_mask"):
                    mask = tf.convert_to_tensor(self.eval_env.get_action_mask()[None, :], dtype=tf.float32)
                    neg_inf = tf.constant(-1e9, dtype=logits.dtype)
                    logits = tf.where(mask > 0, logits, neg_inf)
                probs = tf.nn.softmax(logits)
                action = tf.argmax(probs, axis=-1)[0].numpy()
                obs, reward, done, info = self.eval_env.step(int(action))
                total_r += float(reward)
                if render and hasattr(self.eval_env, "render"):
                    self.eval_env.render()
            returns.append(total_r)
            print(f"Episode {ep+1}: return = {total_r:.2f}")
        print(f"Avg return over {episodes} episodes: {np.mean(returns):.2f}")


# =========================================================
# Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="A2C (TensorFlow 2.x) without Gym")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--episodes", type=int, default=10, help="eval episodes")
    p.add_argument("--checkpoint", type=str, default="checkpoints/a2c_nogym")
    # Attention/architecture
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--hidden", type=int, nargs="*", default=[128])
    return p.parse_args()


# ----- Example stub Env (you must replace with your own) -----
class DummyEnv(EnvInterface):
    """
    예시용 더미 환경: 선형-가우시안 천장 리워드. 실제로는 교체하세요.
    """
    def __init__(self, obs_dim=4, n_actions=2, seed=42):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)
        self._obs = None

    def reset(self):
        self._obs = self.rng.randn(self.obs_dim).astype(np.float32)
        return self._obs

    def step(self, action: int):
        # 간단한 동역학
        self._obs = (0.9 * self._obs + 0.1 * self.rng.randn(self.obs_dim)).astype(np.float32)
        # 임의의 보상(행동이 0일 때 보상 조금 더 줌)
        reward = float(1.0 if action == 0 else 0.9)
        done = bool(self.rng.rand() < 0.01)  # 1% 종료
        info = {}
        if done:
            self._obs = self.reset()
        return self._obs, reward, done, info
# -------------------------------------------------------------


def main():
    args = parse_args()

    # TODO: 여기에 실제 환경을 연결하세요.
    # 예: env = YourCpsEnv(...)
    env = DummyEnv(obs_dim=4, n_actions=2, seed=42)
    eval_env = DummyEnv(obs_dim=4, n_actions=2, seed=43)

    agent = A2C(env=env, eval_env=eval_env,
                gamma=args.gamma,
                n_steps=args.n_steps,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                lr=args.lr,
                max_grad_norm=args.max_grad_norm,
                seed=42,
                d_model=args.d_model,
                num_heads=args.num_heads,
                hidden_sizes=tuple(args.hidden))

    if args.train:
        agent.train(total_steps=args.total_steps,
                    log_interval=args.log_interval,
                    checkpoint_dir=args.checkpoint)

    if args.eval:
        agent.evaluate(episodes=args.episodes,
                       checkpoint=args.checkpoint,
                       render=False)


if __name__ == "__main__":
    main()
