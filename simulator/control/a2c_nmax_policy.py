# a2c_nmax_policy.py
import numpy as np
from typing import List
from A2C.a2c import A2C
# A2C가 요구하는 최소 EnvInterface 껍데기
class _DummyEnv:
    obs_dim = 15      # bandit에서 썼던 15D
    n_actions = 6     # {4,8,12,16,20,24}

    def reset(self): raise NotImplementedError
    def step(self, a): raise NotImplementedError

# A2C 정책 → n_max 값으로 매핑하는 래퍼
class RLNmaxPolicy:
    ACTIONS = [4, 8, 12, 16, 20, 24]

    def __init__(self, checkpoint_dir: str):
        env = _DummyEnv()
        self.agent = A2C(env=env, eval_env=env, n_steps=1, gamma=0.0, lr=3e-4,
                         d_model=64, num_heads=4, hidden_sizes=(128,))
        # 체크포인트 로드 (최신)
        import tensorflow as tf
        ckpt = tf.train.Checkpoint(model=self.agent.net)
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        print(f"[RLNmaxPolicy] loaded checkpoint: {tf.train.latest_checkpoint(checkpoint_dir)}")

    def _obs_from_sim(self, sim) -> np.ndarray:
        """
        간단 15D 요약:
          [ op_progress, agv_util, mach_util,
            sum_q_min, sum_q_mean, sum_q_max,
            rem_min, rem_mean, rem_max,
            outq_min, outq_mean, outq_max,
            cnt_delivery, cnt_fetch, cnt_idle ]
        """
        # 머신 큐 총 처리량(간이): 각 큐에 쌓인 job의 현재 op 평균처리시간 합
        def _op_mean(op):
            d = getattr(op, 'distribution', {}) or {}
            if 'mean' in d: return float(d['mean'])
            if 'low' in d and 'high' in d: return (float(d['low'])+float(d['high']))/2.0
            if 'rate' in d and float(d['rate'])>0: return 1.0/float(d['rate'])
            dur = getattr(op, 'duration', None)
            try: return float(dur) if dur is not None else 0.0
            except: return 0.0

        sum_q = []
        rems  = []
        outq  = []
        busy  = 0
        total_m = len(sim.machines)
        for m in sim.machines:
            # 입력큐 총합
            s = 0.0
            for part in getattr(m, 'queue', []):
                op = part.job.current_op() if hasattr(part, 'job') and hasattr(part.job, 'current_op') else None
                if op: s += _op_mean(op)
            sum_q.append(s)

            # 남은 처리시간(간이): running job이 있으면 그 job의 남은시간 추정치로 1개 넣기
            if getattr(m, 'running_jobs', []):
                busy += 1
                # 추정이 어려우면 1개당 평균처리시간으로 대체
                r = _op_mean(m.running_jobs[0].current_op()) if hasattr(m.running_jobs[0], 'current_op') else 0.0
                rems.append(r)
            else:
                rems.append(0.0)

            # outq 대기수
            outq.append(float(len(getattr(m, 'waiting_for_pickup', []))))

        def _stats(a):
            if not a: return (0.0, 0.0, 0.0)
            arr = np.asarray(a, dtype=np.float32)
            return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))

        s_min, s_mean, s_max = _stats(sum_q)
        r_min, r_mean, r_max = _stats(rems)
        o_min, o_mean, o_max = _stats(outq)

        # AGV 상태 카운트/가동율
        cnt_delivery = cnt_fetch = cnt_idle = 0
        agv_ctrl = sim.models.get('AGVController')
        if agv_ctrl and hasattr(agv_ctrl, 'agvs'):
            for agv in agv_ctrl.agvs.values():
                st = str(getattr(agv, 'status', 'idle'))
                if st == 'delivery': cnt_delivery += 1
                elif st == 'fetch':  cnt_fetch    += 1
                else:                cnt_idle     += 1
        total_agv = max(1, cnt_delivery + cnt_fetch + cnt_idle)
        agv_util = (cnt_delivery + cnt_fetch) / total_agv

        # 머신 가동율
        mach_util = (busy / max(1, total_m))

        # 공정 진행률(간이): 완료 job / 총 job
        gen = sim.models.get('generator')
        total_jobs = len(getattr(gen, '_jobs_by_id', {})) if gen else 0
        comp_jobs = 0
        for m in sim.machines:
            comp_jobs += len(getattr(m, 'finished_jobs', []))
        op_progress = (comp_jobs / max(1, total_jobs))

        obs = np.asarray([
            float(op_progress), float(agv_util), float(mach_util),
            s_min, s_mean, s_max,
            r_min, r_mean, r_max,
            o_min, o_mean, o_max,
            float(cnt_delivery), float(cnt_fetch), float(cnt_idle)
        ], dtype=np.float32)
        assert obs.shape[0] == 15
        return obs

    def pick_nmax(self, sim, all_ops: List[str]) -> int:
        obs = self._obs_from_sim(sim)
        action_idx, _ = self.agent.select_action(obs)
        return self.ACTIONS[action_idx]
