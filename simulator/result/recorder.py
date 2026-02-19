# simulator/result/recorder.py

import os
import pandas as pd

class Recorder:
    records = []
    enabled = True   # trace on/off
    outdir = 'results_reinforecement'
    basedir = ''
    run_meta = {}    # ← 오타 수정

    @classmethod
    def reset(cls):
        cls.records.clear()

    @classmethod
    def _on(cls):
        return bool(getattr(cls, "enabled", True))

    @classmethod
    def log_queue(cls, part, machine, time, operation_id, queue_length, queue_ops):
        if not cls._on(): return
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': operation_id,
            'machine': machine,
            'event': 'queued',
            'time': time,
            'queue_length': queue_length,
            'queue_ops': ','.join(queue_ops),
            'run_meta': cls.run_meta    # ← run_meta로 통일
        })

    @classmethod
    def log_start(cls, part, machine, time, operation_id, queue_length):
        if not cls._on(): return
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': operation_id,
            'machine': machine,
            'event': 'start',
            'time': time,
            'queue_length': queue_length,
            'queue_ops': None,
            'run_meta': cls.run_meta
        })

    @classmethod
    def log_end(cls, part, machine, time, operation_id):
        if not cls._on(): return
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': operation_id,
            'machine': machine,
            'event': 'end',
            'time': time,
            'queue_length': None,
            'queue_ops': None,
            'run_meta': cls.run_meta
        })

    @classmethod
    def log_fetch(cls, part, src_machine, dest_machine, time, delay):
        if not cls._on(): return
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': part.job.current_op().id if part.job.current_op() else None,
            'machine': f"{src_machine}->{dest_machine}",
            'event': 'fetch',
            'time': time,
            'delay': delay,
            'queue_length': None,
            'queue_ops': None,
            'run_meta': cls.run_meta
        })

    @classmethod
    def log_delivery(cls, agv_id, part, src_machine, dest_machine, time, delay):
        if not cls._on(): return
        cls.records.append({
            'part': agv_id,
            'job': part.job.id,
            'operation': part.job.current_op().id if part.job.current_op() else None,
            'machine': f"{src_machine}->{dest_machine}",
            'event': 'delivery',
            'time': time,
            'delay': delay,
            'queue_length': None,
            'queue_ops': None,
            'run_meta': cls.run_meta
        })

    @classmethod
    def log_done(cls, part, time):
        if not cls._on(): return
        cls.records.append({
            'part': part.id,
            'job': part.job.id,
            'operation': None,
            'machine': None,
            'event': 'done',
            'time': time,
            'queue_length': None,
            'queue_ops': None,
            'run_meta': cls.run_meta
        })
    @classmethod
    def log_fetch(cls, agv_id, part, src_machine, dest_machine, time, delay):
        if not cls._on(): return
        cls.records.append({
            'part': agv_id,
            'job': part.job.id,
            'operation': part.job.current_op().id if part.job.current_op() else None,
            'machine': f"{src_machine}->{dest_machine}",
            'event': 'fetch',
            'time': time,
            'delay': delay,
            'queue_length': None,
            'queue_ops': None,
            'run_meta': cls.run_meta
        })

    @classmethod
    def set_base_dir(cls, basedir: str):
        cls.basedir = basedir

    @classmethod
    def set_dir(cls, outdir: str):
        cls.outdir = outdir

    @classmethod
    def set_meta(cls, **kwargs):     # ← 인덴트 통일
        cls.run_meta = dict(kwargs)

    @classmethod
    def save(cls):
        full_dir = os.path.join(cls.basedir, cls.outdir)
        os.makedirs(full_dir, exist_ok=True)
        df = pd.DataFrame(cls.records)
        df.to_csv(os.path.join(full_dir, 'trace.csv'), index=False)
        try:
            df.to_excel(os.path.join(full_dir, 'trace.xlsx'), index=False)
        except ImportError:
            pass
