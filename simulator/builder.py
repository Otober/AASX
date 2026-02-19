# --- simulator/builder.py ---# --- simulator/builder.py ---
import os, json
from simulator.domain.domain import Job, Operation
from simulator.model.machine import Machine
from simulator.model.generator import Generator
from simulator.model.transducer import Transducer
from simulator.control.agv_controller import AGVController
from simulator.model.agv import AGV
from simulator.model.source_station import SourceStation

def load(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        return json.load(f)

class ModelBuilder:
    def __init__(self, subpath, use_dynamic_scheduling=False, agv_count = 1):
        # 절대 경로로 변환
        if os.path.isabs(subpath):
            self.path = subpath
        else:
            base = os.path.dirname(__file__)
            self.path = os.path.join(base, subpath)
        
        self.use_dynamic_scheduling = use_dynamic_scheduling
        self.agv_count = agv_count

    def build(self):
        jobs_j   = load(os.path.join(self.path, 'jobs.json'))
        ops_j    = load(os.path.join(self.path, 'operations.json'))
        dur_j    = load(os.path.join(self.path, 'operation_durations.json'))
        trans    = load(os.path.join(self.path, 'machine_transfer_time.json'))
        # machines.json이 있으면 사용, 없으면 initial_machine_status.json 사용
        machines_file = os.path.join(self.path, 'machines.json')
        
        init_m = load(machines_file)
        releases = load(os.path.join(self.path, 'job_release.json'))
        op_map    = {o['operation_id']: o for o in ops_j}
        
        # 동적 라우팅 사용 (정적 라우팅 제거)
        route_map = {}  # 빈 맵으로 초기화 (동적 할당)

        # release_time 매핑 생성
        release_map = {r['job_id']: r['release_time'] for r in releases}
        due_map = {r['job_id']: r.get('due_time', None) for r in releases}
        
        # 디버깅 정보 제거
        
        jobs = {}
        for j in jobs_j:
            ops = []
            for oid in j['operations']:
                om   = op_map[oid]
                
                # 동적 라우팅 사용 (assigned_machine을 None으로 설정)
                assigned_machine = None  # 동적 할당을 위해 None으로 설정
                
                type_map = dur_j.get(om['type'], {})
                spec_map = {m: type_map.get(m, {}) for m in (om['machines'] or [])}
                
                # Operation에 라우팅된 기계와 후보 리스트, 분포를 전달
                ops.append(Operation(
                    op_id=oid,
                    assigned_machine=assigned_machine,  # 동적 모드에서는 None
                    candidate_machines=om['machines'],
                    distribution=spec_map
                ))
            
            # Job 생성 시 release_time 포함
            job_release_time = release_map.get(j['job_id'], 0.0)
            job_due_time = due_map.get(j['job_id'], 0.0)
            jobs[j['job_id']] = Job(j['job_id'], j['part_id'], ops, job_release_time, job_due_time)
            print(f"[ModelBuilder] Job {j['job_id']} 생성: {len(ops)}개의 Operation, Release Time: {job_release_time}, Due Time: {job_due_time}")

        src = SourceStation()


        machines = []
        for mname, info in init_m.items():
            # 각 머신별 transfer map만 전달
            machine_transfer = trans.get(mname, {})
            machines.append(Machine(mname, machine_transfer, info))

        # AGV 로거 생성 (머신별 AGV에 설정)
        from simulator.control.agv_logger import AGVLogger
        agv_logger = AGVLogger()
        
        # 모든 머신의 AGV에 로거 설정
        for machine in machines:
            if hasattr(machine, 'set_logger'):
                machine.set_logger(agv_logger)

        agvs = []
        for i in range(self.agv_count):
            agv = AGV(str(i+1))
            # ✅ transfer map 세팅
            agv.set_transfer_times(trans)   # trans는 builder에서 이미 로딩한 전체 transfer dict
            agvs.append(agv)
        agv_controller = AGVController(agvs)

        gen = Generator(releases, jobs, optimize_on_release=True, optimizer_model='OptimizationManager', epsilon=1e-6)
        tx  = Transducer()
        return machines, gen, tx, agv_controller, agvs, src, trans