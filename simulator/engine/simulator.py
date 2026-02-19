# --- simulator/engine/simulator.py ---
import heapq
import copy
import random
from enum import Enum, auto
from collections import namedtuple

from simulator import model

class DecisionEpoch(Enum):
    MACHINE_IDLE = auto()
    JOB_RELEASE = auto()
    OPERATION_COMPLETE = auto()

class Action:
    def __init__(self, operation_id, machine_id, insert_position=None):
        self.operation_id = operation_id
        self.machine_id = machine_id
        self.insert_position = insert_position  # 기계 내 삽입 위치 (None이면 큐 끝)
    
    def __repr__(self):
        return f"Action({self.operation_id} -> {self.machine_id}, pos={self.insert_position})"

class SimulatorState:
    def __init__(self, current_time, event_queue, models_state, machines_state, rng_state, running_time, start_time):
        self.current_time = current_time
        self.event_queue = event_queue
        self.models_state = models_state
        self.machines_state = machines_state
        self.rng_state = rng_state
        self.running_time = running_time
        self.start_time = start_time

class Event:
    __slots__ = ('time','event_type','payload','src_model','dest_model','priority')

    def __init__(self, event_type, payload=None, dest_model=None, time=0.0, priority=100):
        self.time = int(round(time))  # 이벤트 시간은 정수로 처리
        self.event_type = event_type
        self.payload = payload or {}
        self.src_model = None
        self.dest_model = dest_model
        self.priority = priority

    def __lt__(self, other):
        if self.time == other.time:
            return self.priority < other.priority
        return self.time < other.time

    def set_src(self, name):
        self.src_model = name

    def set_time(self, t):
        self.time = t

    def __repr__(self):
        return f"Event(time={self.time:.2f}, type={self.event_type}, from={self.src_model}, to={self.dest_model}, payload={self.payload})"

class EoModel:
    push_event = None
    get_time = None

    @classmethod
    def bind(cls, push_fn, time_fn):
        cls.push_event = push_fn
        cls.get_time = time_fn

    def __init__(self, name):
        self.name = name

    def schedule(self, event, delay=0.0):
        if not EoModel.push_event:
            raise RuntimeError("Simulator not bound")
        event.set_src(self.name)
        event.set_time(EoModel.get_time() + delay)
        EoModel.push_event(event)

    def handle_event(self, event):
        raise NotImplementedError
    
    def save_state(self):
        """모델별 직렬화 가능한 dict 반환 (객체 참조는 ID로 치환)"""
        return {}

    def load_state(self, state_dict, registry):
        """저장된 dict 기반으로 상태 복원 (registry로 ID→객체 매핑)"""
        return

class Simulator:
    def __init__(self):
        self.current_time = 0.0
        self.event_queue = []
        self.models = {}
        self.machines = []  # 기계 목록 저장
        self.agvs = []      # AGV 목록 저장
        self.decision_epochs = []  # 결정 시점들
        self.best_objective = float('inf')
        self.best_schedule = None
        self.start_time = None
        self.running_time = 0.0
        self.optimization_time = 0.0

        self.test_type = set()

        EoModel.bind(self.push, self.now)

    def push(self, event):
        event.time = int(round(event.time))  # 이벤트 시간은 정수로 처리
        self.test_type.add(event.event_type)
        if event.event_type == 'handle_delivery_request' or event.event_type == 'agv_fetch_request':
            event.priority = 150
        elif event.event_type == 'optimize_now' : 
            event.priority = 120 
        heapq.heappush(self.event_queue, event)
        print("here")

    def now(self):
        return self.current_time

    def register(self, model):
        model.simulator = self
        self.models[model.name] = model
        # 기계 모델인 경우 별도로 저장
        if hasattr(model, 'queued_jobs'):
            self.machines.append(model)
        if hasattr(model, "carried_jobs"):
            self.agvs.append(model)


    # RNG, 이벤트 큐, 모델 상태를 포함한 시뮬레이터 상태 스냅샷
    def snapshot(self):
        rng_state = random.getstate()
        event_queue_copy = []
        for ev in self.event_queue:
            event_copy = Event(ev.event_type,
                            self._encode_payload(ev.payload),
                            ev.dest_model,
                            ev.time)
            event_copy.src_model = ev.src_model
            event_queue_copy.append(event_copy)
        models_state = {}
        for name, model in self.models.items():
            if hasattr(model, 'save_state'):
                models_state[name] = model.save_state()
        return SimulatorState(self.current_time, event_queue_copy, models_state, {}, rng_state, self.running_time, self.start_time)

    # 스냅샷 상태로 복원
    def restore(self, state: SimulatorState):
        # 1) 시간/난수 상태 복원
        self.current_time = state.current_time
        random.setstate(state.rng_state)
        self.running_time = state.running_time
        self.start_time = state.start_time

        # 2) 먼저 기본 레지스트리를 한 번 만든다 (generator/머신 큐/AGV carried까지)
        base_reg = self._build_registry()

        # 3) 스냅샷(models_state) 안의 current_task에서 필요한 ID를 추출하여 base_reg에 미리 주입
        ct_part_ids = set()
        ct_job_ids  = set()
        for name, st in state.models_state.items():
            ct = st.get('current_task')
            if isinstance(ct, dict) and ct:
                if isinstance(ct.get('part'), dict) and 'id' in ct['part']:
                    ct_part_ids.add(ct['part']['id'])
                if isinstance(ct.get('job'), dict) and 'id' in ct['job']:
                    ct_job_ids.add(ct['job']['id'])

        gen = self.models.get('generator')
        if gen is not None:
            # generator의 맵에서 빠진 객체를 보강
            for pid in ct_part_ids:
                if pid not in base_reg['parts'] and hasattr(gen, '_parts_by_id') and pid in gen._parts_by_id:
                    base_reg['parts'][pid] = gen._parts_by_id[pid]
            for jid in ct_job_ids:
                if jid not in base_reg['jobs'] and hasattr(gen, '_jobs_by_id') and jid in gen._jobs_by_id:
                    base_reg['jobs'][jid] = gen._jobs_by_id[jid]

        # 4) 이제 모든 모델에 대해 동일한 base_reg로 load_state를 수행 (모델별로 reg를 새로 만들지 말 것!)
        for name, model in self.models.items():
            if hasattr(model, 'load_state'):
                model.load_state(state.models_state.get(name, {}), base_reg)

        # 5) 모델들이 살아났으니 최종 reg를 다시 만들어 이벤트 payload를 디코드
        final_reg = self._build_registry()
        self.event_queue = []
        for ev in state.event_queue:
            payload = self._decode_payload(ev.payload, final_reg)
            ev2 = Event(ev.event_type, payload, ev.dest_model, ev.time)
            ev2.src_model = ev.src_model
            heapq.heappush(self.event_queue, ev2)


    def _build_registry(self):
        reg = {
            'machines': {m.name: m for m in self.machines},
            'agvs': {m.name : m for m in self.agvs},
            'jobs': {},
            'parts': {}
        }
        gen = self.models.get('generator')
        if gen and hasattr(gen, '_jobs_by_id'):
            reg['jobs'].update({jid: job for jid, job in gen._jobs_by_id.items()})
        if gen and hasattr(gen, '_parts_by_id'):
            reg['parts'].update({pid: part for pid, part in gen._parts_by_id.items()})
        # 1) AGVController 내부의 AGV들부터 등록
        agv_ctrl = self.models.get('AGVController')
        if agv_ctrl and hasattr(agv_ctrl, 'agvs'):
            for agv_id, agv in agv_ctrl.agvs.items():
                reg['agvs'][str(agv_id)] = agv  # 키를 문자열로 정규화(지문도 문자열 키 사용)
        
        # 2) 모델들 순회 (혹시 개별 AGV가 모델로 등록돼 있으면 추가)
        for model in self.models.values():
            if hasattr(model, 'agv_id'):
                reg['agvs'][str(model.agv_id)] = model

            # 머신에서 jobs/parts 수집
            if hasattr(model, 'queued_jobs'):
                for q in (list(model.queued_jobs) +
                        list(getattr(model, 'running_jobs', [])) +
                        list(getattr(model, 'finished_jobs', []))):
                    reg['jobs'][q.id] = q

            if hasattr(model, 'queue'):  # parts 큐
                for p in list(model.queue):
                    reg['parts'][p.id] = p
            if hasattr(model, 'waiting_for_pickup'):
                for p in list(model.waiting_for_pickup):
                    reg['parts'][p.id] = p
                    if hasattr(p, 'job'):
                        reg['jobs'][p.job.id] = p.job

            # AGV가 들고 있는 파트 (개별 모델이 AGV일 수도 있으니 체크)
            if hasattr(model, 'carried_jobs'):
                for part in getattr(model, 'carried_jobs', []):
                    reg['parts'][part.id] = part
                    if hasattr(part, 'job'):
                        reg['jobs'][part.job.id] = part.job

            ct = getattr(model, 'current_task', None)
            if isinstance(ct, dict) and ct:
                part = ct.get('part')
                job  = ct.get('job') or (getattr(part, 'job', None) if part is not None else None)
                if part is not None and hasattr(part, 'id'):
                    reg['parts'][part.id] = part
                if job is not None and hasattr(job, 'id'):
                    reg['jobs'][job.id] = job
        return reg

    # 이벤트에 관련된 정보를 저장하는 함수, snapshot에 포함
    def _encode_payload(self, payload):
    # 객체 → ID로 치환
        def enc(x):
            from simulator.domain.domain import Job, Part
            if hasattr(x, 'id') and hasattr(x, '__class__'):
                # Part/Job은 id 사용
                if x.__class__.__name__ == 'Part': return {'__type__': 'part', 'id': x.id}
                if x.__class__.__name__ == 'Job':  return {'__type__': 'job',  'id': x.id}
            return x
        if isinstance(payload, dict):
            return {k: enc(v) for k,v in payload.items()}
        return payload

    # 이벤트에 관련된 정보를 복원하는 함수, restore에서 사용
    def _decode_payload(self, payload, reg):
        def dec(x):
            if isinstance(x, dict) and '__type__' in x:
                if x['__type__'] == 'part': return reg['parts'].get(x['id'])
                if x['__type__'] == 'job':  return reg['jobs'].get(x['id'])
            return x
        if isinstance(payload, dict):
            return {k: dec(v) for k,v in payload.items()}
        return payload


    def is_terminal(self):
        """모든 작업이 완료되었는지 확인합니다."""
        for machine in self.machines:
            if machine.queued_jobs or machine.running_jobs:
                return False
        return True

    # 목적함수 여러 개 수정 필요 
    def objective_multi(self):
        """(makespan, agv_total_travel_time) 튜플 반환"""
        # makespan 계산은 기존 objective()와 동일
        all_jobs_completed = True
        for machine in self.machines:
            if machine.queued_jobs or machine.running_jobs:
                all_jobs_completed = False
                break

        due_date_violation = 0.0
        if not all_jobs_completed:
            makespan = float('inf')
        else:
            max_completion_time = 0.0
            for machine in self.machines:
                for job in machine.finished_jobs:
                    if hasattr(job, 'completion_time'):
                        max_completion_time = max(max_completion_time, job.completion_time)
                    if hasattr(job, 'due_time') :
                        due_date_violation += max(0.0, job.completion_time - job.due_time)
            makespan = max_completion_time if max_completion_time > 0 else self.current_time

        # AGV 총 이동시간 합산
        agv_total = 0.0
        agv_ctrl = self.models.get('AGVController')
        if agv_ctrl:
            for agv in agv_ctrl.agvs.values():
                agv_total += float(getattr(agv, 'total_distance', -10.0))

        return makespan, agv_total, due_date_violation


# 실행 관련 코드

    def _pop_next_event(self):
        """이벤트 큐에서 다음 이벤트를 하나 꺼내고 현재시간을 갱신한다."""
        if not self.event_queue:
            return None
        evt = heapq.heappop(self.event_queue)   # event_queue는 heapq 로 관리됨
        self.current_time = getattr(evt, "time", self.current_time)
        return evt

    def step_events(self, max_events: int = 100,
                    print_queues_interval: float | None = None,
                    print_job_summary_interval: float | None = None) -> int:
        """
        시뮬레이터를 '부분 실행'한다: 최대 max_events개의 이벤트만 처리하고 멈춘다.
        통합 루프의 무작위 개입 지점을 만들기 위해 사용.
        반환값: 실제 처리한 이벤트 개수
        """
        processed = 0
        last_print_time = getattr(self, "_last_print_time_step", 0.0)
        last_summary_time = getattr(self, "_last_summary_time_step", 0.0)

        while processed < max_events and self.event_queue:
            if not self.event_queue:
                break

            evt = self._pop_next_event()
            if evt is None:
                break

            # (옵션) 주기적 상태 출력
            if print_queues_interval and (self.current_time - last_print_time) >= print_queues_interval:
                self.print_machine_queues()
                last_print_time = self.current_time

            if print_job_summary_interval and (self.current_time - last_summary_time) >= print_job_summary_interval:
                self.print_job_status_summary()
                last_summary_time = self.current_time

            m = self.models.get(evt.dest_model)
            if not m:
                # 원래 run()은 KeyError를 내지만, 부분 실행에서는 그냥 스킵해도 됨
                # 필요하면 raise로 바꿔도 OK
                continue
            m.handle_event(evt)
            processed += 1

        # 다음 step 호출에도 간격 계산이 이어지도록 내부 상태 저장
        self._last_print_time_step = last_print_time
        self._last_summary_time_step = last_summary_time
        return processed

    def run(self, print_queues_interval=None, print_job_summary_interval=None):
        """
        시뮬레이션 실행
        :param print_queues_interval: 큐 상태를 출력할 시간 간격 (초)
        :param print_job_summary_interval: Job 상태 요약을 출력할 시간 간격 (초)
        """
        last_print_time = 0.0
        last_summary_time = 0.0

        while self.event_queue:
            evt = heapq.heappop(self.event_queue)
            self.current_time = evt.time
            
            # 주기적으로 큐 상태 출력
            if print_queues_interval and self.current_time - last_print_time >= print_queues_interval:
                self.print_machine_queues()
                last_print_time = self.current_time
            
            # 주기적으로 Job 상태 요약 출력
            if print_job_summary_interval and self.current_time - last_summary_time >= print_job_summary_interval:
                self.print_job_status_summary()
                last_summary_time = self.current_time
            
            m = self.models.get(evt.dest_model)
            if not m:
                raise KeyError(f"No model: {evt.dest_model}")
            m.handle_event(evt)
# 검증
    def validate_mathematical_constraints(self):
        """수학적 검증식을 검증합니다."""
        violations = []
        
        # 1) 공정(머신) 쪽 기본 검증
        violations.extend(self._validate_machine_constraints())
        
        return violations

    def print_constraint_violations(self):
        """제약 조건 위반 사항을 출력합니다."""
        violations = self.validate_mathematical_constraints()
        
        if not violations:
            print("✅ 모든 수학적 제약 조건을 만족합니다!")
            return
        
        print(f"\n❌ {len(violations)}개의 제약 조건 위반이 발견되었습니다:")
        print("=" * 80)
        
        for i, violation in enumerate(violations, 1):
            print(f"{i}. {violation['type']}")
            for key, value in violation.items():
                if key != 'type':
                    print(f"   {key}: {value}")
            print()


# 상태 출력 
    def print_machine_queues(self):
        """모든 기계의 큐 상태를 출력"""
        if not self.machines:
            return
            
        print(f"\n=== 시뮬레이션 시간: {self.current_time:.2f} ===")
        for machine in self.machines:
            machine.get_queue_status()

    def get_all_job_status(self):
        """모든 기계에서 관리하는 Job들의 상태를 수집하여 반환합니다."""
        all_jobs = {
            'simulation_time': self.current_time,
            'machines': {}
        }
        
        for machine in self.machines:
            machine_summary = machine.get_job_status_summary()
            all_jobs['machines'][machine.name] = machine_summary
            
        return all_jobs
    
    def print_job_status_summary(self):
        """모든 Job의 상태 요약을 출력합니다."""
        all_jobs = self.get_all_job_status()
        
        print(f"\n=== 전체 Job 상태 요약 (시뮬레이션 시간: {self.current_time:.2f}) ===")
        
        total_queued = 0
        total_running = 0
        total_finished = 0
        
        for machine_name, machine_data in all_jobs['machines'].items():
            queued_count = len(machine_data['queued_jobs'])
            running_count = len(machine_data['running_jobs'])
            finished_count = len(machine_data['finished_jobs'])
            
            total_queued += queued_count
            total_running += running_count
            total_finished += finished_count
            
            print(f"\n{machine_name}:")
            print(f"  대기 중: {queued_count}, 실행 중: {running_count}, 완료: {finished_count}")
            
            if running_count > 0:
                print("  실행 중인 Job들:")
                for job in machine_data['running_jobs']:
                    print(f"    - Job {job['job_id']} (Part {job['part_id']}): {job['current_operation']} - 진행률 {job['progress']:.2f}")
        
        print(f"\n전체 요약:")
        print(f"  총 대기 중: {total_queued}, 총 실행 중: {total_running}, 총 완료: {total_finished}")
        print("=" * 50)