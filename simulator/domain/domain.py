# --- simulator/domain/domain.py ---
import random
from enum import Enum, auto

class JobStatus(Enum):
    QUEUED = auto()
    RUNNING = auto()
    TRANSFER = auto()
    DONE = auto()

class Operation:
    def __init__(self, op_id, assigned_machine, candidate_machines, distribution):
        """
        :param op_id: Operation ID
        :param assigned_machine: 실제 라우팅된 기계 이름 (동적 모드에서는 None)
        :param candidate_machines: 가능한 기계 리스트
        :param distribution: 작업 소요 시간 분포 파라미터
        """
        self.id = op_id
        self.assigned_machine = assigned_machine
        self.candidates = candidate_machines
        self.distribution = distribution
        
        # 수학적 검증을 위한 시간 추적
        self.start_time = None      # s_{i,j}: 작업 시작 시간
        self.end_time = None        # e_{i,j}: 작업 완료 시간
        self.duration = None        # p_{i,j,m}: 실제 처리 시간
        
        # AGV 연계를 위한 시간 추적
        self.agv_pickup_time = None      # L^st_{i,j,k}: AGV 픽업 시작 시간
        self.agv_pickup_end_time = None  # L^en_{i,j,k}: AGV 픽업 완료 시간
        self.agv_delivery_start_time = None  # D^dep_{i,j,k}: AGV 배송 시작 시간
        self.agv_delivery_end_time = None    # D^arr_{i,j,k}: AGV 배송 완료 시간
        self.agv_unload_start_time = None    # U^st_{i,j,k}: AGV 언로드 시작 시간
        self.agv_unload_end_time = None     # U^en_{i,j,k}: AGV 언로드 완료 시간

    def select_machine(self):
        # 동적 할당 모드에서는 None 반환 (실시간 결정 필요)
        if self.assigned_machine is None:
            return None
        # 정적 할당 모드에서는 고정된 기계 반환
        return self.assigned_machine

    def sample_duration(self, machine_id):
        d = self.distribution
        t = d[machine_id]['distribution']
        if t == 'normal':
            return max(0, random.gauss(d[machine_id]['mean'], d[machine_id]['std']))
        if t == 'uniform':
            return random.uniform(d[machine_id]['low'], d[machine_id]['high'])
        if t == 'exponential':
            return random.expovariate(d[machine_id]['rate'])
        raise RuntimeError('Unknown distribution')
    
    def set_start_time(self, time):
        """작업 시작 시간 설정"""
        self.start_time = time
    
    def set_end_time(self, time):
        """작업 완료 시간 설정"""
        self.end_time = time
    
    def set_duration(self, duration):
        """작업 처리 시간 설정"""
        self.duration = duration
    
    def get_actual_duration(self):
        """실제 처리 시간 계산"""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None
    
    def validate_timing_constraints(self):
        """시간 제약 조건 검증"""
        violations = []
        
        # 시작 시간과 완료 시간이 모두 설정된 경우
        if self.start_time is not None and self.end_time is not None:
            # 처리 시간이 음수가 아니어야 함
            if self.end_time < self.start_time:
                violations.append({
                    'type': 'negative_duration',
                    'start_time': self.start_time,
                    'end_time': self.end_time
                })
            
            # 예상 처리 시간과 실제 처리 시간 비교
            if self.duration is not None:
                actual_duration = self.end_time - self.start_time
                if abs(actual_duration - self.duration) > 0.001:
                    violations.append({
                        'type': 'duration_mismatch',
                        'expected': self.duration,
                        'actual': actual_duration
                    })
        
        return violations
    
    def save_state(self) -> dict:
        return {
            'id': self.id,
            'assigned_machine': self.assigned_machine,
            'candidates': list(self.candidates or []),
            'distribution': dict(self.distribution or {}),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'agv': {
                'pickup_start': self.agv_pickup_time,
                'pickup_end': self.agv_pickup_end_time,
                'delivery_start': self.agv_delivery_start_time,
                'delivery_end': self.agv_delivery_end_time,
                'unload_start': self.agv_unload_start_time,
                'unload_end': self.agv_unload_end_time,
            },
        }

    def load_state(self, st: dict):
        self.assigned_machine = st.get('assigned_machine')
        self.candidates = list(st.get('candidates', []) or [])
        self.distribution = dict(st.get('distribution', {}) or {})
        self.start_time = st.get('start_time')
        self.end_time = st.get('end_time')
        self.duration = st.get('duration')
        agv = st.get('agv', {}) or {}
        self.agv_pickup_time = agv.get('pickup_start')
        self.agv_pickup_end_time = agv.get('pickup_end')
        self.agv_delivery_start_time = agv.get('delivery_start')
        self.agv_delivery_end_time = agv.get('delivery_end')
        self.agv_unload_start_time = agv.get('unload_start')
        self.agv_unload_end_time = agv.get('unload_end')

class Job:
    def __init__(self, job_id, part_id, operations, release_time=0.0, due_time=None):
        """
        :param job_id: Job identifier
        :param part_id: Part identifier
        :param operations: list of Operation instances
        :param release_time: Job release time
        """
        self.id = job_id
        self.part_id = part_id
        self.ops = operations
        self.idx = 0
        self.release_time = release_time
        self.due_time = due_time
        if self.due_time is None:
            raise ValueError("due_time must be provided for each job.")
        
        # 상태 관리
        self.status = JobStatus.QUEUED
        self.current_location = None
        self.last_completion_time = None
        self.completion_time = None  # makespan 계산을 위한 속성
        self.total_operations = len(operations)
        self.completed_operations = 0
        
        # 수학적 검증을 위한 시간 추적
        self.queue_entry_time = None  # q_{i,j}: 큐 진입 시간
        self.agv_transfer_times = 0.0  # AGV 전송 시간 추적

    def current_op(self):
        return self.ops[self.idx] if self.idx < len(self.ops) else None

    def advance(self):
        self.idx += 1
        self.completed_operations += 1

    def done(self):
        return self.idx >= len(self.ops)
    
    def set_status(self, status):
        self.status = status
    
    def set_location(self, location):
        self.current_location = location
    
    def set_completion_time(self, timestamp):
        self.last_completion_time = timestamp
        self.completion_time = timestamp  # makespan 계산을 위한 속성
    
    def set_queue_entry_time(self, time):
        """큐 진입 시간 설정"""
        self.queue_entry_time = time
    
    def get_progress(self):
        if self.total_operations == 0:
            return 1.0
        return self.completed_operations / self.total_operations
    
    def get_remaining_operations(self):
        return self.total_operations - self.completed_operations
    
    def validate_precedence_constraints(self):
        """선행 공정 제약 조건 검증: s_{i,j} ≥ e_{i,j-1}"""
        violations = []
        
        for i in range(1, len(self.ops)):
            current_op = self.ops[i]
            prev_op = self.ops[i-1]
            
            if (current_op.start_time is not None and 
                prev_op.end_time is not None):
                
                if current_op.start_time < prev_op.end_time:
                    violations.append({
                        'type': 'precedence_violation',
                        'operation_id': current_op.id,
                        'prev_operation_id': prev_op.id,
                        'current_start': current_op.start_time,
                        'prev_end': prev_op.end_time
                    })
        
        return violations
    
    def validate_queue_timing_constraints(self):
        """큐 타이밍 제약 조건 검증: s_{i,j} ≥ q_{i,j}"""
        violations = []
        
        if self.queue_entry_time is not None:
            for op in self.ops:
                if op.start_time is not None:
                    if op.start_time < self.queue_entry_time:
                        violations.append({
                            'type': 'queue_timing_violation',
                            'operation_id': op.id,
                            'queue_entry': self.queue_entry_time,
                            'start_time': op.start_time
                        })
        
        return violations
    
    def get_all_constraint_violations(self):
        """모든 제약 조건 위반 사항을 수집"""
        violations = []
        
        # 개별 operation의 제약 조건 검증
        for op in self.ops:
            violations.extend(op.validate_timing_constraints())
        
        # Job 수준의 제약 조건 검증
        violations.extend(self.validate_precedence_constraints())
        violations.extend(self.validate_queue_timing_constraints())
        
        return violations
    
    def to_dict(self):
        current_op = self.current_op()
        return {
            'job_id': self.id,
            'part_id': self.part_id,
            'status': self.status.name,
            'current_location': self.current_location,
            'last_completion_time': self.last_completion_time,
            'total_operations': self.total_operations,
            'completed_operations': self.completed_operations,
            'current_operation': current_op.id if current_op else None,
            'progress': self.get_progress(),
            'remaining_operations': self.get_remaining_operations()
        }
    
    def restore_state(self, state):
        self.idx = state['idx']
        self.status = state['status']
        self.current_location = state['current_location']
        self.last_completion_time = state['last_completion_time']
        self.completed_operations = state['completed_operations']

    def save_state(self) -> dict:
        return {
            'job_id': self.id,
            'part_id': self.part_id,
            'idx': int(self.idx),
            'status': self.status.name,
            'current_location': self.current_location,
            'last_completion_time': self.last_completion_time,
            'completion_time': self.completion_time,
            'total_operations': self.total_operations,
            'completed_operations': self.completed_operations,
            'release_time': self.release_time,
            'due_time': self.due_time,
            'queue_entry_time': self.queue_entry_time,
            'ops': [op.save_state() for op in self.ops],
        }

    def load_state(self, st: dict):
        from .domain import JobStatus
        self.idx = int(st.get('idx', 0))
        self.status = JobStatus[st.get('status', 'QUEUED')]
        self.current_location = st.get('current_location')
        self.last_completion_time = st.get('last_completion_time')
        self.completion_time = st.get('completion_time')
        self.total_operations = int(st.get('total_operations', len(self.ops)))
        self.completed_operations = int(st.get('completed_operations', 0))
        self.release_time = float(st.get('release_time', 0.0))
        self.due_time = float(st.get('due_time', 0.0))
        self.queue_entry_time = st.get('queue_entry_time')

        # operation 상태 복원
        ops_state = st.get('ops', [])
        id_to_op = {op.id: op for op in self.ops}
        for op_st in ops_state:
            oid = op_st.get('id')
            if oid in id_to_op:
                id_to_op[oid].load_state(op_st)
class Part:
    def __init__(self, part_id, job):
        self.id = part_id
        self.job = job
        self.status = 'new'

    # ⬇️ 추가: 직렬화/역직렬화 (링크는 job_id로만 들고가고, 복원 시 실제 Job 객체로 연결)
    def save_state(self) -> dict:
        return {
            'part_id': self.id,
            'job_id': getattr(self.job, 'id', None),
            'status': self.status,
        }

    def load_state(self, st: dict, reg: dict):
        self.status = st.get('status', 'new')
        jid = st.get('job_id')
        if jid is None or jid not in reg['jobs']:
            # 요청대로 자동 복구를 하지 않고 바로 실패
            raise ValueError(f"Part({self.id}) restore failed: job_id={jid} not found in registry.")
        self.job = reg['jobs'][jid]