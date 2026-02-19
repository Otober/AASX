# --- simulator/model/agv.py ---
from simulator.engine.simulator import EoModel, Event
from enum import Enum, auto
import math

class AGVStatus(Enum):
    IDLE = auto()          # 유휴 상태
    FETCHING = auto()      # 머신에서 작업 가져오는 중
    LOADING = auto()       # 작업을 AGV에 적재 중
    DELIVERING = auto()    # 목적지로 이동 중
    UNLOADING = auto()     # 목적지에서 작업 하역 중

class AGV(EoModel):
    def __init__(self, agv_id, speed=1.0, capacity=1):
        super().__init__(f"AGV_{agv_id}")
        self.agv_id = agv_id
        self.speed = speed  # 단위: m/s
        self.capacity = capacity  # 동시에 운반할 수 있는 작업 수
        self.transfer_time_map = {}  # (source, destination): time in seconds
        
        # 상태 관리
        self.status = AGVStatus.IDLE
        self.current_location = "GEN"  # 현재 위치 (머신 이름)
        self.destination = None       # 목적지 (머신 이름)
        
        # 작업 관리
        self.carried_jobs = []       # 현재 운반 중인 작업들
        self.current_task = None     # 현재 수행 중인 작업
        
        # 이동 관련
        self.departure_time = 0.0    # 출발 시간
        self.arrival_time = 0.0      # 도착 예정 시간
        self.distance = 0.0          # 이동 거리
        self.total_distance = 0.0    # 누적 이동 거리
        
        # 로깅 관련
        self.logger = None  # AGVLogger 인스턴스
        self.task_start_time = None  # 현재 작업 시작 시간
        
    def save_state(self):
        def task_to_dict(t):
            if not t : return None
            d = dict(t)  # shallow copy
            part_obj = d.get('part', None)
            job_obj  = d.get('job', None)
            op_obj   = d.get('op', None)   # 나중에 2번 패치에서 넣을 예정

            if part_obj is not None:
                d['part'] = {'__type__':'part','id': part_obj.id}
            else:
                d['part'] = None

            if job_obj is not None:
                d['job']  = {'__type__':'job','id': job_obj.id}
            else:
                d['job']  = None

            if op_obj is not None:
                d['op']   = {'__type__':'op','id': getattr(op_obj, 'id', None)}
            # op이 아직 없다면 생략

            return d

        return {
            'agv_id': self.agv_id,
            'status': self.status.name,
            'current_location': self.current_location,
            'destination': self.destination,
            'carried_parts': [p.id for p in self.carried_jobs],
            'current_task': task_to_dict(self.current_task),
            'departure_time': self.departure_time,
            'arrival_time': self.arrival_time,
            'distance': self.distance,
            'task_start_time': self.task_start_time,
            'total_distance': self.total_distance
        }

    def load_state(self, st, reg):
        from enum import Enum
        self.status = type(self.status)[st['status']]
        self.current_location = st.get('current_location')
        self.destination = st.get('destination')
        self.carried_jobs = [reg['parts'][pid] for pid in st.get('carried_parts', []) if pid in reg['parts']]
        ct = st.get('current_task')
        if ct:
            part_obj = reg['parts'].get(ct['part']['id']) if ct.get('part') else None
            job_obj  = reg['jobs'].get(ct['job']['id'])  if ct.get('job')  else None
            if job_obj is None:
                raise ValueError(f"AGV.load_state: missing job for task (agv={self.agv_id})")

            # op 복원(있으면)
            op_obj = None
            if ct.get('op') and job_obj:
                op_id = ct['op']['id']
                op_obj = next((o for o in getattr(job_obj, 'ops', []) if getattr(o, 'id', None) == op_id), None)

            self.current_task = {
                **{k:v for k,v in ct.items() if k not in ('part','job','op')},
                'part': part_obj,
                'job':  job_obj,
                'op':   op_obj,
            }
        else:
            self.current_task = None
        self.departure_time = st.get('departure_time', 0.0)
        self.arrival_time   = st.get('arrival_time', 0.0)
        self.distance       = st.get('distance', 0.0)
        self.task_start_time = st.get('task_start_time', None)
        self.total_distance = st.get('total_distance', 0.0)

    def set_logger(self, logger):
        """로거 설정"""
        self.logger = logger

    def set_transfer_times(self, transfer_map: dict): 
        self.transfer_time_map = transfer_map or {}
    
    def _lookup_transfer_time(self, source: str, destination: str):
        if source in self.transfer_time_map:
            m = self.transfer_time_map[source]
            if source == destination:
                return 0.0
            if isinstance(m, dict):
                spec = m.get(destination)
                if spec:
                    # (a) 숫자일 경우
                    if isinstance(spec, (int, float)):
                        return float(spec)

                    # (b) 분포(dict)일 경우
                    dist = spec.get("distribution")
                    if dist == "normal":
                        return float(spec.get("mean", 0.0))
                    if dist == "uniform":
                        low, high = float(spec.get("low", 0.0)), float(spec.get("high", 0.0))
                        return (low + high) / 2.0
                    if dist == "exponential":
                        rate = float(spec.get("rate", 0.0))
                        return (1.0 / rate) if rate > 0 else None

                    # (c) 단순 시간 필드
                    if "time" in spec:
                        return float(spec["time"])

                    return None
        return self.transfer_time_map.get((source, destination), None)
        
    def _log_event(self, event_type, details):
        """이벤트 로깅"""
        if self.logger:
            self.logger.log_agv_event(f"AGV_{self.agv_id}", event_type, details)
            
    def _log_status_change(self, old_status, new_status):
        """상태 변화 로깅"""
        if self.logger:
            self.logger.log_agv_status_change(
                f"AGV_{self.agv_id}", 
                old_status.name if old_status else "None", 
                new_status.name, 
                self.current_location
            )
            
    def _log_movement(self, from_location, to_location, distance, travel_time):
        """이동 로깅"""
        if self.logger:
            self.logger.log_agv_movement(
                f"AGV_{self.agv_id}",
                from_location,
                to_location,
                distance,
                self.speed,
                travel_time
            )
            
    def _log_task(self, task_type, source_machine, destination_machine, job_id, start_time, end_time):
        """작업 로깅"""
        if self.logger:
            self.logger.log_agv_task(
                f"AGV_{self.agv_id}",
                task_type,
                source_machine,
                destination_machine,
                job_id,
                start_time,
                end_time
            )
        
    def handle_event(self, event):
        """이벤트 처리"""
        if event.event_type == 'agv_fetch_request':
            self._handle_fetch_request(event.payload)
        elif event.event_type == 'agv_delivery_request':
            self._handle_delivery_request(event.payload)
        elif event.event_type == 'agv_fetch_complete':
            self._handle_fetch_complete(event.payload)
        elif event.event_type == 'agv_delivery_complete':
            self._handle_delivery_complete(event.payload)
        elif event.event_type == 'agv_move_complete':
            self._handle_move_complete(event.payload)
    
    def _handle_fetch_request(self, payload):
        """작업 가져오기 요청 처리"""
        if self.status != AGVStatus.IDLE:
            print(f"[{self.name}] 현재 {self.status.name} 상태로 인해 fetch 요청을 거부합니다.")
            return
            
        part = payload.get('part')          # ✅ Part 받기
        job  = part.job if part else payload['job']  # 하위호환
        source_machine = payload['source_machine']

        print(f"[{self.name}] {source_machine}에서 Job {job.id}, Op {job.current_op().id} fetch 시작")

        # 로깅
        self._log_event('fetch_request', {'source_machine': source_machine, 'job_id': job.id})

        
        # current_task를 먼저 설정
        self.current_task = {
            'type': 'fetch',
            'source_machine': source_machine,
            'part': part,                   # ✅ Part 저장
            'job': job                      # (옵션) 참고용
     }
        
        # 작업 시작 시간 기록
        self.task_start_time = EoModel.get_time()

        # AGV를 source machine으로 이동
        self._move_to(source_machine, "fetch")
    
    def _handle_delivery_request(self, payload):
        part = payload['part']
        job = part.job
        current_op = job.current_op()

        # 후보 집합
        candidates = current_op.candidates if (current_op and current_op.candidates) else []
        if not candidates:
            raise ValueError(f"{self.name}: Job {job.id} Op {current_op.id} 다음 작업 후보가 없습니다.")

        # 콜백 필수
        sel_cb = getattr(self, "simulator", None) and getattr(self.simulator, "select_next_machine", None)
        if not sel_cb:
            raise RuntimeError(f"{self.name}: select_next_machine 콜백이 설정되지 않았습니다. (기본 룰: AGV가 delivery 시점에 결정)")

        # 목적지 결정 (반드시 콜백)
        destination = self.simulator.select_next_machine(job.id, current_op.id, candidates)
        if destination not in candidates:
            raise ValueError(f"{self.name}: 콜백이 후보 외 목적지('{destination}')를 반환했습니다. candidates={candidates}")

        # ✅ 여기서 'delivery' 태스크로 전환 + 목적지 기록
        self.current_task = {
            'type': 'delivery',
            'source_machine': self.current_location,   # 참고용(로그 등)
            'destination_machine': destination,
            'part': part,
            'job': job,
            'op': current_op
        }
        self.task_start_time = EoModel.get_time()

        self._log_event('delivery_request', {'current_location': self.current_location, 'job_id': job.id})

        print(f"[{self.name}] {self.current_location}→{destination}로 Job {job.id}, Op {job.current_op().id} delivery 시작")
        self._move_to(destination, "delivery")

    def _move_to(self, destination, task_type):
        """지정된 목적지로 이동"""
        self.destination = destination
        if self.current_location == self.destination:
            # 이미 목적지에 있는 경우
            now = EoModel.get_time()
            self.departure_time = now
            self.arrival_time = now
            self._arrive_at_destination()
            return
            
        old_status = self.status
        self.destination = destination
        self.status = AGVStatus.DELIVERING
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        # 이동 거리 계산 (간단한 유클리드 거리)
        # 실제 구현에서는 좌표 기반 거리 계산 사용
        self.distance = self._lookup_transfer_time(self.current_location, destination)

        if self.distance is None:
            raise ValueError(f"{self.name}: {self.current_location}→{destination} 경로 없음")

        # 이동 시간 계산
        travel_time = self.distance / self.speed
        self.total_distance += self.distance  # 누적 거리 업데이트

        from simulator.result.recorder import Recorder
        if task_type == "delivery":
            try:
                part = self.current_task.get('part') if self.current_task else None
                if part is not None:
                    Recorder.log_delivery(
                        self.agv_id,
                        part,
                        self.current_location,   # 출발지 (예: M1, GEN)
                        destination,             # 목적지 (콜백으로 확정됨)
                        EoModel.get_time(),
                        travel_time
                    )
            except Exception:
                pass
        elif task_type == "fetch":
            try:
                part = self.current_task.get('part') if self.current_task else None
                if part is not None:
                    Recorder.log_fetch(
                        self.agv_id,
                        part,
                        self.current_location,   # 출발지 (예: M1, GEN)
                        destination,             # 목적지 (콜백으로 확정됨)
                        EoModel.get_time(),
                        travel_time
                    )
            except Exception:
                pass
        # 출발 시간과 도착 시간 설정
        current_time = EoModel.get_time()
        self.departure_time = current_time
        self.arrival_time = current_time + travel_time
        
        print(f"[{self.name}] {self.current_location} → {destination} 이동 시작 (거리: {self.distance:.2f}m, 시간: {travel_time:.2f}초)")
        
        # 이동 로깅
        self._log_movement(self.current_location, destination, self.distance, travel_time)
        
        # 이동 완료 이벤트 스케줄링
        ev = Event('agv_move_complete', {
            'agv_id': self.agv_id,
            'destination': destination
        }, dest_model=self.name)
        self.schedule(ev, travel_time)
    
    def _arrive_at_destination(self):
        """목적지 도착 처리"""
        self.current_location = self.destination
        self.destination = None
        
        if self.current_task and self.current_task['type'] == 'fetch':
            self._start_fetching()
        elif self.current_task and self.current_task['type'] == 'delivery':
            self._start_unloading()
    
    def _start_fetching(self):
        """작업 가져오기 시작"""
        old_status = self.status
        self.status = AGVStatus.FETCHING
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        source_machine = self.current_task['source_machine']
        job = self.current_task['job']

        print(f"[{self.name}] {source_machine}에서 Job {job.id}, Op {job.current_op().id} fetch 중...")

        # fetch 완료 이벤트 스케줄링 (간단한 로딩 시간)
        ev = Event('agv_fetch_complete', {
            'agv_id': self.agv_id,
            'job': job,
            'source_machine': source_machine
        }, dest_model=self.name)
        self.schedule(ev, 0.0)  # 2초 로딩 시간 -> 0으로 변경
    
    def _start_unloading(self):
        """작업 하역 시작"""
        old_status = self.status
        self.status = AGVStatus.UNLOADING
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        destination_machine = self.current_task['destination_machine']
        job = self.current_task['job']

        if(job.id == "J26") :
            a=1
        print("test unloading : ", job.id)
        print(f"[{self.name}] {destination_machine}에서 Job {job.id}, Op {job.current_op().id} unload 중...")

        # unload 완료 이벤트 스케줄링 (간단한 하역 시간)
        ev = Event('agv_delivery_complete', {
            'agv_id': self.agv_id,
            'job': job,
            'op_id': job.current_op().id if job.current_op() else None,
            'destination_machine': destination_machine
        }, dest_model=self.name)
        self.schedule(ev, 1.0)  # 2초 하역 시간 -> 1초로 변경

    def _handle_fetch_complete(self, payload):
        """작업 가져오기 완료 처리"""
        part = self.current_task.get('part') if self.current_task else None
        job = payload['job']
        source_machine = payload['source_machine']
        dest = payload.get('destination_machine')
        
        # 작업을 AGV에 적재
        self.carried_jobs.append(part)
        old_status = self.status
        self.status = AGVStatus.LOADING
        pickup_ev = Event("source_pickup",
                          {"part": part},
                          dest_model=source_machine)
        self.schedule(pickup_ev, 0.0)
        self._request_delivery(part)
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)

        

        print(f"[{self.name}] Job {job.id}, Op {job.current_op().id} 적재 완료")

        # 작업 로깅
        if self.task_start_time is not None:
            self._log_task('fetch', source_machine, None, job.id, self.task_start_time, EoModel.get_time())
    
    def _request_delivery(self, part):
        # 목적지 계산 금지! (힌트/콜백 모두 X)
        ev = Event('agv_delivery_request', {
            'agv_id': self.agv_id,
            'part': part
        }, dest_model='AGVController')
        self.schedule(ev, 0)

    
    def _handle_delivery_complete(self, payload):
        """배송 완료 처리"""
        part = self.current_task.get('part') if self.current_task else None
        job  = part.job
        destination_machine = payload['destination_machine']
        
        # 작업을 AGV에서 제거
        if part in self.carried_jobs:
            self.carried_jobs.remove(part)

        print(f"[{self.name}] Job {job.id}, Op {job.current_op().id}를 {destination_machine}에 배송 완료")

        # 작업 로깅
        if self.task_start_time is not None:
            self._log_task('delivery', None, destination_machine, job.id, self.task_start_time, EoModel.get_time())
        
        # 작업을 목적지 기계에 전달
        ev = Event('part_arrival', {
        'part': part,                   # ✅ 여기가 핵심 수정
        'agv_id': self.agv_id
        }, dest_model=destination_machine)
        self.schedule(ev, 0)
        
        # AGV를 유휴 상태로 전환
        self._return_to_idle()
    
    def _handle_move_complete(self, payload):
        """이동 완료 처리"""
        destination = payload['destination']
        print(f"[{self.name}] {destination} 도착 완료")
        self._arrive_at_destination()
    
    def _return_to_idle(self):
        """AGV를 유휴 상태로 전환"""
        old_status = self.status
        self.status = AGVStatus.IDLE
        
        # 상태 변화 로깅
        self._log_status_change(old_status, self.status)
        
        self.current_task = None
        self.task_start_time = None
        print(f"[{self.name}] 유휴 상태로 전환")
        
        # AGV 컨트롤러에 유휴 상태 알림
        ev = Event('agv_idle', {
            'agv_id': self.agv_id,
            'location': self.current_location
        }, dest_model='AGVController')
        self.schedule(ev, 0)
    
    def get_status_info(self):
        """AGV 상태 정보 반환"""
        return {
            'agv_id': self.agv_id,
            'status': self.status.name,
            'current_location': self.current_location,
            'destination': self.destination,
            'carried_jobs': [job.id for job in self.carried_jobs],
            'speed': self.speed,
            'capacity': self.capacity
        }
