# source_station.py
from simulator.engine.simulator import EoModel, Event

class SourceStation(EoModel):
    def __init__(self, name="GEN"):
        super().__init__(name)
        self.waiting_for_pickup = []  # ← 추가: 방금 릴리즈되어 아직 픽업 전인 파트 목록

    def handle_event(self, evt):
        if evt.event_type in ("material_arrival", "part_arrival"):
            part = evt.payload.get("part")
            part.job.current_location = self.name  # 파트의 현재 위치를 소스 스테이션으로 설정
            if part is None:
                print(f"[{self.name}] 경고: {evt.event_type} payload.part=None (src={evt.src_model})")
                return
            # 출발 대기 버퍼에 적재
            if part not in self.waiting_for_pickup:
                self.waiting_for_pickup.append(part)

            # AGV에게 픽업 요청
            job = part.job
            fetch_ev = Event("agv_fetch_request",
                             {"part": part, "job": job, "source_machine": self.name},
                             dest_model="AGVController")
            self.schedule(fetch_ev, 0.0)
            return

        # AGV가 실제로 픽업했음을 알릴 전용 이벤트(아래 2)에서 발행)
        if evt.event_type == "source_pickup":
            part = evt.payload.get("part")
            if part in self.waiting_for_pickup:
                self.waiting_for_pickup.remove(part)
            return
