# --- simulator/control/agv_logger.py ---
import pandas as pd
from datetime import datetime
from collections import defaultdict
import os

class AGVLogger:
    def __init__(self):
        self.logs = []
        self.agv_status_history = []
        self.agv_movement_history = []
        self.agv_task_history = []
        
        # 로그 시작 시간
        self.start_time = datetime.now()
        
    def log_agv_event(self, agv_id, event_type, details, timestamp=None):
        """AGV 이벤트 로그 기록"""
        if timestamp is None:
            timestamp = datetime.now()
            
        log_entry = {
            'timestamp': timestamp,
            'agv_id': agv_id,
            'event_type': event_type,
            'details': details,
            'simulation_time': getattr(timestamp, 'sim_time', 0)
        }
        
        self.logs.append(log_entry)
        
    def log_agv_status_change(self, agv_id, old_status, new_status, location, timestamp=None):
        """AGV 상태 변화 로그 기록"""
        if timestamp is None:
            timestamp = datetime.now()
            
        status_entry = {
            'timestamp': timestamp,
            'agv_id': agv_id,
            'old_status': old_status,
            'new_status': new_status,
            'location': location,
            'simulation_time': getattr(timestamp, 'sim_time', 0)
        }
        
        self.agv_status_history.append(status_entry)
        
    def log_agv_movement(self, agv_id, from_location, to_location, distance, speed, travel_time, timestamp=None):
        """AGV 이동 로그 기록"""
        if timestamp is None:
            timestamp = datetime.now()
            
        movement_entry = {
            'timestamp': timestamp,
            'agv_id': agv_id,
            'from_location': from_location,
            'to_location': to_location,
            'distance_m': distance,
            'speed_mps': speed,
            'travel_time_seconds': travel_time,
            'simulation_time': getattr(timestamp, 'sim_time', 0)
        }
        
        self.agv_movement_history.append(movement_entry)
        
    def log_agv_task(self, agv_id, task_type, source_machine, destination_machine, job_id, start_time, end_time, timestamp=None):
        """AGV 작업 로그 기록"""
        if timestamp is None:
            timestamp = datetime.now()
            
        task_entry = {
            'timestamp': timestamp,
            'agv_id': agv_id,
            'task_type': task_type,  # fetch, delivery, loading, unloading
            'source_machine': source_machine,
            'destination_machine': destination_machine,
            'job_id': job_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': (end_time - start_time) if end_time and start_time else None,
            'simulation_time': getattr(timestamp, 'sim_time', 0)
        }
        
        self.agv_task_history.append(task_entry)
        
    def log_agv_utilization(self, agv_id, status, duration, timestamp=None):
        """AGV 활용도 로그 기록"""
        if timestamp is None:
            timestamp = datetime.now()
            
        utilization_entry = {
            'timestamp': timestamp,
            'agv_id': agv_id,
            'status': status,
            'duration_seconds': duration,
            'simulation_time': getattr(timestamp, 'sim_time', 0)
        }
        
        # 기존 활용도 로그에 추가
        if not hasattr(self, 'agv_utilization_history'):
            self.agv_utilization_history = []
        self.agv_utilization_history.append(utilization_entry)
        
    def get_agv_summary_stats(self):
        """AGV 요약 통계 생성"""
        if not self.agv_task_history:
            return {}
            
        df = pd.DataFrame(self.agv_task_history)
        
        # AGV별 통계
        agv_stats = {}
        for agv_id in df['agv_id'].unique():
            agv_data = df[df['agv_id'] == agv_id]
            
            # 작업별 통계
            task_counts = agv_data['task_type'].value_counts()
            
            # 총 이동 거리
            movement_df = pd.DataFrame(self.agv_movement_history)
            if not movement_df.empty:
                agv_movement = movement_df[movement_df['agv_id'] == agv_id]
                total_distance = agv_movement['distance_m'].sum()
                total_travel_time = agv_movement['travel_time_seconds'].sum()
            else:
                total_distance = 0
                total_travel_time = 0
                
            agv_stats[agv_id] = {
                'total_tasks': len(agv_data),
                'fetch_tasks': task_counts.get('fetch', 0),
                'delivery_tasks': task_counts.get('delivery', 0),
                'loading_tasks': task_counts.get('loading', 0),
                'unloading_tasks': task_counts.get('unloading', 0),
                'total_distance_m': total_distance,
                'total_travel_time_seconds': total_travel_time,
                'average_speed_mps': total_distance / total_travel_time if total_travel_time > 0 else 0
            }
            
        return agv_stats

    def save_to_excel(self, output_dir='results_reinforcement', filename='agv_logs.xlsx'):
        """모든 AGV 로그를 엑셀 파일로 저장"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filepath = os.path.join(output_dir, filename)
        
        # ExcelWriter 객체 생성
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            
            # 1. AGV 이벤트 로그
            if self.logs:
                df_logs = pd.DataFrame(self.logs)
                df_logs.to_excel(writer, sheet_name='AGV_Events', index=False)
                
            # 2. AGV 상태 변화 로그
            if self.agv_status_history:
                df_status = pd.DataFrame(self.agv_status_history)
                df_status.to_excel(writer, sheet_name='AGV_Status_Changes', index=False)
                
            # 3. AGV 이동 로그
            if self.agv_movement_history:
                df_movement = pd.DataFrame(self.agv_movement_history)
                df_movement.to_excel(writer, sheet_name='AGV_Movements', index=False)
                
            # 4. AGV 작업 로그
            if self.agv_task_history:
                df_tasks = pd.DataFrame(self.agv_task_history)
                df_tasks.to_excel(writer, sheet_name='AGV_Tasks', index=False)
                
            # 5. AGV 활용도 로그
            if hasattr(self, 'agv_utilization_history') and self.agv_utilization_history:
                df_utilization = pd.DataFrame(self.agv_utilization_history)
                df_utilization.to_excel(writer, sheet_name='AGV_Utilization', index=False)
                
            # 6. AGV 요약 통계
            summary_stats = self.get_agv_summary_stats()
            if summary_stats:
                df_summary = pd.DataFrame.from_dict(summary_stats, orient='index')
                df_summary.reset_index(inplace=True)
                df_summary.rename(columns={'index': 'agv_id'}, inplace=True)
                df_summary.to_excel(writer, sheet_name='AGV_Summary_Stats', index=False)
                
            # 7. 시스템 요약
            system_summary = {
                'metric': [
                    'Total AGV Events',
                    'Total Status Changes', 
                    'Total Movements',
                    'Total Tasks',
                    'Logging Start Time',
                    'Logging End Time'
                ],
                'value': [
                    len(self.logs),
                    len(self.agv_status_history),
                    len(self.agv_movement_history),
                    len(self.agv_task_history),
                    self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            df_system = pd.DataFrame(system_summary)
            df_system.to_excel(writer, sheet_name='System_Summary', index=False)
            
        print(f"[AGVLogger] AGV 로그가 {filepath}에 저장되었습니다.")
        return filepath
        
    def print_summary(self):
        """AGV 로그 요약 출력"""
        print(f"\n=== AGV 로그 요약 ===")
        print(f"총 이벤트 수: {len(self.logs)}")
        print(f"총 상태 변화 수: {len(self.agv_status_history)}")
        print(f"총 이동 수: {len(self.agv_movement_history)}")
        print(f"총 작업 수: {len(self.agv_task_history)}")
        
        if hasattr(self, 'agv_utilization_history'):
            print(f"총 활용도 기록 수: {len(self.agv_utilization_history)}")
            
        # AGV별 요약 통계
        summary_stats = self.get_agv_summary_stats()
        if summary_stats:
            print(f"\n=== AGV별 통계 ===")
            for agv_id, stats in summary_stats.items():
                print(f"{agv_id}:")
                print(f"  총 작업: {stats['total_tasks']}")
                print(f"  총 이동거리: {stats['total_distance_m']:.2f}m")
                print(f"  총 이동시간: {stats['total_travel_time_seconds']:.2f}초")
                print(f"  평균 속도: {stats['average_speed_mps']:.2f}m/s")
                print()
                
    def clear_logs(self):
        """로그 데이터 초기화"""
        self.logs = []
        self.agv_status_history = []
        self.agv_movement_history = []
        self.agv_task_history = []
        if hasattr(self, 'agv_utilization_history'):
            self.agv_utilization_history = []
        self.start_time = datetime.now()
        print("[AGVLogger] 모든 로그가 초기화되었습니다.")

