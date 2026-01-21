import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math
import uuid

# ==========================================
# 1. 커스텀 물리 엔진 (규칙 준수)
# ==========================================
class Box:
    def __init__(self, name, w, h, d, weight):
        self.name = name
        self.w = float(w)
        self.h = float(h)
        self.d = float(d)
        self.weight = float(weight)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.is_heavy = False
    @property
    def volume(self):
        return self.w * self.h * self.d

class Truck:
    def __init__(self, name, w, h, d, max_weight, cost):
        self.name = name
        self.w = float(w)
        self.h = float(h) # 적재 제한 높이 (1300mm)
        self.d = float(d)
        self.max_weight = float(max_weight)
        self.cost = cost
        self.items = []
        self.total_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

    def put_item(self, item):
        # [규칙 1] 회전 불가 (w, d 고정 사용)
        if self.total_weight + item.weight > self.max_weight:
            return False
        
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0])) # Z, Y, X 순 정렬
        
        for p in self.pivots:
            px, py, pz = p
            # [규칙 2] 높이 제한 및 공간 체크
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h):
                continue
            if self._check_collision(item, px, py, pz):
                continue
            # [규칙 3] 지지 면적 60% 이상
            if not self._check_support(item, px, py, pz):
                continue
                
            item.x, item.y, item.z = px, py, pz
            self.items.append(item)
            self.total_weight += item.weight
            
            # 피벗 추가
            self.pivots.append([item.x + item.w, item.y, item.z])
            self.pivots.append([item.x, item.y + item.d, item.z])
            self.pivots.append([item.x, item.y, item.z + item.h])
            return True
        return False

    def _check_collision(self, item, x, y, z):
        for exist in self.items:
            if (x < exist.x + exist.w and x + item.w > exist.x and
                y < exist.y + exist.d and y + item.d > exist.y and
                z < exist.z + exist.h and z + item.h > exist.z):
                return True
        return False

    def _check_support(self, item, x, y, z):
        if z <= 0.001: return True # 바닥은 100% 지지
        support_area = 0.0
        for exist in self.items:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0.0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy
        return support_area >= (item.w * item.d * 0.6)

# ==========================================
# 2. 설정 및 데이터 (규칙 0 적용)
# ==========================================
TRUCK_DB = {
    "1톤":   {"w": 1600, "l": 2800, "weight": 1490,  "cost": 78000},
    "2.5톤": {"w": 1900, "l": 4200, "weight": 3490,  "cost": 110000},
    "5톤":   {"w": 2100, "l": 6200, "weight": 6900,  "cost": 133000},
    "8톤":   {"w": 2350, "l": 7300, "weight": 9490,  "cost": 153000},
    "11톤":  {"w": 2350, "l": 9200, "weight": 14900, "cost": 188000},
    "15톤":  {"w": 2350, "l": 10200, "weight": 16900, "cost": 211000},
    "18톤":  {"w": 2350, "l": 10200, "weight": 20900, "cost": 242000},
    "22톤":  {"w": 2350, "l": 10200, "weight": 26000, "cost": 308000},
}
LIMIT_H = 1300 # [규칙 2] 높이 제한 공통 적용

# ==========================================
# 3. 로직 함수
# ==========================================
def load_data(df):
    items = []
    # [규칙 4] 상위 10% 중량 파악
    try:
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        if weights:
            heavy_threshold = np.percentile(weights, 90)
        else:
            heavy_threshold = 999999
    except:
        heavy_threshold = 999999

    for _, row in df.iterrows():
        try:
            box = Box(str(row['박스번호']), row['폭'], row['높이'], row['길이'], row['중량'])
            if box.weight >= heavy_threshold:
                box.is_heavy = True # 빨간색 표시 대상
            items.append(box)
        except: continue
    return items

def run_optimization(all_items):
    remaining_items = all_items[:]
    used_trucks = []
    # 단가 기준 오름차순 정렬 (가장 싼 차부터 검토)
    sorted_truck_types = sorted(TRUCK_DB.items(), key=lambda x: x[1]['cost'])

    while remaining_items:
        best_truck_for_batch = None
        
        # 비용 효율을 위해 큰 차부터 실어보고, 남은 게 적으면 작은 차로 전환하는 로직
        # 여기서는 "단가 대비 가장 많은 무게/부피를 실을 수 있는 차"를 찾거나 
        # "현재 남은 것을 다 실을 수 있는 가장 싼 차"를 찾습니다.
        for t_name, spec in sorted_truck_types:
            temp_truck = Truck(t_name, spec['w'], LIMIT_H, spec['l'], spec['weight'], spec['cost'])
            # 부피 큰 순서대로 적재 시도
            test_items = sorted(remaining_items, key=lambda x: x.volume, reverse=True)
            
            packed_in_this_truck = []
            for item in test_items:
                item_copy = Box(item.name, item.w, item.h, item.d, item.weight)
                item_copy.is_heavy = item.is_heavy
                if temp_truck.put_item(item_copy):
                    packed_in_this_truck.append(item.name)
            
            if len(packed_in_this_truck) > 0:
                # 성능 지표: 적재된 박스들의 무게 합 / 차량 단가 (가성비)
                # 만약 남은 박스를 모두 실을 수 있다면 그 중 가장 싼 차가 선택됨
                if len(packed_in_this_truck) == len(remaining_items):
                    best_truck_for_batch = temp_truck
                    break # 가장 싼 차부터 돌았으므로 바로 확정
                
                # 다 못 싣는 경우, 일단 후보로 보관 (더 좋은 가성비의 차가 있는지 확인)
                if best_truck_for_batch is None or \
                   (len(packed_in_this_truck) > len(best_truck_for_batch.items)):
                    best_truck_for_batch = temp_truck

        if best_truck_for_batch:
            best_truck_for_batch.name = f"{best_truck_for_batch.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_truck_for_batch)
            packed_names = [i.name for i in best_truck_for_batch.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else:
            break
    return used_trucks

# ==========================================
# 4. 시각화 및 UI (동일 구조 유지하되 데이터 반영)
# ==========================================
# (기존 draw_truck_3d 함수와 UI 코드는 TRUCK_DB와 LIMIT_H를 참조하도록 연결되어 있으므로 
# 위에서 정의한 변수값에 따라 자동으로 적용됩니다.)
