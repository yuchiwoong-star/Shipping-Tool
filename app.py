import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math
import uuid

# ==========================================
# 1. 커스텀 물리 엔진 (최종 규칙 반영)
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
        self.h = float(h)
        self.d = float(d)
        self.max_weight = float(max_weight)
        self.cost = cost
        self.items = []
        self.total_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

    def put_item(self, item):
        if self.total_weight + item.weight > self.max_weight:
            return False
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))
        for p in self.pivots:
            px, py, pz = p
            # [규칙 1] 회전 불가: w, d를 입력된 그대로 사용
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
        if z <= 0.001: return True
        support_area = 0.0
        for exist in self.items:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0.0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy
        return support_area >= (item.w * item.d * 0.6)

# ==========================================
# 2. 데이터베이스 (규칙 0: 최소 비용 차량)
# ==========================================
st.set_page_config(layout="wide", page_title="Professional Load Planner")

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
LIMIT_H = 1300 # [규칙 2] 높이 제한

# ==========================================
# 3. 로직 함수
# ==========================================
def load_data(df):
    items = []
    try:
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        # [규칙 4] 상위 10% 중량
        heavy_threshold = np.percentile(weights, 90) if weights else 999999
    except: heavy_threshold = 999999

    for _, row in df.iterrows():
        try:
            box = Box(str(row['박스번호']), row['폭'], row['높이'], row['길이'], row['중량'])
            if box.weight >= heavy_threshold: box.is_heavy = True
            items.append(box)
        except: continue
    return items

def run_optimization(all_items):
    remaining_items = all_items[:]
    used_trucks = []
    # 단가 기준 오름차순 (싼 차부터)
    sorted_truck_types = sorted(TRUCK_DB.items(), key=lambda x: x[1]['cost'])

    while remaining_items:
        best_truck = None
        for t_name, spec in sorted_truck_types:
            temp_truck = Truck(t_name, spec['w'], LIMIT_H, spec['l'], spec['weight'], spec['cost'])
            test_items = sorted(remaining_items, key=lambda x: x.volume, reverse=True)
            
            packed_count = 0
            for item in test_items:
                if temp_truck.put_item(Box(item.name, item.w, item.h, item.d, item.weight)):
                    # 복사본에 heavy 속성 수동 전사
                    temp_truck.items[-1].is_heavy = item.is_heavy
                    packed_count += 1
            
            if packed_count > 0:
                # 남은 걸 다 실을 수 있는 가장 싼 차를 우선 선택
                if packed_count == len(remaining_items) or best_truck is None or packed_count > len(best_truck.items):
                    best_truck = temp_truck
                    if packed_count == len(remaining_items): break 

        if best_truck:
            best_truck.name = f"{best_truck.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_truck)
            packed_names = [i.name for i in best_truck.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else: break
    return used_trucks

# ==========================================
# 4. 시각화 (draw_truck_3d 전체 포함)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    spec = TRUCK_DB[truck.name.split(' ')[0]]
    W, L = spec['w'], spec['l']
    H = LIMIT_H

    # 박스 그리기
    for item in truck.items:
        color = 'red' if item.is_heavy else 'orange'
        x, y, z = item.x, item.y, item.z
        w, h, d = item.w, item.h, item.d
        
        # 3D Mesh
        fig.add_trace(go.Mesh3d(
            x=[x, x+w, x+w, x, x, x+w, x+w, x],
            y=[y, y, y+d, y+d, y, y, y+d, y+d],
            z=[z, z, z, z, z+h, z+h, z+h, z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=color, opacity=0.8, flatshading=True, name=item.name
        ))
        # 박스 테두리
        fig.add_trace(go.Scatter3d(
            x=[x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x],
            y=[y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d],
            z=[z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h],
            mode='lines', line=dict(color='black', width=2), showlegend=False
        ))

    # 트럭 바닥 가이드
    fig.add_trace(go.Scatter3d(x=[0, W, W, 0, 0], y=[0, 0, L, L, 0], z=[0, 0, 0, 0, 0], mode='lines', line=dict(color='blue', width=4), name='트레일러 바닥'))

    # 뷰 설정
    if camera_view == "top": eye = dict(x=0, y=0.1, z=2.5)
    elif camera_view == "side": eye = dict(x=2.5, y=0, z=0.2)
    else: eye = dict(x=1.8, y=-1.8, z=1.5)

    fig.update_layout(
        scene=dict(aspectmode='data', camera=dict(eye=eye), xaxis_title="폭", yaxis_title="길이", zaxis_title="높이"),
        margin=dict(l=0, r=0, b=0, t=0), height=600, uirevision=str(uuid.uuid4())
    )
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 Smart Logistics Load Planner")
st.markdown("---")

if 'view_mode' not in st.session_state: st.session_state['view_mode'] = 'iso'

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='cp949') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.subheader("📋 입력 데이터 확인")
    st.dataframe(df.head())

    if st.button("최적 배차 계산 시작", type="primary"):
        items = load_data(df)
        trucks = run_optimization(items)
        st.session_state['results'] = trucks

    if 'results' in st.session_state:
        trucks = st.session_state['results']
        
        col1, col2, col3 = st.columns(3)
        with col1: 
            if st.button("쿼터뷰"): st.session_state['view_mode'] = 'iso'
        with col2: 
            if st.button("탑뷰"): st.session_state['view_mode'] = 'top'
        with col3: 
            if st.button("사이드뷰"): st.session_state['view_mode'] = 'side'

        tabs = st.tabs([t.name for t in trucks])
        for i, tab in enumerate(tabs):
            with tab:
                t = trucks[i]
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("차량 종류", t.name.split(' ')[0])
                    st.metric("총 중량", f"{t.total_weight:,.1f} kg")
                    st.metric("운송 비용", f"{int(t.cost):,} 원")
                    st.write(f"박스 개수: {len(t.items)}개")
                with c2:
                    st.plotly_chart(draw_truck_3d(t, st.session_state['view_mode']), use_container_width=True)
else:
    st.info("왼쪽 사이드바에서 파일을 업로드해 주세요.")
