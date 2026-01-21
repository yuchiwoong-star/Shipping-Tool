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
            # [규칙 1] 회전 불가 / [규칙 2] 높이 제한 체크
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
# 2. 설정 및 데이터 (규칙 0: 운송단가 반영)
# ==========================================
st.set_page_config(layout="wide", page_title="Ultimate Load Planner v2")

TRUCK_DB = {
    "1톤":   {"w": 1600, "real_h": 2000, "l": 2800,  "weight": 1490,  "cost": 78000},
    "2.5톤": {"w": 1900, "real_h": 2200, "l": 4200,  "weight": 3490,  "cost": 110000},
    "5톤":   {"w": 2100, "real_h": 2350, "l": 6200,  "weight": 6900,  "cost": 133000},
    "8톤":   {"w": 2350, "real_h": 2350, "l": 7300,  "weight": 9490,  "cost": 153000},
    "11톤":  {"w": 2350, "real_h": 2350, "l": 9200,  "weight": 14900, "cost": 188000},
    "15톤":  {"w": 2350, "real_h": 2350, "l": 10200, "weight": 16900, "cost": 211000},
    "18톤":  {"w": 2350, "real_h": 2350, "l": 10200, "weight": 20900, "cost": 242000},
    "22톤":  {"w": 2350, "real_h": 2350, "l": 10200, "weight": 26000, "cost": 308000},
}
LIMIT_H = 1300 # [규칙 2] 높이제한

# ==========================================
# 3. 로직 함수
# ==========================================
def load_data(df):
    items = []
    try:
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        # [규칙 4] 상위 10% 중량 박스
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
    # 단가 기준 오름차순 (가장 저렴한 차부터)
    sorted_truck_types = sorted(TRUCK_DB.items(), key=lambda x: x[1]['cost'])

    while remaining_items:
        best_truck = None
        for t_name, spec in sorted_truck_types:
            temp_truck = Truck(t_name, spec['w'], LIMIT_H, spec['l'], spec['weight'], spec['cost'])
            test_items = sorted(remaining_items, key=lambda x: x.volume, reverse=True)
            packed_count = 0
            for item in test_items:
                item_copy = Box(item.name, item.w, item.h, item.d, item.weight)
                item_copy.is_heavy = item.is_heavy
                if temp_truck.put_item(item_copy):
                    packed_count += 1
            
            if packed_count > 0:
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
# 4. 시각화 (최초 제공해주신 디자인 100% 복구)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    spec = TRUCK_DB[truck.name.split(' ')[0]]
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    
    # 1. 섀시
    chassis_h = 180
    fig.add_trace(go.Mesh3d(x=[0, W, W, 0, 0, W, W, 0], y=[0, 0, L, L, 0, 0, L, L], z=[-chassis_h, -chassis_h, -chassis_h, -chassis_h, 0, 0, 0, 0], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color='#222222', flatshading=True, showlegend=False))

    # 2. 바퀴 디자인 함수
    def create_realistic_wheel(cx, cy, cz, r, w):
        theta = np.linspace(0, 2*np.pi, 32)
        x_tire, y_tire, z_tire = [], [], []
        for t in theta:
            x_tire.extend([cx - w/2, cx + w/2])
            y_tire.extend([cy + r*np.cos(t), cy + r*np.cos(t)])
            z_tire.extend([cz + r*np.sin(t), cz + r*np.sin(t)])
        fig.add_trace(go.Mesh3d(x=x_tire, y=y_tire, z=z_tire, alphahull=0, color='#333333', flatshading=True, showlegend=False, lighting=dict(ambient=1.0)))
        hub_r, hub_w = r * 0.6, w * 0.1
        theta_hub = np.linspace(0, 2*np.pi, 16)
        x_hub, y_hub, z_hub = [cx + w/2 + hub_w], [cy], [cz]
        for t in theta_hub:
            x_hub.append(cx + w/2); y_hub.append(cy + hub_r*math.cos(t)); z_hub.append(cz + hub_r*math.sin(t))
        fig.add_trace(go.Mesh3d(x=x_hub, y=y_hub, z=z_hub, i=[0]*16, j=list(range(1,17)), k=list(range(2,17))+[1], color='#dddddd', flatshading=True, showlegend=False, lighting=dict(ambient=0.9)))

    wheel_r, wheel_w, wheel_z = 450, 280, -chassis_h - 100
    wheel_pos = [(-wheel_w/2, L*0.15), (W+wheel_w/2, L*0.15), (-wheel_w/2, L*0.85), (W+wheel_w/2, L*0.85)]
    for wx, wy in wheel_pos: create_realistic_wheel(wx, wy, wheel_z, wheel_r, wheel_w)

    # 3. 적재함 (Surface 사용 - 대각선 제거)
    wall_rgba = 'rgba(230, 230, 230, 0.4)'
    fig.add_trace(go.Surface(x=[[0, 0], [0, 0]], y=[[0, L], [0, L]], z=[[0, 0], [Real_H, Real_H]], colorscale=[[0, wall_rgba], [1, wall_rgba]], showscale=False, opacity=0.4))
    fig.add_trace(go.Surface(x=[[W, W], [W, W]], y=[[0, L], [0, L]], z=[[0, 0], [Real_H, Real_H]], colorscale=[[0, wall_rgba], [1, wall_rgba]], showscale=False, opacity=0.4))
    fig.add_trace(go.Surface(x=[[0, W], [0, W]], y=[[L, L], [L, L]], z=[[0, 0], [Real_H, Real_H]], colorscale=[[0, wall_rgba], [1, wall_rgba]], showscale=False, opacity=0.4))
    fig.add_trace(go.Surface(x=[[0, W], [0, W]], y=[[0, 0], [0, 0]], z=[[0, 0], [Real_H, Real_H]], colorscale=[[0, wall_rgba], [1, wall_rgba]], showscale=False, opacity=0.4))

    # 4. 치수선 및 높이제한 라인
    OFFSET = 1000
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), name='높이제한'))

    # 5. 박스 및 라벨
    annotations = []
    for item in truck.items:
        color = '#FF0000' if item.is_heavy else '#f39c12'
        x, y, z, w, h, d = item.x, item.y, item.z, item.w, item.h, item.d
        fig.add_trace(go.Mesh3d(x=[x,x+w,x+w,x, x,x+w,x+w,x], y=[y,y,y+d,y+d, y,y,y+d,y+d], z=[z,z,z,z, z+h,z+h,z+h,z+h], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color=color, flatshading=True, name=item.name))
        fig.add_trace(go.Scatter3d(x=[x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x], y=[y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d], z=[z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h], mode='lines', line=dict(color='black', width=2), showlegend=False))
        annotations.append(dict(x=x+w/2, y=y+d/2, z=z+h/2, text=f"<b>{item.name}</b>", showarrow=False, font=dict(color="white" if item.is_heavy else "black", size=12), bgcolor="rgba(0,0,0,0.5)" if item.is_heavy else "rgba(255,255,255,0.5)"))

    # 뷰 설정
    if camera_view == "top": eye = dict(x=0, y=0.01, z=2.5); up = dict(x=0, y=1, z=0)
    elif camera_view == "side": eye = dict(x=2.5, y=0, z=0.5); up = dict(x=0, y=0, z=1)
    else: eye = dict(x=2.0, y=-1.5, z=1.2); up = dict(x=0, y=0, z=1)

    fig.update_layout(scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), camera=dict(eye=eye, up=up), annotations=annotations), margin=dict(l=0,r=0,b=0,t=0), height=700, uirevision=str(uuid.uuid4()))
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 Ultimate Load Planner (Final_v2)")
if 'view_mode' not in st.session_state: st.session_state['view_mode'] = 'iso'
uploaded_file = st.sidebar.file_uploader("파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='cp949') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    if st.button("최적 배차 실행", type="primary"):
        st.session_state['run_result'] = load_data(df)
    
    if 'run_result' in st.session_state:
        trucks = run_optimization(st.session_state['run_result'])
        if trucks:
            c1, c2, c3, _ = st.columns([1,1,1,5])
            with c1: 
                if st.button("쿼터뷰"): st.session_state['view_mode'] = 'iso'
            with c2: 
                if st.button("탑뷰"): st.session_state['view_mode'] = 'top'
            with c3: 
                if st.button("사이드뷰"): st.session_state['view_mode'] = 'side'
            
            tabs = st.tabs([t.name for t in trucks])
            for i, tab in enumerate(tabs):
                with tab:
                    t = trucks[i]
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"### **{t.name}**")
                        st.metric("총 중량", f"{t.total_weight:,} kg")
                        st.metric("운송비용", f"{int(t.cost):,} 원")
                        st.write(f"- 박스: {len(t.items)}개")
                    with col2:
                        st.plotly_chart(draw_truck_3d(t, st.session_state['view_mode']), use_container_width=True)
