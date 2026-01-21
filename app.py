import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import uuid

# ==========================================
# 0. 세션 및 캐시 초기화 (데이터 꼬임 방지)
# ==========================================
if 'final_fix_v10' not in st.session_state:
    st.session_state.clear()
    st.session_state['final_fix_v10'] = str(uuid.uuid4())

# ==========================================
# 1. 물리 엔진 (SafeBox, SafeTruck)
# ==========================================
class SafeBox:
    def __init__(self, name, w, h, d, weight):
        self.name = str(name)
        self.w = float(w); self.h = float(h); self.d = float(d)
        self.weight = float(weight)
        self.x = 0.0; self.y = 0.0; self.z = 0.0
        self.is_heavy = False
    @property
    def volume(self): return self.w * self.h * self.d

class SafeTruck:
    def __init__(self, name, w, h, d, max_weight, cost):
        self.name = name
        self.w = float(w); self.h = float(h); self.d = float(d)
        self.max_weight = float(max_weight); self.cost = cost
        self.loaded = []; self.current_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

    def try_loading(self, item):
        if self.current_weight + item.weight > self.max_weight: return False
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))
        for p in self.pivots:
            px, py, pz = p
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h): continue
            if self._collision_check(item, px, py, pz): continue
            if not self._support_check(item, px, py, pz): continue
            item.x, item.y, item.z = px, py, pz
            self.loaded.append(item); self.current_weight += item.weight
            self.pivots.append([item.x + item.w, item.y, item.z])
            self.pivots.append([item.x, item.y + item.d, item.z])
            self.pivots.append([item.x, item.y, item.z + item.h])
            return True
        return False

    def _collision_check(self, item, x, y, z):
        for exist in self.loaded:
            if (x < exist.x + exist.w and x + item.w > exist.x and
                y < exist.y + exist.d and y + item.d > exist.y and
                z < exist.z + exist.h and z + item.h > exist.z): return True
        return False

    def _support_check(self, item, x, y, z):
        if z <= 0.001: return True
        support_area = 0.0
        for exist in self.loaded:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0.0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy
        return support_area >= (item.w * item.d * 0.6)

# ==========================================
# 2. 데이터 (5톤 폭 2100 반영 및 단가표 수정)
# ==========================================
st.set_page_config(layout="wide", page_title="Load Planner (Perfect Data)")

# [최종] 사진 데이터와 100% 일치화
TRUCK_DATA = {
    "1톤":    {"w": 1600, "l": 2800, "real_h": 1700, "weight": 1000,  "cost": 45000},
    "2.5톤":  {"w": 1800, "l": 4300, "real_h": 2000, "weight": 2500,  "cost": 100000},
    "5톤":    {"w": 2100, "l": 6200, "real_h": 2350, "weight": 5000,  "cost": 145000}, # 폭 2100 수정완료
    "8톤":    {"w": 2350, "l": 7300, "real_h": 2350, "weight": 8000,  "cost": 170000},
    "11톤":   {"w": 2350, "l": 9600, "real_h": 2400, "weight": 11000, "cost": 200000},
    "15톤":   {"w": 2350, "l": 9600, "real_h": 2500, "weight": 15000, "cost": 240000},
    "18톤":   {"w": 2350, "l": 10200, "real_h": 2500, "weight": 18000, "cost": 270000},
    "22톤":   {"w": 2350, "l": 10200, "real_h": 2500, "weight": 22000, "cost": 310000},
}

# ==========================================
# 3. 로직 함수
# ==========================================
def parse_file(df):
    results = []
    try:
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        if weights:
            weights.sort(reverse=True)
            cutoff = max(0, int(len(weights)*0.1)-1)
            heavy_limit = weights[cutoff]
        else: heavy_limit = 999999999
    except: heavy_limit = 999999999
    for _, row in df.iterrows():
        try:
            b = SafeBox(row['박스번호'], row['폭'], row['높이'], row['길이'], row['중량'])
            b.is_heavy = (b.weight >= heavy_limit and b.weight > 0)
            results.append(b)
        except: continue
    return results

def solve_loading(items):
    if not items: return []
    remaining = items[:]
    results = []
    
    # 비용 최적화 (가장 싼 차부터 시도)
    types_cost_asc = sorted(TRUCK_DATA.keys(), key=lambda k: TRUCK_DATA[k]['cost'])
    for t_name in types_cost_asc:
        spec = TRUCK_DATA[t_name]
        t = SafeTruck(t_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
        temp_items = sorted(remaining, key=lambda x: x.volume, reverse=True)
        success = True
        for it in temp_items:
            it_cp = SafeBox(it.name, it.w, it.h, it.d, it.weight)
            it_cp.is_heavy = it.is_heavy
            if not t.try_loading(it_cp):
                success = False; break
        if success:
            t.name = f"{t_name} (단일)"
            return [t]

    # 다중 차량 (큰 차 우선)
    types_size_desc = sorted(TRUCK_DATA.keys(), key=lambda k: TRUCK_DATA[k]['weight'], reverse=True)
    while remaining:
        best_t = None; max_cnt = -1; packed_names = []
        for t_name in types_size_desc:
            spec = TRUCK_DATA[t_name]
            t = SafeTruck(t_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
            temp_items = sorted(remaining, key=lambda x: x.volume, reverse=True)
            cnt = 0; curr_names = []
            for it in temp_items:
                it_cp = SafeBox(it.name, it.w, it.h, it.d, it.weight)
                it_cp.is_heavy = it.is_heavy
                if t.try_loading(it_cp):
                    cnt += 1; curr_names.append(it.name)
            if cnt > max_cnt:
                max_cnt = cnt; best_t = t; packed_names = curr_names
        if best_t and max_cnt > 0:
            best_t.name = f"{best_t.name} #{len(results)+1}"
            results.append(best_t)
            remaining = [r for r in remaining if r.name not in packed_names]
        else: break
    return results

# ==========================================
# 4. 시각화 (No Numpy)
# ==========================================
def draw_3d(truck, view):
    fig = go.Figure()
    base_name = truck.name.split(' ')[0]
    spec = TRUCK_DATA.get(base_name)
    W, L, H = spec['w'], spec['l'], spec['real_h']
    
    fig.add_trace(go.Mesh3d(x=[0,W,W,0,0,W,W,0], y=[0,0,L,L,0,0,L,L], z=[-180]*4+[0]*4, color='#222222', alphahull=0, showlegend=False))
    def add_wheel(cx, cy):
        steps = 16
        tx, ty, tz = [], [], []
        for i in range(steps):
            rad = (2*math.pi/steps)*i
            tx.extend([cx-100, cx+100])
            ty.extend([cy+450*math.cos(rad), cy+450*math.cos(rad)])
            tz.extend([-250+450*math.sin(rad), -250+450*math.sin(rad)])
        fig.add_trace(go.Mesh3d(x=tx, y=ty, z=tz, alphahull=0, color='#333333', showlegend=False))
    for yp in [L*0.2, L*0.8]:
        add_wheel(-140, yp); add_wheel(W+140, yp)
    wall_c = [[0,'rgba(200,200,200,0.1)'],[1,'rgba(200,200,200,0.1)']]
    fig.add_trace(go.Surface(x=[[0,0],[0,0]], y=[[0,L],[0,L]], z=[[0,0],[H,H]], colorscale=wall_c, showscale=False, opacity=0.1))
    fig.add_trace(go.Surface(x=[[W,W],[W,W]], y=[[0,L],[0,L]], z=[[0,0],[H,H]], colorscale=wall_c, showscale=False, opacity=0.1))
    fig.add_trace(go.Surface(x=[[0,W],[0,W]], y=[[L,L],[L,L]], z=[[0,0],[H,H]], colorscale=wall_c, showscale=False, opacity=0.1))
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0, 0,W,W,0,0], y=[0,0,L,L,0, 0,0,L,L,0], z=[0,0,0,0,0, H,H,H,H,H], mode='lines', line=dict(color='black', width=4), showlegend=False))
    for px, py in [(0,0), (W,0), (W,L), (0,L)]:
        fig.add_trace(go.Scatter3d(x=[px,px], y=[py,py], z=[0,H], mode='lines', line=dict(color='black', width=4), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[1300]*5, mode='lines', line=dict(color='red', width=5, dash='dash'), showlegend=False))
    for b in truck.loaded:
        c = '#FF0000' if b.is_heavy else '#f39c12'
        x, y, z, w, h, d = b.x, b.y, b.z, b.w, b.h, b.d
        fig.add_trace(go.Mesh3d(x=[x,x+w,x+w,x, x,x+w,x+w,x], y=[y,y,y+d,y+d, y,y,y+d,y+d], z=[z,z,z,z, z+h,z+h,z+h,z+h], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color=c, opacity=1.0, flatshading=True, name=b.name))
        fig.add_trace(go.Scatter3d(x=[x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x], y=[y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d], z=[z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h], mode='lines', line=dict(color='black', width=2), showlegend=False))
    eye = dict(x=2.0, y=-1.5, z=1.2)
    if view == 'Top': eye = dict(x=0, y=0.1, z=2.5)
    fig.update_layout(scene=dict(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor='white', camera=dict(eye=eye)), margin=dict(l=0,r=0,b=0,t=0), height=600)
    return fig

# ==========================================
# 5. UI
# ==========================================
st.title("🚛 Final Load Planner (Perfect Data V10)")
st.caption("✅ 5톤 폭 2100mm | 사진 단가 100% 반영 | 8종 차량")

up_file = st.file_uploader("파일 업로드", type=['xlsx', 'csv'])
if up_file:
    try:
        if up_file.name.endswith('.csv'): df = pd.read_csv(up_file, encoding='cp949')
        else: df = pd.read_excel(up_file)
        df.columns = [c.strip() for c in df.columns]
        st.dataframe(df.head())
        if st.button("배차 실행"):
            items = parse_file(df)
            if items:
                trucks = solve_loading(items)
                if trucks:
                    st.success(f"완료! 총 비용: {sum(t.cost for t in trucks):,}원")
                    view = st.radio("뷰", ["ISO", "Top"], horizontal=True)
                    tabs = st.tabs([t.name for t in trucks])
                    for i, tab in enumerate(tabs):
                        with tab:
                            t = trucks[i]
                            st.write(f"**{t.name}** | 비용: {t.cost:,}원 | {len(t.loaded)}개 적재")
                            st.plotly_chart(draw_3d(t, view), use_container_width=True)
    except Exception as e: st.error(f"오류: {e}")
