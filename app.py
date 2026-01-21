import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math

# ==========================================
# 0. 생존 신고 (제목 확인용)
# ==========================================
st.set_page_config(layout="wide", page_title="업데이트 확인용")

# ==========================================
# 1. 커스텀 물리 엔진
# ==========================================
class Box:
    def __init__(self, name, w, h, d, weight):
        self.name = name
        self.w = float(w)
        self.h = float(h)
        self.d = float(d)
        self.weight = float(weight)
        self.x = 0.0; self.y = 0.0; self.z = 0.0
        self.is_heavy = False
    @property
    def volume(self): return self.w * self.h * self.d

class Truck:
    def __init__(self, name, w, h, d, max_weight):
        self.name = name
        self.w = float(w); self.h = float(h); self.d = float(d)
        self.max_weight = float(max_weight)
        self.items = []
        self.total_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

    def put_item(self, item):
        fit = False
        if self.total_weight + item.weight > self.max_weight: return False
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))
        for p in self.pivots:
            px, py, pz = p
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h): continue
            if self._check_collision(item, px, py, pz): continue
            if not self._check_support(item, px, py, pz): continue
            item.x, item.y, item.z = px, py, pz
            self.items.append(item)
            self.total_weight += item.weight
            fit = True
            break
        if fit:
            self.pivots.append([item.x + item.w, item.y, item.z])
            self.pivots.append([item.x, item.y + item.d, item.z])
            self.pivots.append([item.x, item.y, item.z + item.h])
        return fit

    def _check_collision(self, item, x, y, z):
        for exist in self.items:
            if (x < exist.x + exist.w and x + item.w > exist.x and
                y < exist.y + exist.d and y + item.d > exist.y and
                z < exist.z + exist.h and z + item.h > exist.z): return True
        return False

    def _check_support(self, item, x, y, z):
        if z <= 0.001: return True
        support_area = 0.0
        for exist in self.items:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0.0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy
        return support_area >= item.w * item.d * 0.6

# ==========================================
# 2. 데이터 및 로직
# ==========================================
TRUCK_DB = {
    "5톤":  {"w": 2350, "real_h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "real_h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "real_h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "real_h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "real_h": 2350, "l": 10200, "weight": 24000},
}

def load_data(df):
    items = []
    try:
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        if weights:
            sorted_weights = sorted(weights, reverse=True)
            cutoff_index = max(0, int(len(weights) * 0.1) - 1)
            heavy_threshold = sorted_weights[cutoff_index]
        else: heavy_threshold = 999999999
    except: heavy_threshold = 999999999

    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            w, h, l, weight = float(row['폭']), float(row['높이']), float(row['길이']), float(row['중량'])
            box = Box(name, w, h, l, weight)
            box.is_heavy = (weight >= heavy_threshold and weight > 0)
            items.append(box)
        except: continue
    return items

def run_optimization(all_items):
    remaining_items = all_items[:]
    used_trucks = [] 
    truck_types = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])
    while remaining_items:
        best_truck = None
        best_score = -1
        for t_name in truck_types:
            spec = TRUCK_DB[t_name]
            temp_truck = Truck(t_name, spec['w'], 1300, spec['l'], spec['weight'])
            test_items = sorted(remaining_items, key=lambda x: x.volume, reverse=True)
            packed_count = 0
            for item in test_items:
                item_copy = Box(item.name, item.w, item.h, item.d, item.weight)
                item_copy.is_heavy = item.is_heavy 
                if temp_truck.put_item(item_copy): packed_count += 1
            if packed_count > 0:
                if packed_count == len(remaining_items): score = 100000 - spec['weight']
                else: score = (temp_truck.total_weight / spec['weight']) * 100
                if score > best_score: best_score = score; best_truck = temp_truck
        if best_truck and len(best_truck.items) > 0:
            best_truck.name = f"{best_truck.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_truck)
            packed_names = [i.name for i in best_truck.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else: break
    return used_trucks

# ==========================================
# 3. 시각화 (여기가 핵심!)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    spec = TRUCK_DB[truck.name.split(' ')[0]]
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    
    # 1. 섀시 & 바퀴
    fig.add_trace(go.Mesh3d(x=[0, W, W, 0, 0, W, W, 0], y=[0, 0, L, L, 0, 0, L, L], z=[-150]*4+[0]*4, color='#222222', i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], showlegend=False))
    
    # 바퀴 (디테일 휠)
    def create_wheel(cx, cy):
        # 타이어
        th = np.linspace(0, 2*np.pi, 32)
        x, y, z = [], [], []
        for t in th:
            x.extend([cx-100, cx+100])
            y.extend([cy+450*np.cos(t), cy+450*np.cos(t)])
            z.extend([-150-100+450*np.sin(t), -150-100+450*np.sin(t)])
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, alphahull=0, color='#111111', lighting=dict(ambient=0.5, diffuse=0.8), showlegend=False))
        # 휠 허브 (은색)
        xh, yh, zh = [], [], []
        xh.append(cx+120); yh.append(cy); zh.append(-250) # 중심
        for t in th:
            xh.append(cx+100)
            yh.append(cy+250*np.cos(t))
            zh.append(-250+250*np.sin(t))
        i, j, k = [0]*32, list(range(1,33)), list(range(2,33))+[1]
        fig.add_trace(go.Mesh3d(x=xh, y=yh, z=zh, i=i, j=j, k=k, color='#dddddd', lighting=dict(specular=1.0), showlegend=False))

    for wy in [L*0.15, L*0.30, L*0.70, L*0.85]:
        create_wheel(-140, wy); create_wheel(W+140, wy)

    # 2. 적재함 벽면 (Surface 사용 -> 대각선 선이 절대 나올 수 없음)
    # 벽면이 투명한 유리처럼 보이게 설정
    wall_c = [[0, 'rgba(200,200,200,0.2)'], [1, 'rgba(200,200,200,0.2)']]
    # 좌/우/앞/뒤 (평면 사각형 4개)
    fig.add_trace(go.Surface(x=[[0,0],[0,0]], y=[[0,L],[0,L]], z=[[0,0],[Real_H,Real_H]], colorscale=wall_c, showscale=False)) # 좌
    fig.add_trace(go.Surface(x=[[W,W],[W,W]], y=[[0,L],[0,L]], z=[[0,0],[Real_H,Real_H]], colorscale=wall_c, showscale=False)) # 우
    fig.add_trace(go.Surface(x=[[0,W],[0,W]], y=[[L,L],[L,L]], z=[[0,0],[Real_H,Real_H]], colorscale=wall_c, showscale=False)) # 앞
    fig.add_trace(go.Surface(x=[[0,W],[0,W]], y=[[0,0],[0,0]], z=[[0,0],[Real_H,Real_H]], colorscale=wall_c, showscale=False)) # 뒤 (문)

    # 3. 프레임 (진한 회색 선)
    lx = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    ly = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lz = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='#333333', width=5), showlegend=False))

    # 4. 치수선
    def dim_line(p1, p2, txt, c='black'):
        fig.add_trace(go.Scatter3d(x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]], mode='lines+text', text=[f"", f"", f"<b>{txt}</b>"], textposition="middle center", line=dict(color=c, width=2), showlegend=False))
        # 화살표(Cone)
        v = np.array(p2)-np.array(p1); v = v/np.linalg.norm(v)
        fig.add_trace(go.Cone(x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]], u=[-v[0],v[0]], v=[-v[1],v[1]], w=[-v[2],v[2]], sizemode="absolute", sizeref=150, showscale=False, colorscale=[[0,c],[1,c]]))

    offset = 1200
    dim_line([0,-offset,0], [W,-offset,0], f"폭: {int(W)}")
    dim_line([-offset,0,0], [-offset,L,0], f"길이: {int(L)}")
    dim_line([-offset,L,0], [-offset,L,1300], f"높이제한: 1300", c='red')
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[1300]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), showlegend=False))

    # 5. 박스 그리기
    for item in truck.items:
        color = '#FF0000' if getattr(item, 'is_heavy', False) else '#f39c12'
        x, y, z = item.x, item.y, item.z; w, h, d = item.w, item.h, item.d
        fig.add_trace(go.Mesh3d(x=[x,x+w,x+w,x,x,x+w,x+w,x], y=[y,y,y+d,y+d,y,y,y+d,y+d], z=[z,z,z,z,z+h,z+h,z+h,z+h], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color=color, opacity=1.0, flatshading=True, name=item.name))
        # 테두리
        fig.add_trace(go.Scatter3d(x=[x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x], y=[y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d], z=[z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h], mode='lines', line=dict(color='black', width=2), showlegend=False))
        # 라벨
        fig.add_trace(go.Scatter3d(x=[x+w/2], y=[y], z=[z+h/2], mode='text', text=[f"<b>{item.name}</b>"], textfont=dict(size=14, color="white" if color=='#FF0000' else "black")))

    # 카메라
    eye = dict(x=2.0, y=-1.5, z=1.2)
    if camera_view == 'top': eye = dict(x=0, y=0.1, z=2.5)
    elif camera_view == 'side': eye = dict(x=2.5, y=0, z=0.5)
    
    fig.update_layout(scene=dict(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor='white', camera=dict(eye=eye)), margin=dict(l=0,r=0,b=0,t=0), height=700)
    return fig

# ==========================================
# 4. 메인 화면
# ==========================================
st.title("🚀 업데이트 성공!! (제발 바껴라)") # <--- 이 제목이 보여야 합니다!!!
st.caption("✅ 물리엔진 | 회전금지 | 1.3m 제한 | 뷰 컨트롤 | 디자인 최종 수정")

if 'view_mode' not in st.session_state: st.session_state['view_mode'] = 'iso'
uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file, encoding='cp949')
    else: df = pd.read_excel(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    st.dataframe(df)
    
    if st.button("최적 배차 실행", type="primary"):
        items = load_data(df)
        if not items: st.error("데이터 오류")
        else:
            trucks = run_optimization(items)
            if trucks:
                st.success(f"배차 완료: 총 {len(trucks)}대")
                c1, c2, c3, _ = st.columns([1,1,1,5])
                if c1.button("쿼터뷰"): st.session_state['view_mode']='iso'
                if c2.button("탑뷰"): st.session_state['view_mode']='top'
                if c3.button("사이드뷰"): st.session_state['view_mode']='side'
                
                tabs = st.tabs([t.name for t in trucks])
                for i, tab in enumerate(tabs):
                    with tab:
                        st.plotly_chart(draw_truck_3d(trucks[i], st.session_state['view_mode']), use_container_width=True)
            else: st.warning("적재 불가")
