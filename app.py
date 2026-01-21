import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import uuid

# ==========================================
# 0. 세션 초기화 (가장 먼저 실행)
# ==========================================
# 페이지가 새로고침될 때마다 불필요한 데이터 충돌을 막기 위해 초기화 토큰을 확인합니다.
if 'app_reset_token' not in st.session_state:
    st.session_state.clear()
    st.session_state['app_reset_token'] = str(uuid.uuid4())

# ==========================================
# 1. 물리 엔진 (SafeBox, SafeTruck)
# ==========================================
class SafeBox:
    def __init__(self, name, w, h, d, weight):
        self.name = str(name)
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

class SafeTruck:
    def __init__(self, name, w, h, d, max_weight, cost):
        self.name = name
        self.w = float(w)
        self.h = float(h)
        self.d = float(d)
        self.max_weight = float(max_weight)
        self.cost = cost
        self.loaded = []
        self.current_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

    def try_loading(self, item):
        if self.current_weight + item.weight > self.max_weight:
            return False
        
        # 피벗 정렬: Z(아래) -> Y(안쪽) -> X(왼쪽)
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))

        for p in self.pivots:
            px, py, pz = p
            # 1. 공간 체크
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h):
                continue
            # 2. 충돌 체크
            if self._collision_check(item, px, py, pz):
                continue
            # 3. 지지 체크
            if not self._support_check(item, px, py, pz):
                continue

            item.x, item.y, item.z = px, py, pz
            self.loaded.append(item)
            self.current_weight += item.weight
            
            self.pivots.append([item.x + item.w, item.y, item.z])
            self.pivots.append([item.x, item.y + item.d, item.z])
            self.pivots.append([item.x, item.y, item.z + item.h])
            return True
        return False

    def _collision_check(self, item, x, y, z):
        for exist in self.loaded:
            if (x < exist.x + exist.w and x + item.w > exist.x and
                y < exist.y + exist.d and y + item.d > exist.y and
                z < exist.z + exist.h and z + item.h > exist.z):
                return True
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
# 2. 데이터 (비용 포함)
# ==========================================
TRUCK_DATA = {
    "1톤":    {"w": 1600, "l": 2800, "h": 1700, "weight": 1000, "cost": 100000},
    "1.4톤":  {"w": 1650, "l": 3400, "h": 1800, "weight": 1400, "cost": 130000},
    "2.5톤":  {"w": 1800, "l": 4300, "h": 2000, "weight": 2500, "cost": 180000},
    "3.5톤":  {"w": 2000, "l": 4800, "h": 2000, "weight": 3500, "cost": 220000},
    "5톤":    {"w": 2350, "l": 6200, "h": 2350, "weight": 5000, "cost": 300000},
    "5톤축":  {"w": 2350, "l": 7300, "h": 2350, "weight": 8000, "cost": 350000},
    "11톤":   {"w": 2350, "l": 9600, "h": 2400, "weight": 11000, "cost": 450000},
    "18톤":   {"w": 2350, "l": 10200, "h": 2500, "weight": 18000, "cost": 550000},
    "25톤":   {"w": 2350, "l": 10200, "h": 2500, "weight": 25000, "cost": 650000},
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
        else:
            heavy_limit = 999999999
    except:
        heavy_limit = 999999999

    for _, row in df.iterrows():
        try:
            b = SafeBox(row['박스번호'], row['폭'], row['높이'], row['길이'], row['중량'])
            if b.weight >= heavy_limit and b.weight > 0:
                b.is_heavy = True
            results.append(b)
        except:
            continue
    return results

def solve_loading(items):
    if not items: return []
    remaining = items[:]
    results = []
    
    types_cost = sorted(TRUCK_DATA.keys(), key=lambda k: TRUCK_DATA[k]['cost'])
    types_size = sorted(TRUCK_DATA.keys(), key=lambda k: TRUCK_DATA[k]['weight'], reverse=True)
    
    # 1. 단일 차량 시도
    for t_name in types_cost:
        spec = TRUCK_DATA[t_name]
        t = SafeTruck(t_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
        
        temp_items = sorted(remaining, key=lambda x: x.volume, reverse=True)
        success = True
        for item in temp_items:
            i_copy = SafeBox(item.name, item.w, item.h, item.d, item.weight)
            i_copy.is_heavy = item.is_heavy
            if not t.try_loading(i_copy):
                success = False
                break
        
        if success:
            t.name = f"{t_name} (단일차량)"
            return [t]

    # 2. 다중 차량
    while remaining:
        best_t = None
        max_cnt = -1
        packed_names = []
        
        for t_name in types_size:
            spec = TRUCK_DATA[t_name]
            t = SafeTruck(t_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
            
            temp_items = sorted(remaining, key=lambda x: x.volume, reverse=True)
            cnt = 0
            curr_names = []
            
            for item in temp_items:
                i_copy = SafeBox(item.name, item.w, item.h, item.d, item.weight)
                i_copy.is_heavy = item.is_heavy
                if t.try_loading(i_copy):
                    cnt += 1
                    curr_names.append(item.name)
            
            if cnt > max_cnt:
                max_cnt = cnt
                best_t = t
                packed_names = curr_names
        
        if best_t and max_cnt > 0:
            best_t.name = f"{best_t.name} #{len(results)+1}"
            results.append(best_t)
            remaining = [r for r in remaining if r.name not in packed_names]
        else:
            break
            
    return results

# ==========================================
# 4. 시각화 (No Numpy)
# ==========================================
def draw_3d(truck, view):
    fig = go.Figure()
    spec = TRUCK_DATA[truck.name.split(' ')[0]]
    W, L = spec['w'], spec['l']
    
    # 1. 섀시
    fig.add_trace(go.Mesh3d(x=[0,W,W,0,0,W,W,0], y=[0,0,L,L,0,0,L,L], z=[-180]*4+[0]*4, color='#222222', alphahull=0, showlegend=False))
    
    # 2. 바퀴
    def add_wheel(cx, cy):
        steps = 16
        tx, ty, tz = [], [], []
        for i in range(steps):
            rad = (2*math.pi/steps)*i
            tx.extend([cx-100, cx+100])
            ty.extend([cy+450*math.cos(rad), cy+450*math.cos(rad)])
            tz.extend([-250+450*math.sin(rad), -250+450*math.sin(rad)])
        fig.add_trace(go.Mesh3d(x=tx, y=ty, z=tz, alphahull=0, color='#333333', showlegend=False))
    
    for y_pos in [L*0.2, L*0.8]:
        add_wheel(-140, y_pos)
        add_wheel(W+140, y_pos)

    # 3. 벽면 (투명)
    wall_c = [[0,'rgba(200,200,200,0.1)'],[1,'rgba(200,200,200,0.1)']]
    H = 2300
    fig.add_trace(go.Surface(x=[[0,0],[0,0]], y=[[0,L],[0,L]], z=[[0,0],[H,H]], colorscale=wall_c, showscale=False, opacity=0.1))
    fig.add_trace(go.Surface(x=[[W,W],[W,W]], y=[[0,L],[0,L]], z=[[0,0],[H,H]], colorscale=wall_c, showscale=False, opacity=0.1))
    fig.add_trace(go.Surface(x=[[0,W],[0,W]], y=[[L,L],[L,L]], z=[[0,0],[H,H]], colorscale=wall_c, showscale=False, opacity=0.1))
    
    # 4. 프레임
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[0,0,0,0,0], mode='lines', line=dict(color='black', width=4), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[H,H,H,H,H], mode='lines', line=dict(color='black', width=4), showlegend=False))
    for px, py in [(0,0), (W,0), (W,L), (0,L)]:
        fig.add_trace(go.Scatter3d(x=[px,px], y=[py,py], z=[0,H], mode='lines', line=dict(color='black', width=4), showlegend=False))

    # 5. 제한선 (1.3m)
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[1300]*5, mode='lines', line=dict(color='red', width=5, dash='dash'), showlegend=False))

    # 6. 화물
    for b in truck.loaded:
        c = '#FF0000' if b.is_heavy else '#f39c12'
        x, y, z, w, h, d = b.x, b.y, b.z, b.w, b.h, b.d
        fig.add_trace(go.Mesh3d(
            x=[x,x+w,x+w,x, x,x+w,x+w,x], y=[y,y,y+d,y+d, y,y,y+d,y+d], z=[z,z,z,z, z+h,z+h,z+h,z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=c, opacity=1.0, flatshading=True, name=b.name
        ))
        fig.add_trace(go.Scatter3d(
            x=[x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x],
            y=[y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d],
            z=[z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h],
            mode='lines', line=dict(color='black', width=2), showlegend=False
        ))

    eye = dict(x=2.0, y=-1.5, z=1.2)
    if view == 'Top': eye = dict(x=0, y=0.1, z=2.5)
    
    fig.update_layout(scene=dict(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor='white', camera=dict(eye=eye)), margin=dict(l=0,r=0,b=0,t=0), height=600)
    return fig

# ==========================================
# 5. UI
# ==========================================
st.title("🚛 Final Safe Load Planner")
st.caption("안전 모드: Numpy 제거, 순수 Python 로직")

if 'view_mode' not in st.session_state: st.session_state['view_mode'] = 'ISO'

up_file = st.file_uploader("파일 업로드 (xlsx, csv)", type=['xlsx', 'csv'])

if up_file:
    try:
        if up_file.name.endswith('.csv'): df = pd.read_csv(up_file, encoding='cp949')
        else: df = pd.read_excel(up_file)
        
        df.columns = [c.strip() for c in df.columns]
        st.dataframe(df.head())
        
        if st.button("배차 실행"):
            items = parse_file(df)
            if not items:
                st.error("데이터를 읽을 수 없습니다.")
            else:
                trucks = solve_loading(items)
                
                if trucks:
                    cost_sum = sum(t.cost for t in trucks)
                    st.success(f"완료! 총 {len(trucks)}대 배차 (예상 비용: {cost_sum:,}원)")
                    
                    c1, c2 = st.columns([1, 5])
                    if c1.button("뷰 전환"):
                        st.session_state['view_mode'] = 'Top' if st.session_state['view_mode'] == 'ISO' else 'ISO'
                    
                    tabs = st.tabs([t.name for t in trucks])
                    for i, tab in enumerate(tabs):
                        with tab:
                            st.plotly_chart(draw_3d(trucks[i], st.session_state['view_mode']), use_container_width=True)
                else:
                    st.warning("적재 실패: 화물이 너무 크거나 무겁습니다.")
                    
    except Exception as e:
        st.error(f"오류: {e}")
