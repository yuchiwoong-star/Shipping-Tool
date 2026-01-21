import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import uuid

# ==========================================
# 0. 세션 및 캐시 강제 초기화
# ==========================================
if 'reset_counter' not in st.session_state:
    st.session_state.clear()
    st.session_state['reset_counter'] = 0

# ==========================================
# 1. 커스텀 물리 엔진 (간소화 및 변수명 통일)
# ==========================================
class BoxItem: # 클래스 이름 변경으로 충돌 방지
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

class TruckObject: # 클래스 이름 변경으로 충돌 방지
    def __init__(self, name, w, h, d, max_weight, cost):
        self.name = name
        self.w = float(w)
        self.h = float(h)
        self.d = float(d)
        self.max_weight = float(max_weight)
        self.cost = cost
        self.cargo_list = [] # items -> cargo_list 로 확실히 변경
        self.total_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

    def try_load(self, box): # put_item -> try_load 이름 변경
        if self.total_weight + box.weight > self.max_weight:
            return False
        
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))

        for p in self.pivots:
            px, py, pz = p
            # 공간 체크
            if (px + box.w > self.w) or (py + box.d > self.d) or (pz + box.h > self.h):
                continue
            # 충돌 체크
            if self._check_collision(box, px, py, pz):
                continue
            # 지지 체크
            if not self._check_support(box, px, py, pz):
                continue

            # 적재 확정
            box.x, box.y, box.z = px, py, pz
            self.cargo_list.append(box)
            self.total_weight += box.weight
            
            # 피봇 추가
            self.pivots.append([box.x + box.w, box.y, box.z])
            self.pivots.append([box.x, box.y + box.d, box.z])
            self.pivots.append([box.x, box.y, box.z + box.h])
            return True
        return False

    def _check_collision(self, new_box, x, y, z):
        for exist in self.cargo_list:
            if (x < exist.x + exist.w and x + new_box.w > exist.x and
                y < exist.y + exist.d and y + new_box.d > exist.y and
                z < exist.z + exist.h and z + new_box.h > exist.z):
                return True
        return False

    def _check_support(self, new_box, x, y, z):
        if z <= 0.001: return True
        support_area = 0.0
        for exist in self.cargo_list:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + new_box.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0.0, min(y + new_box.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy
        return support_area >= (new_box.w * new_box.d * 0.6)

# ==========================================
# 2. 데이터 및 설정
# ==========================================
st.set_page_config(layout="wide", page_title="Load Planner v5 (Fixed)")

TRUCK_DB = {
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
def load_data_from_df(df):
    boxes = []
    try:
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        if weights:
            weights.sort(reverse=True)
            cutoff = max(0, int(len(weights) * 0.1) - 1)
            heavy_limit = weights[cutoff]
        else:
            heavy_limit = 999999999
    except:
        heavy_limit = 999999999

    for index, row in df.iterrows():
        try:
            b = BoxItem(row['박스번호'], row['폭'], row['높이'], row['길이'], row['중량'])
            b.is_heavy = (b.weight >= heavy_limit and b.weight > 0)
            boxes.append(b)
        except:
            continue
    return boxes

def run_solver(all_boxes):
    if not all_boxes: return []
    
    remaining = all_boxes[:]
    dispatched_trucks = []
    
    # 비용 순 정렬
    truck_types_cost_asc = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['cost'])
    truck_types_weight_desc = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'], reverse=True)
    
    # 1. 단일 차량 시도
    for t_name in truck_types_cost_asc:
        spec = TRUCK_DB[t_name]
        t = TruckObject(t_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
        
        test_boxes = sorted(remaining, key=lambda x: x.volume, reverse=True)
        success = True
        for b in test_boxes:
            # 객체 복사
            b_copy = BoxItem(b.name, b.w, b.h, b.d, b.weight)
            b_copy.is_heavy = b.is_heavy
            if not t.try_load(b_copy):
                success = False
                break
        
        if success:
            t.name = f"{t_name} (단일차량)"
            return [t]

    # 2. 다중 차량 배차 (Greedy)
    while remaining:
        best_t = None
        max_cnt = -1
        packed_names = []
        
        for t_name in truck_types_weight_desc:
            spec = TRUCK_DB[t_name]
            t = TruckObject(t_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
            
            test_boxes = sorted(remaining, key=lambda x: x.volume, reverse=True)
            cnt = 0
            current_names = []
            
            for b in test_boxes:
                b_copy = BoxItem(b.name, b.w, b.h, b.d, b.weight)
                b_copy.is_heavy = b.is_heavy
                if t.try_load(b_copy):
                    cnt += 1
                    current_names.append(b.name)
            
            if cnt > max_cnt:
                max_cnt = cnt
                best_t = t
                packed_names = current_names

        if best_t and max_cnt > 0:
            best_t.name = f"{best_t.name} (No.{len(dispatched_trucks)+1})"
            dispatched_trucks.append(best_t)
            remaining = [b for b in remaining if b.name not in packed_names]
        else:
            break
            
    return dispatched_trucks

# ==========================================
# 4. 시각화 (math 사용)
# ==========================================
def draw_scene(truck, view_mode="iso"):
    fig = go.Figure()
    spec = TRUCK_DB[truck.name.split(' ')[0]]
    W, L, H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    
    # 1. 섀시
    fig.add_trace(go.Mesh3d(x=[0, W, W, 0, 0, W, W, 0], y=[0, 0, L, L, 0, 0, L, L], z=[-180]*4+[0]*4, color='#222222', i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], showlegend=False))

    # 2. 바퀴
    def add_wheel(cx, cy):
        steps=24
        tx, ty, tz = [], [], [] # 타이어
        hx, hy, hz = [], [], [] # 허브
        lx, ly, lz = [], [], [] # 트레드
        
        for i in range(steps):
            a = (2*math.pi/steps)*i
            # 타이어
            tx.extend([cx-100, cx+100])
            ty.extend([cy+450*math.cos(a), cy+450*math.cos(a)])
            tz.extend([-250+450*math.sin(a), -250+450*math.sin(a)])
            # 허브
            hx.append(cx+100); hy.append(cy+250*math.cos(a)); hz.append(-250+250*math.sin(a))
        
        # 타이어 Mesh
        fig.add_trace(go.Mesh3d(x=tx, y=ty, z=tz, alphahull=0, color='#333333', lighting=dict(ambient=1.0, diffuse=0.0), showlegend=False))
        
        # 허브 Mesh
        hx.insert(0, cx+110); hy.insert(0, cy); hz.insert(0, -250)
        i_idx = [0]*steps
        j_idx = list(range(1, steps+1))
        k_idx = list(range(2, steps+1)) + [1]
        fig.add_trace(go.Mesh3d(x=hx, y=hy, z=hz, i=i_idx, j=j_idx, k=k_idx, color='#dddddd', lighting=dict(ambient=1.0, diffuse=0.0), showlegend=False))

        # 트레드 Line
        for i in range(16):
            a = (2*math.pi/16)*i
            lx.extend([cx-100, cx+100, None])
            ly.extend([cy+450*math.cos(a), cy+450*math.cos(a), None])
            lz.extend([-250+450*math.sin(a), -250+450*math.sin(a), None])
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=3), showlegend=False))

    for wy in [L*0.15, L*0.30, L*0.70, L*0.85]:
        add_wheel(-140, wy); add_wheel(W+140, wy)

    # 3. 적재함 (Surface)
    c_map = [[0, 'rgba(200,200,200,0.1)'], [1, 'rgba(200,200,200,0.1)']]
    fig.add_trace(go.Surface(x=[[0,0],[0,0]], y=[[0,L],[0,L]], z=[[0,0],[H,H]], colorscale=c_map, showscale=False, opacity=0.1))
    fig.add_trace(go.Surface(x=[[W,W],[W,W]], y=[[0,L],[0,L]], z=[[0,0],[H,H]], colorscale=c_map, showscale=False, opacity=0.1))
    fig.add_trace(go.Surface(x=[[0,W],[0,W]], y=[[L,L],[L,L]], z=[[0,0],[H,H]], colorscale=c_map, showscale=False, opacity=0.1))
    fig.add_trace(go.Surface(x=[[0,W],[0,W]], y=[[0,0],[0,0]], z=[[0,0],[H,H]], colorscale=c_map, showscale=False, opacity=0.1))

    # 4. 프레임
    lx = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]; ly = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]; lz = [0,0,0,0,0, H,H,H,H,H, 0,H,H,0, 0,H]
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='#333333', width=6), showlegend=False))

    # 5. 치수선
    def dim(p1, p2, txt):
        fig.add_trace(go.Scatter3d(x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]], mode='lines+text', text=["", "", f"<b>{txt}</b>"], textposition="middle center", line=dict(color='black', width=2), showlegend=False))
        dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
        ln = math.sqrt(dx**2+dy**2+dz**2)
        if ln>0:
            vx, vy, vz = dx/ln, dy/ln, dz/ln
            fig.add_trace(go.Cone(x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]], u=[-vx,vx], v=[-vy,vy], w=[-vz,vz], sizemode="absolute", sizeref=150, showscale=False, colorscale=[[0,'black'],[1,'black']]))

    off = 1200
    dim([0,-off,0], [W,-off,0], f"폭: {int(W)}")
    dim([-off,0,0], [-off,L,0], f"길이: {int(L)}")
    dim([-off,L,0], [-off,L,LIMIT_H], f"높이제한: {int(LIMIT_H)}")
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), showlegend=False))

    # 6. 박스
    for b in truck.cargo_list:
        c = '#FF0000' if getattr(b, 'is_heavy', False) else '#f39c12'
        x, y, z, w, h, d = b.x, b.y, b.z, b.w, b.h, b.d
        fig.add_trace(go.Mesh3d(x=[x,x+w,x+w,x,x,x+w,x+w,x], y=[y,y,y+d,y+d,y,y,y+d,y+d], z=[z,z,z,z,z+h,z+h,z+h,z+h], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color=c, opacity=1.0, flatshading=True, name=b.name))
        fig.add_trace(go.Scatter3d(x=[x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x], y=[y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d], z=[z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h], mode='lines', line=dict(color='black', width=2), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[x+w/2], y=[y], z=[z+h/2], mode='text', text=[f"<b>{b.name}</b>"], textfont=dict(size=14, color="white" if c=='#FF0000' else "black"), showlegend=False))

    eye = dict(x=2.0, y=-1.5, z=1.2)
    if view_mode == 'top': eye = dict(x=0, y=0.1, z=2.5)
    elif view_mode == 'side': eye = dict(x=2.5, y=0, z=0.5)
    
    fig.update_layout(scene=dict(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor='white', camera=dict(eye=eye)), margin=dict(l=0,r=0,b=0,t=0), height=700, uirevision=str(uuid.uuid4()))
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 Ultimate Load Planner (v5 Fixed)")
st.caption("✅ 비용최적화 | 회전금지 | 1.3m 제한 | 60% 지지 | 상위10% 빨강")

if 'view_mode' not in st.session_state: st.session_state['view_mode'] = 'iso'
uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file, encoding='cp949')
        else: df = pd.read_excel(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
        st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        st.dataframe(df)
        
        if st.button("최적 배차 실행", type="primary"): st.session_state['run_result'] = load_data_from_df(df)
        
        if 'run_result' in st.session_state:
            box_data = st.session_state['run_result']
            if not box_data: st.error("데이터 변환 실패.")
            else:
                with st.spinner("계산 중..."):
                    trucks = run_solver(box_data)
                
                if trucks:
                    total_cost = sum([t.cost for t in trucks])
                    st.success(f"✅ 배차 완료: 총 {len(trucks)}대 (예상 비용: {total_cost:,}원)")
                    c1, c2, c3, _ = st.columns([1, 1, 1, 5])
                    if c1.button("↗️ 쿼터뷰"): st.session_state['view_mode'] = 'iso'
                    if c2.button("⬆️ 탑뷰"): st.session_state['view_mode'] = 'top'
                    if c3.button("➡️ 사이드뷰"): st.session_state['view_mode'] = 'side'
                    
                    tabs = st.tabs([t.name for t in trucks])
                    for i, tab in enumerate(tabs):
                        with tab:
                            t = trucks[i]
                            c_left, c_right = st.columns([1, 4])
                            with c_left:
                                st.markdown(f"### **{t.name}**")
                                st.write(f"- 비용: **{t.cost:,}원**")
                                st.write(f"- 박스: **{len(t.cargo_list)}개**")
                                st.write(f"- 중량: **{t.total_weight:,} / {t.max_weight:,} kg**")
                                with st.expander("목록 보기"): st.write(", ".join([b.name for b in t.cargo_list]))
                            with c_right:
                                st.plotly_chart(draw_scene(t, st.session_state['view_mode']), use_container_width=True)
                else: st.warning("적재 가능한 차량을 찾지 못했습니다.")
    except Exception as e: st.error(f"오류 발생: {e}")
