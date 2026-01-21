import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math

# ==========================================
# 1. 커스텀 물리 엔진 (순수 파이썬 구현 - 수정 없음)
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
        # 피봇 초기화 (0,0,0)
        self.pivots = [[0.0, 0.0, 0.0]]

    def put_item(self, item):
        fit = False
        if self.total_weight + item.weight > self.max_weight:
            return False
        
        # 피봇 정렬: Z(아래) -> Y(안쪽) -> X(왼쪽) 우선
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))

        for p in self.pivots:
            px, py, pz = p
            
            # 1. 공간 벗어남 체크
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h):
                continue
            
            # 2. 충돌 체크
            if self._check_collision(item, px, py, pz):
                continue
            
            # 3. 바닥 지지(60%) 체크
            if not self._check_support(item, px, py, pz):
                continue

            # 적재 성공
            item.x, item.y, item.z = px, py, pz
            self.items.append(item)
            self.total_weight += item.weight
            fit = True
            break
        
        if fit:
            # 새 피봇 추가 (박스 우측, 뒤쪽, 위쪽)
            self.pivots.append([item.x + item.w, item.y, item.z])
            self.pivots.append([item.x, item.y + item.d, item.z])
            self.pivots.append([item.x, item.y, item.z + item.h])
            
        return fit

    def _check_collision(self, item, x, y, z):
        for exist in self.items:
            # AABB 충돌 검사 (축 정렬 경계 상자)
            if (x < exist.x + exist.w and x + item.w > exist.x and
                y < exist.y + exist.d and y + item.d > exist.y and
                z < exist.z + exist.h and z + item.h > exist.z):
                return True
        return False

    def _check_support(self, item, x, y, z):
        # 바닥이면 무조건 OK
        if z <= 0.001: return True
        
        support_area = 0.0
        required_area = item.w * item.d * 0.6 # 60% 이상 지지 필요
        
        for exist in self.items:
            # 바로 아래층(오차 1mm)에 있는 박스와 겹치는 면적 계산
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0.0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy
                
        return support_area >= required_area

# ==========================================
# 2. 데이터 설정 (규칙 0 반영)
# ==========================================
st.set_page_config(layout="wide", page_title="Load Planner - Reset Version")

# 차량 제원 및 비용 테이블 (단위: mm, kg, 원)
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

def load_data(df):
    items = []
    try:
        # 중량 데이터 전처리
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        if weights:
            sorted_weights = sorted(weights, reverse=True)
            top10_count = max(1, int(len(weights) * 0.1) - 1)
            heavy_threshold = sorted_weights[top10_count]
        else:
            heavy_threshold = 999999999
    except:
        heavy_threshold = 999999999

    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            box = Box(name, w, h, l, weight)
            box.is_heavy = (weight >= heavy_threshold and weight > 0)
            items.append(box)
        except:
            continue
    return items

def run_optimization(all_items):
    remaining_items = all_items[:]
    used_trucks = [] 
    
    # 비용이 낮은 순서대로 차량 정렬
    truck_types_by_cost = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['cost'])
    
    # 1. 단일 차량으로 모두 적재 가능한지 테스트 (비용 싼 순서로)
    for t_name in truck_types_by_cost:
        spec = TRUCK_DB[t_name]
        limit_h = 1300 # 규칙 2: 높이 제한 1.3m
        
        temp_truck = Truck(t_name, spec['w'], limit_h, spec['l'], spec['weight'], spec['cost'])
        test_items = sorted(remaining_items, key=lambda x: x.volume, reverse=True)
        
        success = True
        for item in test_items:
            item_copy = Box(item.name, item.w, item.h, item.d, item.weight)
            item_copy.is_heavy = item.is_heavy
            if not temp_truck.put_item(item_copy):
                success = False
                break
        
        if success:
            temp_truck.name = f"{t_name} (단일차량)"
            return [temp_truck]

    # 2. 단일 차량 불가 시, 큰 차부터 채우기 (Greedy)
    truck_types_desc = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'], reverse=True)
    
    while remaining_items:
        best_truck = None
        max_packed_count = -1
        best_truck_packed_names = []
        
        for t_name in truck_types_desc:
            spec = TRUCK_DB[t_name]
            limit_h = 1300
            temp_truck = Truck(t_name, spec['w'], limit_h, spec['l'], spec['weight'], spec['cost'])
            
            test_items = sorted(remaining_items, key=lambda x: x.volume, reverse=True)
            packed_count = 0
            current_packed_items = []
            
            for item in test_items:
                item_copy = Box(item.name, item.w, item.h, item.d, item.weight)
                item_copy.is_heavy = item.is_heavy
                if temp_truck.put_item(item_copy):
                    packed_count += 1
                    current_packed_items.append(item.name)
            
            if packed_count > max_packed_count:
                max_packed_count = packed_count
                best_truck = temp_truck
                best_truck_packed_names = current_packed_items

        if best_truck and max_packed_count > 0:
            best_truck.name = f"{best_truck.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_truck)
            # 실린 짐 제거 (이름 기준)
            remaining_items = [i for i in remaining_items if i.name not in best_truck_packed_names]
        else:
            break 
            
    return used_trucks

# ==========================================
# 4. 시각화 (math만 사용, numpy 제거)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    spec = TRUCK_DB[truck.name.split(' ')[0]]
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    
    # 1. 섀시 그리기
    chassis_h = 180
    fig.add_trace(go.Mesh3d(
        x=[0, W, W, 0, 0, W, W, 0],
        y=[0, 0, L, L, 0, 0, L, L],
        z=[-chassis_h]*4+[0]*4,
        color='#222222',
        i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
        showlegend=False
    ))

    # 2. 바퀴 그리기 함수 (순수 math 사용)
    def create_wheel(cx, cy):
        steps = 24 # 부드러움 정도
        # 타이어
        tx, ty, tz = [], [], []
        for i in range(steps):
            angle = (2 * math.pi / steps) * i
            tx.extend([cx-100, cx+100])
            ty.extend([cy+450*math.cos(angle), cy+450*math.cos(angle)])
            tz.extend([-250+450*math.sin(angle), -250+450*math.sin(angle)])
        
        # alphahull로 감싸기
        fig.add_trace(go.Mesh3d(x=tx, y=ty, z=tz, alphahull=0, color='#333333', showlegend=False))
        
        # 휠 허브 (단순 원)
        hx, hy, hz = [], [], []
        hx.append(cx+110); hy.append(cy); hz.append(-250) # 중심
        for i in range(steps):
            angle = (2 * math.pi / steps) * i
            hx.append(cx+100); hy.append(cy+250*math.cos(angle)); hz.append(-250+250*math.sin(angle))
        
        i_idx = [0]*steps
        j_idx = list(range(1, steps+1))
        k_idx = list(range(2, steps+1)) + [1]
        fig.add_trace(go.Mesh3d(x=hx, y=hy, z=hz, i=i_idx, j=j_idx, k=k_idx, color='#dddddd', showlegend=False))

    # 바퀴 위치 (2축)
    wheel_positions = [L*0.15, L*0.30, L*0.70, L*0.85]
    for wy in wheel_positions:
        create_wheel(-140, wy)
        create_wheel(W+140, wy)

    # 3. 적재함 벽면 (Surface 사용)
    wall_c = [[0, 'rgba(200,200,200,0.1)'], [1, 'rgba(200,200,200,0.1)']]
    # 좌우앞뒤
    fig.add_trace(go.Surface(x=[[0,0],[0,0]], y=[[0,L],[0,L]], z=[[0,0],[Real_H,Real_H]], colorscale=wall_c, showscale=False, opacity=0.1))
    fig.add_trace(go.Surface(x=[[W,W],[W,W]], y=[[0,L],[0,L]], z=[[0,0],[Real_H,Real_H]], colorscale=wall_c, showscale=False, opacity=0.1))
    fig.add_trace(go.Surface(x=[[0,W],[0,W]], y=[[L,L],[L,L]], z=[[0,0],[Real_H,Real_H]], colorscale=wall_c, showscale=False, opacity=0.1))
    fig.add_trace(go.Surface(x=[[0,W],[0,W]], y=[[0,0],[0,0]], z=[[0,0],[Real_H,Real_H]], colorscale=wall_c, showscale=False, opacity=0.1))

    # 4. 프레임 (선)
    lx = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    ly = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lz = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='#333333', width=5), showlegend=False))

    # 5. 치수선
    def add_dim(p1, p2, text, c='black'):
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines+text', text=["", "", f"<b>{text}</b>"], textposition="middle center",
            line=dict(color=c, width=2), showlegend=False
        ))
        # 화살표 (Cone)
        dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        if length > 0:
            vx, vy, vz = dx/length, dy/length, dz/length
            fig.add_trace(go.Cone(x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]], u=[-vx,vx], v=[-vy,vy], w=[-vz,vz], sizemode="absolute", sizeref=100, showscale=False, colorscale=[[0,c],[1,c]]))

    offset = 1200
    add_dim([0,-offset,0], [W,-offset,0], f"폭: {int(W)}")
    add_dim([-offset,0,0], [-offset,L,0], f"길이: {int(L)}")
    add_dim([-offset,L,0], [-offset,L,LIMIT_H], f"높이제한: {int(LIMIT_H)}", c='red')
    
    # 높이 제한선
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), showlegend=False))

    # 6. 박스 그리기
    for item in truck.items:
        color = '#FF0000' if getattr(item, 'is_heavy', False) else '#f39c12'
        x, y, z = item.x, item.y, item.z
        w, h, d = item.w, item.h, item.d
        
        # 박스 면
        fig.add_trace(go.Mesh3d(
            x=[x,x+w,x+w,x, x,x+w,x+w,x],
            y=[y,y,y+d,y+d, y,y,y+d,y+d],
            z=[z,z,z,z, z+h,z+h,z+h,z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=color, opacity=1.0, flatshading=True, name=item.name
        ))
        
        # 박스 테두리
        bx = [x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x]
        by = [y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d]
        bz = [z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h]
        fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', line=dict(color='black', width=2), showlegend=False))
        
        # 라벨
        fig.add_trace(go.Scatter3d(
            x=[x+w/2], y=[y+d/2], z=[z+h/2],
            mode='text', text=[f"<b>{item.name}</b>"],
            textfont=dict(size=12, color="white" if color=='#FF0000' else "black"),
            showlegend=False
        ))

    # 카메라 설정
    eye = dict(x=2.0, y=-1.5, z=1.2)
    if camera_view == 'top': eye = dict(x=0, y=0.1, z=2.5)
    elif camera_view == 'side': eye = dict(x=2.5, y=0, z=0.5)
    
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='white', camera=dict(eye=eye)
        ),
        margin=dict(l=0,r=0,b=0,t=0), height=700
    )
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 Ultimate Load Planner (Reset Version)")
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
        
        if st.button("최적 배차 실행", type="primary"):
            st.session_state['run_result'] = load_data(df)
        
        if 'run_result' in st.session_state:
            items = st.session_state['run_result']
            if not items:
                st.error("데이터 변환 실패.")
            else:
                with st.spinner("비용 최소화 배차 계산 중..."):
                    trucks = run_optimization(items)
                
                if trucks:
                    total_cost = sum([t.cost for t in trucks])
                    st.success(f"✅ 배차 완료: 총 {len(trucks)}대 (예상 비용: {total_cost:,}원)")
                    
                    c1, c2, c3, _ = st.columns([1, 1, 1, 5])
                    with c1: 
                        if st.button("↗️ 쿼터뷰"): st.session_state['view_mode'] = 'iso'
                    with c2: 
                        if st.button("⬆️ 탑뷰"): st.session_state['view_mode'] = 'top'
                    with c3: 
                        if st.button("➡️ 사이드뷰"): st.session_state['view_mode'] = 'side'
                    
                    tabs = st.tabs([t.name for t in trucks])
                    for i, tab in enumerate(tabs):
                        with tab:
                            t = trucks[i]
                            c_left, c_right = st.columns([1, 4])
                            with c_left:
                                st.markdown(f"### **{t.name}**")
                                st.write(f"- 운송 비용: **{t.cost:,}원**")
                                st.write(f"- 박스 수: **{len(t.items)}개**")
                                st.write(f"- 적재 중량: **{t.total_weight:,} / {t.max_weight:,} kg**")
                                with st.expander("목록 보기"):
                                    st.write(", ".join([b.name for b in t.items]))
                            with c_right:
                                st.plotly_chart(draw_truck_3d(t, st.session_state['view_mode']), use_container_width=True)
                else:
                    st.warning("적재 가능한 차량을 찾지 못했습니다.")
                    
    except Exception as e:
        st.error(f"오류 발생: {e}")
