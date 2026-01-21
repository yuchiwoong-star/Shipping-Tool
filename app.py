import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math

# ==========================================
# 1. 커스텀 물리 엔진 (기존 로직 100% 동결)
# ==========================================
# ※ 로직 수정 없음 (회전금지, 중력, 높이제한, 최적화 그대로)

class Box:
    def __init__(self, name, w, h, d, weight):
        self.name = name
        self.w = w
        self.h = h
        self.d = d
        self.weight = weight
        self.x = 0
        self.y = 0
        self.z = 0
        self.is_heavy = False

    @property
    def volume(self):
        return self.w * self.h * self.d

class Truck:
    def __init__(self, name, w, h, d, max_weight):
        self.name = name
        self.w = w
        self.h = h          # 제한 높이 (1300)
        self.d = d          # 길이
        self.max_weight = max_weight
        self.items = []     # 적재된 박스들
        self.total_weight = 0
        self.pivots = [[0, 0, 0]] 

    def put_item(self, item):
        fit = False
        if self.total_weight + item.weight > self.max_weight:
            return False
        
        # Z -> Y -> X 순 정렬
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))

        for p in self.pivots:
            px, py, pz = p
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h):
                continue
            if self._check_collision(item, px, py, pz):
                continue
            if not self._check_support(item, px, py, pz):
                continue

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
                z < exist.z + exist.h and z + item.h > exist.z):
                return True
        return False

    def _check_support(self, item, x, y, z):
        if z == 0: return True 
        support_area = 0
        required_area = item.w * item.d * 0.6 
        for exist in self.items:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy
        return support_area >= required_area

# ==========================================
# 2. 설정 및 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="Ultimate Load Planner")

TRUCK_DB = {
    "5톤":  {"w": 2350, "real_h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "real_h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "real_h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "real_h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "real_h": 2350, "l": 10200, "weight": 24000},
}

# ==========================================
# 3. 로직 함수
# ==========================================

def load_data(df):
    items = []
    try:
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        sorted_weights = sorted(weights, reverse=True)
        top10_idx = max(0, int(len(weights) * 0.1) - 1)
        heavy_threshold = sorted_weights[top10_idx] if weights else 999999
    except:
        heavy_threshold = 999999

    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            box = Box(name, w, h, l, weight)
            box.is_heavy = (weight >= heavy_threshold)
            items.append(box)
        except:
            continue
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
            limit_h = 1300 
            temp_truck = Truck(t_name, spec['w'], limit_h, spec['l'], spec['weight'])
            
            test_items = sorted(remaining_items, key=lambda x: x.volume, reverse=True)
            packed_count = 0
            
            for item in test_items:
                item_copy = Box(item.name, item.w, item.h, item.d, item.weight)
                if temp_truck.put_item(item_copy):
                    packed_count += 1
            
            if packed_count > 0:
                if packed_count == len(remaining_items):
                    score = 100000 - spec['weight']
                else:
                    util_w = temp_truck.total_weight / spec['weight']
                    util_v = sum([i.volume for i in temp_truck.items]) / (spec['w'] * limit_h * spec['l'])
                    score = (util_w + util_v) * 100
                
                if score > best_score:
                    best_score = score
                    best_truck = temp_truck

        if best_truck and len(best_truck.items) > 0:
            best_truck.name = f"{best_truck.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_truck)
            packed_names = [i.name for i in best_truck.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else:
            break
    return used_trucks

# ==========================================
# 4. 고퀄리티 3D 시각화 (디자인 구체화)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    spec = TRUCK_DB[truck.name.split(' ')[0]]
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    
    # --- [1] 디테일한 트럭 모델링 ---
    
    # 1. 섀시 (Chassis) - 검은색 하부 프레임
    chassis_h = 150
    fig.add_trace(go.Mesh3d(
        x=[0, W, W, 0, 0, W, W, 0],
        y=[0, 0, L, L, 0, 0, L, L],
        z=[-chassis_h, -chassis_h, -chassis_h, -chassis_h, 0, 0, 0, 0],
        i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
        color='#1a1a1a', flatshading=True, name='섀시'
    ))

    # 2. 바퀴 (Wheels) - 8각형 근사 원기둥
    def create_wheel(center_x, center_y, z_pos, radius, width):
        # 8각형 좌표 계산
        angles = [i * (2 * math.pi / 8) for i in range(8)]
        xs, ys, zs = [], [], []
        # 바깥면
        for a in angles:
            xs.append(center_x + width/2)
            ys.append(center_y + radius * math.cos(a))
            zs.append(z_pos + radius * math.sin(a))
        # 안쪽면
        for a in angles:
            xs.append(center_x - width/2)
            ys.append(center_y + radius * math.cos(a))
            zs.append(z_pos + radius * math.sin(a))
        
        return go.Mesh3d(
            x=xs, y=ys, z=zs,
            # 8각 기둥 Mesh 인덱스 (단순화하여 박스 형태로 대체하되 조금 더 둥글게 보임)
            alphahull=0, 
            color='#111111', name='바퀴'
        )

    wheel_r = 400
    wheel_w = 250
    wheel_z = -chassis_h - 100
    
    # 바퀴 4개 배치
    fig.add_trace(create_wheel(-wheel_w/2, L*0.15, wheel_z, wheel_r, wheel_w))
    fig.add_trace(create_wheel(W + wheel_w/2, L*0.15, wheel_z, wheel_r, wheel_w))
    fig.add_trace(create_wheel(-wheel_w/2, L*0.85, wheel_z, wheel_r, wheel_w))
    fig.add_trace(create_wheel(W + wheel_w/2, L*0.85, wheel_z, wheel_r, wheel_w))

    # 3. 헤드 (Cabin) - 구체화
    cabin_len = 1600
    cabin_h = 2400
    cy = L + 150 # 섀시와 약간 띄움
    
    # 메인 바디 (파랑)
    fig.add_trace(go.Mesh3d(
        x=[0, W, W, 0, 0, W, W, 0],
        y=[cy, cy, cy+cabin_len, cy+cabin_len, cy, cy, cy+cabin_len, cy+cabin_len],
        z=[0, 0, 0, 0, cabin_h, cabin_h, cabin_h, cabin_h],
        i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
        color='#2980b9', flatshading=True, name='트럭 헤드'
    ))

    # 범퍼 & 그릴 (앞쪽 하단)
    fig.add_trace(go.Mesh3d(
        x=[0, W, W, 0, 0, W, W, 0],
        y=[cy+cabin_len, cy+cabin_len, cy+cabin_len+100, cy+cabin_len+100, cy+cabin_len, cy+cabin_len, cy+cabin_len+100, cy+cabin_len+100],
        z=[0, 0, 0, 0, 600, 600, 600, 600],
        i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
        color='#333333', name='범퍼'
    ))

    # 윈드쉴드 (앞유리)
    fig.add_trace(go.Mesh3d(
        x=[50, W-50, W-50, 50],
        y=[cy+cabin_len+10, cy+cabin_len+10, cy+cabin_len+10, cy+cabin_len+10],
        z=[1100, 1100, 2100, 2100],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#85c1e9', opacity=0.8, name='앞유리'
    ))

    # 4. 적재함 벽면 (반투명)
    wall_color = '#ecf0f1'
    wall_op = 0.1
    def wall_mesh(xs, ys, zs):
        return go.Mesh3d(x=xs, y=ys, z=zs, color=wall_color, opacity=wall_op, showlegend=False)

    fig.add_trace(wall_mesh([0,0,0,0], [0,L,L,0], [0,0,Real_H,Real_H])) # 좌
    fig.add_trace(wall_mesh([W,W,W,W], [0,L,L,0], [0,0,Real_H,Real_H])) # 우
    fig.add_trace(wall_mesh([0,W,W,0], [L,L,L,L], [0,0,Real_H,Real_H])) # 앞

    # 프레임 (외곽선)
    lx = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    ly = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lz = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='#7f8c8d', width=3), showlegend=False))


    # --- [2] 치수선 (가독성 위해 멀리 배치) ---
    OFFSET = 1200 # 간격 더 벌림
    
    def add_dim(p1, p2, text, color='black'):
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines', line=dict(color=color, width=2), showlegend=False
        ))
        mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]
        # 배경색 있는 라벨 사용 (가독성 UP)
        fig.add_trace(go.Scatter3d(
            x=[mid[0]], y=[mid[1]], z=[mid[2]],
            mode='text', text=[f"<b>{text}</b>"], 
            textfont=dict(size=13, color=color),
            showlegend=False
        ))

    # 폭(W)
    add_dim((0, -OFFSET, 0), (W, -OFFSET, 0), f"폭 {W}")
    # 길이(L)
    add_dim((-OFFSET, 0, 0), (-OFFSET, L, 0), f"길이 {L}")
    # 높이(H)
    add_dim((-OFFSET, L, 0), (-OFFSET, L, LIMIT_H), f"제한 {LIMIT_H}", color='red')
    
    # 1.3m 제한선 (빨간 점선)
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=4, dash='dash')))


    # --- [3] 박스 및 2D 라벨 (Annotations) ---
    annotations = []
    
    for item in truck.items:
        x, y, z = item.x, item.y, item.z
        w, h, d = item.w, item.h, item.d
        
        # 색상
        color = '#c0392b' if item.is_heavy else '#f39c12'
        
        # 박스 Mesh
        fig.add_trace(go.Mesh3d(
            x=[x,x+w,x+w,x, x,x+w,x+w,x],
            y=[y,y,y+d,y+d, y,y,y+d,y+d],
            z=[z,z,z,z, z+h,z+h,z+h,z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=color, opacity=1.0, flatshading=True, name=item.name
        ))
        # 테두리
        ex = [x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x]
        ey = [y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d]
        ez = [z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='black', width=2), showlegend=False))
        
        # [핵심] 2D Annotation 라벨 생성 (화면 위에 뜸)
        # 박스의 중심점
        cx, cy, cz = x + w/2, y + d/2, z + h/2
        
        # 사이드뷰(옆면)일 때 잘 보이도록 y좌표(깊이) 조정
        annotations.append(dict(
            x=cx, y=cy, z=cz,
            text=item.name,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(color="black", size=11, family="Arial Black"),
            bgcolor="rgba(255, 255, 255, 0.7)", # 반투명 흰색 배경
            borderpad=2
        ))

    # --- [4] 카메라 뷰 설정 ---
    if camera_view == "top":
        eye = dict(x=0, y=0.1, z=2.5) # 위에서
        up = dict(x=0, y=1, z=0)
    elif camera_view == "side":
        eye = dict(x=2.5, y=0, z=0.5) # 옆에서 (길이 방향)
        up = dict(x=0, y=0, z=1)
    else: # iso (default)
        eye = dict(x=2.0, y=-1.5, z=1.2)
        up = dict(x=0, y=0, z=1)

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='white',
            camera=dict(eye=eye, up=up),
            annotations=annotations # 2D 라벨 적용
        ),
        margin=dict(l=0,r=0,b=0,t=0),
        height=700
    )
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 Ultimate Load Planner")
st.caption("✅ 물리엔진 | 회전금지 | 1.3m 제한 | 뷰 컨트롤 | 고퀄리티 디자인")

# 세션 상태 초기화 (뷰 버튼용)
if 'view_mode' not in st.session_state:
    st.session_state['view_mode'] = 'iso'

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='cp949')
        else:
            df = pd.read_excel(uploaded_file)
        
        df.columns = [c.strip() for c in df.columns]
        
        st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        st.dataframe(df)

        if st.button("최적 배차 실행", type="primary"):
            st.session_state['run_result'] = load_data(df)

        # 결과가 있으면 표시
        if 'run_result' in st.session_state:
            items = st.session_state['run_result']
            if not items:
                st.error("데이터 변환 실패.")
            else:
                trucks = run_optimization(items)
                
                if trucks:
                    t_names = [t.name.split(' ')[0] for t in trucks]
                    from collections import Counter
                    cnt = Counter(t_names)
                    summary = ", ".join([f"{k} {v}대" for k,v in cnt.items()])
                    
                    st.success(f"✅ 분석 완료: 총 {len(trucks)}대 ({summary})")
                    
                    # 뷰 컨트롤 버튼
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
                            col1, col2 = st.columns([1, 4])
                            t = trucks[i]
                            with col1:
                                st.markdown(f"### **{t.name}**")
                                st.write(f"- 박스: **{len(t.items)}개**")
                                st.write(f"- 중량: **{t.total_weight:,} kg**")
                                with st.expander("목록 보기"):
                                    st.write(", ".join([b.name for b in t.items]))
                            with col2:
                                # 선택된 뷰 모드 적용
                                st.plotly_chart(draw_truck_3d(t, st.session_state['view_mode']), use_container_width=True)
                else:
                    st.warning("적재 가능한 차량을 찾지 못했습니다.")

    except Exception as e:
        st.error(f"오류 발생: {e}")
