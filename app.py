import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# ==========================================
# 1. 커스텀 물리 엔진 (Gravity & Collision)
# ==========================================

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
        # 기준점 (Pivot): (x, y, z) 후보군
        self.pivots = [[0, 0, 0]] 

    def put_item(self, item):
        """
        박스 적재 시도 (충돌 체크 + 지지 기반 체크)
        """
        fit = False
        
        # 무게 초과 체크
        if self.total_weight + item.weight > self.max_weight:
            return False

        # Z(높이) -> Y(안쪽) -> X(왼쪽) 순으로 정렬하여 
        # "바닥부터", "안쪽부터" 채우도록 유도
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))

        for p in self.pivots:
            px, py, pz = p
            
            # 1. 트럭 범위 벗어나는지 체크
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h):
                continue

            # 2. 다른 박스와 충돌 체크
            if self._check_collision(item, px, py, pz):
                continue

            # 3. [핵심] 바닥 지지 여부 체크 (Gravity)
            # 공중에 뜨지 않으려면 아래쪽에 60% 이상의 면적이 받쳐줘야 함
            if not self._check_support(item, px, py, pz):
                continue

            # 적재 성공
            item.x, item.y, item.z = px, py, pz
            self.items.append(item)
            self.total_weight += item.weight
            fit = True
            break
        
        if fit:
            # 새로운 기준점 생성 (새 박스의 오른쪽, 뒤쪽, 위쪽)
            self.pivots.append([item.x + item.w, item.y, item.z])
            self.pivots.append([item.x, item.y + item.d, item.z])
            self.pivots.append([item.x, item.y, item.z + item.h])
            # (최적화를 위해 불필요한 Pivot 제거 로직은 생략)
            
        return fit

    def _check_collision(self, item, x, y, z):
        """기존 박스들과 겹치는지 확인 (AABB 충돌)"""
        for exist in self.items:
            if (x < exist.x + exist.w and x + item.w > exist.x and
                y < exist.y + exist.d and y + item.d > exist.y and
                z < exist.z + exist.h and z + item.h > exist.z):
                return True
        return False

    def _check_support(self, item, x, y, z):
        """
        박스 아래가 비어있는지 확인 (Support Logic)
        z=0이면 바닥이니 OK.
        z>0이면 바로 아래(z-height)에 있는 박스들과 접촉 면적 계산.
        """
        if z == 0: return True # 바닥은 무조건 지지됨

        support_area = 0
        required_area = item.w * item.d * 0.6 # 최소 60%는 걸쳐져 있어야 함

        for exist in self.items:
            # 바로 아래층에 있는 박스인가? (오차범위 1mm)
            if abs((exist.z + exist.h) - z) < 1.0:
                # 겹치는 면적 계산 (Intersection of Rectangles)
                ox = max(0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy

        return support_area >= required_area

# ==========================================
# 2. 설정 및 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="물류 적재 시뮬레이터 (Physics Engine)")

TRUCK_DB = {
    "5톤":  {"w": 2350, "real_h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "real_h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "real_h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "real_h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "real_h": 2350, "l": 10200, "weight": 24000},
}

# ==========================================
# 3. 데이터 로드 및 최적화 로직
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
            # [규칙] 회전 절대 금지 (파일 그대로)
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
    
    # 작은 차 -> 큰 차 순서
    truck_types = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])

    while remaining_items:
        best_truck = None
        best_score = -1
        
        # 모든 차종 시뮬레이션
        for t_name in truck_types:
            spec = TRUCK_DB[t_name]
            limit_h = 1300 # [규칙] 높이 제한 1.3m
            
            temp_truck = Truck(t_name, spec['w'], limit_h, spec['l'], spec['weight'])
            
            # [전략] 부피가 큰 짐부터 넣어야 바닥이 안정적으로 깔림
            test_items = sorted(remaining_items, key=lambda x: x.volume, reverse=True)
            packed_count = 0
            
            for item in test_items:
                # 상태 복사해서 시도
                item_copy = Box(item.name, item.w, item.h, item.d, item.weight)
                if temp_truck.put_item(item_copy):
                    packed_count += 1
            
            if packed_count > 0:
                # 점수 계산 (많이 실을수록, 작은 차일수록 좋음)
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
# 4. 고퀄리티 3D 시각화 (디자인 개선)
# ==========================================
def draw_truck_3d(truck):
    fig = go.Figure()
    spec = TRUCK_DB[truck.name.split(' ')[0]]
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    
    # --- [트럭 디자인] ---
    
    # 1. 섀시(바닥 프레임) - 진한 회색
    fig.add_trace(go.Mesh3d(
        x=[0, W, W, 0], y=[0, 0, L, L], z=[0, 0, 0, 0],
        color='rgb(50, 50, 50)', opacity=1.0, name='섀시'
    ))

    # 2. 바퀴 (단순화된 검은 박스 4개)
    wheel_w, wheel_r, wheel_h = 300, 500, 300 # 바퀴 크기
    wheel_z = -300
    wheel_positions = [
        (0 - wheel_w, L*0.15), (W, L*0.15), # 앞바퀴
        (0 - wheel_w, L*0.85), (W, L*0.85)  # 뒷바퀴
    ]
    for wx, wy in wheel_positions:
        fig.add_trace(go.Mesh3d(
            x=[wx, wx+wheel_w, wx+wheel_w, wx, wx, wx+wheel_w, wx+wheel_w, wx],
            y=[wy, wy, wy+wheel_r, wy+wheel_r, wy, wy, wy+wheel_r, wy+wheel_r],
            z=[wheel_z, wheel_z, wheel_z, wheel_z, 0, 0, 0, 0],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color='black', flatshading=True, showlegend=False
        ))

    # 3. 헤드 (Cabin) - 더 디테일하게
    cabin_len = 1800
    cabin_h = 2500
    cy_start = L + 100 # 적재함과 약간 띄움
    
    # 헤드 본체
    fig.add_trace(go.Mesh3d(
        x=[0, W, W, 0, 0, W, W, 0],
        y=[cy_start, cy_start, cy_start+cabin_len, cy_start+cabin_len, cy_start, cy_start, cy_start+cabin_len, cy_start+cabin_len],
        z=[0, 0, 0, 0, cabin_h, cabin_h, cabin_h, cabin_h],
        i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
        color='rgb(30, 100, 180)', name='트럭 헤드'
    ))
    
    # 4. 적재함 벽면 (반투명 아크릴 느낌)
    wall_color = 'rgba(200, 220, 255, 0.2)'
    # 좌, 우, 앞(운전석쪽), 뒤(문)
    # 좌측
    fig.add_trace(go.Mesh3d(x=[0,0,0,0], y=[0,L,L,0], z=[0,0,Real_H,Real_H], color=wall_color, showlegend=False))
    # 우측
    fig.add_trace(go.Mesh3d(x=[W,W,W,W], y=[0,L,L,0], z=[0,0,Real_H,Real_H], color=wall_color, showlegend=False))
    # 앞쪽 (헤드 쪽)
    fig.add_trace(go.Mesh3d(x=[0,W,W,0], y=[L,L,L,L], z=[0,0,Real_H,Real_H], color='rgba(150, 170, 200, 0.4)', showlegend=False))

    # 5. 프레임 (외곽선)
    lines_x = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    lines_y = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lines_z = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='black', width=3), showlegend=False))
    
    # 6. 높이 제한선 (1.3m)
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[1300]*5, mode='lines', line=dict(color='red', width=5, dash='dash'), name='높이제한(1.3m)'))

    # --- [박스 그리기] ---
    for item in truck.items:
        x, y, z = item.x, item.y, item.z
        w, h, d = item.w, item.h, item.d
        
        # 색상 (상위 10% 강조)
        color = '#FF4B4B' if item.is_heavy else '#E0E0E0'
        opacity = 1.0 if item.is_heavy else 0.9
        
        # 박스 메쉬
        fig.add_trace(go.Mesh3d(
            x=[x,x+w,x+w,x, x,x+w,x+w,x],
            y=[y,y,y+d,y+d, y,y,y+d,y+d],
            z=[z,z,z,z, z+h,z+h,z+h,z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=color, opacity=opacity, flatshading=True, name=item.name
        ))
        
        # 박스 테두리
        ex = [x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x]
        ey = [y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d]
        ez = [z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='black', width=2), showlegend=False))
        
        # 박스 번호 (측면)
        fig.add_trace(go.Scatter3d(
            x=[x + w/2], y=[y], z=[z + h/2],
            mode='text', text=[item.name], textposition="middle center",
            textfont=dict(size=14, color='black', weight='bold'), showlegend=False
        ))

    # 카메라 및 축 설정
    fig.update_layout(
        scene=dict(
            aspectmode='data', # 비율 유지
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=2.0, y=1.5, z=1.5)) # 시점 조정
        ), 
        margin=dict(l=0,r=0,b=0,t=0), 
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("🚛 물류 적재 시뮬레이터 (Physics Engine)")
st.caption("✅ 물리엔진 적용: 공중부양 방지(Gravity) | 회전금지 | 1.3m 제한 | 11/5톤 최적화")

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
            items = load_data(df)
            if not items:
                st.error("데이터 변환 실패.")
            else:
                with st.spinner("물리 엔진으로 적재 시뮬레이션 중... (Support Check)"):
                    trucks = run_optimization(items)
                    
                    if trucks:
                        t_names = [t.name.split(' ')[0] for t in trucks]
                        from collections import Counter
                        cnt = Counter(t_names)
                        summary = ", ".join([f"{k} {v}대" for k,v in cnt.items()])
                        
                        st.success(f"✅ 분석 완료: 총 {len(trucks)}대 ({summary})")
                        
                        tabs = st.tabs([t.name for t in trucks])
                        for i, tab in enumerate(tabs):
                            with tab:
                                c1, c2 = st.columns([1, 3])
                                t = trucks[i]
                                with c1:
                                    st.markdown(f"### **{t.name}**")
                                    st.write(f"- 박스 수: {len(t.items)}개")
                                    st.write(f"- 총 중량: {t.total_weight:,} kg")
                                    with st.expander("박스 목록 보기"):
                                        st.write(", ".join([b.name for b in t.items]))
                                with c2:
                                    st.plotly_chart(draw_truck_3d(t), use_container_width=True)
                    else:
                        st.warning("적재 가능한 차량을 찾지 못했습니다.")
    except Exception as e:
        st.error(f"오류 발생: {e}")
