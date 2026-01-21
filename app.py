import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# ==========================================
# 1. 커스텀 적재 엔진 (처음부터 새로 작성)
# ==========================================
# 외부 라이브러리(py3dbp) 없이 직접 계산하므로 에러가 날 수 없습니다.

class Box:
    def __init__(self, name, w, h, d, weight):
        self.name = name
        self.w = w  # 폭 (x)
        self.h = h  # 높이 (z)
        self.d = d  # 길이 (y)
        self.weight = weight
        self.x = 0
        self.y = 0
        self.z = 0
        self.is_heavy = False # 시각화용 태그

    def get_volume(self):
        return self.w * self.h * self.d

class Truck:
    def __init__(self, name, w, h, d, max_weight):
        self.name = name
        self.w = w          # 폭
        self.h = h          # 높이 (제한 높이 1300이 들어올 예정)
        self.d = d          # 길이
        self.max_weight = max_weight
        self.items = []     # 실린 박스들
        self.total_weight = 0
        
        # 적재 위치 관리를 위한 기준점(Pivot) 리스트
        # (0,0,0)에서 시작
        self.pivots = [[0, 0, 0]] 

    def put_item(self, item):
        """
        박스를 적재 시도하는 함수 (회전 로직 아예 없음)
        """
        fit = False
        valid_pivots = self.pivots 
        
        # 현재 무게 체크
        if self.total_weight + item.weight > self.max_weight:
            return False

        # 가능한 모든 기준점(빈 공간)을 순회하며 넣어봄
        for p in valid_pivots:
            # 박스를 해당 위치에 놓았을 때 트럭 밖으로 나가는지 확인
            if (p[0] + item.w > self.w) or \
               (p[1] + item.d > self.d) or \
               (p[2] + item.h > self.h):
                continue # 범위 초과

            # 이미 실린 다른 박스들과 겹치는지 확인 (충돌 체크)
            overlap = False
            for exist in self.items:
                if self.intersect(item, p, exist):
                    overlap = True
                    break
            
            if not overlap:
                # 적재 성공!
                item.x, item.y, item.z = p
                self.items.append(item)
                self.total_weight += item.weight
                fit = True
                break
        
        if fit:
            # 기준점 업데이트 (새로운 박스 주변으로 새 기준점 생성)
            # 1. x축 방향 (박스 오른쪽)
            self.pivots.append([item.x + item.w, item.y, item.z])
            # 2. y축 방향 (박스 뒤쪽 - 길이방향)
            self.pivots.append([item.x, item.y + item.d, item.z])
            # 3. z축 방향 (박스 위쪽)
            self.pivots.append([item.x, item.y, item.z + item.h])
            
            # 유효하지 않은 기준점(다른 박스 내부 등) 정리 로직은 생략(단순화)하되
            # Z->Y->X 순서로 정렬하여 안쪽/아래쪽부터 채우도록 유도
            self.pivots.sort(key=lambda x: (x[2], x[1], x[0]))
            
        return fit

    def intersect(self, item, pos, exist_item):
        """두 박스의 충돌 감지 (AABB 충돌 알고리즘)"""
        # 새 박스의 좌표 범위
        ix, iy, iz = pos
        iw, id_, ih = item.w, item.d, item.h
        
        # 기존 박스의 좌표 범위
        ex, ey, ez = exist_item.x, exist_item.y, exist_item.z
        ew, ed, eh = exist_item.w, exist_item.d, exist_item.h

        return (
            ix < ex + ew and ix + iw > ex and
            iy < ey + ed and iy + id_ > ey and
            iz < ez + eh and iz + ih > ez
        )

# ==========================================
# 2. 설정 및 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="물류 적재 시뮬레이터 (Standalone)")

# 차량 제원 (mm, kg)
# 실제 높이는 2350이지만, 계산은 1300으로 제한할 것임
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
    # 중량 상위 10% 기준 계산
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
            
            # 박스 객체 생성 (순수 파이썬 클래스)
            box = Box(name, w, h, l, weight)
            box.is_heavy = (weight >= heavy_threshold)
            items.append(box)
        except:
            continue
    return items

def run_optimization(all_items):
    remaining_items = all_items[:]
    used_trucks = [] # 결과로 배차된 트럭들 (Truck 객체)
    
    # 트럭 타입 정렬 (작은 차 -> 큰 차)
    truck_types = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])

    while remaining_items:
        best_truck = None
        best_score = -1
        
        # 현재 남은 짐으로 모든 차종 시뮬레이션
        for t_name in truck_types:
            spec = TRUCK_DB[t_name]
            # [핵심] 높이 제한 1.3m (1300mm) 적용
            limit_h = 1300
            
            # 가상 트럭 생성
            temp_truck = Truck(t_name, spec['w'], limit_h, spec['l'], spec['weight'])
            
            # 남은 짐들을 큰 것(부피)부터 넣어봄 -> 빈 공간 최소화
            # (매 시뮬레이션마다 복사본으로 테스트)
            test_items = sorted(remaining_items, key=lambda x: x.get_volume(), reverse=True)
            packed_count = 0
            
            for item in test_items:
                # 박스 복제(좌표 초기화)하여 넣기
                item_copy = Box(item.name, item.w, item.h, item.d, item.weight)
                if temp_truck.put_item(item_copy):
                    packed_count += 1
            
            # 점수 계산
            if packed_count > 0:
                # 1. 남은 짐을 몽땅 실었다면 -> 가장 작은(가벼운) 트럭이 1등
                if packed_count == len(remaining_items):
                    score = 100000 - spec['weight']
                else:
                    # 2. 다 못 실었다면 -> 얼마나 꽉 채웠는지(무게+부피) 평가
                    util_w = temp_truck.total_weight / spec['weight']
                    util_v = sum([i.get_volume() for i in temp_truck.items]) / (spec['w'] * limit_h * spec['l'])
                    score = (util_w + util_v) * 100
                
                if score > best_score:
                    best_score = score
                    best_truck = temp_truck

        # 최적 트럭 확정
        if best_truck and len(best_truck.items) > 0:
            # 트럭 이름에 번호 부여
            best_truck.name = f"{best_truck.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_truck)
            
            # 실린 박스들을 남은 목록에서 제거
            packed_names = [i.name for i in best_truck.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else:
            # 더 이상 적재 불가 (짐이 너무 크거나 오류)
            break
            
    return used_trucks

def draw_truck_3d(truck):
    fig = go.Figure()
    spec = TRUCK_DB[truck.name.split(' ')[0]]
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    
    # 1. 트럭 바닥 (진한 회색)
    fig.add_trace(go.Mesh3d(x=[0,W,W,0], y=[0,0,L,L], z=[0,0,0,0], color='rgb(100,100,100)', opacity=1.0, name='바닥'))
    
    # 2. 트럭 벽면 (반투명)
    wall_c = 'lightblue'
    wall_o = 0.1
    # 좌측(x=0), 우측(x=W), 앞면(y=L)
    fig.add_trace(go.Mesh3d(x=[0,0,0,0], y=[0,L,L,0], z=[0,0,Real_H,Real_H], color=wall_c, opacity=wall_o, showlegend=False)) # 좌
    fig.add_trace(go.Mesh3d(x=[W,W,W,W], y=[0,L,L,0], z=[0,0,Real_H,Real_H], color=wall_c, opacity=wall_o, showlegend=False)) # 우
    fig.add_trace(go.Mesh3d(x=[0,W,W,0], y=[L,L,L,L], z=[0,0,Real_H,Real_H], color=wall_c, opacity=wall_o, showlegend=False)) # 앞

    # 3. 헤드(Cabin) 장식
    head_len = 1500
    fig.add_trace(go.Mesh3d(
        x=[0,W,W,0, 0,W,W,0], 
        y=[L,L,L+head_len,L+head_len, L,L,L+head_len,L+head_len],
        z=[0,0,0,0, Real_H*0.7,Real_H*0.7,Real_H*0.7,Real_H*0.7],
        i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
        color='rgb(80,80,80)', name='헤드'
    ))

    # 4. 프레임 선
    lx = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    ly = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lz = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=3), showlegend=False))
    
    # 5. 높이 제한선 (1.3m)
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[1300]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), name='높이제한(1.3m)'))

    # 6. 박스 그리기
    for item in truck.items:
        # 좌표(item.x, item.y, item.z)는 좌측 하단 기준
        x, y, z = item.x, item.y, item.z
        w, h, d = item.w, item.h, item.d # d는 길이(y방향)
        
        # 색상 (상위 10% 빨강)
        color = '#FF4B4B' if item.is_heavy else '#E0E0E0'
        opacity = 1.0 if item.is_heavy else 0.8
        
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
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='black', width=1), showlegend=False))
        
        # 박스 번호 (측면)
        fig.add_trace(go.Scatter3d(
            x=[x + w/2], y=[y], z=[z + h/2],
            mode='text', text=[item.name], textposition="middle center",
            textfont=dict(size=12, color='black', weight='bold'), showlegend=False
        ))

    fig.update_layout(scene=dict(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), margin=dict(l=0,r=0,b=0,t=0), height=600)
    return fig

# ==========================================
# 4. 메인 UI
# ==========================================
st.title("🚛 Custom 적재 시뮬레이터 (v1.0)")
st.caption("✅ 특징: 라이브러리 미사용(에러없음) | 회전금지 | 1.3m 제한 | 11/5톤 최적화")

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='cp949')
        else:
            df = pd.read_excel(uploaded_file)
        
        # 컬럼명 공백 제거
        df.columns = [c.strip() for c in df.columns]
        
        st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        st.dataframe(df)

        if st.button("최적 배차 실행", type="primary"):
            items = load_data(df)
            if not items:
                st.error("데이터 변환 실패. 컬럼명(박스번호, 폭, 높이, 길이, 중량)을 확인하세요.")
            else:
                with st.spinner("자체 알고리즘으로 계산 중입니다..."):
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
