import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import py3dbp
from py3dbp import Packer, Bin, Item, RotationType

# ==========================================
# 0. 핵심: 회전 금지 강제 패치 (Monkey Patch)
# ==========================================
# py3dbp가 자동으로 박스를 돌려보는 것을 원천 차단합니다.
# 오직 RT_WHD (기본 방향: Width, Height, Depth 그대로)만 시도하게 만듭니다.

def patched_put_item(self, item, pivot):
    # 회전하지 않고(RotationType.RT_WHD = 0) 원래 데이터 그대로만 적재 시도
    # py3dbp 좌표계: Axis 0=Width, Axis 1=Height, Axis 2=Depth
    # 엑셀 매핑: 폭->Width, 높이->Height, 길이->Depth
    valid_rotations = [RotationType.RT_WHD] 
    
    fit = False
    valid_item_position = item.position
    item.position = pivot
    
    for rotation_type in valid_rotations:
        item.rotation_type = rotation_type
        dimension = item.get_dimension()
        if self.can_hold(item, pivot, dimension):
            fit = True
            self.items.append(item)
            self.total_weight += item.weight
            break
            
    if not fit:
        item.position = valid_item_position
    return fit

# 라이브러리 함수 덮어쓰기
py3dbp.Bin.put_item = patched_put_item


# ==========================================
# 1. 설정 및 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="물류 적재 시뮬레이터 (회전금지)")

# 차량 제원 (단위: mm, kg)
TRUCK_DB = {
    "5톤":  {"w": 2350, "h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 24000},
}

# ==========================================
# 2. 로직 함수
# ==========================================

def create_items_from_df(df):
    items = []
    # 중량 상위 10% 기준점 계산 (색상 구분용)
    try:
        weights = df['중량'].tolist()
        sorted_weights = sorted(weights, reverse=True)
        top10_idx = max(0, int(len(weights) * 0.1) - 1)
        heavy_threshold = sorted_weights[top10_idx] if weights else 999999
    except:
        heavy_threshold = 999999

    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            # 사용자 요청: 파일의 폭/높이/길이를 그대로 적용 (회전 X)
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            # py3dbp Item(name, width, height, depth, weight)
            # 여기 넣은 순서대로 w, h, l이 고정됩니다.
            item = Item(name, w, h, l, weight)
            
            # 시각화용 속성 추가
            item.is_heavy = (weight >= heavy_threshold)
            items.append(item)
        except Exception as e:
            st.error(f"{index}행 데이터 변환 오류: {e}")
            continue
    return items

def get_optimized_trucks(items):
    remaining_items = items[:]
    used_trucks = []
    
    # 트럭 정렬 (작은 차 -> 큰 차)
    truck_types = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])

    while remaining_items:
        best_bin = None
        best_score = -1
        
        # 어떤 트럭이 남은 짐을 가장 '잘' 실을지 테스트
        for t_name in truck_types:
            spec = TRUCK_DB[t_name]
            
            # [제약] 계산용 높이는 1.3m로 제한 (적재 단수 제한 효과)
            CALC_HEIGHT = 1300
            
            packer = Packer()
            # Bin(name, width, height, depth, max_weight)
            packer.add_bin(Bin(t_name, spec['w'], CALC_HEIGHT, spec['l'], spec['weight']))
            
            for item in remaining_items:
                packer.add_item(item)
            
            # 패킹 실행 (회전 금지 패치 적용됨)
            packer.pack(bigger_first=True)
            
            temp_bin = packer.bins[0]
            packed_count = len(temp_bin.items)
            
            if packed_count > 0:
                # 점수 산정 로직
                # 1. 남은 짐을 몽땅 실을 수 있다면? -> 가장 작은 트럭을 선호 (비용 절감)
                if packed_count == len(remaining_items):
                    # 트럭이 작을수록(무게가 가벼울수록) 높은 점수
                    score = 100000 - spec['weight'] 
                else:
                    # 2. 다 못 싣는다면? -> 최대한 꽉 채우는 트럭 선호 (효율성)
                    # 중량 적재율 + 부피 적재율
                    util_weight = temp_bin.get_total_weight() / spec['weight']
                    util_vol = sum([i.width * i.height * i.depth for i in temp_bin.items]) / (spec['w'] * CALC_HEIGHT * spec['l'])
                    score = (util_weight + util_vol) * 100
                
                if score > best_score:
                    best_score = score
                    best_bin = temp_bin

        if best_bin and len(best_bin.items) > 0:
            best_bin.name = f"{best_bin.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_bin)
            
            packed_names = [i.name for i in best_bin.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else:
            # 더 이상 짐이 안 들어감 (규격 초과 등)
            break
            
    return used_trucks

def create_3d_figure(bin_obj):
    fig = go.Figure()
    
    # 트럭 정보 (시각화는 실제 높이 2350mm로 그림)
    truck_type = bin_obj.name.split(' ')[0]
    spec = TRUCK_DB.get(truck_type, TRUCK_DB["22톤"])
    W, Real_H, L = spec['w'], spec['h'], spec['l']
    
    # -- 1. 트럭 디자인 --
    # 바닥 (회색)
    fig.add_trace(go.Mesh3d(x=[0,W,W,0], y=[0,0,L,L], z=[0,0,0,0], color='gray', opacity=0.5, name='바닥'))
    
    # 벽면 (반투명 파랑) - 좌, 우, 앞
    wall_x = [0,0,0,0, W,W,W,W, 0,W,W,0]
    wall_y = [0,L,L,0, 0,L,L,0, L,L,L,L]
    wall_z = [0,0,Real_H,Real_H, 0,0,Real_H,Real_H, 0,0,Real_H,Real_H]
    fig.add_trace(go.Mesh3d(x=wall_x, y=wall_y, z=wall_z, color='lightblue', opacity=0.1, name='적재함 벽', hoverinfo='skip'))

    # 프레임 (검은 선)
    lines_x = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    lines_y = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lines_z = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='black', width=2), showlegend=False))

    # 높이 제한선 (1.3m)
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[1300]*5, mode='lines', line=dict(color='red', dash='dash'), name='높이제한(1.3m)'))

    # -- 2. 박스 그리기 --
    for item in bin_obj.items:
        x, y, z = float(item.position[0]), float(item.position[2]), float(item.position[1])
        w, h, d = float(item.width), float(item.depth), float(item.height)
        
        # 색상: 상위 10% 빨강, 나머지 회색
        color = 'red' if getattr(item, 'is_heavy', False) else '#dddddd'
        opacity = 1.0 if getattr(item, 'is_heavy', False) else 0.8
        
        # 박스 메쉬
        fig.add_trace(go.Mesh3d(
            x=[x,x+w,x+w,x, x,x+w,x+w,x],
            y=[y,y,y+d,y+d, y,y,y+d,y+d],
            z=[z,z,z,z, z+h,z+h,z+h,z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=color, opacity=opacity, flatshading=True, name=item.name
        ))
        
        # 박스 테두리
        edge_x = [x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x]
        edge_y = [y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d]
        edge_z = [z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h]
        fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='black', width=1), showlegend=False))
        
        # 박스 번호 (측면)
        fig.add_trace(go.Scatter3d(
            x=[x + w/2], y=[y], z=[z + h/2],
            mode='text', text=[item.name], textposition="middle center",
            textfont=dict(size=12, color='black'), showlegend=False
        ))

    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0,r=0,b=0,t=0), height=600)
    return fig

# ==========================================
# 3. 메인 UI
# ==========================================
st.title("🚛 물류 적재 시뮬레이터 (v3.0 Final)")
st.caption("✅ 회전금지(원본치수 유지) | 높이 1.3m 제한 | 5/11톤 최적조합 | 상위 10% 강조")

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    # 파일 읽기
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    st.subheader(f"📋 데이터 확인 ({len(df)}건)")
    st.dataframe(df)

    if st.button("최적 배차 실행", type="primary"):
        items = create_items_from_df(df)
        if items:
            with st.spinner("최적의 차량을 계산 중입니다... (회전 없이 적재)"):
                trucks = get_optimized_trucks(items)
            
            if trucks:
                # 결과 요약
                t_names = [t.name.split(' ')[0] for t in trucks]
                from collections import Counter
                cnt = Counter(t_names)
                summary = ", ".join([f"{k} {v}대" for k,v in cnt.items()])
                
                st.success(f"✅ 배차 완료: 총 {len(trucks)}대 ({summary})")
                
                tabs = st.tabs([t.name for t in trucks])
                for i, tab in enumerate(tabs):
                    with tab:
                        col1, col2 = st.columns([1, 3])
                        t = trucks[i]
                        with col1:
                            st.markdown(f"**{t.name}**")
                            st.write(f"- 박스 수: {len(t.items)}개")
                            st.write(f"- 적재 중량: {t.get_total_weight():,} kg")
                            # 적재된 박스 목록
                            with st.expander("적재 박스 목록"):
                                st.write(", ".join([item.name for item in t.items]))
                        with col2:
                            st.plotly_chart(create_3d_figure(t), use_container_width=True)
            else:
                st.error("적재할 수 있는 차량을 찾지 못했습니다.")
