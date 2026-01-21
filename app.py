import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from py3dbp import Packer, Bin, Item

# ==========================================
# 1. 설정 및 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="물류 적재 시뮬레이터 (최종)")

# 차량 제원 (실제 물리적 크기)
TRUCK_DB = {
    "5톤":  {"w": 2350, "h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 24000},
}

# ==========================================
# 2. 핵심 로직
# ==========================================

def create_items_from_df(df):
    items = []
    # 색상 구분을 위해 전체 중량 순으로 정렬 후 상위 10% 기준점 계산
    try:
        sorted_weights = sorted(df['중량'].tolist(), reverse=True)
        top_10_count = max(1, int(len(df) * 0.1)) # 최소 1개는 표시
        heavy_threshold = sorted_weights[top_10_count - 1] if sorted_weights else 0
    except:
        heavy_threshold = 999999

    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            # 상위 10% 여부 판단하여 이름에 태그 추가 (hack)
            is_heavy = weight >= heavy_threshold
            item_obj = Item(name, w, h, l, weight)
            
            # 객체에 사용자 정의 속성 추가 (색상 결정용)
            item_obj.is_heavy = is_heavy 
            items.append(item_obj)
            
        except Exception as e:
            st.error(f"{index}행 데이터 오류: {e}")
            continue
    return items

def get_optimized_trucks(items):
    remaining_items = items[:]
    used_trucks = []
    
    # [로직 변경] 무조건 큰 차(22톤)가 아니라, "효율이 좋은 차"를 찾기 위해 모든 차종을 후보로 둠
    # 작은 차부터 큰 차 순서로 정렬
    truck_types = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])

    while remaining_items:
        best_bin = None
        best_efficiency = -1
        
        # 현재 남은 짐들에 대해, 각 트럭별로 "얼마나 꽉 차는지(효율)" 시뮬레이션
        for t_name in truck_types:
            spec = TRUCK_DB[t_name]
            
            # [규칙 2] 적재 높이 1.3m 제한 (계산용 높이)
            CALC_HEIGHT = 1300 
            
            packer = Packer()
            # Bin(이름, 폭, 높이, 길이, 무게)
            packer.add_bin(Bin(t_name, spec['w'], CALC_HEIGHT, spec['l'], spec['weight']))
            
            for item in remaining_items:
                packer.add_item(item)
            
            # [규칙 1] 회전 최소화: py3dbp 특성상 높이(1.3m)가 낮으면 
            # 긴 박스를 세울 수 없어 자연스럽게 눕혀집니다.
            packer.pack(bigger_first=True, number_of_decimals=0)
            
            temp_bin = packer.bins[0]
            
            # 효율 계산: (적재된 무게 / 차량 허용 무게) + (적재된 부피 / 차량 부피) 
            # 여기서는 단순하게 '가장 많은 아이템을 실은 트럭' 중 '가장 작은 트럭'을 선호하도록 로직 구성
            packed_count = len(temp_bin.items)
            
            if packed_count > 0:
                # 1. 짐을 다 실을 수 있는 가장 작은 차 발견 -> 즉시 선택 (비용 절감)
                if packed_count == len(remaining_items):
                    best_bin = temp_bin
                    break # 더 큰 차 볼 필요 없음
                
                # 2. 다 못 싣는다면, "적재율"이 가장 높은 차를 선택
                # (트럭 용량 대비 얼마나 채웠는가?)
                # 여기서는 간단히 '적재된 무게 비율'로 효율을 따져봅니다.
                efficiency = temp_bin.get_total_weight() / spec['weight']
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_bin = temp_bin

        if best_bin and len(best_bin.items) > 0:
            # 트럭 확정
            best_bin.name = f"{best_bin.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_bin)
            
            # 실린 짐 제거
            packed_names = [item.name for item in best_bin.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else:
            break # 더 이상 적재 불가
            
    return used_trucks

def create_3d_figure(bin_obj):
    fig = go.Figure()
    truck_type = bin_obj.name.split(' ')[0]
    real_spec = TRUCK_DB.get(truck_type, TRUCK_DB["22톤"])
    
    W, Real_H, D = real_spec['w'], real_spec['h'], real_spec['l']
    
    # 1. 트럭 프레임 (실제 높이 2350mm 표현)
    lines_x = [0, W, W, 0, 0, 0, W, W, 0, 0, W, W, 0, 0, W, W]
    lines_y = [0, 0, D, D, 0, 0, 0, D, D, 0, 0, 0, D, D, D, D]
    lines_z = [0, 0, 0, 0, 0, Real_H, Real_H, Real_H, Real_H, Real_H, Real_H, 0, 0, Real_H, Real_H, 0]
    
    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines', line=dict(color='lightgrey', width=3), hoverinfo='none', name='적재함'
    ))
    
    # 높이 제한선 (1.3m)
    fig.add_trace(go.Scatter3d(
        x=[0, W, W, 0, 0], y=[0, 0, D, D, 0], z=[1300, 1300, 1300, 1300, 1300],
        mode='lines', line=dict(color='red', width=2, dash='dash'), name='높이제한(1.3m)'
    ))

    # 2. 박스 그리기
    for item in bin_obj.items:
        # 좌표 변환
        x, y, z = float(item.position[0]), float(item.position[2]), float(item.position[1])
        w, h, d = float(item.width), float(item.depth), float(item.height)
        
        # [규칙 3] 색상 적용 (상위 10% = Red, 나머지 = Grey)
        # item 객체에 아까 심어둔 is_heavy 속성 확인
        is_heavy = getattr(item, 'is_heavy', False)
        box_color = '#FF0000' if is_heavy else '#E0E0E0' # 빨강 vs 연회색
        opacity_val = 1.0 if is_heavy else 0.4 # 중요하지 않은건 약간 투명하게

        # 박스 메쉬
        fig.add_trace(go.Mesh3d(
            x=[x, x+w, x+w, x, x, x+w, x+w, x],
            y=[y, y, y+d, y+d, y, y, y+d, y+d],
            z=[z, z, z, z, z+h, z+h, z+h, z+h],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=box_color, opacity=opacity_val, flatshading=True, name=item.name
        ))
        
        # 박스 테두리 (선명하게)
        wire_x = [x, x+w, x+w, x, x, x, x+w, x+w, x, x, x+w, x+w, x+w, x+w, x, x]
        wire_y = [y, y, y+d, y+d, y, y, y, y+d, y+d, y, y, y, y+d, y+d, y+d, y+d]
        wire_z = [z, z, z, z, z, z+h, z+h, z+h, z+h, z+h, z+h, z, z, z+h, z+h, z]
        fig.add_trace(go.Scatter3d(
            x=wire_x, y=wire_y, z=wire_z,
            mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='none'
        ))

        # [규칙 4] 박스 번호 표시 (측면)
        fig.add_trace(go.Scatter3d(
            x=[x + w/2], y=[y], z=[z + h/2],
            mode='text', text=[str(item.name)],
            textposition="middle center",
            textfont=dict(size=12, color="black", weight="bold"),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='폭 (W)', range=[-100, 2450], showbackground=False),
            yaxis=dict(title='길이 (L)', range=[-100, 10300], showbackground=False),
            zaxis=dict(title='높이 (H)', range=[-100, 2450], showbackground=False),
            aspectratio=dict(x=1, y=3, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=600
    )
    return fig

# ==========================================
# 3. UI 구성
# ==========================================
st.title("🚛 물류 적재 최적화 (규칙 적용 완료)")
st.caption("✅ 적용 규칙: 상위 10% 중량 강조(빨강) | 1.3m 높이 제한 | 박스 서있음 방지 | 최적 효율 배차")

uploaded_file = st.sidebar.file_uploader("엑셀 파일 업로드", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader(f"📋 입력 데이터 (총 {len(df)}건)")
    st.dataframe(df, use_container_width=True)

    if st.button("최적 적재 실행", type="primary"):
        items = create_items_from_df(df)
        if items:
            with st.spinner("최적의 차량 조합을 계산 중입니다..."):
                trucks = get_optimized_trucks(items)
            
            if trucks:
                st.success(f"✅ 분석 완료! 총 **{len(trucks)}대** 필요")
                
                # 탭 생성
                tabs = st.tabs([t.name for t in trucks])
                for i, tab in enumerate(tabs):
                    truck = trucks[i]
                    with tab:
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.markdown(f"### {truck.name}")
                            st.write(f"- 박스 수: {len(truck.items)}개")
                            st.write(f"- 적재 중량: {truck.get_total_weight():,} kg")
                            st.warning(f"**상위 10% 고중량 박스는 빨간색**으로 표시됩니다.")
                            
                        with c2:
                            fig = create_3d_figure(truck)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("적재 가능한 차량을 찾지 못했습니다.")
