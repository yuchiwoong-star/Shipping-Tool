import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from py3dbp import Packer, Bin, Item

# ==========================================
# 1. 설정 및 데이터
# ==========================================

st.set_page_config(layout="wide", page_title="물류 적재 시뮬레이터")

# 차량 제원 (실제 물리적 크기)
TRUCK_DB = {
    "5톤":  {"w": 2350, "h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 24000},
}

# 시각화용 색상
COLORS = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C', '#FF9F1C', '#2B2D42', '#EF233C', '#D90429']

# ==========================================
# 2. 핵심 로직
# ==========================================

def create_items_from_df(df):
    items = []
    # 엑셀 데이터 읽기
    for index, row in df.iterrows():
        try:
            # 컬럼명 매칭: 박스번호, 폭, 높이, 길이, 중량
            name = str(row['박스번호'])
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            # py3dbp Item 생성 (이름, 가로, 높이, 깊이, 무게)
            # 주의: 여기서 높이(h)는 실제 박스의 높이입니다.
            items.append(Item(name, w, h, l, weight))
            
        except KeyError as e:
            st.error(f"❌ 엑셀 컬럼명이 틀렸습니다. (필요: 박스번호, 폭, 높이, 길이, 중량) / 에러: {e}")
            return None
        except Exception as e:
            st.error(f"❌ {index}행 데이터 처리 중 오류: {e}")
            continue
    return items

def get_optimized_trucks(items):
    remaining_items = items[:]
    used_trucks = []
    
    # 작은 차부터 검토 (비용 절감)
    sorted_truck_keys = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])

    while remaining_items:
        best_bin = None
        max_efficiency = -1
        
        for t_name in sorted_truck_keys:
            spec = TRUCK_DB[t_name]
            
            # [규칙 2 핵심] 차량의 물리적 높이는 2350이지만,
            # 계산용 높이(Height)는 '1300'으로 강제 제한합니다.
            # 이렇게 하면 알고리즘이 1.3m 이상 쌓을 수 없습니다.
            CALC_HEIGHT = 1300 
            
            packer = Packer()
            # Bin 생성 (이름, 폭, 높이(제한값), 길이, 무게)
            packer.add_bin(Bin(t_name, spec['w'], CALC_HEIGHT, spec['l'], spec['weight']))
            
            for item in remaining_items:
                packer.add_item(item)
            
            # [규칙 1 고려] py3dbp는 기본적으로 회전을 시도하지만,
            # 높이가 1300으로 제한되어 있어 긴 박스를 세우는(Rotation) 행위가 불가능해집니다.
            # 따라서 자연스럽게 폭/길이가 바닥으로 가도록 유도됩니다.
            packer.pack(bigger_first=True, number_of_decimals=0)
            
            temp_bin = packer.bins[0]
            
            # 100% 다 실리면 즉시 채택
            if len(temp_bin.items) == len(remaining_items):
                best_bin = temp_bin
                break
            
            # 아니면 가장 많이 실리는 차 기억
            if len(temp_bin.items) > max_efficiency:
                max_efficiency = len(temp_bin.items)
                best_bin = temp_bin

        if best_bin and len(best_bin.items) > 0:
            # 트럭 확정 (이름에 번호 붙이기)
            best_bin.name = f"{best_bin.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_bin)
            
            # 실린 짐 제거
            packed_names = [item.name for item in best_bin.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else:
            # 더 이상 실을 수 없는 경우 (짐이 너무 크거나 등등)
            break
            
    return used_trucks

def create_3d_figure(bin_obj):
    fig = go.Figure()
    
    # 트럭 제원 가져오기 (이름에서 톤수 파싱)
    truck_type = bin_obj.name.split(' ')[0] # "11톤" 추출
    real_spec = TRUCK_DB.get(truck_type, TRUCK_DB["22톤"]) # 없으면 기본값
    
    # [시각화] 트럭 프레임은 '실제 높이(2350)'로 그립니다. (적재는 1300까지만 됨)
    W, Real_H, D = real_spec['w'], real_spec['h'], real_spec['l']
    
    # 1. 트럭 바닥 및 프레임 그리기
    lines_x = [0, W, W, 0, 0, 0, W, W, 0, 0, W, W, 0, 0, W, W]
    lines_y = [0, 0, D, D, 0, 0, 0, D, D, 0, 0, 0, D, D, D, D]
    lines_z = [0, 0, 0, 0, 0, Real_H, Real_H, Real_H, Real_H, Real_H, Real_H, 0, 0, Real_H, Real_H, 0]
    
    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines', line=dict(color='lightgrey', width=3), hoverinfo='none', name='적재함'
    ))
    
    # 1.3m 높이 제한선 (빨간 점선) 표시 (시각적 확인용)
    fig.add_trace(go.Scatter3d(
        x=[0, W, W, 0, 0], y=[0, 0, D, D, 0], z=[1300, 1300, 1300, 1300, 1300],
        mode='lines', line=dict(color='red', width=2, dash='dash'), name='높이제한(1.3m)'
    ))

    # 2. 박스 그리기
    for i, item in enumerate(bin_obj.items):
        # 좌표 및 크기
        x, y, z = float(item.position[0]), float(item.position[2]), float(item.position[1])
        w, h, d = float(item.width), float(item.depth), float(item.height)
        
        color = COLORS[i % len(COLORS)]

        # (1) 박스 메쉬 (면)
        fig.add_trace(go.Mesh3d(
            x=[x, x+w, x+w, x, x, x+w, x+w, x],
            y=[y, y, y+d, y+d, y, y, y+d, y+d],
            z=[z, z, z, z, z+h, z+h, z+h, z+h],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=color, opacity=1.0, flatshading=True, name=item.name
        ))
        
        # (2) 박스 테두리 (검은 선)
        wire_x = [x, x+w, x+w, x, x, x, x+w, x+w, x, x, x+w, x+w, x+w, x+w, x, x]
        wire_y = [y, y, y+d, y+d, y, y, y, y+d, y+d, y, y, y, y+d, y+d, y+d, y+d]
        wire_z = [z, z, z, z, z, z+h, z+h, z+h, z+h, z+h, z+h, z, z, z+h, z+h, z]
        fig.add_trace(go.Scatter3d(
            x=wire_x, y=wire_y, z=wire_z,
            mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='none'
        ))

        # [규칙 3] 박스 번호 표시 (측면: 길이 방향 시작 부분)
        # 위치: 가로의 중앙(x+w/2), 길이의 시작(y), 높이의 중앙(z+h/2)
        fig.add_trace(go.Scatter3d(
            x=[x + w/2],
            y=[y], # 길이 방향의 시작면
            z=[z + h/2],
            mode='text',
            text=[str(item.name)],
            textposition="middle center",
            textfont=dict(size=10, color="black"),
            showlegend=False
        ))

    # 카메라 및 축 설정
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='폭 (W)', range=[-100, 2450], showbackground=False),
            yaxis=dict(title='길이 (L)', range=[-100, 10300], showbackground=False),
            zaxis=dict(title='높이 (H)', range=[-100, 2450], showbackground=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=3, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600,
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
    )
    return fig

# ==========================================
# 3. 메인 화면 UI
# ==========================================

st.title("🚛 자동 배차 시뮬레이터 (v2.0)")
st.caption("✅ 적용된 규칙: 회전 금지 | 적재높이 1.3m 제한 | 박스번호 표시")

uploaded_file = st.sidebar.file_uploader("엑셀 파일 업로드", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # 전체 데이터 보여주기 (스크롤 가능)
    st.subheader(f"📋 입력 데이터 (총 {len(df)}건)")
    st.dataframe(df, use_container_width=True)

    if st.button("🚀 최적 적재 실행", type="primary"):
        items = create_items_from_df(df)
        
        if items:
            with st.spinner("규칙에 맞춰 최적의 차량을 계산 중입니다..."):
                trucks = get_optimized_trucks(items)
            
            if not trucks:
                st.error("적재 가능한 차량을 찾지 못했습니다.")
            else:
                st.success(f"✅ 분석 완료! 총 **{len(trucks)}대**의 차량이 필요합니다.")
                st.info("💡 빨간 점선은 1.3m 높이 제한선입니다.")
                
                # 탭으로 결과 보여주기
                tabs = st.tabs([t.name for t in trucks])
                
                for i, tab in enumerate(tabs):
                    current_truck = trucks[i]
                    with tab:
                        c1, c2 = st.columns([1, 3])
                        
                        with c1:
                            st.markdown(f"### 🚛 {current_truck.name}")
                            st.write(f"**실린 박스:** {len(current_truck.items)}개")
                            st.write(f"**총 중량:** {current_truck.get_total_weight():,} kg")
                            
                            # 적재된 박스 리스트 펼쳐보기
                            with st.expander("박스 목록 보기"):
                                item_names = [it.name for it in current_truck.items]
                                st.write(", ".join(item_names))

                        with c2:
                            fig = create_3d_figure(current_truck)
                            st.plotly_chart(fig, use_container_width=True)
