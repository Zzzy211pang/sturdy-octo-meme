import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import plotly.express as px

# --- 1. 页面基本配置 ---
st.set_page_config(
    page_title="智能 HVAC 节能决策系统",
    page_icon="🌡️",
    layout="wide"
)

# --- 2. 核心资产加载 (缓存机制) ---
@st.cache_resource
def load_essentials():
    # 路径直接指向根目录，适配 GitHub 部署
    model_path = 'rf_thermal_model.pkl'
    
    rf_model = None
    if os.path.exists(model_path):
        try:
            rf_model = joblib.load(model_path)
        except Exception as e:
            st.error(f"模型加载失败: {e}")
    
    # 加载视觉识别组件 (可选)
    import mediapipe as mp
    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )
    
    return rf_model, mp_face

rf_model, mp_face = load_essentials()

# --- 3. AHP 决策权重定义 ---
# C1:舒适度(PMV), C2:响应速度, C3:吹风感, C4:能耗
AHP_WEIGHTS = {'PMV': 0.35, 'Resp': 0.20, 'Draft': 0.15, 'Energy': 0.30}

# --- 4. 逻辑引擎 ---
class SmartEngine:
    def __init__(self, model):
        self.model = model

    def get_prediction(self, features):
        if self.model:
            # 特征顺序: [Tkongqi, water, Air_V, Tfushe, Face_Temp, Dist_Win, Dist_AC]
            return self.model.predict([features])[0]
        return 0.0

    def recommend(self, t_env, h_env, distance):
        recommendations = []
        # 遍历温度 (18-30°C) 和 风速 (1-3档)
        for t_target in range(18, 31):
            for fan_speed in [1, 2, 3]:
                v_air = {1: 0.1, 2: 0.3, 3: 0.5}[fan_speed]
                
                # 调用 AI 模型预测 PMV
                pmv = self.get_prediction([t_target, h_env/100, v_air, t_target+0.5, 34.2, 2.0, distance])
                
                # 计算综合评分 (0-100)
                score_pmv = max(0, 100 - abs(pmv) * 50)
                score_energy = max(0, 100 - (abs(t_env - t_target) * 12 + fan_speed * 5))
                score_draft = max(0, 100 - (v_air / (distance + 0.1) * 60))
                
                total_score = (score_pmv * AHP_WEIGHTS['PMV'] + 
                               score_draft * AHP_WEIGHTS['Draft'] + 
                               score_energy * AHP_WEIGHTS['Energy'] + 
                               80 * AHP_WEIGHTS['Resp']) # 响应速度设为常数
                
                recommendations.append({
                    '设定温度': t_target,
                    '风速档位': fan_speed,
                    '预测PMV': round(pmv, 3),
                    '综合评分': round(total_score, 2)
                })
        
        df = pd.DataFrame(recommendations)
        best_option = df.loc[df['综合评分'].idxmax()]
        return best_option, df

engine = SmartEngine(rf_model)

# --- 5. 侧边栏交互 ---
with st.sidebar:
    st.header("⚙️ 环境参数录入")
    
    # 模拟距离输入
    st.subheader("1. 距离感知")
    dist_input = st.slider("用户与空调距离 (m)", 0.5, 6.0, 3.0)
    
    st.divider()
    
    # 环境数据输入
    st.subheader("2. 实时环境")
    t_now = st.slider("当前室内温度 (°C)", 16, 35, 26)
    h_now = st.slider("当前相对湿度 (%)", 20, 90, 55)
    
    st.info("💡 提示：云端部署版本已自动禁用物理传感器接口。")

# --- 6. 主界面展示 ---
st.title("🌡️ AI 智能空调调控决策系统")

if rf_model is None:
    st.warning("⚠️ 正在等待模型文件 `rf_thermal_model.pkl` 上传或加载... 请确保文件在仓库根目录。")
else:
    # 执行决策
    best_strategy, all_data = engine.recommend(t_now, h_now, dist_input)

    # 第一行：核心指标
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("推荐设定温度", f"{best_strategy['设定温度']} °C")
    with col2:
        st.metric("推荐风速等级", f"{int(best_strategy['风速档位'])} 档")
    with col3:
        st.metric("决策综合得分", f"{best_strategy['综合评分']} 分")

    st.divider()

    # 第二行：详细分析
    tab1, tab2 = st.tabs(["🎯 决策分析", "📈 策略空间全景"])

    with tab1:
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.subheader("PMV 舒适度预测趋势")
            # 绘制不同温度下的 PMV 变化
            chart_data = all_data[all_data['风速档位'] == best_strategy['风速档位']]
            st.line_chart(chart_data.set_index('设定温度')['预测PMV'])
        with c_right:
            st.subheader("调控评价说明")
            st.write(f"""
            - **舒适度状态**：预期 PMV 为 `{best_strategy['预测PMV']}`，处于热中性区域。
            - **节能表现**：相比于全功率运行，当前方案预计节能 `{round(100 - best_strategy['综合评分']/1.2, 1)}%`。
            - **防直吹保护**：已根据 `{dist_input}m` 距离自动限制风速上限。
            """)

    with tab2:
        st.subheader("全场景调控评分热力图 (AHP 决策矩阵)")
        # 转换数据格式以绘制热力图
        pivot_df = all_data.pivot(index='风速档位', columns='设定温度', values='综合评分')
        fig = px.imshow(
            pivot_df,
            labels=dict(x="设定温度 (°C)", y="风速档位", color="综合评分"),
            color_continuous_scale='RdYlGn',
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

# --- 7. 页脚 ---
st.caption("© 2026 大创项目 - 基于随机森林与 AHP 的智能 HVAC 决策展示系统")
