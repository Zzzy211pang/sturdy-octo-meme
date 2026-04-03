import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import plotly.express as px

# --- 1. 页面基本配置 ---
st.set_page_config(
    page_title="AI 智能 HVAC 节能决策系统",
    page_icon="🌡️",
    layout="wide"
)

# --- 2. 核心资产加载 (带容错机制) ---
@st.cache_resource
def load_essentials():
    # 路径直接指向根目录
    model_path = 'rf_thermal_model.pkl'
    rf_model = None
    
    # 1. 加载随机森林模型
    if os.path.exists(model_path):
        try:
            rf_model = joblib.load(model_path)
        except Exception as e:
            st.error(f"模型加载失败 (版本或路径问题): {e}")
    else:
        st.warning(f"未找到模型文件: {model_path}，请确认文件已上传至 GitHub 根目录。")

    # 2. 加载 Mediapipe (针对云端 libGL 环境做容错)
    mp_face = None
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    except Exception as e:
        # 如果云端缺少 libGL 库，跳过视觉初始化，但不崩溃
        st.sidebar.warning("注意：视觉识别组件未能在当前环境启动（缺少底层库）。")
    
    return rf_model, mp_face

rf_model, mp_face = load_essentials()

# --- 3. AHP 决策权重定义 ---
# 权重分配：舒适度(PMV) 35%, 响应速度 20%, 吹风感 15%, 能耗 30%
AHP_WEIGHTS = {'PMV': 0.35, 'Resp': 0.20, 'Draft': 0.15, 'Energy': 0.30}

# --- 4. 决策逻辑引擎 ---
class SmartEngine:
    def __init__(self, model):
        self.model = model

    def get_prediction(self, features):
        if self.model:
            try:
                # 确保特征维度正确
                return self.model.predict([features])[0]
            except:
                return 0.0
        return 0.0

    def recommend(self, t_env, h_env, distance):
        recommendations = []
        # 模拟决策空间：温度 18-30°C，风速 1-3 档
        for t_target in range(18, 31):
            for fan_speed in [1, 2, 3]:
                v_air = {1: 0.1, 2: 0.3, 3: 0.5}[fan_speed]
                
                # 构建特征向量 (需与训练时保持一致)
                # [Tkongqi, humidity, Air_V, Tfushe, Face_Temp, Dist_Win, Dist_AC]
                features = [t_target, h_env/100, v_air, t_target+0.5, 34.2, 2.0, distance]
                pmv = self.get_prediction(features)
                
                # 计算 AHP 各项得分 (0-100)
                score_pmv = max(0, 100 - abs(pmv) * 50)
                score_energy = max(0, 100 - (abs(t_env - t_target) * 12 + fan_speed * 5))
                score_draft = max(0, 100 - (v_air / (distance + 0.1) * 60))
                
                total_score = (score_pmv * AHP_WEIGHTS['PMV'] + 
                               score_draft * AHP_WEIGHTS['Draft'] + 
                               score_energy * AHP_WEIGHTS['Energy'] + 
                               80 * AHP_WEIGHTS['Resp'])
                
                recommendations.append({
                    '设定温度': t_target,
                    '风速档位': fan_speed,
                    '预测PMV': round(pmv, 3),
                    '综合评分': round(total_score, 2)
                })
        
        df = pd.DataFrame(recommendations)
        if not df.empty:
            best_option = df.loc[df['综合评分'].idxmax()]
            return best_option, df
        return None, None

engine = SmartEngine(rf_model)

# --- 5. 侧边栏交互 ---
with st.sidebar:
    st.header("⚙️ 环境参数模拟")
    
    st.subheader("1. 距离感知")
    dist_input = st.slider("用户与空调距离 (m)", 0.5, 6.0, 3.0)
    
    st.divider()
    
    st.subheader("2. 实时环境录入")
    t_now = st.slider("当前室内温度 (°C)", 16, 35, 26)
    h_now = st.slider("当前相对湿度 (%)", 20, 90, 55)
    
    st.info("💡 提示：该版本已适配云端部署。")

# --- 6. 主界面展示 ---
st.title("🌡️ AI 智能 HVAC 节能决策系统")

if rf_model is None:
    st.error("❌ 核心模型未就绪。请检查 GitHub 仓库中是否存在 `rf_thermal_model.pkl`。")
else:
    # 执行决策运算
    best_strategy, all_data = engine.recommend(t_now, h_now, dist_input)

    if best_strategy is not None:
        # 第一行：核心指标看板
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("💡 推荐设定温度", f"{best_strategy['设定温度']} °C")
        with m2:
            st.metric("🌀 推荐风速等级", f"{int(best_strategy['风速档位'])} 档")
        with m3:
            st.metric("📈 决策综合得分", f"{best_strategy['综合评分']} 分")

        st.divider()

        # 第二行：可视化图表
        tab1, tab2 = st.tabs(["🎯 舒适度趋势", "🗺️ 策略评分全景图"])

        with tab1:
            st.subheader(f"{int(best_strategy['风速档位'])} 档风速下的 PMV 预测曲线")
            line_data = all_data[all_data['风速档位'] == best_strategy['风速档位']]
            st.line_chart(line_data.set_index('设定温度')['预测PMV'])
            st.caption("PMV 说明：0 为最舒适，正数为热感，负数为凉感。")

        with tab2:
            st.subheader("全场景调控评分热力图 (AHP 权重分析)")
            pivot_df = all_data.pivot(index='风速档位', columns='设定温度', values='综合评分')
            fig = px.imshow(
                pivot_df,
                labels=dict(x="设定温度 (°C)", y="风速档位", color="综合评分"),
                color_continuous_scale='RdYlGn', # 绿高红低
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)

# --- 7. 页脚 ---
st.divider()
st.caption("© 2026 大创项目 - 基于随机森林与 AHP 权重的智能空调决策系统演示")
