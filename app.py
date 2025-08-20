import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd

# ========== 1. 准备训练数据 ==========
X = pd.DataFrame({
    "顶格资金": [506.14, 1336.34, 1776.34, 1276.51, 1060.2, 950],
    "网上发行": [1404, 2914.6, 2040.66, 1064.29, 1900, 1900],
    "申购价格": [7.21, 9.17, 17.41, 23.99, 11.16, 10],
    "资金规模比": [506.14/1404, 1336.34/2914.6, 1776.34/2040.66,
               1276.51/1064.29, 1060.2/1900, 950/1900],
    "价格调整因子": [7.21*1404, 9.17*2914.6, 17.41*2040.66,
               23.99*1064.29, 11.16*1900, 10*1900]
})
y = np.array([5645, 7409, 7315, 6146, 6289, 5527])

# ========== 2. 训练模型 ==========
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
model.fit(X, y)

# ========== 3. 页面设置 ==========
st.title("申购总额预测工具")
st.write("输入参数，点击按钮即可预测。仅供参考，结果不保证精准！")

# 用户输入
top_funds = st.number_input("顶格资金（万元）", value=1424.15)
online_shares = st.number_input("网上发行（万股）", value=1805.0)
price = st.number_input("申购价格", value=15.78)

# 生成特征
ratio = top_funds / online_shares if online_shares != 0 else 0
adjust_factor = price * online_shares

new_data = pd.DataFrame([{
    "顶格资金": top_funds,
    "网上发行": online_shares,
    "申购价格": price,
    "资金规模比": ratio,
    "价格调整因子": adjust_factor
}])

# 预测按钮
if st.button("预测申购总额"):
    prediction = model.predict(new_data)[0]
    st.success(f"预测申购总额 ≈ {prediction:.2f} 万元")
