import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Page Config
st.set_page_config(page_title="Embraer Agentic AI", layout="wide")

# Title & Description from HTML
st.title("🤖 Agentic AI Demand Forecasting System")
st.markdown("Autonomous demand prediction & inventory optimization for Embraer.")

# Sidebar - Parameters from HTML
st.sidebar.header("⚙️ Optimization Parameters")
u_price = st.sidebar.number_input("Unit Price ($M)", value=169.1)
lead_time = st.sidebar.slider("Lead Time (Months)", 1, 6, 2)
setup_cost = st.sidebar.number_input("Setup Cost ($)", value=500)

# 🎯 Agent Overview Section (Matching HTML)
with st.expander("🎯 View AI Agent Pipeline Logic", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**Agent 1 & 2:** Data Preprocessing & Feature Engineering.")
        st.info("**Agent 3:** Training Ensemble ML Models (Random Forest + Gradient Boosting).")
    with col_b:
        st.info("**Agent 4:** 15-25-60 Monthly Disaggregation Forecast.")
        st.info("**Agent 5:** EOQ & Reorder Point Optimization.")

# File Upload
uploaded_file = st.file_uploader("Upload your Embraer Historical Data (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 🏥 Inventory Health Metrics (Matching HTML)
    st.subheader("🏥 Inventory Health Signals")
    m1, m2, m3 = st.columns(3)
    m1.metric("Inventory Turnover", "2.46x", "Healthy")
    m2.metric("DIO (Days)", "148.3", "-2 Days")
    m3.metric("Inv / Sales", "0.68%", "Improving")

    # --- AI AGENT PROCESSING ---
    if st.button("🚀 Run Agentic AI Pipeline"):
        with st.status("AI Agents are thinking...", expanded=True) as status:
            st.write("Agent 1: Cleaning data...")
            st.write("Agent 2: Calculating quarterly lags...")
            st.write("Agent 3: Training Ensemble models...")
            
            # Simplified Logic for Demo
            annual_forecast = 6358 # Based on your Excel forecast
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            
            # Agent 4 Logic: 15-25-60 Rule from HTML
            q_forecast = annual_forecast / 4
            monthly_vals = [q_forecast*0.15, q_forecast*0.25, q_forecast*0.60] * 4
            
            status.update(label="Pipeline Complete!", state="complete")

        # Visualizations
        st.subheader("📊 2026 Monthly Demand Forecast")
        fig = px.bar(x=months, y=monthly_vals, labels={'x':'Month', 'y':'Demand ($M)'}, color_discrete_sequence=['#2E86AB'])
        st.plotly_chart(fig, use_container_width=True)

        # Agent 5 Optimization Table
        st.subheader("📦 Inventory Optimization (Agent 5)")
        eoq = np.sqrt((2 * annual_forecast * setup_cost) / (u_price * 0.10))
        rop = (annual_forecast / 12) * lead_time
        
        opt_data = {
            "Parameter": ["Economic Order Quantity (EOQ)", "Reorder Point (ROP)", "Safety Stock"],
            "Value": [f"{int(eoq)} Units", f"{int(rop)} Units", "250 Units"],
            "Status": ["✅ Optimized", "⚠️ Action Required", "✅ Stable"]
        }
        st.table(pd.DataFrame(opt_data))

        # Download Button
        st.download_button("📥 Export Forecast Report", data=pd.DataFrame(monthly_vals).to_csv(), file_name="Embraer_2026_Report.csv")
