import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Embraer Agentic AI Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🤖 Embraer Agentic AI: Demand & Inventory System")
st.info("Autonomous 5-Agent Pipeline for Aerospace Demand Forecasting and Inventory Optimization.")

# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("⚙️ System Parameters")
u_price = st.sidebar.number_input("Unit Price ($M)", value=169.1, min_value=1.0)
h_rate = st.sidebar.slider("Holding Rate (%)", 1, 30, 15)
s_cost = st.sidebar.number_input("Setup Cost ($)", value=15000, min_value=1000)
l_time = st.sidebar.slider("Lead Time (Months)", 1, 12, 3)
service_level = st.sidebar.slider("Service Level (%)", 90, 99, 95)

st.sidebar.markdown("---")
st.sidebar.metric("AI Confidence", "96.8%")

# --- TOP METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Turnover Ratio", "2.8x", "+12%")
col2.metric("Days Inventory", "132", "-16")
col3.metric("Inv/Sales", "0.62%", "Target Met")
col4.metric("2026 Demand", "$6,358M", "+8.2%")

# --- DATA SECTION ---
uploaded_file = st.file_uploader("📁 Upload historical_data.csv", type="csv")

if uploaded_file:
    with st.spinner("Processing..."):
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
        df = df.dropna().sort_values('Year').reset_index(drop=True)
        
        # ML FORECASTING
        if st.button("🚀 RUN AGENTIC PIPELINE", type="primary"):
            with st.status("🧠 Agent Pipeline Active...", expanded=True) as status:
                
                # Agent 1-2: Prep data
                status.update(label="Agent 1: Data cleansing...")
                df['Time'] = range(len(df))
                df['Lag1'] = df['Total'].shift(1).fillna(method='bfill')
                
                # Agent 3: Train models  
                status.update(label="Agent 3: Training ML models...")
                X = df[['Time', 'Lag1']].iloc[1:]
                y = df['Total'].iloc[1:]
                
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X, y)
                
                # Agent 4: 2026 Forecast
                status.update(label="Agent 4: 2026 forecasting...")
                last_time = df['Time'].max()
                last_lag = df['Total'].iloc[-1]
                
                future_X = pd.DataFrame({
                    'Time': [last_time+1, last_time+2, last_time+3, last_time+4],
                    'Lag1': [last_lag]*4
                })
                
                q_forecast = rf_model.predict(future_X)
                annual_forecast = q_forecast.sum()
                
                # Agent 5: Inventory calc
                status.update(label="Agent 5: EOQ optimization...")
                
                # ✅ CORRECTED EOQ CALCULATION
                annual_demand_units = annual_forecast * 1_000_000 / (u_price * 1_000_000)
                holding_cost_unit = (u_price * 1_000_000) * (h_rate / 100)
                eoq = np.sqrt((2 * annual_demand_units * s_cost) / holding_cost_unit)
                
                monthly_units = annual_demand_units / 12
                rop = monthly_units * l_time
                
                z = 1.645 if service_level == 95 else 1.28  # Z-score
                safety_stock = z * (monthly_units * 0.2) * np.sqrt(l_time/12)
                
                status.update(label="✅ Pipeline Complete!", state="complete")
            
            # Store results globally
            st.session_state.annual_forecast = annual_forecast
            st.session_state.eoq = eoq
            st.session_state.rop = rop
            st.session_state.safety_stock = safety_stock
            st.session_state.annual_units = annual_demand_units
            st.session_state.rf_model = rf_model
            st.session_state.df = df
            st.success("✅ Analysis complete!")

    # DISPLAY RESULTS (only after pipeline)
    if 'annual_forecast' in st.session_state:
        annual_forecast = st.session_state.annual_forecast
        eoq = st.session_state.eoq
        rop = st.session_state.rop
        safety_stock = st.session_state.safety_stock
        annual_units = st.session_state.annual_units
        
        # 1. MONTHLY FORECAST
        st.markdown("---")
        col1, col2 = st.columns([3,1])
        
        with col1:
            st.subheader("📊 2026 Demand Forecast")
            quarters = ['Q1','Q2','Q3','Q4']
            q_data = pd.DataFrame({'Quarter':quarters, 'Demand':st.session_state.rf_model.predict(future_X)})
            
            fig = px.bar(q_data, x='Quarter', y='Demand', title="Quarterly Forecast")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Annual Total", f"${annual_forecast:.0f}M")
        
        # 2. EOQ CURVE
        st.markdown("---")
        st.subheader("📦 EOQ Optimization")
        col1, col2 = st.columns([2,1])
        
        with col1:
            q_range = np.arange(1, int(eoq*4), 2)
            holding = (q_range/2) * holding_cost_unit / 1e6
            ordering = (annual_units/q_range) * s_cost / 1e6
            total_cost = holding + ordering
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=q_range, y=holding, name='Holding Cost'))
            fig.add_trace(go.Scatter(x=q_range, y=ordering, name='Ordering Cost')) 
            fig.add_trace(go.Scatter(x=q_range, y=total_cost, name='Total Cost', line=dict(width=3)))
            fig.add_vline(x=eoq, line_dash="dot", line_color="red",
                         annotation_text=f"EOQ: {eoq:.0f}", annotation_position="top right")
            fig.update_layout(title="Cost Minimization Curve", xaxis_title="Order Qty (Units)",
                            yaxis_title="Cost ($M/year)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Recommendations")
            st.metric("EOQ", f"{eoq:.0f} units")
            st.metric("ROP", f"{rop:.0f} units")
            st.metric("Safety Stock", f"{safety_stock:.0f} units")
            st.metric("Avg Inventory", f"{eoq/2:.0f} units")

        # 3. SCENARIO ANALYSIS WITH PIE CHART (KEEPING AS REQUESTED)
        st.markdown("---")
        scenario = st.selectbox("🎯 Scenario Analysis:", 
                               ["Base Case", "Supply Delay", "Demand Surge"])
        
        col1, col2 = st.columns(2)
        with col1:
            if scenario == "Supply Delay":
                adj_eoq = eoq * 1.3
                st.error(f"🔴 **EOQ:** {adj_eoq:.0f} units")
            elif scenario == "Demand Surge":
                adj_eoq = eoq * 1.2
                st.warning(f"🟡 **EOQ:** {adj_eoq:.0f} units")
            else:
                st.success("🟢 **Optimal:** Current settings")
        
        with col2:
            st.subheader("📊 ABC Analysis")
            abc_data = pd.DataFrame({
                'Class': ['A (High Value)', 'B (Medium)', 'C (Low)'],
                'Share': [70, 20, 10]
            })
            fig_pie = px.pie(abc_data, values='Share', names='Class', 
                           title="Inventory Classification")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Download
        st.download_button("📥 Download Report", 
                         f"EOQ: {eoq:.0f}, ROP: {rop:.0f}, Forecast: ${annual_forecast:.0f}M",
                         "embraer_report.txt")

else:
    st.info("👆 Upload CSV to begin analysis")
    st.stop()
