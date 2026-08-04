import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

st.set_page_config(page_title="Frugal AI Portfolio Dashboard", page_icon="🌱", layout="wide")

st.title("🌱 Eco-Aware Machine Learning: Frugal AI Pipeline")
st.caption("Benchmarking Model Architecture, Compression, and Hardware-Aware Sobriété Numérique.")

st.divider()

# --- SECTION 1: CARBON EMISSIONS OVERVIEW ---
if os.path.exists("emissions.csv"):
    df_emissions = pd.read_csv("emissions.csv")
    df_emissions["emissions_g"] = df_emissions["emissions"] * 1000

    st.subheader("1. Environmental Audit (Training Stage)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total CO₂ Tracked", f"{df_emissions['emissions_g'].sum():.4f} gCO₂e")
    c2.metric("Energy Consumption", f"{df_emissions['energy_consumed'].sum():.6f} kWh")
    c3.metric("Total Training Time", f"{df_emissions['duration'].sum():.2f} sec")
    
    st.divider()

# --- SECTION 2: 3-WAY MODEL COMPARISON ---
st.subheader("2. Multi-Model Architecture & Optimization Benchmark")

if os.path.exists("all_models_comparison.json"):
    with open("all_models_comparison.json", "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Highlight Metrics Cards
    col1, col2, col3 = st.columns(3)
    
    base_acc = df.iloc[0]["Accuracy (%)"]
    adv_acc = df.iloc[2]["Accuracy (%)"]
    acc_diff = adv_acc - base_acc

    base_size = df.iloc[0]["Size (KB)"]
    adv_size = df.iloc[2]["Size (KB)"]
    
    col1.metric(
        label="Accuracy Evolution", 
        value=f"{adv_acc}%", 
        delta=f"{acc_diff:+.2f}% vs Baseline", 
        delta_color="normal"
    )
    col2.metric(
        label="MobileNet Model Size", 
        value=f"{adv_size:.1f} KB", 
        delta=f"{(base_size - adv_size):.1f} KB saved", 
        delta_color="normal"
    )
    col3.metric(
        label="Inference Latency", 
        value=f"{df.iloc[2]['Latency (ms/img)']} ms/img", 
        delta="Optimized for Edge CPU"
    )

    st.write("")
    st.markdown("### Comparative Performance Charts")

    # Color Palette for 3 models
    colors = ["#EF553B", "#FFA15A", "#00CC96"]

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    # Chart 1: Accuracy
    with chart_col1:
        fig_acc = px.bar(
            df, x="Model", y="Accuracy (%)", 
            text="Accuracy (%)", title="Accuracy Comparison (%)",
            color="Model", color_discrete_sequence=colors
        )
        fig_acc.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig_acc, width="stretch")


    # Chart 2: Size
    with chart_col2:
        fig_size = px.bar(
            df, x="Model", y="Size (KB)", 
            text="Size (KB)", title="Memory Footprint (KB)",
            color="Model", color_discrete_sequence=colors
        )
        fig_size.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig_size, width="stretch")

    # Chart 3: Latency
    with chart_col3:
        fig_lat = px.bar(
            df, x="Model", y="Latency (ms/img)", 
            text="Latency (ms/img)", title="Inference Latency (ms/img)",
            color="Model", color_discrete_sequence=colors
        )
        fig_lat.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig_lat, width="stretch")

    # Data Table
    st.markdown("### Raw Comparison Matrix")
    st.dataframe(df, width="stretch")


    # Portfolio Summary Box
    st.info("""
    **Key Master's Thesis Takeaway (*Sobriété Numérique*):**  
    By applying Transfer Learning (MobileNetV2), Weight Pruning (20%), and Post-Training FX Static Quantization (INT8), 
    we drastically boost classification accuracy while locking the mathematical operations to pure 8-bit integer math suitable for microcontrollers and edge hardware.
    """)

else:
    st.warning("Please run `python evaluate_all.py` first to generate benchmark metrics.")