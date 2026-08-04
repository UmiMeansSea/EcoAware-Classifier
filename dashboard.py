import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

st.set_page_config(page_title="Frugal AI Dashboard", page_icon="🌱", layout="wide")
st.title("🌱 Eco-Aware AI: Carbon Footprint & Optimization Dashboard")
st.write("Benchmarking accuracy, memory footprint, and digital sobriety (*sobriété numérique*).")

# --- SECTION 1: CARBON EMISSIONS ---
if os.path.exists("emissions.csv"):
    df_emissions = pd.read_csv("emissions.csv")
    df_emissions["emissions_g"] = df_emissions["emissions"] * 1000

    st.subheader("1. Environmental Impact (Training)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Emissions", f"{df_emissions['emissions_g'].sum():.4f} gCO₂e")
    col2.metric("Energy Consumed", f"{df_emissions['energy_consumed'].sum():.6f} kWh")
    col3.metric("Training Duration", f"{df_emissions['duration'].sum():.2f} sec")
    
    st.divider()

# --- SECTION 2: MODEL COMPARISON ---
st.subheader("2. Heavy FP32 vs. Frugal INT8 Model Comparison")

if os.path.exists("model_comparison.json"):
    with open("model_comparison.json", "r") as f:
        comp_data = json.load(f)
    
    df_comp = pd.DataFrame(comp_data)
    
    # Calculate exact accuracy drop and size compression ratio
    heavy_acc = df_comp.loc[df_comp['Model'] == 'Heavy Model (FP32)', 'Accuracy (%)'].values[0]
    tiny_acc = df_comp.loc[df_comp['Model'] == 'Frugal Model (INT8)', 'Accuracy (%)'].values[0]
    acc_loss = heavy_acc - tiny_acc

    heavy_size = df_comp.loc[df_comp['Model'] == 'Heavy Model (FP32)', 'Size (KB)'].values[0]
    tiny_size = df_comp.loc[df_comp['Model'] == 'Frugal Model (INT8)', 'Size (KB)'].values[0]
    compression_ratio = heavy_size / tiny_size

    # Key Performance Metrics Cards
    m1, m2, m3 = st.columns(3)
    m1.metric("FP32 Accuracy", f"{heavy_acc}%")
    m2.metric("INT8 Accuracy", f"{tiny_acc}%", delta=f"-{acc_loss:.2f}%", delta_color="inverse")
    m3.metric("Size Reduction", f"{compression_ratio:.1f}x smaller", delta=f"-{heavy_size - tiny_size:.1f} KB")

    st.markdown("### Trade-off Visualizations")
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    # Chart 1: Accuracy Comparison
    with chart_col1:
        fig_acc = px.bar(
            df_comp, x="Model", y="Accuracy (%)", 
            text="Accuracy (%)", title="Model Accuracy (%)",
            color="Model"
        )
        fig_acc.update_layout(showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)

    # Chart 2: Model Size Comparison
    with chart_col2:
        fig_size = px.bar(
            df_comp, x="Model", y="Size (KB)", 
            text="Size (KB)", title="Memory Footprint (KB)",
            color="Model"
        )
        fig_size.update_layout(showlegend=False)
        st.plotly_chart(fig_size, use_container_width=True)

    # Chart 3: Inference Latency Comparison
    with chart_col3:
        fig_lat = px.bar(
            df_comp, x="Model", y="Latency (ms/img)", 
            text="Latency (ms/img)", title="Inference Latency (ms/img)",
            color="Model"
        )
        fig_lat.update_layout(showlegend=False)
        st.plotly_chart(fig_lat, use_container_width=True)

else:
    st.warning("Please run `python quantize.py` first to generate comparison metrics.")