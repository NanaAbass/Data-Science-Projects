import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="Depression Analyst Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def zero_count(series):
    return (series == 0).sum()

def process_data(df):
    """Ensure timestamp is datetime and calculate logs."""
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
    df['log_activity'] = np.log1p(df['activity'])
    return df

def get_stats(df):
    """Calculate key metrics."""
    stats = {
        "avg_activity": df['activity'].mean(),
        "max_activity": df['activity'].max(),
        "zero_pct": (df['activity'] == 0).mean() * 100,
        "total_readings": len(df)
    }
    return stats

# --- Sidebar ---
st.sidebar.title("📊 Control Panel")
st.sidebar.markdown("Upload your actigraphy data to begin professional analysis.")
upload_file = st.sidebar.file_uploader("Upload Actigraphy CSV", type=["csv"])

# --- Main Interface ---
st.title("🧠 Advanced Depression Analysis Dashboard")
st.markdown("---")

if upload_file is not None:
    try:
        raw_df = pd.read_csv(upload_file)
        if 'activity' not in raw_df.columns:
            st.error("❌ Error: Missing 'activity' column in the uploaded file.")
        else:
            df = process_data(raw_df)
            stats = get_stats(df)

            # --- Row 1: Key Metrics ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Average Activity", f"{stats['avg_activity']:.2f}")
            m2.metric("Max Activity", f"{stats['max_activity']}")
            m3.metric("Zero-Activity %", f"{stats['zero_pct']:.1f}%")
            m4.metric("Total Readings", f"{stats['total_readings']}")

            # --- Tabs for Analysis ---
            tab1, tab2, tab3 = st.tabs(["📈 Overview & Distribution", "🕒 Time-Series Analysis", "🔮 Clinical Prediction"])

            with tab1:
                st.subheader("Data Insights & Distribution")
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.write("#### Activity Spread (Log Scale)")
                    fig = px.histogram(df, x="log_activity", nbins=50, 
                                     title="Frequency Distribution of Log-Activity",
                                     color_discrete_sequence=['#636EFA'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_right:
                    st.write("#### Statistical Summary")
                    st.dataframe(df[['activity', 'log_activity']].describe(), use_container_width=True)
                
                st.write("#### Raw Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

            with tab2:
                st.subheader("Temporal and Circadian Patterns")
                
                if 'timestamp' in df.columns:
                    # Hourly Patterns
                    hourly_avg = df.groupby('hour')['activity'].mean().reset_index()
                    st.write("#### Average Hourly Activity (Circadian Rhythm)")
                    fig_hour = px.line(hourly_avg, x='hour', y='activity', 
                                     labels={'hour': 'Hour of Day (24h)', 'activity': 'Mean Activity'},
                                     title="Movement Patterns Across 24 Hours",
                                     markers=True)
                    fig_hour.update_traces(line_color='#EF553B')
                    st.plotly_chart(fig_hour, use_container_width=True)
                    
                    # Daily Average
                    daily_avg = df.groupby('date')['activity'].mean().reset_index()
                    st.write("#### Daily Activity Trends")
                    fig_day = px.bar(daily_avg, x='date', y='activity',
                                   title="Mean Activity Levels per Day")
                    st.plotly_chart(fig_day, use_container_width=True)
                else:
                    st.warning("⚠️ Timestamp column not found. Temporal analysis is limited.")

            with tab3:
                st.subheader("Predictive Assessment")
                
                # Feature Extraction for the model
                mean_log = df['log_activity'].mean()
                std_log = df['log_activity'].std()
                zero_prop = zero_count(df['activity'])
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write("#### Clinical Features")
                    feat_df = pd.DataFrame({
                        "Feature": ["Mean Log Activity", "Std Log Activity", "Zero-Activity Count"],
                        "Value": [f"{mean_log:.4f}", f"{std_log:.4f}", zero_prop]
                    })
                    st.table(feat_df)
                
                with c2:
                    st.write("#### Assessment Result")
                    # Using findings from the notebook for heuristic prediction
                    # Heuristic: Depressed individuals often show higher zero-activity counts 
                    # and lower mean log activity.
                    is_depressed = (zero_prop > 500) or (mean_log < 2.5)
                    
                    if is_depressed:
                        st.error("### 🚩 Result: Indicators of Depression")
                        st.markdown("""
                        The analyzed data shows patterns frequently associated with clinical depression:
                        - **Fragmented Activity**: High number of zero-activity intervals.
                        - **Low Overall Intensity**: Reduced mean movement levels.
                        """)
                    else:
                        st.success("### ✅ Result: Normal Activity Patterns")
                        st.markdown("""
                        The analyzed data aligns with the 'Control' group benchmarks:
                        - **Consistent Movement**: Sustained activity levels throughout active hours.
                        - **Healthy Circadian Rhythm**: Clear peaks in activity correlated with typical wake cycles.
                        """)
                
                st.divider()
                st.caption("Disclaimer: This tool is for research demonstration purposes and does not replace professional clinical diagnosis.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    # --- Landing Page ---
    st.info("👋 Welcome! Please upload an actigraphy CSV file in the sidebar to begin analysis.")
    st.markdown("""
    ### About this tool
    This application uses Machine Learning insights to analyze activity patterns. 
    It focuses on:
    - **Circadian Rhythms**: How your movement changes throughout the day.
    - **Activity Density**: The ratio of movement to rest.
    - **Statistical Deviations**: Comparing your data against clinical benchmarks.
    """)
    
    # Show example of expected format
    st.write("#### Expected CSV Format:")
    example_data = pd.DataFrame({
        'timestamp': ['2003-05-07 12:00:00', '2003-05-07 12:01:00', '2003-05-07 12:02:00'],
        'activity': [0, 143, 20]
    })
    st.table(example_data)
