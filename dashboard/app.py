import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import mlflow
import sys
from pathlib import Path

# =====================================================
# PROJECT PATHS AND CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.alerts import check_resource_health

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "metrics.csv"
INFRA_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "infrastructure_metrics.csv"
API_URL = "http://127.0.0.1:8001/predict"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

MLFLOW_EXPERIMENT = "CloudPulse-Resource-Prediction"

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CloudPulse | Cloud Intelligence",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# PROFESSIONAL DARK THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #0B1120;
    color: #E2E8F0;
}
header[data-testid="stHeader"] {
    background: #0B1120;
}
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #253247;
}
h1, h2, h3 {
    color: #F8FAFC !important;
}
p, label {
    color: #CBD5E1;
}
div[data-testid="stMetric"] {
    background: #151F32;
    border: 1px solid #263750;
    padding: 20px;
    border-radius: 12px;
}
div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}
div[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
}
hr {
    border-color: #253247;
}
.cloud-card {
    background: #151F32;
    border: 1px solid #263750;
    border-radius: 12px;
    padding: 18px;
}
.muted {
    color: #94A3B8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# CHART THEME
# =====================================================

CHART_BG = "#111827"
PAPER_BG = "#111827"
GRID_COLOR = "#263750"
TEXT_COLOR = "#CBD5E1"
CPU_COLOR = "#3B82F6"
MEMORY_COLOR = "#A78BFA"


def style_chart(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=TEXT_COLOR, family="Arial"),
        margin=dict(l=15, r=15, t=45, b=15),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(showgrid=False, color=TEXT_COLOR),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            color=TEXT_COLOR
        )
    )
    return fig


# =====================================================
# MLFLOW MODEL HISTORY
# =====================================================

def load_model_history():

    experiment = mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT
    )

    if experiment is None:
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    return runs


def get_run_value(run, column, default=None):
    """Safely read a value from an MLflow run."""
    value = run.get(column, default)

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    return value


def format_metric(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def strategy_label(strategy):
    if strategy == "random_forest":
        return "Random Forest"
    if strategy == "previous_value":
        return "Previous Value Baseline"
    return str(strategy or "Unavailable")


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        "<h1 style='color:#60A5FA;'>☁️ CloudPulse</h1>",
        unsafe_allow_html=True
    )

    st.caption("INTELLIGENT CLOUD MONITORING")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Resource Analytics",
            "Infrastructure Monitoring",
            "ML Predictions",
            "System Health",
            "Model Performance"
        ]
    )

    st.divider()
    st.markdown("### Platform Status")
    st.success("Monitoring enabled")
    st.caption("Automatic refresh: 10 seconds")

    st.divider()
    st.caption("CloudPulse v2.0")
    st.caption("MLOps Cloud Computing Project")


# =====================================================
# HEADER
# =====================================================

st.title("Cloud Infrastructure Dashboard")

st.markdown(
    "<p class='muted'>"
    "Real-time resource monitoring and machine learning insights"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# GAUGE CHART
# =====================================================

def create_gauge(title, value, maximum, color, suffix):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": suffix,
                "font": {"color": "#F8FAFC", "size": 30}
            },
            title={
                "text": title,
                "font": {"color": TEXT_COLOR, "size": 16}
            },
            gauge={
                "axis": {
                    "range": [0, maximum],
                    "tickcolor": TEXT_COLOR
                },
                "bar": {"color": color},
                "bgcolor": CHART_BG,
                "bordercolor": GRID_COLOR,
                "steps": [
                    {
                        "range": [0, maximum * 0.7],
                        "color": "#1E293B"
                    },
                    {
                        "range": [maximum * 0.7, maximum * 0.9],
                        "color": "#78350F"
                    },
                    {
                        "range": [maximum * 0.9, maximum],
                        "color": "#7F1D1D"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=250,
        paper_bgcolor=PAPER_BG,
        font={"color": TEXT_COLOR},
        margin=dict(l=25, r=25, t=55, b=15)
    )

    return fig


# =====================================================
# CPU HISTORY CHART
# =====================================================

def create_cpu_chart(df):

    chart_df = df.copy()

    chart_df["cpu_average"] = (
        chart_df["cpu_percent"]
        .rolling(window=5, min_periods=1)
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["cpu_percent"],
            mode="lines",
            name="CPU Usage",
            line=dict(
                color=CPU_COLOR,
                width=2.5,
                shape="spline"
            ),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.12)",
            hovertemplate=(
                "Time: %{x}<br>"
                "CPU: %{y:.2f}%<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["cpu_average"],
            mode="lines",
            name="5-point Average",
            line=dict(
                color="#FBBF24",
                width=2,
                dash="dash"
            )
        )
    )

    fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="#FBBF24",
        annotation_text="Warning 70%"
    )

    fig.add_hline(
        y=90,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="Critical 90%"
    )

    fig.update_yaxes(
        title="CPU Usage (%)",
        rangemode="tozero"
    )

    fig.update_xaxes(rangeslider_visible=True)

    return style_chart(fig, 400)


# =====================================================
# MEMORY HISTORY CHART
# =====================================================

def create_memory_chart(df):

    chart_df = df.copy()

    chart_df["memory_average"] = (
        chart_df["memory_mb"]
        .rolling(window=5, min_periods=1)
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["memory_mb"],
            mode="lines",
            name="Memory Usage",
            line=dict(
                color=MEMORY_COLOR,
                width=2.5,
                shape="spline"
            ),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.12)",
            hovertemplate=(
                "Time: %{x}<br>"
                "Memory: %{y:.2f} MB<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["timestamp"],
            y=chart_df["memory_average"],
            mode="lines",
            name="5-point Average",
            line=dict(
                color="#FBBF24",
                width=2,
                dash="dash"
            )
        )
    )

    fig.add_hline(
        y=1024,
        line_dash="dot",
        line_color="#FBBF24",
        annotation_text="Warning 1024 MB"
    )

    fig.add_hline(
        y=1536,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="Critical 1536 MB"
    )

    fig.update_yaxes(
        title="Memory Usage (MB)",
        rangemode="tozero"
    )

    fig.update_xaxes(rangeslider_visible=True)

    return style_chart(fig, 400)


# =====================================================
# SMART ALERT DISPLAY
# =====================================================

def display_smart_alerts(health_result):

    status = health_result["status"]

    st.subheader("Smart Resource Alerts")

    if status == "CRITICAL":
        st.error("🚨 CRITICAL: Immediate attention required.")
    elif status == "WARNING":
        st.warning("⚠️ WARNING: Resource usage needs attention.")
    else:
        st.success("✅ HEALTHY: All monitored resources are normal.")

    st.markdown("#### Detected Alerts")

    for alert in health_result["alerts"]:

        if "CRITICAL" in alert:
            st.error(alert)
        elif "WARNING" in alert:
            st.warning(alert)
        else:
            st.success(alert)

    st.markdown("#### Optimization Recommendations")

    for index, recommendation in enumerate(
        health_result["recommendations"], start=1
    ):
        st.markdown(f"**{index}.** {recommendation}")


# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

def display_model_performance():

    st.subheader("Model Performance Monitoring")

    st.caption(
        "Monitor model selection, prediction errors, "
        "baseline comparisons and MLflow training history."
    )

    try:
        runs = load_model_history()

    except Exception as error:
        st.error(f"Could not load MLflow history: {error}")
        return

    if runs.empty:
        st.warning("No MLflow training experiments found.")
        return

    latest_run = runs.iloc[0]

    # Latest run metrics from the updated training script

    cpu_rf_mae = get_run_value(
        latest_run, "metrics.cpu_random_forest_mae"
    )
    cpu_baseline_mae = get_run_value(
        latest_run, "metrics.cpu_baseline_mae"
    )
    cpu_r2 = get_run_value(
        latest_run, "metrics.cpu_random_forest_r2"
    )

    memory_rf_mae = get_run_value(
        latest_run, "metrics.memory_random_forest_mae"
    )
    memory_baseline_mae = get_run_value(
        latest_run, "metrics.memory_baseline_mae"
    )
    memory_r2 = get_run_value(
        latest_run, "metrics.memory_random_forest_r2"
    )

    cpu_strategy = get_run_value(
        latest_run, "params.cpu_selected_strategy", "Unknown"
    )
    memory_strategy = get_run_value(
        latest_run, "params.memory_selected_strategy", "Unknown"
    )

    # -------------------------------------------------
    # SELECTED STRATEGIES
    # -------------------------------------------------

    st.markdown("### Currently Selected Prediction Strategies")

    strategy_col1, strategy_col2 = st.columns(2)

    with strategy_col1:
        st.markdown("#### CPU Prediction")
        st.info(strategy_label(cpu_strategy))

    with strategy_col2:
        st.markdown("#### Memory Prediction")
        st.info(strategy_label(memory_strategy))

    st.divider()

    # -------------------------------------------------
    # RANDOM FOREST METRICS
    # -------------------------------------------------

    st.markdown("### Random Forest Evaluation Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("CPU RF MAE", format_metric(cpu_rf_mae))
    col2.metric("CPU RF R²", format_metric(cpu_r2))
    col3.metric("Memory RF MAE", format_metric(memory_rf_mae))
    col4.metric("Memory RF R²", format_metric(memory_r2))

    st.divider()

    # -------------------------------------------------
    # BASELINE COMPARISON
    # -------------------------------------------------

    st.markdown("### Random Forest vs Previous-Value Baseline")

    comparison_rows = []

    if cpu_rf_mae is not None and cpu_baseline_mae is not None:
        comparison_rows.append({
            "Resource": "CPU",
            "Random Forest MAE": float(cpu_rf_mae),
            "Baseline MAE": float(cpu_baseline_mae),
            "Selected Strategy": strategy_label(cpu_strategy)
        })

    if memory_rf_mae is not None and memory_baseline_mae is not None:
        comparison_rows.append({
            "Resource": "Memory",
            "Random Forest MAE": float(memory_rf_mae),
            "Baseline MAE": float(memory_baseline_mae),
            "Selected Strategy": strategy_label(memory_strategy)
        })

    if comparison_rows:

        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        chart_rows = []

        for row in comparison_rows:
            chart_rows.append({
                "Resource": row["Resource"],
                "Strategy": "Random Forest",
                "MAE": row["Random Forest MAE"]
            })
            chart_rows.append({
                "Resource": row["Resource"],
                "Strategy": "Previous-Value Baseline",
                "MAE": row["Baseline MAE"]
            })

        chart_df = pd.DataFrame(chart_rows)

        comparison_fig = go.Figure()

        for strategy_name, color in [
            ("Random Forest", CPU_COLOR),
            ("Previous-Value Baseline", MEMORY_COLOR)
        ]:
            selected = chart_df[
                chart_df["Strategy"] == strategy_name
            ]

            comparison_fig.add_trace(
                go.Bar(
                    x=selected["Resource"],
                    y=selected["MAE"],
                    name=strategy_name,
                    marker_color=color
                )
            )

        comparison_fig.update_layout(
            title="Prediction Error Comparison (Lower is Better)",
            yaxis_title="Mean Absolute Error",
            barmode="group"
        )

        st.plotly_chart(
            style_chart(comparison_fig),
            use_container_width=True,
            key="baseline_comparison"
        )

    else:
        st.info(
            "The latest MLflow run does not contain the new "
            "baseline comparison metrics. Run model training again."
        )

    st.divider()

    # -------------------------------------------------
    # MODEL SELECTION EXPLANATION
    # -------------------------------------------------

    st.markdown("### Automatic Model Selection")

    st.write(
        "CloudPulse evaluates the Random Forest model against a "
        "simple previous-value baseline. The strategy with the "
        "lower Mean Absolute Error (MAE) is selected independently "
        "for CPU and memory predictions."
    )

    st.caption(
        "A lower MAE indicates a smaller average prediction error. "
        "The selected strategy is saved and used by the FastAPI service."
    )

    st.divider()

    # -------------------------------------------------
    # MLFLOW TRAINING HISTORY
    # -------------------------------------------------

    st.markdown("### MLflow Training History")

    history = runs.copy()

    history["Training Time"] = pd.to_datetime(
        history["start_time"],
        errors="coerce"
    )

    history["CPU RF MAE"] = history.get(
        "metrics.cpu_random_forest_mae",
        pd.Series(index=history.index, dtype=float)
    )

    history["CPU Baseline MAE"] = history.get(
        "metrics.cpu_baseline_mae",
        pd.Series(index=history.index, dtype=float)
    )

    history["CPU RF R²"] = history.get(
        "metrics.cpu_random_forest_r2",
        pd.Series(index=history.index, dtype=float)
    )

    history["Memory RF MAE"] = history.get(
        "metrics.memory_random_forest_mae",
        pd.Series(index=history.index, dtype=float)
    )

    history["Memory Baseline MAE"] = history.get(
        "metrics.memory_baseline_mae",
        pd.Series(index=history.index, dtype=float)
    )

    history["Memory RF R²"] = history.get(
        "metrics.memory_random_forest_r2",
        pd.Series(index=history.index, dtype=float)
    )

    history["CPU Strategy"] = history.get(
        "params.cpu_selected_strategy",
        pd.Series("Not recorded", index=history.index)
    )

    history["Memory Strategy"] = history.get(
        "params.memory_selected_strategy",
        pd.Series("Not recorded", index=history.index)
    )

    display_columns = [
        "Training Time",
        "CPU RF MAE",
        "CPU Baseline MAE",
        "CPU RF R²",
        "Memory RF MAE",
        "Memory Baseline MAE",
        "Memory RF R²",
        "CPU Strategy",
        "Memory Strategy"
    ]

    st.dataframe(
        history[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.caption(f"Total recorded training runs: {len(history)}")

    # -------------------------------------------------
    # HISTORICAL PERFORMANCE CHART
    # -------------------------------------------------

    st.divider()

    st.markdown("### Model Performance Across Training Runs")

    metric_options = [
        "CPU RF MAE",
        "CPU Baseline MAE",
        "CPU RF R²",
        "Memory RF MAE",
        "Memory Baseline MAE",
        "Memory RF R²"
    ]

    metric_choice = st.selectbox(
        "Select performance metric",
        metric_options
    )

    chart_data = history.sort_values("Training Time").copy()

    chart_data[metric_choice] = pd.to_numeric(
        chart_data[metric_choice],
        errors="coerce"
    )

    chart_data = chart_data.dropna(
        subset=["Training Time", metric_choice]
    )

    if not chart_data.empty:

        history_fig = go.Figure()

        history_fig.add_trace(
            go.Scatter(
                x=chart_data["Training Time"],
                y=chart_data[metric_choice],
                mode="lines+markers",
                name=metric_choice,
                line=dict(
                    color=CPU_COLOR,
                    width=3
                ),
                marker=dict(size=9)
            )
        )

        history_fig.update_layout(
            title=f"{metric_choice} Training History",
            xaxis_title="Training Time",
            yaxis_title=metric_choice
        )

        st.plotly_chart(
            style_chart(history_fig),
            use_container_width=True,
            key="model_history_chart"
        )

    else:
        st.info("No historical values available for this metric.")

    # -------------------------------------------------
    # LATEST RUN INFORMATION
    # -------------------------------------------------

    st.divider()

    st.markdown("### Latest MLflow Run Information")

    st.code(
        str(latest_run.get("run_id", "Unavailable")),
        language="text"
    )

    st.caption("Experiment: CloudPulse-Resource-Prediction")



# =====================================================
# INFRASTRUCTURE MONITORING PAGE
# =====================================================

def display_infrastructure_monitoring():

    st.subheader("Infrastructure Monitoring")
    st.caption(
        "Extended infrastructure metrics collected through Prometheus, "
        "cAdvisor and Node Exporter."
    )

    if not INFRA_DATA_FILE.exists():
        st.warning(
            "Infrastructure metrics file not found. "
            "Run the metrics collector first."
        )
        return

    try:
        infra_df = pd.read_csv(INFRA_DATA_FILE)

        required_columns = [
            "timestamp",
            "host_cpu_percent",
            "host_memory_used_mb",
            "host_memory_percent",
            "disk_percent",
            "network_receive_bytes_per_sec",
            "network_transmit_bytes_per_sec"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in infra_df.columns
        ]

        if missing_columns:
            st.error(
                "Infrastructure CSV is missing these columns: "
                + ", ".join(missing_columns)
            )
            return

        infra_df["timestamp"] = pd.to_datetime(
            infra_df["timestamp"],
            errors="coerce"
        )

        numeric_columns = [
            "host_cpu_percent",
            "host_memory_used_mb",
            "host_memory_percent",
            "disk_percent",
            "network_receive_bytes_per_sec",
            "network_transmit_bytes_per_sec"
        ]

        for column in numeric_columns:
            infra_df[column] = pd.to_numeric(
                infra_df[column],
                errors="coerce"
            )

        infra_df = infra_df.dropna(
            subset=["timestamp"] + numeric_columns
        ).sort_values("timestamp")

    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read infrastructure metrics: {error}")
        return

    if infra_df.empty:
        st.warning("No valid infrastructure observations are available yet.")
        return

    latest_infra = infra_df.iloc[-1]

    host_cpu = float(latest_infra["host_cpu_percent"])
    host_memory_used = float(latest_infra["host_memory_used_mb"])
    host_memory_percent = float(latest_infra["host_memory_percent"])
    disk_percent = float(latest_infra["disk_percent"])
    network_receive = float(
        latest_infra["network_receive_bytes_per_sec"]
    )
    network_transmit = float(
        latest_infra["network_transmit_bytes_per_sec"]
    )

    # Current infrastructure metrics
    st.markdown("### Latest Infrastructure Metrics")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Host CPU", f"{host_cpu:.2f}%")
    metric2.metric("Host Memory Used", f"{host_memory_used:.2f} MB")
    metric3.metric("Host Memory", f"{host_memory_percent:.2f}%")
    metric4.metric("Disk Usage", f"{disk_percent:.2f}%")

    metric5, metric6 = st.columns(2)

    metric5.metric(
        "Network Receive",
        f"{network_receive / 1024:.2f} KB/s"
    )
    metric6.metric(
        "Network Transmit",
        f"{network_transmit / 1024:.2f} KB/s"
    )

    st.caption(
        "Latest sample: "
        + latest_infra["timestamp"].strftime("%d %b %Y, %H:%M:%S")
    )

    st.info(
        "Node Exporter reports metrics from the Linux environment used by "
        "Docker Desktop. These values should not be interpreted as direct "
        "measurements of the physical Windows host."
    )

    if disk_percent == 0:
        st.warning(
            "Disk usage is currently reported as 0%. Verify the Node Exporter "
            "filesystem query and mountpoint labels before relying on this value."
        )

    st.divider()

    # Host CPU history
    st.markdown("### Host CPU Utilization")

    cpu_fig = go.Figure()
    cpu_fig.add_trace(
        go.Scatter(
            x=infra_df["timestamp"],
            y=infra_df["host_cpu_percent"],
            mode="lines",
            name="Host CPU",
            line=dict(color=CPU_COLOR, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.12)"
        )
    )
    cpu_fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="#FBBF24",
        annotation_text="Warning 70%"
    )
    cpu_fig.add_hline(
        y=90,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="Critical 90%"
    )
    cpu_fig.update_yaxes(
        title="CPU Utilization (%)",
        rangemode="tozero"
    )
    st.plotly_chart(
        style_chart(cpu_fig),
        use_container_width=True,
        key="infra_host_cpu"
    )

    # Host memory history
    st.markdown("### Host Memory Utilization")

    memory_fig = go.Figure()
    memory_fig.add_trace(
        go.Scatter(
            x=infra_df["timestamp"],
            y=infra_df["host_memory_percent"],
            mode="lines",
            name="Host Memory",
            line=dict(color=MEMORY_COLOR, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.12)"
        )
    )
    memory_fig.update_yaxes(
        title="Memory Utilization (%)",
        rangemode="tozero"
    )
    st.plotly_chart(
        style_chart(memory_fig),
        use_container_width=True,
        key="infra_host_memory"
    )

    # Disk history
    st.markdown("### Disk Utilization")

    disk_fig = go.Figure()
    disk_fig.add_trace(
        go.Scatter(
            x=infra_df["timestamp"],
            y=infra_df["disk_percent"],
            mode="lines+markers",
            name="Disk Usage",
            line=dict(color="#F59E0B", width=2.5)
        )
    )
    disk_fig.update_yaxes(
        title="Disk Utilization (%)",
        rangemode="tozero"
    )
    st.plotly_chart(
        style_chart(disk_fig),
        use_container_width=True,
        key="infra_disk"
    )

    # Network history
    st.markdown("### Network Throughput")

    network_fig = go.Figure()
    network_fig.add_trace(
        go.Scatter(
            x=infra_df["timestamp"],
            y=infra_df["network_receive_bytes_per_sec"] / 1024,
            mode="lines",
            name="Receive",
            line=dict(color=CPU_COLOR, width=2.5)
        )
    )
    network_fig.add_trace(
        go.Scatter(
            x=infra_df["timestamp"],
            y=infra_df["network_transmit_bytes_per_sec"] / 1024,
            mode="lines",
            name="Transmit",
            line=dict(color=MEMORY_COLOR, width=2.5)
        )
    )
    network_fig.update_yaxes(
        title="Throughput (KB/s)",
        rangemode="tozero"
    )
    st.plotly_chart(
        style_chart(network_fig),
        use_container_width=True,
        key="infra_network"
    )

    st.divider()

    st.markdown("### Recent Infrastructure Records")

    display_df = infra_df.tail(20).sort_values(
        "timestamp",
        ascending=False
    ).copy()

    display_df["network_receive_kb_per_sec"] = (
        display_df["network_receive_bytes_per_sec"] / 1024
    )
    display_df["network_transmit_kb_per_sec"] = (
        display_df["network_transmit_bytes_per_sec"] / 1024
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Total infrastructure observations: {len(infra_df)}"
    )


# =====================================================
# MAIN DASHBOARD
# =====================================================

@st.fragment(run_every="10s")
def monitoring_dashboard():

    # LOAD DATA

    try:

        df = pd.read_csv(DATA_FILE)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df["cpu_percent"] = pd.to_numeric(
            df["cpu_percent"],
            errors="coerce"
        )

        df["memory_mb"] = pd.to_numeric(
            df["memory_mb"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp", "cpu_percent", "memory_mb"]
        )

        df = df.sort_values("timestamp")

    except Exception as error:

        st.error(f"Could not load metrics: {error}")
        return

    if df.empty:

        st.warning("No monitoring data available yet.")
        return

    latest = df.iloc[-1]

    cpu = float(latest["cpu_percent"])
    memory = float(latest["memory_mb"])

   
    # FASTAPI PREDICTIONS

    predictions = None
    api_error = None

    try:
        # Get the latest five observations in chronological order
        recent_metrics = df.tail(5)

        if len(recent_metrics) < 5:
            raise ValueError(
                "At least five observations are required for forecasting."
            )

        cpu_history = (
            recent_metrics["cpu_percent"].astype(float).tolist()
        )

        memory_history = (
            recent_metrics["memory_mb"].astype(float).tolist()
        )

        response = requests.post(
            API_URL,
            json={
                "cpu_history": cpu_history,
                "memory_history": memory_history
            },
            timeout=10
        )

        response.raise_for_status()
        predictions = response.json()

    except (requests.RequestException, ValueError) as error:
        api_error = str(error)


    # SMART ALERT ENGINE

    health_result = check_resource_health(cpu, memory)
    health = health_result["status"]

    health_message = {
        "HEALTHY": "Resources are within configured limits.",
        "WARNING": "Resource usage needs attention.",
        "CRITICAL": "High resource usage detected."
    }[health]

    # PLATFORM STATUS

    status1, status2 = st.columns(2)

    status1.success("Monitoring: Connected")

    if predictions is not None:
        status2.success("Prediction API: Connected")
    else:
        status2.error("Prediction API: Disconnected")

    if api_error:
        st.warning(
            "Live predictions are temporarily unavailable. "
            f"API details: {api_error}"
        )

    # =================================================
    # OVERVIEW
    # =================================================

    if page == "Overview":

        st.subheader("Infrastructure Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("CPU Utilization", f"{cpu:.2f}%")
        col2.metric("Memory Usage", f"{memory:.2f} MB")

        if predictions is not None:

            predicted_cpu = float(
                predictions["predicted_cpu_percent"]
            )

            predicted_memory = float(
                predictions["predicted_memory_mb"]
            )

            col3.metric("Predicted CPU", f"{predicted_cpu:.2f}%")
            col4.metric("Predicted Memory", f"{predicted_memory:.2f} MB")

        else:

            col3.metric("Predicted CPU", "Unavailable")
            col4.metric("Predicted Memory", "Unavailable")

        st.divider()

        st.subheader("System Health")

        if health == "HEALTHY":
            st.success("HEALTHY — " + health_message)
        elif health == "WARNING":
            st.warning("WARNING — " + health_message)
        else:
            st.error("CRITICAL — " + health_message)

        display_smart_alerts(health_result)

        st.divider()

        st.subheader("Live Resource Gauges")

        gauge1, gauge2 = st.columns(2)

        with gauge1:

            st.plotly_chart(
                create_gauge(
                    "CPU Utilization",
                    cpu,
                    100,
                    CPU_COLOR,
                    "%"
                ),
                use_container_width=True,
                key="cpu_gauge"
            )

        with gauge2:

            memory_max = max(2048, memory * 1.2)

            st.plotly_chart(
                create_gauge(
                    "Memory Usage",
                    memory,
                    memory_max,
                    MEMORY_COLOR,
                    " MB"
                ),
                use_container_width=True,
                key="memory_gauge"
            )

        st.divider()

        st.subheader("Resource Performance")

        st.markdown("**CPU Utilization Over Time**")

        st.plotly_chart(
            create_cpu_chart(df),
            use_container_width=True,
            key="overview_cpu"
        )

        st.markdown("**Memory Utilization Over Time**")

        st.plotly_chart(
            create_memory_chart(df),
            use_container_width=True,
            key="overview_memory"
        )

        st.caption(
            f"Showing {len(df)} actual collected observations."
        )

    # =================================================
    # RESOURCE ANALYTICS
    # =================================================

    elif page == "Resource Analytics":

        st.subheader("Resource Analytics")

        start_date = df["timestamp"].min().date()
        end_date = df["timestamp"].max().date()

        selected_dates = st.date_input(
            "Filter monitoring dates",
            value=(start_date, end_date)
        )

        if (
            isinstance(selected_dates, (tuple, list))
            and len(selected_dates) == 2
        ):

            start, end = selected_dates

            filtered_df = df[
                (df["timestamp"].dt.date >= start)
                & (df["timestamp"].dt.date <= end)
            ]

        else:

            filtered_df = df

        if filtered_df.empty:

            st.warning("No data available for this date range.")
            return

        st.subheader("CPU Performance")

        st.plotly_chart(
            create_cpu_chart(filtered_df),
            use_container_width=True,
            key="analytics_cpu"
        )

        st.subheader("Memory Performance")

        st.plotly_chart(
            create_memory_chart(filtered_df),
            use_container_width=True,
            key="analytics_memory"
        )

        st.divider()

        st.subheader("Resource Statistics")

        stat1, stat2, stat3 = st.columns(3)

        stat1.metric(
            "Average CPU",
            f"{filtered_df['cpu_percent'].mean():.2f}%"
        )

        stat2.metric(
            "Maximum CPU",
            f"{filtered_df['cpu_percent'].max():.2f}%"
        )

        stat3.metric(
            "Minimum CPU",
            f"{filtered_df['cpu_percent'].min():.2f}%"
        )

        stat4, stat5, stat6 = st.columns(3)

        stat4.metric(
            "Average Memory",
            f"{filtered_df['memory_mb'].mean():.2f} MB"
        )

        stat5.metric(
            "Maximum Memory",
            f"{filtered_df['memory_mb'].max():.2f} MB"
        )

        stat6.metric(
            "Minimum Memory",
            f"{filtered_df['memory_mb'].min():.2f} MB"
        )

        st.divider()

        st.subheader("Recent Monitoring Records")

        st.dataframe(
            filtered_df.tail(20).sort_values(
                "timestamp",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    # =================================================
    # INFRASTRUCTURE MONITORING
    # =================================================

    elif page == "Infrastructure Monitoring":

        display_infrastructure_monitoring()

    # =================================================
    # ML PREDICTIONS
    # =================================================

    elif page == "ML Predictions":

        st.subheader("Machine Learning Predictions")

        st.caption(
            "Predictions generated using CloudPulse's "
            "automatically selected prediction strategies."
        )

        if predictions is None:

            st.warning(
                "Prediction API is unavailable. "
                "Start FastAPI to view predictions."
            )

        else:

            predicted_cpu = float(
                predictions["predicted_cpu_percent"]
            )

            predicted_memory = float(
                predictions["predicted_memory_mb"]
            )

            cpu_strategy = predictions.get(
                "cpu_strategy", "Unknown"
            )

            memory_strategy = predictions.get(
                "memory_strategy", "Unknown"
            )

            # Selected strategies

            st.markdown("### Active Prediction Strategies")

            strategy_col1, strategy_col2 = st.columns(2)

            strategy_col1.info(
                f"CPU: {strategy_label(cpu_strategy)}"
            )

            strategy_col2.info(
                f"Memory: {strategy_label(memory_strategy)}"
            )

            st.divider()

            # Prediction metrics

            col1, col2 = st.columns(2)

            col1.metric(
                "Current CPU",
                f"{cpu:.2f}%",
                delta=f"{predicted_cpu - cpu:+.2f} percentage points",
                delta_color="off"
            )

            col2.metric(
                "Predicted CPU",
                f"{predicted_cpu:.2f}%"
            )

            col3, col4 = st.columns(2)

            col3.metric(
                "Current Memory",
                f"{memory:.2f} MB",
                delta=f"{predicted_memory - memory:+.2f} MB",
                delta_color="off"
            )

            col4.metric(
                "Predicted Memory",
                f"{predicted_memory:.2f} MB"
            )

            st.divider()

            # CPU comparison

            cpu_fig = go.Figure()

            cpu_fig.add_trace(
                go.Bar(
                    x=["Current", "Predicted"],
                    y=[cpu, predicted_cpu],
                    marker_color=[CPU_COLOR, MEMORY_COLOR],
                    name="CPU"
                )
            )

            cpu_fig.update_layout(
                title="CPU: Current vs Predicted",
                yaxis_title="CPU (%)"
            )

            st.plotly_chart(
                style_chart(cpu_fig),
                use_container_width=True,
                key="prediction_cpu"
            )

            # Memory comparison

            memory_fig = go.Figure()

            memory_fig.add_trace(
                go.Bar(
                    x=["Current", "Predicted"],
                    y=[memory, predicted_memory],
                    marker_color=[CPU_COLOR, MEMORY_COLOR],
                    name="Memory"
                )
            )

            memory_fig.update_layout(
                title="Memory: Current vs Predicted",
                yaxis_title="Memory (MB)"
            )

            st.plotly_chart(
                style_chart(memory_fig),
                use_container_width=True,
                key="prediction_memory"
            )

            st.info(
                "Predictions are experimental estimates, "
                "not guaranteed future measurements. "
                "The previous-value strategy returns the latest "
                "observed value as its prediction."
            )

    # =================================================
    # SYSTEM HEALTH
    # =================================================

    elif page == "System Health":

        st.subheader("System Health Center")

        display_smart_alerts(health_result)

        st.divider()

        st.subheader("CPU Health")

        if cpu >= 90:
            st.error(f"Critical CPU usage: {cpu:.2f}%")
        elif cpu >= 70:
            st.warning(f"Elevated CPU usage: {cpu:.2f}%")
        else:
            st.success(f"CPU usage is normal: {cpu:.2f}%")

        st.subheader("Memory Health")

        if memory >= 1536:
            st.error(f"Critical memory usage: {memory:.2f} MB")
        elif memory >= 1024:
            st.warning(f"Elevated memory usage: {memory:.2f} MB")
        else:
            st.success(f"Memory usage is normal: {memory:.2f} MB")

        st.divider()

        st.subheader("Alert Thresholds")

        threshold_df = pd.DataFrame({
            "Resource": ["CPU", "Memory"],
            "Warning": ["70%", "1024 MB"],
            "Critical": ["90%", "1536 MB"]
        })

        st.dataframe(
            threshold_df,
            use_container_width=True,
            hide_index=True
        )

    # =================================================
    # MODEL PERFORMANCE
    # =================================================

    elif page == "Model Performance":

        display_model_performance()

    # =================================================
    # FOOTER
    # =================================================

    st.divider()

    st.caption(
        "Last data timestamp: "
        + latest["timestamp"].strftime("%d %b %Y, %H:%M:%S")
    )

    st.caption(f"Total observations: {len(df)}")


# =====================================================
# RUN DASHBOARD
# =====================================================

monitoring_dashboard()

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748B; font-size:12px;">
        CloudPulse | Intelligent Cloud Resource Monitoring
        <br>
        Python • Prometheus • FastAPI • Streamlit
        • Random Forest • MLflow
    </div>
    """,
    unsafe_allow_html=True
)
