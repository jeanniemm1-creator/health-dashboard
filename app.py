import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

# -- Page Config --
st.set_page_config(
    page_title="Jeannie's Health Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Password Gate --
def check_password():
    """Simple password protection."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    st.markdown("## 🔒 Jeannie's Health Dashboard")
    st.markdown("Enter your password to continue.")
    password = st.text_input("Password", type="password", key="pw_input")
    if st.button("Sign In"):
        if password == st.secrets.get("password", "jeannie2026"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not check_password():
    st.stop()

# -- Custom CSS --
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; font-size: 14px; opacity: 0.85; }
    .metric-card h1 { margin: 5px 0; font-size: 32px; }
    .metric-card p { margin: 0; font-size: 13px; opacity: 0.75; }
    .green-card { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .orange-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .blue-card { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .purple-card { background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%); }
    .section-header {
        font-size: 22px; font-weight: 600; margin-top: 30px;
        margin-bottom: 15px; padding-bottom: 8px;
        border-bottom: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)
# -- Targets --
TARGETS = {
    "Calories": 1800, "Protein (g)": 125, "Carbs (g)": 160,
    "Fat (g)": 60, "Fiber (g)": 28, "Water (cups)": 8, "Sleep (hrs)": 7.5,
}

# -- Data Loading --
@st.cache_data
def load_data(file):
    df = pd.read_excel(file, sheet_name="Daily Log")
    df = df[df["Date"] != "Target"].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    num_cols = ["Sleep (hrs)", "Calories", "Protein (g)", "Carbs (g)",
                "Fat (g)", "Fiber (g)", "Caffeine (mg)", "Water (cups)"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    yn_cols = ["Omega 3", "Vit D", "Mag Glyc x2", "Vit B12",
               "Neck/Shoulder Stretch", "Deep Breathing"]
    for c in yn_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: True if str(x).strip().upper() in ["Y", "YES", "TRUE"] else False)
    if "Headache" in df.columns:
        df["Headache_Detail"] = df["Headache"].fillna("None")
        df["Headache"] = df["Headache"].apply(
            lambda x: False if str(x).strip().upper() in ["N", "NONE", "", "NAN", "NO"] else True
        )
    def categorize_exercise(ex):
        if pd.isna(ex): return "No Data"
        ex_lower = str(ex).lower()
        if "rest" in ex_lower: return "Rest"
        elif any(w in ex_lower for w in ["treadmill", "run", "walk", "bike", "cardio"]):
            if any(w in ex_lower for w in ["weight", "squat", "press", "thrust", "pulldown", "row", "curl", "dip", "fly", "crunch", "kick"]):
                return "Cardio + Weights"
            return "Cardio"
        elif any(w in ex_lower for w in ["weight", "squat", "press", "thrust", "upper", "lower", "pulldown", "row", "curl", "dip", "fly", "crunch", "kick", "split"]):
            return "Weights"
        else: return "Other"
    df["Exercise Type"] = df["Exercise"].apply(categorize_exercise)
    return df

# -- Sidebar --
st.sidebar.markdown("# 💪 Health Dashboard")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Health Tracker (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.markdown("## Welcome, Jeannie!")
    st.markdown("Upload your **health_tracker.xlsx** file using the sidebar to see your dashboard.")
    st.code("OneDrive > Documents > Health > Fitness > health_tracker.xlsx")
    st.stop()
df = load_data(uploaded_file)
st.sidebar.markdown("### Date Range")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input("Filter dates", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if len(date_range) == 2:
    df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]
st.sidebar.markdown(f"**{len(df)} days** of data loaded")

# -- OVERVIEW --
st.markdown("# 📊 Overview")
col1, col2, col3, col4, col5 = st.columns(5)
avg_cal = df["Calories"].mean()
avg_protein = df["Protein (g)"].mean()
avg_sleep = df["Sleep (hrs)"].mean()
avg_water = df["Water (cups)"].dropna().mean()
cal_delta = avg_cal - TARGETS["Calories"]
pro_delta = avg_protein - TARGETS["Protein (g)"]
with col1:
    st.markdown(f'<div class="metric-card"><h3>Avg Calories</h3><h1>{avg_cal:,.0f}</h1><p>Target: {TARGETS["Calories"]} | {"▲" if cal_delta > 0 else "▼"} {abs(cal_delta):.0f}</p></div>', unsafe_allow_html=True)
with col2:
    cc = "metric-card green-card" if avg_protein >= TARGETS["Protein (g)"] else "metric-card orange-card"
    st.markdown(f'<div class="{cc}"><h3>Avg Protein</h3><h1>{avg_protein:,.0f}g</h1><p>Target: {TARGETS["Protein (g)"]}g | {"▲" if pro_delta > 0 else "▼"} {abs(pro_delta):.0f}g</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card blue-card"><h3>Avg Sleep</h3><h1>{avg_sleep:.1f} hrs</h1><p>Target: {TARGETS["Sleep (hrs)"]} hrs</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card purple-card"><h3>Avg Water</h3><h1>{avg_water:.1f} cups</h1><p>Target: {TARGETS["Water (cups)"]}+ cups</p></div>', unsafe_allow_html=True)
with col5:
    headache_days = df["Headache"].sum() if "Headache" in df.columns else 0
    pct = (headache_days / len(df) * 100) if len(df) > 0 else 0
    st.markdown(f'<div class="metric-card orange-card"><h3>Headache Days</h3><h1>{headache_days}/{len(df)}</h1><p>{pct:.0f}% of tracked days</p></div>', unsafe_allow_html=True)
# -- CALORIE & PROTEIN TRENDS --
st.markdown('<div class="section-header">🔥 Calorie & Protein Trends</div>', unsafe_allow_html=True)
fig_cal = make_subplots(specs=[[{"secondary_y": True}]])
fig_cal.add_trace(go.Bar(x=df["Date"], y=df["Calories"], name="Calories",
    marker_color=["#f5576c" if c > TARGETS["Calories"] else "#4facfe" for c in df["Calories"]], opacity=0.7), secondary_y=False)
fig_cal.add_trace(go.Scatter(x=df["Date"], y=df["Protein (g)"], name="Protein (g)",
    mode="lines+markers", line=dict(color="#38ef7d", width=3), marker=dict(size=8)), secondary_y=True)
fig_cal.add_hline(y=TARGETS["Calories"], line_dash="dash", line_color="rgba(245,87,108,0.5)",
    annotation_text=f"Cal Target ({TARGETS['Calories']})", secondary_y=False)
fig_cal.add_hline(y=TARGETS["Protein (g)"], line_dash="dash", line_color="rgba(56,239,125,0.5)",
    annotation_text=f"Protein Target ({TARGETS['Protein (g)']}g)", secondary_y=True)
fig_cal.update_layout(height=400, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=40, b=40))
fig_cal.update_yaxes(title_text="Calories", secondary_y=False)
fig_cal.update_yaxes(title_text="Protein (g)", secondary_y=True)
st.plotly_chart(fig_cal, use_container_width=True)
cal_on_target = ((df["Calories"] >= TARGETS["Calories"] - 100) & (df["Calories"] <= TARGETS["Calories"] + 100)).sum()
pro_on_target = (df["Protein (g)"] >= TARGETS["Protein (g)"]).sum()
col_a, col_b = st.columns(2)
col_a.metric("Days within 100 cal of target", f"{cal_on_target}/{len(df)}")
col_b.metric("Days hitting protein goal", f"{pro_on_target}/{len(df)}")
# -- MACRO BREAKDOWN --
st.markdown('<div class="section-header">🥗 Macro Breakdown</div>', unsafe_allow_html=True)
col_pie, col_stack = st.columns([1, 2])
with col_pie:
    avg_macros = {"Protein": df["Protein (g)"].mean()*4, "Carbs": df["Carbs (g)"].mean()*4, "Fat": df["Fat (g)"].mean()*9}
    fig_pie = px.pie(names=list(avg_macros.keys()), values=list(avg_macros.values()),
        color_discrete_sequence=["#38ef7d", "#4facfe", "#f5576c"], hole=0.45)
    fig_pie.update_layout(height=350, margin=dict(t=30, b=30), title=dict(text="Avg Calorie Sources", font=dict(size=14)))
    fig_pie.update_traces(textinfo="label+percent", textposition="outside")
    st.plotly_chart(fig_pie, use_container_width=True)
with col_stack:
    fig_stack = go.Figure()
    fig_stack.add_trace(go.Bar(x=df["Date"], y=df["Protein (g)"]*4, name="Protein", marker_color="#38ef7d"))
    fig_stack.add_trace(go.Bar(x=df["Date"], y=df["Carbs (g)"]*4, name="Carbs", marker_color="#4facfe"))
    fig_stack.add_trace(go.Bar(x=df["Date"], y=df["Fat (g)"]*9, name="Fat", marker_color="#f5576c"))
    fig_stack.update_layout(barmode="stack", height=350, template="plotly_white",
        title=dict(text="Daily Macro Calories", font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=50, b=40))
    fig_stack.add_hline(y=TARGETS["Calories"], line_dash="dash", line_color="gray", annotation_text="Calorie Target")
    st.plotly_chart(fig_stack, use_container_width=True)
# -- INDIVIDUAL MACROS VS TARGETS --
st.markdown('<div class="section-header">🎯 Daily Macros vs Targets</div>', unsafe_allow_html=True)
macro_cols = ["Protein (g)", "Carbs (g)", "Fat (g)", "Fiber (g)"]
macro_colors = ["#38ef7d", "#4facfe", "#f5576c", "#a18cd1"]
cols = st.columns(2)
for i, (macro, color) in enumerate(zip(macro_cols, macro_colors)):
    with cols[i % 2]:
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=df["Date"], y=df[macro], mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=6), name=macro,
            fill="tozeroy", fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.1])}"))
        target_val = TARGETS.get(macro)
        if target_val:
            fig_m.add_hline(y=target_val, line_dash="dash", line_color="gray", annotation_text=f"Target: {target_val}")
        fig_m.update_layout(height=250, template="plotly_white", title=dict(text=macro, font=dict(size=14)),
            margin=dict(t=40, b=30, l=40, r=20), showlegend=False)
        st.plotly_chart(fig_m, use_container_width=True)

# -- SLEEP --
st.markdown('<div class="section-header">😴 Sleep</div>', unsafe_allow_html=True)
fig_sleep = go.Figure()
fig_sleep.add_trace(go.Scatter(x=df["Date"], y=df["Sleep (hrs)"], mode="lines+markers",
    line=dict(color="#764ba2", width=3), marker=dict(size=8), fill="tozeroy", fillcolor="rgba(118,75,162,0.1)"))
fig_sleep.add_hline(y=7, line_dash="dash", line_color="rgba(0,0,0,0.3)", annotation_text="Min Target (7)")
fig_sleep.add_hline(y=8, line_dash="dash", line_color="rgba(0,0,0,0.3)", annotation_text="Max Target (8)")
fig_sleep.update_layout(height=300, template="plotly_white", yaxis_title="Hours", margin=dict(t=20, b=40))
st.plotly_chart(fig_sleep, use_container_width=True)
# -- SUPPLEMENT COMPLIANCE --
st.markdown('<div class="section-header">💊 Supplement & Wellness Compliance</div>', unsafe_allow_html=True)
supp_cols = ["Omega 3", "Vit D", "Mag Glyc x2", "Vit B12", "Neck/Shoulder Stretch", "Deep Breathing"]
available_supps = [c for c in supp_cols if c in df.columns]
if available_supps:
    heatmap_data = []
    for col in available_supps:
        row = df[col].apply(lambda x: 1 if x else 0).tolist()
        heatmap_data.append(row)
    fig_heat = go.Figure(data=go.Heatmap(z=heatmap_data, x=df["Date"].dt.strftime("%b %d"),
        y=available_supps, colorscale=[[0, "#f0f0f0"], [1, "#38ef7d"]], showscale=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:d}<extra></extra>"))
    fig_heat.update_layout(height=300, template="plotly_white", margin=dict(t=20, b=40, l=150), xaxis=dict(tickangle=-45))
    st.plotly_chart(fig_heat, use_container_width=True)
    comp_cols = st.columns(len(available_supps))
    for i, col_name in enumerate(available_supps):
        pct = (df[col_name].sum() / len(df) * 100)
        comp_cols[i].metric(col_name, f"{pct:.0f}%")
# -- EXERCISE --
st.markdown('<div class="section-header">🏋️ Exercise</div>', unsafe_allow_html=True)
col_ex1, col_ex2 = st.columns([1, 2])
with col_ex1:
    ex_counts = df["Exercise Type"].value_counts()
    fig_ex = px.pie(names=ex_counts.index, values=ex_counts.values,
        color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
    fig_ex.update_layout(height=300, margin=dict(t=30, b=30), title=dict(text="Workout Distribution", font=dict(size=14)))
    st.plotly_chart(fig_ex, use_container_width=True)
with col_ex2:
    type_colors = {"Weights": "#f5576c", "Cardio": "#4facfe", "Cardio + Weights": "#38ef7d",
                   "Rest": "#e0e0e0", "Other": "#a18cd1", "No Data": "#f0f0f0"}
    fig_timeline = go.Figure()
    for ex_type in df["Exercise Type"].unique():
        mask = df["Exercise Type"] == ex_type
        fig_timeline.add_trace(go.Bar(x=df.loc[mask, "Date"], y=[1]*mask.sum(),
            name=ex_type, marker_color=type_colors.get(ex_type, "#999"),
            hovertext=df.loc[mask, "Exercise"].fillna(""), hovertemplate="<b>%{x}</b><br>%{hovertext}<extra></extra>"))
    fig_timeline.update_layout(barmode="stack", height=300, template="plotly_white",
        title=dict(text="Daily Activity", font=dict(size=14)), showlegend=True, yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=50, b=40))
    st.plotly_chart(fig_timeline, use_container_width=True)
# -- HEADACHE ANALYSIS --
st.markdown('<div class="section-header">🧠 Headache Analysis</div>', unsafe_allow_html=True)
if "Headache" in df.columns:
    ha_df = df.copy()
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        ha_yes = ha_df[ha_df["Headache"]==True]["Sleep (hrs)"].mean()
        ha_no = ha_df[ha_df["Headache"]==False]["Sleep (hrs)"].mean()
        fig_ha_sleep = go.Figure(data=[go.Bar(x=["Headache Days", "No Headache Days"],
            y=[ha_yes, ha_no], marker_color=["#f5576c", "#38ef7d"],
            text=[f"{ha_yes:.1f} hrs", f"{ha_no:.1f} hrs"], textposition="outside")])
        fig_ha_sleep.update_layout(height=300, template="plotly_white",
            title=dict(text="Avg Sleep: Headache vs No Headache", font=dict(size=14)),
            margin=dict(t=50, b=40), yaxis_title="Hours of Sleep")
        st.plotly_chart(fig_ha_sleep, use_container_width=True)
    with col_h2:
        ha_yes_caf = ha_df[ha_df["Headache"]==True]["Caffeine (mg)"].mean()
        ha_no_caf = ha_df[ha_df["Headache"]==False]["Caffeine (mg)"].mean()
        fig_ha_caf = go.Figure(data=[go.Bar(x=["Headache Days", "No Headache Days"],
            y=[ha_yes_caf, ha_no_caf], marker_color=["#f5576c", "#4facfe"],
            text=[f"{ha_yes_caf:.0f} mg", f"{ha_no_caf:.0f} mg"], textposition="outside")])
        fig_ha_caf.update_layout(height=300, template="plotly_white",
            title=dict(text="Avg Caffeine: Headache vs No Headache", font=dict(size=14)),
            margin=dict(t=50, b=40), yaxis_title="Caffeine (mg)")
        st.plotly_chart(fig_ha_caf, use_container_width=True)
    fig_ha_time = go.Figure()
    fig_ha_time.add_trace(go.Scatter(x=df["Date"], y=df["Headache"].astype(int), mode="markers+lines",
        marker=dict(size=12, color=["#f5576c" if h else "#e0e0e0" for h in df["Headache"]]),
        line=dict(color="#e0e0e0", width=1)))
    fig_ha_time.update_layout(height=200, template="plotly_white",
        title=dict(text="Headache Timeline", font=dict(size=14)),
        yaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]), margin=dict(t=40, b=30))
    st.plotly_chart(fig_ha_time, use_container_width=True)

# -- WATER & CAFFEINE --
st.markdown('<div class="section-header">💧 Hydration & Caffeine</div>', unsafe_allow_html=True)
col_w, col_c = st.columns(2)
with col_w:
    fig_water = go.Figure()
    fig_water.add_trace(go.Bar(x=df["Date"], y=df["Water (cups)"],
        marker_color=["#4facfe" if w >= 8 else "#f5576c" for w in df["Water (cups)"].fillna(0)], opacity=0.8))
    fig_water.add_hline(y=8, line_dash="dash", line_color="gray", annotation_text="Target: 8 cups")
    fig_water.update_layout(height=300, template="plotly_white",
        title=dict(text="Daily Water Intake", font=dict(size=14)), yaxis_title="Cups", margin=dict(t=50, b=40))
    st.plotly_chart(fig_water, use_container_width=True)
with col_c:
    fig_caf = go.Figure()
    fig_caf.add_trace(go.Scatter(x=df["Date"], y=df["Caffeine (mg)"], mode="lines+markers",
        line=dict(color="#f093fb", width=2), marker=dict(size=7), fill="tozeroy", fillcolor="rgba(240,147,251,0.1)"))
    fig_caf.update_layout(height=300, template="plotly_white",
        title=dict(text="Daily Caffeine", font=dict(size=14)), yaxis_title="mg", margin=dict(t=50, b=40))
    st.plotly_chart(fig_caf, use_container_width=True)
# -- TIRZ SHOT LOG --
if "Tirz Shot" in df.columns:
    tirz_days = df[df["Tirz Shot"].apply(lambda x: str(x).strip() not in ["N", "n", "", "nan", "None", "0"])]
    if len(tirz_days) > 0:
        st.markdown('<div class="section-header">💉 Tirz Shot Log</div>', unsafe_allow_html=True)
        for _, row in tirz_days.iterrows():
            st.markdown(f"**{row['Date'].strftime('%b %d, %Y')}** — {row['Tirz Shot']}")

# -- WEEKLY SUMMARY --
st.markdown('<div class="section-header">📋 Weekly Summary</div>', unsafe_allow_html=True)
df_weekly = df.copy()
df_weekly["Week"] = df_weekly["Date"].dt.isocalendar().week
weekly_agg = df_weekly.groupby("Week").agg({
    "Calories": "mean", "Protein (g)": "mean", "Carbs (g)": "mean",
    "Fat (g)": "mean", "Fiber (g)": "mean", "Sleep (hrs)": "mean",
    "Water (cups)": "mean", "Date": ["min", "max"]}).reset_index()
weekly_agg.columns = ["Week", "Avg Calories", "Avg Protein", "Avg Carbs", "Avg Fat",
    "Avg Fiber", "Avg Sleep", "Avg Water", "Start", "End"]
weekly_agg["Period"] = weekly_agg.apply(lambda r: f"{r['Start'].strftime('%b %d')} – {r['End'].strftime('%b %d')}", axis=1)
display_df = weekly_agg[["Period", "Avg Calories", "Avg Protein", "Avg Carbs",
    "Avg Fat", "Avg Fiber", "Avg Sleep", "Avg Water"]].copy()
display_df = display_df.round(1)
st.dataframe(display_df, use_container_width=True, hide_index=True)

# -- SYMPTOMS LOG --
st.markdown('<div class="section-header">📝 Symptoms & Notes Log</div>', unsafe_allow_html=True)
symptoms_df = df[df["Symptoms"].notna() & (df["Symptoms"] != "None")][["Date", "Symptoms", "Notes"]].copy()
if len(symptoms_df) > 0:
    symptoms_df["Date"] = symptoms_df["Date"].dt.strftime("%b %d, %Y")
    st.dataframe(symptoms_df, use_container_width=True, hide_index=True)
else:
    st.info("No symptoms logged in this period.")

# -- Footer --
st.markdown("---")
st.markdown("*Dashboard updates each time you upload a new tracker file.*")