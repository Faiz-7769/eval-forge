import streamlit as st
import pandas as pd
import sqlite3
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Benchmark Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Design tokens ─────────────────────────────────────────────────────
# Dark terminal-inspired palette — deliberate choice for an eval/ML tool
# Signature element: monospace metric cards with colored left-border accents
COLORS = {
    "bg":        "#0D1117",
    "surface":   "#161B22",
    "border":    "#30363D",
    "text":      "#E6EDF3",
    "muted":     "#8B949E",
    "ms-phi":  "#58A6FF",   # blue  — local model
    "groq":      "#3FB950",   # green — fastest
    "mistral":   "#F0883E",   # orange — quality winner
    "accent":    "#BC8CFF",   # purple — highlights
}

MODEL_COLORS  = {
    "ms-phi":          COLORS["ms-phi"],
    "groq/qwen3-32b":        COLORS["groq"],
    "mistral-small-2506": COLORS["mistral"],
}
MODEL_LABELS = {
    "ms-phi":           "Microsoft-phi-14B (Local)",
    "groq/qwen3-32b":         "Qwen3-32b (Groq)",
    "mistral-small-2506": "Mistral-Small (API)",
}

# ── Global CSS ────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] {{
      background-color: {COLORS['bg']};
      color: {COLORS['text']};
      font-family: 'Inter', sans-serif;
  }}
  .stApp {{ background-color: {COLORS['bg']}; }}

  /* Hero */
  .hero {{
      padding: 3rem 0 2rem 0;
      border-bottom: 1px solid {COLORS['border']};
      margin-bottom: 2rem;
  }}
  .hero-tag {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      color: {COLORS['accent']};
      letter-spacing: 0.15em;
      text-transform: uppercase;
      margin-bottom: 0.75rem;
  }}
  .hero-title {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 2.6rem;
      font-weight: 700;
      color: {COLORS['text']};
      line-height: 1.1;
      margin-bottom: 0.75rem;
  }}
  .hero-sub {{
      font-size: 1rem;
      color: {COLORS['muted']};
      max-width: 620px;
      line-height: 1.6;
  }}

  /* Section headers */
  .section-label {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      color: {COLORS['accent']};
      letter-spacing: 0.18em;
      text-transform: uppercase;
      margin-bottom: 0.4rem;
  }}
  .section-title {{
      font-size: 1.35rem;
      font-weight: 600;
      color: {COLORS['text']};
      margin-bottom: 1.25rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid {COLORS['border']};
  }}

  /* Metric cards — signature element */
  .metric-card {{
      background: {COLORS['surface']};
      border: 1px solid {COLORS['border']};
      border-radius: 8px;
      padding: 1.1rem 1.25rem;
      margin-bottom: 0.75rem;
  }}
  .metric-card.ms-phi  {{ border-left: 3px solid {COLORS['ms-phi']}; }}
  .metric-card.groq     {{ border-left: 3px solid {COLORS['groq']}; }}
  .metric-card.mistral   {{ border-left: 3px solid {COLORS['mistral']}; }}

  .card-model {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      color: {COLORS['muted']};
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 0.3rem;
  }}
  .card-value {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.9rem;
      font-weight: 700;
      color: {COLORS['text']};
      line-height: 1;
  }}
  .card-label {{
      font-size: 0.8rem;
      color: {COLORS['muted']};
      margin-top: 0.25rem;
  }}
  .card-badge {{
      display: inline-block;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.65rem;
      padding: 0.15rem 0.5rem;
      border-radius: 20px;
      margin-top: 0.4rem;
      font-weight: 600;
  }}
  .badge-best  {{ background: rgba(63,185,80,0.15);  color: {COLORS['groq']}; }}
  .badge-local {{ background: rgba(88,166,255,0.15); color: {COLORS['ms-phi']}; }}
  .badge-qual  {{ background: rgba(240,136,62,0.15); color: {COLORS['mistral']}; }}
  .badge-tps   {{ background: rgba(188,140,255,0.15); color: {COLORS['accent']}; }}

  /* Divider */
  .divider {{
      border: none;
      border-top: 1px solid {COLORS['border']};
      margin: 2.5rem 0;
  }}

  /* GPU stat pills */
  .gpu-pill {{
      display: inline-block;
      background: {COLORS['surface']};
      border: 1px solid {COLORS['border']};
      border-radius: 6px;
      padding: 0.5rem 1rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      color: {COLORS['ms-phi']};
      margin-right: 0.5rem;
  }}
  .gpu-pill span {{ color: {COLORS['muted']}; font-size: 0.7rem; display: block; }}

  /* Q&A browser */
  .qa-card {{
      background: {COLORS['surface']};
      border: 1px solid {COLORS['border']};
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin-bottom: 0.75rem;
      font-size: 0.88rem;
  }}
  .qa-q {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      color: {COLORS['accent']};
      margin-bottom: 0.5rem;
  }}
  .qa-a {{ color: {COLORS['text']}; line-height: 1.6; }}
  .qa-score {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
      color: {COLORS['muted']};
      margin-top: 0.5rem;
  }}

  /* Download button */
  .stDownloadButton button {{
      background: transparent;
      border: 1px solid {COLORS['accent']};
      color: {COLORS['accent']};
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      border-radius: 6px;
      padding: 0.4rem 1rem;
  }}
  .stDownloadButton button:hover {{
      background: rgba(188,140,255,0.1);
  }}

  /* Plotly chart background override */
  .js-plotly-plot {{ border-radius: 8px; }}

  /* Hide Streamlit branding */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 1rem; padding-bottom: 3rem; }}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Try SQLite first, fall back to CSVs
    if os.path.exists("benchmark.db"):
        conn = sqlite3.connect("benchmark.db")
        df = pd.read_sql("SELECT * FROM benchmark_results", conn)
        conn.close()
    else:
        import glob
        csvs = glob.glob("evaluations/*_eval.csv")
        frames = []
        for c in csvs:
            frames.append(pd.read_csv(c))
        df = pd.concat(frames, ignore_index=True)

    return df

@st.cache_data
def load_benchmark():
    if os.path.exists("data/benchmark.json"):
        with open("data/benchmark.json") as f:
            return json.load(f)
    return []

df        = load_data()
benchmark = load_benchmark()

# Precompute per-model summary
summary = df.groupby('model').agg(
    avg_ar       =('answer_relevancy', 'mean'),
    avg_faith    =('faithfulness',     'mean'),
    avg_latency  =('latency_sec',      'mean'),
    avg_tps      =('tokens_per_sec',   'mean'),
    avg_gpu_pct  =('gpu_usage_pct',    'mean'),
    avg_gpu_temp =('gpu_temp_c',       'mean'),
    avg_gpu_mem  =('gpu_mem_used_mb',  'mean'),
).round(3)

MODEL_ORDER = ["ms-phi", "groq/qwen3-32b", "mistral-small-2506"]

# ── HERO ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">⚡ LLM Eval Forge · 30-Question Benchmark</div>
  <div class="hero-title">Which model wins<br>when it actually matters?</div>
  <div class="hero-sub">
    A multi-metric evaluation of Phi-4 (local), Qwen3-32B (Groq), and Mistral-Small (API) —
    scored on latency, GPU utilization, faithfulness, and answer relevancy
    by an independent Gemini judge.
  </div>
</div>
""", unsafe_allow_html=True)


# ── SECTION 1 — WINNER CARDS ──────────────────────────────────────────
st.markdown('<div class="section-label">At a glance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Scorecard</div>', unsafe_allow_html=True)

# Compute real winners dynamically — never hardcode badges again
local_model        = "ms-phi"  # the only locally-hosted model in this run
fastest_model      = summary['avg_latency'].idxmin()
best_quality_model = summary['avg_ar'].idxmax()

# Build badge text per model — a model can win more than one category,
# in which case badges stack (joined with " · ")
def get_badges(model):
    badges = []
    if model == local_model:
        badges.append(("badge-local", "🖥 Local / Quantized"))
    if model == fastest_model:
        badges.append(("badge-best", "⚡ Fastest"))
    if model == best_quality_model:
        badges.append(("badge-qual", "🏆 Best Quality"))
    return badges

c1, c2, c3 = st.columns(3)

for col, model, slug in [
    (c1, "ms-phi",              "ms-phi"),
    (c2, "groq/qwen3-32b",      "groq"),
    (c3, "mistral-small-2506",  "mistral"),
]:
    row = summary.loc[model] if model in summary.index else None
    if row is not None:
        badges_html = "".join(
            f'<span class="card-badge {cls}">{txt}</span> '
            for cls, txt in get_badges(model)
        )
        col.markdown(f"""
        <div class="metric-card {slug}">
          <div class="card-model">{MODEL_LABELS[model]}</div>
          <div class="card-value">{row['avg_ar']:.2f}</div>
          <div class="card-label">Answer Relevancy (avg)</div>
          <div class="card-value" style="font-size:1.3rem; margin-top:0.6rem">{row['avg_faith']:.2f}</div>
          <div class="card-label">Faithfulness · factual Qs</div>
          <div class="card-value" style="font-size:1.3rem; margin-top:0.6rem">
            {"%.0f" % row['avg_latency'] if row['avg_latency'] > 1 else "%.2f" % row['avg_latency']}s
          </div>
          <div class="card-label">Avg Latency per query</div>
          {badges_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── SECTION 2 — RADAR CHART ───────────────────────────────────────────
st.markdown('<div class="section-label">Multi-metric comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Radar — All Metrics at Once</div>', unsafe_allow_html=True)

# Normalise metrics 0-1 for radar
# latency: invert + normalise (lower=better)
max_lat = summary['avg_latency'].max()
min_lat = summary['avg_latency'].min()

def norm_latency(v):
    if max_lat == min_lat:
        return 1.0
    return 1 - (v - min_lat) / (max_lat - min_lat)

max_tps = summary['avg_tps'].max()

radar_categories = ['Answer\nRelevancy', 'Faithfulness', 'Speed\n(norm)', 'Tokens/sec\n(norm)', 'Constraint\nFollowing']

fig_radar = go.Figure()
for model in MODEL_ORDER:
    if model not in summary.index:
        continue
    row   = summary.loc[model]
    label = MODEL_LABELS[model]
    color = MODEL_COLORS[model]

    # constraint AR
    constraint_ar = df[(df['model'] == model) & (df['question_type'] == 'constraint')]['answer_relevancy'].mean()

    values = [
        row['avg_ar'],
        row['avg_faith'] if not np.isnan(row['avg_faith']) else 0,
        norm_latency(row['avg_latency']),
        row['avg_tps'] / max_tps,
        constraint_ar if not np.isnan(constraint_ar) else 0,
    ]
    values_closed = values + [values[0]]
    cats_closed   = radar_categories + [radar_categories[0]]

    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill='toself',
        name=label,
        line=dict(color=color, width=2),
        fillcolor='rgba({},{},{},{})'.format(
            int(color[1:3], 16),
            int(color[3:5], 16),
            int(color[5:7], 16),
            0.12
        ),
    ))

fig_radar.update_layout(
    polar=dict(
        bgcolor=COLORS['surface'],
        radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS['border'],
                        tickfont=dict(color=COLORS['muted'], size=10), tickformat='.1f'),
        angularaxis=dict(gridcolor=COLORS['border'],
                         tickfont=dict(color=COLORS['text'], size=11)),
    ),
    paper_bgcolor=COLORS['bg'],
    plot_bgcolor=COLORS['bg'],
    legend=dict(font=dict(color=COLORS['text'], size=12),
                bgcolor=COLORS['surface'], bordercolor=COLORS['border']),
    margin=dict(l=60, r=60, t=30, b=30),
    height=420,
)
st.plotly_chart(fig_radar, width='stretch')

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── SECTION 3 — LATENCY & SPEED ───────────────────────────────────────
st.markdown('<div class="section-label">Performance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Latency & Throughput</div>', unsafe_allow_html=True)

col_lat, col_tps = st.columns(2)

with col_lat:
    lat_df = summary.reset_index()
    lat_df['label'] = lat_df['model'].map(MODEL_LABELS)
    lat_df['color'] = lat_df['model'].map(MODEL_COLORS)
    fig_lat = go.Figure(go.Bar(
        x=lat_df['label'],
        y=lat_df['avg_latency'],
        marker_color=lat_df['color'].tolist(),
        text=[f"{v:.2f}s" for v in lat_df['avg_latency']],
        textposition='outside',
        textfont=dict(color=COLORS['text'], family='JetBrains Mono', size=11),
    ))
    fig_lat.update_layout(
        title=dict(text="Avg Latency per Query (seconds)", font=dict(color=COLORS['muted'], size=12)),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['surface'],
        xaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text'])),
        yaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['muted'])),
        margin=dict(l=20, r=20, t=40, b=20), height=320,
    )
    st.plotly_chart(fig_lat, width='stretch')

with col_tps:
    fig_tps = go.Figure(go.Bar(
        x=lat_df['label'],
        y=lat_df['avg_tps'],
        marker_color=lat_df['color'].tolist(),
        text=[f"{v:.0f}" for v in lat_df['avg_tps']],
        textposition='outside',
        textfont=dict(color=COLORS['text'], family='JetBrains Mono', size=11),
    ))
    fig_tps.update_layout(
        title=dict(text="Avg Tokens per Second", font=dict(color=COLORS['muted'], size=12)),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['surface'],
        xaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text'])),
        yaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['muted'])),
        margin=dict(l=20, r=20, t=40, b=20), height=320,
    )
    st.plotly_chart(fig_tps, width='stretch')

# Per-question latency scatter
fig_scatter = go.Figure()
for model in MODEL_ORDER:
    mdf   = df[df['model'] == model].sort_values('question_id')
    label = MODEL_LABELS[model]
    color = MODEL_COLORS[model]
    fig_scatter.add_trace(go.Scatter(
        x=mdf['question_id'],
        y=mdf['latency_sec'],
        mode='lines+markers',
        name=label,
        line=dict(color=color, width=2),
        marker=dict(size=5, color=color),
    ))
fig_scatter.update_layout(
    title=dict(text="Latency per Question (all 30)", font=dict(color=COLORS['muted'], size=12)),
    paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['surface'],
    xaxis=dict(title="Question ID", gridcolor=COLORS['border'],
               tickfont=dict(color=COLORS['muted'])),
    yaxis=dict(title="Latency (s)", gridcolor=COLORS['border'],
               tickfont=dict(color=COLORS['muted'])),
    legend=dict(font=dict(color=COLORS['text']), bgcolor=COLORS['surface'],
                bordercolor=COLORS['border']),
    margin=dict(l=20, r=20, t=40, b=20), height=300,
)
st.plotly_chart(fig_scatter, width='stretch')

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── SECTION 4 — GPU (Phi-4 local model only) ──────────────────────────
st.markdown('<div class="section-label">Hardware · Phi-4 local only</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">GPU Utilisation (nvidia-ml-py)</div>', unsafe_allow_html=True)

ds = df[df['model'] == 'ms-phi'].sort_values('question_id')
gpu_row = summary.loc['ms-phi']

# Pill stats
st.markdown(f"""
<div style="margin-bottom:1.25rem">
  <span class="gpu-pill">{gpu_row['avg_gpu_pct']:.0f}%<span>avg GPU usage</span></span>
  <span class="gpu-pill">{gpu_row['avg_gpu_temp']:.0f}°C<span>avg temperature</span></span>
  <span class="gpu-pill">{gpu_row['avg_gpu_mem']:.0f} MB<span>avg VRAM used</span></span>
  <span class="gpu-pill">{ds['gpu_temp_c'].max():.0f}°C<span>peak temp</span></span>
  <span class="gpu-pill">{ds['gpu_mem_used_mb'].max():.0f} MB<span>peak VRAM</span></span>
</div>
""", unsafe_allow_html=True)

gcol1, gcol2, gcol3 = st.columns(3)

for gcol, col_name, label, color in [
    (gcol1, 'gpu_usage_pct',   'GPU Usage (%)',     COLORS['ms-phi']),
    (gcol2, 'gpu_temp_c',      'GPU Temp (°C)',     '#FF7B72'),
    (gcol3, 'gpu_mem_used_mb', 'VRAM Used (MB)',    COLORS['accent']),
]:
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig = go.Figure(go.Scatter(
        x=ds['question_id'], y=ds[col_name],
        fill='tozeroy',
        line=dict(color=color, width=2),
        fillcolor=f'rgba({r},{g},{b},0.13)',
    ))
    fig.update_layout(
        title=dict(text=label, font=dict(color=COLORS['muted'], size=12)),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['surface'],
        xaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['muted']),
                   title="Question ID"),
        yaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['muted'])),
        margin=dict(l=20, r=10, t=40, b=20), height=240,
        showlegend=False,
    )
    gcol.plotly_chart(fig, width='stretch')

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── SECTION 5 — QUALITY METRICS ───────────────────────────────────────
st.markdown('<div class="section-label">Output quality · Gemini judge</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Answer Relevancy & Faithfulness</div>', unsafe_allow_html=True)

qcol1, qcol2 = st.columns(2)

with qcol1:
    # AR by model grouped bar — factual vs constraint
    ar_data = df.groupby(['model', 'question_type'])['answer_relevancy'].mean().reset_index()
    ar_data['label'] = ar_data['model'].map(MODEL_LABELS)
    fig_ar = go.Figure()
    for qt, pattern in [('factual', ''), ('constraint', '/')]:
        sub = ar_data[ar_data['question_type'] == qt]
        fig_ar.add_trace(go.Bar(
            name=qt.capitalize(),
            x=sub['label'],
            y=sub['answer_relevancy'],
            marker_color=[MODEL_COLORS[m] for m in sub['model']],
            marker_pattern_shape=pattern,
            text=[f"{v:.2f}" for v in sub['answer_relevancy']],
            textposition='outside',
            textfont=dict(color=COLORS['text'], size=10),
        ))
    fig_ar.update_layout(
        title=dict(text="Answer Relevancy by Type", font=dict(color=COLORS['muted'], size=12)),
        barmode='group',
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['surface'],
        xaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text'])),
        yaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['muted']),
                   range=[0, 1.15]),
        legend=dict(font=dict(color=COLORS['text']), bgcolor=COLORS['surface']),
        margin=dict(l=20, r=20, t=40, b=20), height=340,
    )
    st.plotly_chart(fig_ar, width='stretch')

with qcol2:
    # Faithfulness — factual only
    faith_df = df[df['question_type'] == 'factual'].groupby('model')['faithfulness'].mean().reset_index()
    faith_df['label'] = faith_df['model'].map(MODEL_LABELS)
    faith_df['color'] = faith_df['model'].map(MODEL_COLORS)
    fig_faith = go.Figure(go.Bar(
        x=faith_df['label'],
        y=faith_df['faithfulness'],
        marker_color=faith_df['color'].tolist(),
        text=[f"{v:.2f}" for v in faith_df['faithfulness']],
        textposition='outside',
        textfont=dict(color=COLORS['text'], family='JetBrains Mono', size=11),
    ))
    fig_faith.update_layout(
        title=dict(text="Faithfulness · Factual Questions Only", font=dict(color=COLORS['muted'], size=12)),
        paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['surface'],
        xaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text'])),
        yaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['muted']),
                   range=[0, 1.15]),
        margin=dict(l=20, r=20, t=40, b=20), height=340,
    )
    st.plotly_chart(fig_faith, width='stretch')

# Score heatmap — per question
st.markdown("**Score Heatmap — Answer Relevancy across all 30 questions**",
            unsafe_allow_html=False)
heat_df = df.pivot_table(index='model', columns='question_id',
                         values='answer_relevancy').fillna(0)
heat_df.index = [MODEL_LABELS.get(m, m) for m in heat_df.index]

fig_heat = go.Figure(go.Heatmap(
    z=heat_df.values,
    x=[f"Q{int(c)+1}" for c in heat_df.columns],
    y=list(heat_df.index),
    colorscale=[[0, '#1a1a2e'], [0.5, COLORS['ms-phi']], [1, COLORS['mistral']]],
    text=[[f"{v:.2f}" for v in row] for row in heat_df.values],
    texttemplate="%{text}",
    textfont=dict(size=9, color=COLORS['text']),
    showscale=True,
    zmin=0, zmax=1,
))
fig_heat.update_layout(
    paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['surface'],
    xaxis=dict(tickfont=dict(color=COLORS['muted'], size=9)),
    yaxis=dict(tickfont=dict(color=COLORS['text'], size=11)),
    margin=dict(l=20, r=20, t=20, b=20), height=220,
)
st.plotly_chart(fig_heat, width='stretch')

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── SECTION 6 — CATEGORY BREAKDOWN ───────────────────────────────────
st.markdown('<div class="section-label">By domain</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Performance by Question Category</div>', unsafe_allow_html=True)

# Merge benchmark category into df
bm_df = pd.DataFrame(benchmark)[['id', 'category', 'difficulty']]
bm_df.columns = ['question_id_1based', 'category', 'difficulty']
df_cat = df.copy()
df_cat['question_id_1based'] = df_cat['question_id'] + 1
df_cat = df_cat.merge(bm_df, on='question_id_1based', how='left')

cat_summary = df_cat.groupby(['model', 'category'])[['answer_relevancy', 'faithfulness']].mean().reset_index()
cat_summary['label'] = cat_summary['model'].map(MODEL_LABELS)

fig_cat = px.bar(
    cat_summary,
    x='category', y='answer_relevancy',
    color='label',
    barmode='group',
    color_discrete_map={v: MODEL_COLORS[k] for k, v in MODEL_LABELS.items()},
    text=cat_summary['answer_relevancy'].round(2),
)
fig_cat.update_traces(textposition='outside', textfont=dict(color=COLORS['text'], size=10))
fig_cat.update_layout(
    paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['surface'],
    xaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['text'])),
    yaxis=dict(gridcolor=COLORS['border'], tickfont=dict(color=COLORS['muted']),
               range=[0, 1.2], title='Answer Relevancy'),
    legend=dict(title='', font=dict(color=COLORS['text']), bgcolor=COLORS['surface']),
    margin=dict(l=20, r=20, t=20, b=20), height=350,
)
st.plotly_chart(fig_cat, width='stretch')

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── SECTION 7 — Q&A BROWSER ──────────────────────────────────────────
st.markdown('<div class="section-label">Drill down</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Question Browser</div>', unsafe_allow_html=True)

# Filters
fcol1, fcol2, fcol3 = st.columns(3)
categories  = ['All'] + sorted(df_cat['category'].dropna().unique().tolist())
difficulties = ['All'] + ['Easy', 'Medium', 'Hard']
qtypes      = ['All', 'factual', 'constraint']

sel_cat   = fcol1.selectbox("Category",   categories,   key="cat")
sel_diff  = fcol2.selectbox("Difficulty", difficulties, key="diff")
sel_qtype = fcol3.selectbox("Type",       qtypes,       key="qtype")

bm_filtered = pd.DataFrame(benchmark)
if sel_cat  != 'All': bm_filtered = bm_filtered[bm_filtered['category']   == sel_cat]
if sel_diff != 'All': bm_filtered = bm_filtered[bm_filtered['difficulty']  == sel_diff]

q_ids_available = bm_filtered['id'].tolist()  # 1-based

if not q_ids_available:
    st.info("No questions match the selected filters.")
else:
    sel_q = st.selectbox(
        "Select question",
        options=q_ids_available,
        format_func=lambda qid: f"Q{qid} — {next((q['prompt'][:100]+'...' for q in benchmark if q['id']==qid), '')}"
    )

    bm_row   = next((q for q in benchmark if q['id'] == sel_q), None)
    q_df     = df_cat[df_cat['question_id_1based'] == sel_q]

    if bm_row:
        st.markdown(f"""
        <div class="qa-card">
          <div class="qa-q">QUESTION · {bm_row['category']} · {bm_row['difficulty']}</div>
          <div class="qa-a">{bm_row['prompt']}</div>
        </div>
        <div class="qa-card" style="border-left: 2px solid {COLORS['accent']}">
          <div class="qa-q">EXPECTED ANSWER</div>
          <div class="qa-a">{bm_row['expected_answer']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Show all 3 model answers side by side
    m_cols = st.columns(len(MODEL_ORDER))
    for mcol, model in zip(m_cols, MODEL_ORDER):
        mrow = q_df[q_df['model'] == model]
        if mrow.empty:
            muted_color = COLORS['muted']
            mcol.markdown(f"<div class='qa-card'><div class='qa-q'>{MODEL_LABELS[model]}</div><div class='qa-a' style='color:{muted_color}'>No data</div></div>", unsafe_allow_html=True)
            continue
        mrow  = mrow.iloc[0]
        color = MODEL_COLORS[model]
        ar    = mrow['answer_relevancy']
        faith = mrow['faithfulness']
        lat   = mrow['latency_sec']
        mcol.markdown(f"""
        <div class="qa-card" style="border-left: 2px solid {color}; height: 100%">
          <div class="qa-q" style="color:{color}">{MODEL_LABELS[model]}</div>
          <div class="qa-a">{str(mrow['answer'])[:2000]}{'...' if len(str(mrow['answer']))>2000 else ''}</div>
          <div class="qa-score">
            AR: {ar:.2f} · Faith: {f"{faith:.2f}" if not (isinstance(faith, float) and np.isnan(faith)) else 'N/A'} · {lat:.2f}s
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── SECTION 8 — BENCHMARK DOWNLOAD ───────────────────────────────────
st.markdown('<div class="section-label">Open benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Download the Benchmark Dataset</div>', unsafe_allow_html=True)

dl_col, info_col = st.columns([1, 2])

with dl_col:
    if benchmark:
        st.download_button(
            label="⬇ benchmark.json",
            data=json.dumps(benchmark, indent=2),
            file_name="benchmark.json",
            mime="application/json",
        )
    st.markdown(f"""
    <div style="margin-top:1rem; font-size:0.85rem; color:{COLORS['muted']}; font-family:'JetBrains Mono', monospace; line-height:2">
      30 questions<br>
      3 categories<br>
      3 difficulty levels<br>
      MIT licensed · free to use
    </div>
    """, unsafe_allow_html=True)

with info_col:
    cat_counts = pd.DataFrame(benchmark)['category'].value_counts().reset_index()
    cat_counts.columns = ['category', 'count']
    fig_pie = go.Figure(go.Pie(
        labels=cat_counts['category'],
        values=cat_counts['count'],
        hole=0.55,
        marker=dict(colors=[COLORS['ms-phi'], COLORS['groq'], COLORS['mistral']]),
        textinfo='label+value',
        textfont=dict(color=COLORS['text'], size=11),
    ))
    fig_pie.update_layout(
        paper_bgcolor=COLORS['bg'],
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=15),
        height=220,
    )
    st.plotly_chart(fig_pie, width='stretch')

# ── FOOTER ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding: 2rem 0 1rem; font-family:'JetBrains Mono',monospace;
            font-size:0.72rem; color:{COLORS['muted']}; border-top: 1px solid {COLORS['border']}; margin-top:1rem">
  LLM Eval Forge · Phi-4 vs Qwen3-32B (Groq) vs Mistral-Small · Judged by Gemini 3.1 Flash Lite ·
  Built with DeepEval · nvidia-ml-py · Streamlit
</div>
""", unsafe_allow_html=True)