# app.py
import os
import math
import time
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.exceptions import NotFittedError

# Optional: GPT insights
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------------------
# Helper utilities
# ---------------------------
def load_models():
    """Load models from repo root. Return (model, vectorizer, encoder) or raise helpful error."""
    missing = []
    for fname in ("logistic_model.joblib", "vectorizer.joblib", "label_encoder.joblib"):
        if not os.path.exists(fname):
            missing.append(fname)
    if missing:
        raise FileNotFoundError(
            f"Missing model files in repo root: {', '.join(missing)}. "
            "Place logistic_model.joblib, vectorizer.joblib, label_encoder.joblib in the repo root."
        )
    m = joblib.load("logistic_model.joblib")
    v = joblib.load("vectorizer.joblib")
    le = joblib.load("label_encoder.joblib")
    return m, v, le

def css():
    st.markdown(
        """
        <style>
        /* page background gradient */
        .stApp {
          background: linear-gradient(135deg, #0f1724 0%, #0b3d91 40%, #0b6b6b 100%);
          color: #f0f6ff;
        }
        .glass {
          background: rgba(255,255,255,0.06);
          border-radius: 12px;
          padding: 18px;
          border: 1px solid rgba(255,255,255,0.06);
        }
        .title {
          font-size: 32px;
          font-weight: 800;
          color: white;
        }
        .muted { color: #cfe8ff; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def make_wordcloud(text, width=800, height=400, bgcolor="black"):
    wc = WordCloud(width=width, height=height, background_color=bgcolor).generate(text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def plot_sentiment_gauge(score):
    """Plotly gauge showing score (0..1)"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Positive Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00cc96"},
            'steps': [
                {'range': [0, 40], 'color': "#ef553b"},
                {'range': [40, 70], 'color': "#fbc02d"},
                {'range': [70, 100], 'color': "#00cc96"}]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def top_keywords(text, n=10):
    tokens = [w.lower().strip() for w in text.split() if w.lower() not in ENGLISH_STOP_WORDS]
    s = pd.Series(tokens).value_counts().head(n)
    return s

def create_rotating_sphere(freq_series, frames=40):
    """3D rotating mood sphere (frequencies -> radii). Returns Plotly figure with animation frames."""
    labels = list(freq_series.index)
    vals = freq_series.values
    n = len(vals)
    # map frequency -> sphere points (small cluster per keyword)
    xs, ys, zs, tlabels, ts = [], [], [], [], []
    maxv = float(vals.max()) if len(vals) else 1.0
    for i, (lab, v) in enumerate(zip(labels, vals)):
        r = 0.4 + (v / maxv) * 1.6
        # create a small spherical cluster for each word around different angles
        theta = i * (2 * math.pi / max(1, n))
        for k in range(10):
            phi = np.random.rand() * math.pi
            x = (np.cos(theta) * r) + (0.1 * np.random.randn())
            y = (np.sin(theta) * r) + (0.1 * np.random.randn())
            z = np.cos(phi) * (0.2 + 0.3 * np.random.rand())
            xs.append(x); ys.append(y); zs.append(z)
            tlabels.append(lab); ts.append(v)

    # base scatter
    scatter = go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                           marker=dict(size=6, color=ts, colorscale='Viridis', showscale=True),
                           text=tlabels, hoverinfo='text')
    fig = go.Figure(data=[scatter])
    # generate camera frames for rotation animation
    frames_list = []
    for k in range(frames):
        angle = 2 * math.pi * (k / frames)
        cam = dict(eye=dict(x=2.0 * np.cos(angle), y=2.0 * np.sin(angle), z=0.6))
        frames_list.append(dict(name=str(k), layout=dict(scene_camera=cam)))

    fig.frames = frames_list
    fig.update_layout(scene=dict(aspectmode='auto'),
                      updatemenus=[dict(type="buttons", showactive=False,
                                        y=1, x=0.8,
                                        xanchor="right", yanchor="top",
                                        pad=dict(t=0, r=10),
                                        buttons=[dict(label="Play",
                                                      method="animate",
                                                      args=[None, {"frame": {"duration": 80, "redraw": True},
                                                                   "fromcurrent": True, "transition": {"duration": 0}}]),
                                                 dict(label="Pause",
                                                      method="animate",
                                                      args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                     "mode": "immediate",
                                                                     "transition": {"duration": 0}}])])])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=500)
    return fig

def openai_insight(prompt, key, max_tokens=150):
    """Call OpenAI ChatCompletion (gpt-4 style) for insights. Return string or error text."""
    if not OPENAI_AVAILABLE:
        return "OpenAI not installed in environment. Add openai to requirements to enable."
    try:
        openai.api_key = key
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list() else "gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"

# ---------------------------
# App start
# ---------------------------
st.set_page_config(page_title="Emotion AI — Advanced Dashboard", layout="wide", page_icon="🧠")
css()
st.sidebar.title("Emotion AI — Menu")

menu = st.sidebar.radio("Navigate to:", ["Home", "Predict", "Analytics", "History", "GPT Insights"])

# Try load models
model = vectorizer = label_encoder = None
model_err = None
try:
    model, vectorizer, label_encoder = load_models()
except Exception as e:
    model_err = str(e)

# Header
st.markdown("<div class='title'>Emotion AI — Advanced Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Professional UI • 3D visuals • Animated charts • GPT-powered insights (optional)</div>", unsafe_allow_html=True)
st.write("")

# HOME
if menu == "Home":
    st.markdown("## Welcome")
    st.info("This dashboard includes a professional UI, interactive visualizations, export features, and optional GPT insights.")
    st.markdown("---")
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        st.metric("Model loaded", "Yes" if model and vectorizer and label_encoder else "No")
        st.metric("History rows", len(st.session_state.get("history", [])))
    with c2:
        st.image("https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200", use_column_width=True)
    with c3:
        st.markdown("#### Quick tips")
        st.write("- Paste journal text in Predict or Analytics.")
        st.write("- Use GPT Insights to get suggested interventions (requires key).")
        st.write("- Download CSV from History.")

# PREDICT
elif menu == "Predict":
    st.markdown("## Predict Emotion")
    if model_err:
        st.error("Model load error: " + model_err)
    else:
        text = st.text_area("Enter journal text to analyze:", height=220)
        col1, col2 = st.columns([2, 1])
        if col2.button("Analyze"):
            if not text.strip():
                st.warning("Enter text first.")
            else:
                try:
                    X = vectorizer.transform([text])
                    pred = model.predict(X)[0]
                    probs = None
                    try:
                        probs = model.predict_proba(X)[0]
                    except Exception:
                        pass
                    emotion = label_encoder.inverse_transform([pred])[0]
                    st.success(f"**Detected emotion:** {emotion}")
                    if probs is not None:
                        st.plotly_chart(plot_sentiment_gauge(probs[1]), use_container_width=True)
                    # wordcloud and keywords preview
                    wc_text = " ".join([w for w in text.split() if w.lower() not in ENGLISH_STOP_WORDS])
                    fig_wc = make_wordcloud(wc_text, bgcolor="white")
                    st.pyplot(fig_wc)
                    kw = top_keywords(text, n=12)
                    st.subheader("Top keywords")
                    st.bar_chart(kw)
                    # save to history
                    hist = st.session_state.get("history", [])
                    hist.append({"text": text, "emotion": emotion, "score": float(probs[1]) if probs is not None else None, "time": pd.Timestamp.now()})
                    st.session_state["history"] = hist
                except NotFittedError:
                    st.error("Model appears not fitted correctly. Recreate and upload proper joblib files.")

# ANALYTICS
elif menu == "Analytics":
    st.markdown("## Deep Analytics & 3D Visuals")
    raw = st.text_area("Paste many journal entries or a long text:", height=220)
    if st.button("Run Analytics"):
        if not raw.strip():
            st.warning("Enter text first.")
        else:
            # Wordcloud
            wc_text = " ".join([w for w in raw.split() if w.lower() not in ENGLISH_STOP_WORDS])
            st.subheader("Word Cloud")
            fig_wc = make_wordcloud(wc_text, bgcolor="black")
            st.pyplot(fig_wc)

            # Frequency and 2D chart
            st.subheader("Keyword Frequency (top 20)")
            freq = top_keywords(raw, n=20)
            st.plotly_chart(go.Figure(go.Bar(x=freq.index, y=freq.values)), use_container_width=True)

            # 3D rotating sphere (animated)
            st.subheader("3D Rotating Mood Sphere")
            try:
                fig3d = create_rotating_sphere(freq.head(12), frames=60)
                st.plotly_chart(fig3d, use_container_width=True)
            except Exception as e:
                st.error("3D chart error: " + str(e))

# HISTORY
elif menu == "History":
    st.markdown("## Prediction History")
    hist = st.session_state.get("history", [])
    if len(hist) == 0:
        st.info("No history yet — run predictions from Predict page.")
    else:
        df = pd.DataFrame(hist)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "emotion_history.csv", "text/csv")
        if st.button("Clear History"):
            st.session_state["history"] = []
            st.success("History cleared.")

# GPT INSIGHTS
elif menu == "GPT Insights":
    st.markdown("## GPT-powered Insights (optional)")
    st.markdown("Provide an OpenAI API key (or set OPENAI_API_KEY env var). The model will summarize notes and suggest small interventions.")
    key_input = st.text_input("OpenAI API Key (sk-...)", type="password")
    user_notes = st.text_area("Paste the text/journal entry you want insights for:", height=240)
    if st.button("Generate Insights"):
        if not user_notes.strip():
            st.warning("Paste text to analyze.")
        else:
            api_key = key_input.strip() or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                st.error("No OpenAI API key provided.")
            else:
                with st.spinner("Contacting OpenAI..."):
                    prompt = (
                        "You are an expert mental-health-aware assistant. "
                        "Summarize this journal entry in 2-3 lines and give 3 short evidence-based self-care suggestions."
                        f"\n\nJournal entry:\n{user_notes}\n\nSummary + 3 suggestions:"
                    )
                    out = openai_insight(prompt, api_key, max_tokens=220)
                    st.markdown("### GPT Summary & Suggestions")
                    st.info(out)

# Footer
st.markdown("---")
st.markdown("<div class='muted'>Built with ❤️ • Upload your model files to the repo root before deployment.</div>", unsafe_allow_html=True)
