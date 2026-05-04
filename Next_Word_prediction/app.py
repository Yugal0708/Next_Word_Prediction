import streamlit as st
import numpy as np
import pickle
import tf_keras as keras
from tf_keras.models import load_model
from tf_keras.preprocessing.sequence import pad_sequences
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

  /* ── Root Variables ── */
  :root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --border:    #1e1e2e;
    --accent:    #7c3aed;
    --accent2:   #06b6d4;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --glow:      rgba(124, 58, 237, 0.35);
  }

  /* ── Global Reset ── */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2.5rem !important; max-width: 780px; }

  /* ── Noise texture overlay ── */
  body::before {
    content: "";
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0; opacity: 0.4;
  }

  /* ── Hero heading ── */
  .hero-title {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 60%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
  }
  .hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
  }

  /* ── Card container ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: "";
    position: absolute;
    top: -60px; left: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, var(--glow) 0%, transparent 70%);
    pointer-events: none;
  }

  /* ── Text input override ── */
  .stTextInput > div > div > input {
    background: #0d0d14 !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--glow) !important;
    outline: none !important;
  }
  .stTextInput label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  /* ── Slider override ── */
  .stSlider label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .stSlider > div > div > div > div {
    background: var(--accent) !important;
  }

  /* ── Button ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent), #4f46e5) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    letter-spacing: 0.03em;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px var(--glow) !important;
  }
  .stButton > button:active {
    transform: translateY(0px) !important;
  }

  /* ── Prediction chips ── */
  .pred-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1.2rem;
  }
  .pred-chip {
    background: linear-gradient(135deg, #1e1b4b, #1a1a2e);
    border: 1.5px solid #312e81;
    border-radius: 8px;
    padding: 0.5rem 1.1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: #a5b4fc;
    cursor: pointer;
    transition: all 0.15s;
  }
  .pred-chip:hover {
    border-color: var(--accent);
    color: white;
    background: var(--accent);
    transform: translateY(-2px);
    box-shadow: 0 6px 18px var(--glow);
  }
  .pred-rank {
    font-size: 0.6rem;
    color: var(--muted);
    margin-right: 0.4rem;
    vertical-align: super;
  }

  /* ── Sentence output ── */
  .sentence-box {
    background: #0d0d14;
    border: 1.5px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    color: var(--text);
    margin-top: 1.2rem;
    line-height: 1.7;
    word-break: break-word;
  }
  .sentence-box .highlight {
    color: #a78bfa;
    font-weight: 700;
  }

  /* ── Section label ── */
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .section-label::before {
    content: "";
    display: inline-block;
    width: 16px; height: 2px;
    background: var(--accent);
    border-radius: 1px;
  }

  /* ── Divider ── */
  hr { border-color: var(--border) !important; margin: 1.6rem 0 !important; }

  /* ── Footer ── */
  .footer-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-align: center;
    margin-top: 2.5rem;
    letter-spacing: 0.08em;
  }
  .footer-tag a { color: #7c3aed; text-decoration: none; }
  .footer-tag a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Artifacts ────────────────────────────────────────────────────
import os

@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    model     = load_model(os.path.join(base, "lstm_model.h5"))
    tokenizer = pickle.load(open(os.path.join(base, "tokenizer.pkl"), "rb"))
    max_len   = pickle.load(open(os.path.join(base, "max_len.pkl"),   "rb"))
    return model, tokenizer, max_len
try:
    model, tokenizer, max_len = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)


# ─── Prediction Logic ──────────────────────────────────────────────────────────
def predict_next_words(text: str, n: int = 5):
    """Return top-n predicted next words with their probabilities."""
    token_seq = tokenizer.texts_to_sequences([text])[0]
    padded    = pad_sequences([token_seq], maxlen=max_len - 1, padding="pre")
    preds     = model.predict(padded, verbose=0)[0]

    # Get top-n indices
    top_n_idx   = np.argsort(preds)[-n:][::-1]
    word_index  = tokenizer.word_index
    idx_to_word = {v: k for k, v in word_index.items()}

    results = []
    for idx in top_n_idx:
        word = idx_to_word.get(idx, None)
        if word:
            results.append((word, float(preds[idx])))
    return results


# ─── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔮 Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">LSTM · Deep Learning · NLP</div>', unsafe_allow_html=True)

if not model_loaded:
    st.error(f"**Model load failed:** {load_error}\n\nMake sure `lstm_model.h5`, `tokenizer.pkl`, and `max_len.pkl` are in the same directory as `app.py`.")
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────────
if "input_text"        not in st.session_state: st.session_state.input_text        = ""
if "predictions"       not in st.session_state: st.session_state.predictions       = []
if "auto_sentence"     not in st.session_state: st.session_state.auto_sentence     = ""
if "auto_sentence_raw" not in st.session_state: st.session_state.auto_sentence_raw = ""

# ── Controls ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input(
        "Seed text",
        value=st.session_state.input_text,
        placeholder="Type a sentence...",
        key="input_widget",
    )
with col2:
    top_n = st.slider("Top N", min_value=1, max_value=10, value=5, key="topn")

predict_clicked = st.button("⚡ Predict Next Word")

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_clicked and user_input.strip():
    preds = predict_next_words(user_input.strip(), n=top_n)
    st.session_state.predictions       = preds
    st.session_state.input_text        = user_input.strip()
    st.session_state.auto_sentence     = user_input.strip()
    st.session_state.auto_sentence_raw = user_input.strip()

# ── Chip click handler via query param trick ───────────────────────────────────
# We'll render chips as buttons inside columns
if st.session_state.predictions:
    st.markdown('<div class="section-label">Predicted Words</div>', unsafe_allow_html=True)

    chips = st.session_state.predictions
    cols  = st.columns(len(chips))
    for i, (word, prob) in enumerate(chips):
        with cols[i]:
            if st.button(f"{word}\n{prob*100:.1f}%", key=f"chip_{i}"):
                st.session_state.auto_sentence     += f" {word}"
                st.session_state.auto_sentence_raw += f" {word}"
                new_preds = predict_next_words(st.session_state.auto_sentence, n=top_n)
                st.session_state.predictions = new_preds
                st.rerun()

    st.markdown("---")

    # ── Sentence so far ──────────────────────────────────────────────────────
    if st.session_state.auto_sentence:
        seed_len   = len(st.session_state.input_text.split())
        words      = st.session_state.auto_sentence.split()
        seed_part  = " ".join(words[:seed_len])
        added_part = " ".join(words[seed_len:])

        display_html = f'<div class="sentence-box">'
        display_html += f'<span style="color:#94a3b8">{seed_part}</span>'
        if added_part:
            display_html += f' <span class="highlight">{added_part}</span>'
        display_html += "</div>"

        st.markdown('<div class="section-label">Built Sentence</div>', unsafe_allow_html=True)
        st.markdown(display_html, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Reset Sentence"):
                st.session_state.auto_sentence     = st.session_state.input_text
                st.session_state.auto_sentence_raw = st.session_state.input_text
                st.session_state.predictions       = predict_next_words(st.session_state.input_text, n=top_n)
                st.rerun()
        with c2:
            if st.button("📋 Copy to Input"):
                st.session_state.input_text = st.session_state.auto_sentence
                st.rerun()

# ── Auto-Complete Mode ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">Auto-Complete Mode</div>', unsafe_allow_html=True)

auto_col1, auto_col2 = st.columns([3, 1])
with auto_col1:
    auto_seed  = st.text_input("Seed for auto-complete", placeholder="e.g. The quick brown", key="auto_seed")
with auto_col2:
    auto_words = st.slider("Words to generate", 1, 30, 10, key="auto_len")

if st.button("✨ Auto-Complete Sentence"):
    if auto_seed.strip():
        generated = auto_seed.strip()
        for _ in range(auto_words):
            p = predict_next_words(generated, n=1)
            if p:
                generated += f" {p[0][0]}"
        # Display
        original_words = len(auto_seed.strip().split())
        gen_words      = generated.split()
        seed_disp      = " ".join(gen_words[:original_words])
        added_disp     = " ".join(gen_words[original_words:])
        st.markdown(
            f'<div class="sentence-box">'
            f'<span style="color:#94a3b8">{seed_disp}</span>'
            f' <span class="highlight">{added_disp}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="footer-tag">Built with 🔮 by <a href="https://github.com/Yugal0708" target="_blank">Yugal</a> · LSTM Next Word Prediction · Streamlit + TensorFlow</div>',
    unsafe_allow_html=True,
)
