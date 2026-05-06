import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroPredict · Next Word AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load CSS from file (avoids Python TokenError with inline CSS) ──────────────
def load_css():
    base     = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base, "style.css")
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base      = os.path.dirname(os.path.abspath(__file__))
    model     = load_model(os.path.join(base, "lstm_model.h5"))
    tokenizer = pickle.load(open(os.path.join(base, "tokenizer.pkl"), "rb"))
    max_len   = pickle.load(open(os.path.join(base, "max_len.pkl"),   "rb"))
    return model, tokenizer, max_len

try:
    model, tokenizer, max_len = load_artifacts()
    model_loaded = True
    vocab_size   = len(tokenizer.word_index) + 1
except Exception as e:
    model_loaded = False
    load_error   = str(e)
    vocab_size   = 0

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_next_words(text, n=5):
    token_seq = tokenizer.texts_to_sequences([text])[0]
    padded    = pad_sequences([token_seq], maxlen=max_len - 1, padding="pre")
    preds     = model.predict(padded, verbose=0)[0]
    top_idx   = np.argsort(preds)[-n:][::-1]
    idx2word  = {v: k for k, v in tokenizer.word_index.items()}
    return [(idx2word[i], float(preds[i])) for i in top_idx if i in idx2word]

# ══════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-eyebrow">[ NEURAL LANGUAGE MODEL <em>// LSTM ARCHITECTURE</em> ]</div>'
    '<div class="hero-title">NEUROPREDICT</div>'
    '<div class="hero-sub">Next Word AI <span class="d">&#9670;</span> Deep Learning <span class="d">&#9670;</span> NLP</div>'
    '<div class="hero-rule">'
    '  <div class="hr-bar l"></div>'
    '  <div class="hr-gem"></div>'
    '  <div class="hr-bar r"></div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Error guard ────────────────────────────────────────────────────────────────
if not model_loaded:
    st.markdown(
        '<div class="err-box"><strong>SYSTEM ERROR:</strong> ' + load_error +
        '<br><br>Place <code>lstm_model.h5</code>, <code>tokenizer.pkl</code>,'
        ' <code>max_len.pkl</code> in the same folder as <code>app.py</code>.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Stats cards ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="stats-row">'
    '  <div class="stat-card"><div class="stat-val">' + f"{vocab_size:,}" + '</div><div class="stat-lbl">Vocabulary</div></div>'
    '  <div class="stat-card"><div class="stat-val">' + str(max_len) + '</div><div class="stat-lbl">Max Seq Len</div></div>'
    '  <div class="stat-card"><div class="stat-val">LSTM</div><div class="stat-lbl">Architecture</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("input_text",""), ("predictions",[]), ("sentence",""), ("seed_len",0), ("run",0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════
# PREDICT PANEL
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div class="neo-panel">'
    '  <div class="panel-top-line"></div>'
    '  <div class="panel-glow"></div>'
    '  <div class="c tl"></div><div class="c tr"></div>'
    '  <div class="c bl"></div><div class="c br"></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="slbl"><span class="slbl-arrow">&#9658;</span> Input Sequence <span class="tag">ENCODE</span></div>',
    unsafe_allow_html=True,
)

ci, cn = st.columns([4, 1])
with ci:
    user_input = st.text_input(
        "seed", value=st.session_state.input_text,
        placeholder="Enter seed text...",
        label_visibility="collapsed",
    )
with cn:
    top_n = st.slider("N", 1, 10, 5, label_visibility="collapsed")

st.caption(f"Top-**{top_n}** predictions")

if st.button("PREDICT NEXT WORD"):
    if user_input.strip():
        preds = predict_next_words(user_input.strip(), n=top_n)
        st.session_state.predictions = preds
        st.session_state.input_text  = user_input.strip()
        st.session_state.sentence    = user_input.strip()
        st.session_state.seed_len    = len(user_input.strip().split())
        st.session_state.run        += 1

st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════
if st.session_state.predictions:
    preds    = st.session_state.predictions
    max_prob = preds[0][1] if preds else 1

    # Probability bars
    st.markdown(
        '<div class="slbl" style="margin-top:1.5rem">'
        '<span class="slbl-arrow">&#9658;</span> Probability Distribution '
        '<span class="tag">SOFTMAX</span></div>',
        unsafe_allow_html=True,
    )
    bars_html = ""
    for w, p in preds:
        bar_w = int((p / max_prob) * 100)
        bars_html += (
            '<div class="prob-row">'
            f'  <div class="prob-word">{w}</div>'
            f'  <div class="prob-bar-bg"><div class="prob-bar-fg" style="width:{bar_w}%"></div></div>'
            f'  <div class="prob-pct">{p*100:.1f}%</div>'
            '</div>'
        )
    st.markdown(bars_html, unsafe_allow_html=True)

    # Clickable word buttons
    st.markdown(
        '<div class="slbl" style="margin-top:1.2rem">'
        '<span class="slbl-arrow">&#9658;</span> Select Next Word '
        '<span class="tag">INTERACTIVE</span></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(preds))
    for i, (word, prob) in enumerate(preds):
        with cols[i]:
            label = f"{word}\n{prob*100:.0f}%"
            if st.button(label, key=f"w_{i}_{st.session_state.run}"):
                st.session_state.sentence    += f" {word}"
                st.session_state.predictions  = predict_next_words(st.session_state.sentence, n=top_n)
                st.session_state.run         += 1
                st.rerun()

    # Built sentence display
    if st.session_state.sentence:
        words  = st.session_state.sentence.split()
        seed_p = " ".join(words[:st.session_state.seed_len])
        add_p  = " ".join(words[st.session_state.seed_len:])
        added_html = f'<span class="added"> {add_p}</span>' if add_p else ""
        st.markdown(
            '<div class="sbox">'
            f'  <span class="seed">{seed_p}</span>'
            f'  {added_html}'
            '  <span class="cursor"></span>'
            '</div>',
            unsafe_allow_html=True,
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            if st.button("RESET"):
                st.session_state.sentence    = st.session_state.input_text
                st.session_state.predictions = predict_next_words(st.session_state.input_text, top_n)
                st.session_state.run        += 1
                st.rerun()
        with r2:
            if st.button("USE AS SEED"):
                st.session_state.input_text = st.session_state.sentence
                st.session_state.seed_len   = len(words)
                st.rerun()
        with r3:
            wc = len(words) - st.session_state.seed_len
            st.markdown(
                f'<div style="text-align:center;font-family:JetBrains Mono,monospace;'
                f'font-size:0.62rem;color:#4a4a7a;padding-top:0.72rem">+{wc} WORD{"S" if wc!=1 else ""} ADDED</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════
# AUTO-COMPLETE
# ══════════════════════════════════════════════════════════
st.markdown(
    '<div class="auto-panel">'
    '  <div class="auto-top-line"></div>'
    '  <div class="c tl m"></div><div class="c tr m"></div>'
    '  <div class="c bl m"></div><div class="c br m"></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="slbl m"><span class="slbl-arrow">&#9658;</span> Auto-Complete Engine '
    '<span class="tag">GENERATIVE</span></div>',
    unsafe_allow_html=True,
)

a1, a2 = st.columns([3, 1])
with a1:
    auto_seed = st.text_input("ac", placeholder="Type a seed phrase...",
                              label_visibility="collapsed", key="aseed")
with a2:
    auto_n = st.slider("W", 3, 40, 12, label_visibility="collapsed", key="an")

st.caption(f"Generating **{auto_n}** words")

if st.button("AUTO-COMPLETE", key="acbtn"):
    if auto_seed.strip():
        with st.spinner("Generating..."):
            gen = auto_seed.strip()
            for _ in range(auto_n):
                p = predict_next_words(gen, n=1)
                if p:
                    gen += f" {p[0][0]}"
        ow = len(auto_seed.strip().split())
        gw = gen.split()
        sd = " ".join(gw[:ow])
        ad = " ".join(gw[ow:])
        st.markdown(
            '<div class="sbox" style="border-left-color:var(--magenta)">'
            f'  <span class="seed">{sd}</span>'
            f'  <span class="added" style="color:var(--magenta)"> {ad}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="neo-footer">'
    '  <div class="footer-inner">'
    '    <span>NeuroPredict</span><span class="sep">&#9670;</span>'
    '    <span>LSTM &middot; TensorFlow &middot; Streamlit</span><span class="sep">&#9670;</span>'
    '    <a href="https://github.com/Yugal0708" target="_blank">@Yugal0708</a>'
    '    <span class="sep">&#9670;</span><span>Nagpur, IN</span>'
    '  </div>'
    '</div>',
    unsafe_allow_html=True,
)
