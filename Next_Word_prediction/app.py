import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroPredict · Next Word AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── MEGA CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=JetBrains+Mono:ital,wght@0,300;0,400;0,700;1,300&family=Rajdhani:wght@300;400;600;700&display=swap');

:root {
  --bg0:     #03030a;
  --bg1:     #07071a;
  --bg2:     #0d0d28;
  --panel:   #0a0a20;
  --border:  #1a1a40;
  --cyan:    #00e5ff;
  --magenta: #ff2d78;
  --text:    #e8e8ff;
  --muted:   #4a4a7a;
  --glow-c:  rgba(0,229,255,0.18);
  --glow-m:  rgba(255,45,120,0.18);
}

html, body, [class*="css"], .stApp {
  font-family: 'Rajdhani', sans-serif !important;
  background: var(--bg0) !important;
  color: var(--text) !important;
}

.stApp {
  background-image:
    linear-gradient(rgba(0,229,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.025) 1px, transparent 1px) !important;
  background-size: 44px 44px !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 820px !important; }

/* ── Keyframes ── */
@keyframes fadeUp    { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
@keyframes chipPop   { 0%{transform:scale(0.75);opacity:0} 70%{transform:scale(1.05)} 100%{transform:scale(1);opacity:1} }
@keyframes blink     { 0%,100%{opacity:1} 50%{opacity:0} }
@keyframes scanH     { 0%{top:-2px} 100%{top:100%} }
@keyframes glowPulse { 0%,100%{box-shadow:0 0 15px var(--glow-c)} 50%{box-shadow:0 0 35px var(--glow-c),0 0 60px rgba(0,229,255,0.08)} }

/* ── Hero ── */
.hero-wrap { text-align:center; padding:2rem 0 1.2rem; animation:fadeUp 0.8s ease both; }
.hero-eyebrow {
  font-family:'JetBrains Mono',monospace; font-size:0.6rem;
  letter-spacing:0.3em; text-transform:uppercase; color:var(--cyan);
  margin-bottom:0.6rem; opacity:0.75;
}
.hero-eyebrow em { color:var(--magenta); font-style:normal; }
.hero-title {
  font-family:'Orbitron',monospace; font-weight:900;
  font-size:clamp(2.2rem,5.5vw,3.8rem); line-height:1;
  background:linear-gradient(135deg,#00e5ff 0%,#ffffff 45%,#ff2d78 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  letter-spacing:-0.01em; margin-bottom:0.4rem;
}
.hero-sub {
  font-family:'JetBrains Mono',monospace; font-size:0.68rem;
  color:var(--muted); letter-spacing:0.18em; text-transform:uppercase;
}
.hero-sub .d { color:var(--cyan); margin:0 0.5rem; }
.hero-rule {
  display:flex; align-items:center; justify-content:center; gap:0.7rem; margin:1.1rem 0 2rem;
}
.hr-bar  { height:1px; width:72px; }
.hr-bar.l{ background:linear-gradient(90deg,transparent,var(--cyan)); }
.hr-bar.r{ background:linear-gradient(90deg,var(--cyan),transparent); }
.hr-gem  { width:7px;height:7px;background:var(--cyan);transform:rotate(45deg);
           box-shadow:0 0 10px var(--cyan),0 0 22px var(--cyan); }

/* ── Stats row ── */
.stats-row {
  display:grid; grid-template-columns:repeat(3,1fr); gap:0.75rem;
  margin-bottom:1.75rem; animation:fadeUp 0.85s ease 0.1s both;
}
.stat-card {
  background:var(--bg1); border:1px solid rgba(0,229,255,0.1);
  border-radius:10px; padding:0.85rem 1rem; text-align:center;
}
.stat-val {
  font-family:'Orbitron',monospace; font-size:1.35rem; font-weight:700;
  color:var(--cyan); line-height:1; text-shadow:0 0 14px var(--glow-c);
}
.stat-lbl {
  font-family:'JetBrains Mono',monospace; font-size:0.52rem;
  text-transform:uppercase; letter-spacing:0.15em; color:var(--muted); margin-top:0.3rem;
}

/* ── Panel ── */
.neo-panel {
  background:var(--panel); border:1px solid rgba(0,229,255,0.13);
  border-radius:16px; padding:1.75rem 2rem; position:relative;
  overflow:hidden; animation:fadeUp 0.9s ease 0.15s both;
}
.neo-panel::before {
  content:""; position:absolute; top:0;left:0;right:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
}
.neo-panel::after {
  content:""; position:absolute; top:-80px;right:-80px;
  width:250px;height:250px;
  background:radial-gradient(circle,rgba(0,229,255,0.04) 0%,transparent 70%);
  pointer-events:none;
}
/* corner brackets */
.c { position:absolute; width:14px; height:14px; pointer-events:none; }
.c.tl { top:8px;left:8px;   border-top:2px solid var(--cyan);border-left:2px solid var(--cyan); }
.c.tr { top:8px;right:8px;  border-top:2px solid var(--cyan);border-right:2px solid var(--cyan); }
.c.bl { bottom:8px;left:8px; border-bottom:2px solid var(--cyan);border-left:2px solid var(--cyan); }
.c.br { bottom:8px;right:8px;border-bottom:2px solid var(--cyan);border-right:2px solid var(--cyan); }
.c.m  { border-color:var(--magenta)!important; }

/* ── Section label ── */
.slbl {
  font-family:'JetBrains Mono',monospace; font-size:0.6rem;
  text-transform:uppercase; letter-spacing:0.2em; color:var(--cyan);
  margin-bottom:0.7rem; display:flex; align-items:center; gap:0.55rem;
}
.slbl::before { content:"▶"; font-size:0.48rem; color:var(--magenta); }
.slbl .tag {
  background:rgba(0,229,255,0.07); border:1px solid rgba(0,229,255,0.18);
  border-radius:4px; padding:1px 6px; font-size:0.5rem; color:var(--cyan); margin-left:auto;
}
.slbl.m  { color:var(--magenta); }
.slbl.m .tag { background:rgba(255,45,120,0.07);border-color:rgba(255,45,120,0.2);color:var(--magenta); }

/* ── Input ── */
.stTextInput > div > div > input {
  background:#050510!important; border:1.5px solid rgba(0,229,255,0.18)!important;
  border-radius:10px!important; color:var(--cyan)!important;
  font-family:'JetBrains Mono',monospace!important; font-size:0.98rem!important;
  padding:0.78rem 1rem!important; letter-spacing:0.04em!important;
  transition:all 0.2s!important; caret-color:var(--magenta)!important;
}
.stTextInput > div > div > input::placeholder { color:var(--muted)!important; font-style:italic; }
.stTextInput > div > div > input:focus {
  border-color:var(--cyan)!important;
  box-shadow:0 0 0 2px rgba(0,229,255,0.1),0 0 18px rgba(0,229,255,0.08)!important;
}
.stTextInput label {
  font-family:'JetBrains Mono',monospace!important; font-size:0.6rem!important;
  color:var(--muted)!important; text-transform:uppercase; letter-spacing:0.15em!important;
}

/* ── Slider ── */
.stSlider label {
  font-family:'JetBrains Mono',monospace!important; font-size:0.6rem!important;
  color:var(--muted)!important; text-transform:uppercase; letter-spacing:0.15em!important;
}
div[data-baseweb="slider"] > div > div > div {
  background:linear-gradient(90deg,var(--cyan),var(--magenta))!important; height:3px!important;
}
div[data-baseweb="slider"] [role="slider"] {
  background:var(--cyan)!important; border:2px solid var(--bg0)!important;
  box-shadow:0 0 10px var(--cyan)!important; width:15px!important; height:15px!important;
}

/* ── Button ── */
.stButton > button {
  background:transparent!important; border:1.5px solid var(--cyan)!important;
  border-radius:10px!important; color:var(--cyan)!important;
  font-family:'Orbitron',monospace!important; font-weight:700!important;
  font-size:0.72rem!important; padding:0.65rem 1.4rem!important;
  width:100%!important; letter-spacing:0.18em!important; text-transform:uppercase!important;
  transition:all 0.22s!important;
}
.stButton > button:hover {
  background:var(--cyan)!important; color:#03030a!important;
  box-shadow:0 0 20px var(--glow-c),0 0 40px var(--glow-c)!important;
  transform:translateY(-2px)!important;
}
.stButton > button:active { transform:translateY(0)!important; }

/* ── Prob bars ── */
.prob-row {
  display:flex; align-items:center; gap:0.65rem;
  margin-bottom:0.42rem; animation:fadeUp 0.4s ease both;
}
.prob-word {
  font-family:'JetBrains Mono',monospace; font-size:0.78rem;
  color:var(--cyan); width:110px; flex-shrink:0;
}
.prob-bar-bg {
  flex:1; height:5px; background:rgba(255,255,255,0.05);
  border-radius:3px; overflow:hidden;
}
.prob-bar-fg {
  height:100%; border-radius:3px;
  background:linear-gradient(90deg,var(--cyan),var(--magenta));
  box-shadow:0 0 7px var(--glow-c);
}
.prob-pct {
  font-family:'JetBrains Mono',monospace; font-size:0.65rem;
  color:var(--muted); width:40px; text-align:right;
}

/* ── Sentence box ── */
.sbox {
  background:#04040e; border:1px solid rgba(0,229,255,0.1);
  border-left:3px solid var(--cyan); border-radius:10px;
  padding:1rem 1.25rem; font-family:'JetBrains Mono',monospace;
  font-size:0.96rem; line-height:1.85; word-break:break-word;
  position:relative; margin:0.8rem 0; animation:fadeUp 0.4s ease both;
}
.sbox::before {
  content:"OUTPUT"; position:absolute; top:-8px; left:14px;
  background:var(--bg0); padding:0 6px;
  font-family:'JetBrains Mono',monospace; font-size:0.47rem;
  letter-spacing:0.2em; color:var(--cyan);
}
.sbox .seed  { color:#4a4a7a; }
.sbox .added { color:var(--cyan); font-weight:700; }
.sbox .cursor {
  display:inline-block; width:2px; height:1em;
  background:var(--magenta); vertical-align:text-bottom;
  margin-left:2px; animation:blink 1s step-end infinite;
}

/* ── Auto panel ── */
.auto-panel {
  background:linear-gradient(135deg,#080818,#0a0520);
  border:1px solid rgba(255,45,120,0.14); border-radius:16px;
  padding:1.75rem 2rem; position:relative; overflow:hidden;
  margin-top:1.5rem; animation:fadeUp 0.9s ease 0.25s both;
}
.auto-panel::before {
  content:""; position:absolute; top:0;left:0;right:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--magenta),transparent);
}

/* ── Error ── */
.err-box {
  background:rgba(255,45,120,0.06); border:1px solid rgba(255,45,120,0.28);
  border-left:3px solid var(--magenta); border-radius:10px;
  padding:1rem 1.25rem; font-family:'JetBrains Mono',monospace;
  font-size:0.78rem; color:#ff8ab0; line-height:1.7;
}
.err-box strong { color:var(--magenta); }

/* ── Footer ── */
.neo-footer {
  text-align:center; padding:2rem 0 1rem;
  animation:fadeUp 1s ease 0.35s both;
}
.footer-inner {
  display:inline-flex; align-items:center; gap:0.55rem;
  font-family:'JetBrains Mono',monospace; font-size:0.58rem;
  color:var(--muted); letter-spacing:0.1em; text-transform:uppercase;
}
.footer-inner .sep { color:var(--cyan); }
.footer-inner a { color:var(--cyan); text-decoration:none; }
.footer-inner a:hover { text-shadow:0 0 8px var(--cyan); }

/* stagger delays */
.prob-row:nth-child(1){animation-delay:.04s}
.prob-row:nth-child(2){animation-delay:.10s}
.prob-row:nth-child(3){animation-delay:.16s}
.prob-row:nth-child(4){animation-delay:.22s}
.prob-row:nth-child(5){animation-delay:.28s}
.prob-row:nth-child(6){animation-delay:.34s}
.prob-row:nth-child(7){animation-delay:.40s}
.prob-row:nth-child(8){animation-delay:.46s}
</style>
""", unsafe_allow_html=True)


# ─── Load artifacts ────────────────────────────────────────────────────────────
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


# ─── Prediction ────────────────────────────────────────────────────────────────
def predict_next_words(text: str, n: int = 5):
    token_seq   = tokenizer.texts_to_sequences([text])[0]
    padded      = pad_sequences([token_seq], maxlen=max_len - 1, padding="pre")
    preds       = model.predict(padded, verbose=0)[0]
    top_n_idx   = np.argsort(preds)[-n:][::-1]
    idx2word    = {v: k for k, v in tokenizer.word_index.items()}
    return [(idx2word[i], float(preds[i])) for i in top_n_idx if i in idx2word]


# ══════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">[ NEURAL LANGUAGE MODEL <em>// LSTM ARCHITECTURE</em> ]</div>
  <div class="hero-title">NEUROPREDICT</div>
  <div class="hero-sub">Next Word AI <span class="d">◆</span> Deep Learning <span class="d">◆</span> NLP</div>
  <div class="hero-rule">
    <div class="hr-bar l"></div>
    <div class="hr-gem"></div>
    <div class="hr-bar r"></div>
  </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.markdown(f'<div class="err-box"><strong>⚠ SYSTEM ERROR :</strong> {load_error}<br><br>Place <code>lstm_model.h5</code>, <code>tokenizer.pkl</code>, <code>max_len.pkl</code> in the same folder as <code>app.py</code>.</div>', unsafe_allow_html=True)
    st.stop()

# Stats
st.markdown(f"""
<div class="stats-row">
  <div class="stat-card"><div class="stat-val">{vocab_size:,}</div><div class="stat-lbl">Vocabulary</div></div>
  <div class="stat-card"><div class="stat-val">{max_len}</div><div class="stat-lbl">Max Seq Len</div></div>
  <div class="stat-card"><div class="stat-val">LSTM</div><div class="stat-lbl">Architecture</div></div>
</div>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
for k, v in [("input_text",""),("predictions",[]),("sentence",""),("seed_len",0),("run",0)]:
    if k not in st.session_state: st.session_state[k] = v


# ══════════════════════════════════════════════════════════
# PREDICT PANEL
# ══════════════════════════════════════════════════════════
st.markdown('<div class="neo-panel"><div class="c tl"></div><div class="c tr"></div><div class="c bl"></div><div class="c br"></div>', unsafe_allow_html=True)
st.markdown('<div class="slbl">Input Sequence <span class="tag">ENCODE</span></div>', unsafe_allow_html=True)

ci, cn = st.columns([4, 1])
with ci:
    user_input = st.text_input("seed", value=st.session_state.input_text,
                               placeholder="Enter seed text…", label_visibility="collapsed")
with cn:
    top_n = st.slider("N", 1, 10, 5, label_visibility="collapsed")

st.caption(f"🎛 Top-**{top_n}** predictions")

if st.button("⚡  RUN PREDICTION"):
    if user_input.strip():
        preds = predict_next_words(user_input.strip(), n=top_n)
        st.session_state.update(predictions=preds, input_text=user_input.strip(),
                                sentence=user_input.strip(),
                                seed_len=len(user_input.strip().split()), run=st.session_state.run+1)
st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════
if st.session_state.predictions:
    preds    = st.session_state.predictions
    max_prob = preds[0][1] if preds else 1

    # Probability bars
    st.markdown('<div class="slbl" style="margin-top:1.5rem">Probability Distribution <span class="tag">SOFTMAX</span></div>', unsafe_allow_html=True)
    bars = "".join(f"""
      <div class="prob-row">
        <div class="prob-word">{w}</div>
        <div class="prob-bar-bg"><div class="prob-bar-fg" style="width:{int(p/max_prob*100)}%"></div></div>
        <div class="prob-pct">{p*100:.1f}%</div>
      </div>""" for w, p in preds)
    st.markdown(bars, unsafe_allow_html=True)

    # Clickable word buttons
    st.markdown('<div class="slbl" style="margin-top:1.2rem">Select Next Word <span class="tag">INTERACTIVE</span></div>', unsafe_allow_html=True)
    cols = st.columns(len(preds))
    for i, (word, prob) in enumerate(preds):
        with cols[i]:
            if st.button(f"{word}\n{prob*100:.0f}%", key=f"w_{i}_{st.session_state.run}"):
                st.session_state.sentence += f" {word}"
                st.session_state.predictions = predict_next_words(st.session_state.sentence, n=top_n)
                st.session_state.run += 1
                st.rerun()

    # Sentence display
    if st.session_state.sentence:
        words  = st.session_state.sentence.split()
        seed_p = " ".join(words[:st.session_state.seed_len])
        add_p  = " ".join(words[st.session_state.seed_len:])
        added_html = f'<span class="added"> {add_p}</span>' if add_p else ""
        st.markdown(f'<div class="sbox"><span class="seed">{seed_p}</span>{added_html}<span class="cursor"></span></div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            if st.button("↺  RESET"):
                st.session_state.update(sentence=st.session_state.input_text,
                                        predictions=predict_next_words(st.session_state.input_text, top_n),
                                        run=st.session_state.run+1)
                st.rerun()
        with r2:
            if st.button("⊕  USE AS SEED"):
                st.session_state.update(input_text=st.session_state.sentence,
                                        seed_len=len(words))
                st.rerun()
        with r3:
            wc = len(words) - st.session_state.seed_len
            st.markdown(f'<div style="text-align:center;font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#4a4a7a;padding-top:0.72rem">+{wc} WORD{"S" if wc!=1 else ""} ADDED</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# AUTO-COMPLETE
# ══════════════════════════════════════════════════════════
st.markdown('<div class="auto-panel"><div class="c tl m"></div><div class="c tr m"></div><div class="c bl m"></div><div class="c br m"></div>', unsafe_allow_html=True)
st.markdown('<div class="slbl m">Auto-Complete Engine <span class="tag">GENERATIVE</span></div>', unsafe_allow_html=True)

a1, a2 = st.columns([3, 1])
with a1:
    auto_seed = st.text_input("ac", placeholder="Type a seed phrase…", label_visibility="collapsed", key="aseed")
with a2:
    auto_n = st.slider("W", 3, 40, 12, label_visibility="collapsed", key="an")

st.caption(f"🔁 Generating **{auto_n}** words")

if st.button("✦  AUTO-COMPLETE", key="acbtn"):
    if auto_seed.strip():
        with st.spinner("Generating…"):
            gen = auto_seed.strip()
            for _ in range(auto_n):
                p = predict_next_words(gen, n=1)
                if p: gen += f" {p[0][0]}"
            ow    = len(auto_seed.strip().split())
            gw    = gen.split()
            sd    = " ".join(gw[:ow])
            ad    = " ".join(gw[ow:])
        st.markdown(f'<div class="sbox" style="border-left-color:var(--magenta)"><span style="position:absolute;top:-8px;left:14px;background:var(--bg0);padding:0 6px;font-family:JetBrains Mono,monospace;font-size:0.47rem;letter-spacing:0.2em;color:var(--magenta)">AUTO-OUTPUT</span><span class="seed">{sd}</span> <span class="added" style="color:var(--magenta)">{ad}</span></div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="neo-footer">
  <div class="footer-inner">
    <span>NeuroPredict</span><span class="sep">◆</span>
    <span>LSTM · TensorFlow · Streamlit</span><span class="sep">◆</span>
    <a href="https://github.com/Yugal0708" target="_blank">@Yugal0708</a><span class="sep">◆</span>
    <span>Nagpur, IN</span>
  </div>
</div>
""", unsafe_allow_html=True)
