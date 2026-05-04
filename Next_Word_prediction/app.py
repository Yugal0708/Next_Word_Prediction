import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

st.set_page_config(page_title="NextWord AI", page_icon="✨", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main-header {font-size: 3rem; font-weight: 700; background: linear-gradient(90deg, #667eea, #764ba2);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
.word-chip {display: inline-block; background: linear-gradient(45deg, #667eea, #764ba2); color: white;
padding: 10px 20px; margin: 6px; border-radius: 25px; font-weight: 600; cursor: pointer;}
</style>
""", unsafe_allow_html=True)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model_and_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "lstm_model.h5")
    
    try:
        # Try modern Keras 3 approach
        os.environ['TF_USE_LEGACY_KERAS'] = '0'   # Use new Keras
        
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )
        
        with open(os.path.join(current_dir, "tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)
        with open(os.path.join(current_dir, "max_len.pkl"), "rb") as f:
            max_len = pickle.load(f)
            
        return model, tokenizer, max_len, "✅ Loaded successfully (Keras 3)"
        
    except Exception as e1:
        st.warning("Trying legacy mode...")
        try:
            # Fallback to legacy Keras
            os.environ['TF_USE_LEGACY_KERAS'] = '1'
            model = tf.keras.models.load_model(model_path, compile=False)
            
            with open(os.path.join(current_dir, "tokenizer.pkl"), "rb") as f:
                tokenizer = pickle.load(f)
            with open(os.path.join(current_dir, "max_len.pkl"), "rb") as f:
                max_len = pickle.load(f)
                
            return model, tokenizer, max_len, "✅ Loaded with Legacy Keras"
        except Exception as e2:
            st.error(f"Both loading methods failed: {e2}")
            st.write("Files in folder:", os.listdir(current_dir))
            st.stop()

# Load
model, tokenizer, max_len, status = load_model_and_files()
st.success(status)

# ====================== UI ======================
st.markdown('<h1 class="main-header">✨ NextWord AI</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.write("**Vocab Size:**", len(tokenizer.word_index))
    st.write("**Max Length:**", max_len)

input_text = st.text_area("Type here...", placeholder="The weather today is", height=120)

col1, col2 = st.columns(2)
with col1:
    num_pred = st.slider("Suggestions", 3, 8, 5)
with col2:
    temp = st.slider("Temperature", 0.1, 1.5, 0.8, 0.1)

if st.button("🔮 Predict", type="primary"):
    if input_text.strip():
        with st.spinner("Predicting..."):
            sequence = tokenizer.texts_to_sequences([input_text])[0]
            padded = pad_sequences([sequence], maxlen=max_len, padding='pre')
            
            pred = model.predict(padded, verbose=0)[0]
            pred = np.log(pred + 1e-8) / temp
            pred = np.exp(pred) / np.sum(pred)
            
            top_idx = np.argsort(pred)[-num_pred:][::-1]
            preds = [(tokenizer.index_word.get(i, "?"), float(pred[i])) for i in top_idx if i != 0]
            
            st.session_state.preds = preds
            st.session_state.text = input_text

if 'preds' in st.session_state:
    st.subheader("Top Predictions")
    for word, prob in st.session_state.preds:
        full = f"{st.session_state.text} {word}"
        st.markdown(f"""
        <div style="padding:12px; background:#f0f2f6; border-radius:10px; margin:8px 0;">
            <span class="word-chip" onclick="navigator.clipboard.writeText('{full}')">{word}</span>
            <span style="float:right; color:#333;">{prob*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
