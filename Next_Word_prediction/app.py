import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

st.set_page_config(
    page_title="NextWord AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
        font-family: 'Consolas', monospace;
    }
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .word-chip {
        display: inline-block;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 8px 18px;
        margin: 5px;
        border-radius: 25px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
    }
    .word-chip:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ====================== LOAD MODEL & FILES ======================
@st.cache_resource
def load_model_and_files():
    model = tf.keras.models.load_model("lstm_model (1).h5")
    
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    
    return model, tokenizer, max_len

try:
    model, tokenizer, max_len = load_model_and_files()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/files: {e}")
    st.stop()

# ====================== UI ======================
st.markdown('<h1 class="main-header">✨ NextWord AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent LSTM-powered next word prediction</p>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    input_text = st.text_area(
        "Type something...",
        placeholder="The future of artificial intelligence is",
        height=150,
        help="Write a few words and let AI complete your sentence"
    )

with col2:
    st.markdown("### Settings")
    num_predictions = st.slider("Number of suggestions", 3, 10, 5)
    temperature = st.slider("Creativity (Temperature)", 0.1, 1.5, 0.7, 0.1)
    
    if st.button("🔮 Predict Next Words", type="primary", use_container_width=True):
        if input_text.strip():
            with st.spinner("Thinking..."):
                # Preprocess
                sequence = tokenizer.texts_to_sequences([input_text])[0]
                padded = pad_sequences([sequence], maxlen=max_len, padding='pre')
                
                # Predict
                pred = model.predict(padded, verbose=0)[0]
                
                # Apply temperature
                pred = np.log(pred + 1e-8) / temperature
                exp_pred = np.exp(pred)
                pred = exp_pred / np.sum(exp_pred)
                
                # Get top predictions
                top_indices = np.argsort(pred)[-num_predictions:][::-1]
                top_words = [(tokenizer.index_word.get(i, "<UNK>"), float(pred[i])) 
                           for i in top_indices if i != 0]
                
                # Store in session state
                st.session_state.predictions = top_words
                st.session_state.input_text = input_text
        else:
            st.warning("Please enter some text")

# Show predictions
if 'predictions' in st.session_state:
    st.markdown("### 🎯 Top Predictions")
    
    for word, prob in st.session_state.predictions:
        prob_percent = prob * 100
        st.markdown(f"""
        <div class="prediction-card">
            <span class="word-chip" onclick="navigator.clipboard.writeText('{st.session_state.input_text} {word}')">
                {word}
            </span>
            <span style="float:right; margin-top:8px; color:#666; font-size:0.9rem;">
                {prob_percent:.1f}% confidence
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Full sentence examples
    st.markdown("### 📝 Full Sentence Examples")
    for word, _ in st.session_state.predictions[:3]:
        full_sentence = f"{st.session_state.input_text} **{word}**..."
        st.info(full_sentence)

# Sidebar
with st.sidebar:
    st.header("📊 Model Info")
    st.write("**Model:** LSTM")
    st.write("**Vocabulary Size:**", len(tokenizer.word_index))
    st.write("**Max Sequence Length:**", max_len)
    
    st.divider()
    st.markdown("### How to use")
    st.markdown("""
    1. Type a few words
    2. Click **Predict**
    3. Click on any word chip to copy the full sentence
    """)
    
    st.divider()
    st.caption("Built with ❤️ using Streamlit + TensorFlow")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#888;'>NextWord AI • Authored by Yugal Bilawane </p>", unsafe_allow_html=True)