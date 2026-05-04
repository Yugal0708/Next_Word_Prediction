import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# PAGE CONFIG 
st.set_page_config(
    page_title="NextWord AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS 
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
    .sub-header { text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem; }
    .word-chip {
        display: inline-block;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 10px 20px;
        margin: 6px;
        border-radius: 25px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
    }
    .word-chip:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# LOAD MODEL & FILES 
@st.cache_resource
def load_model_and_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(current_dir, "lstm_model (1).h5")
    tokenizer_path = os.path.join(current_dir, "tokenizer.pkl")
    maxlen_path = os.path.join(current_dir, "max_len.pkl")
    
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.write("📁 Files in current directory:", os.listdir(current_dir))
            st.stop()
        
        model = tf.keras.models.load_model(model_path)
        
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        
        with open(maxlen_path, "rb") as f:
            max_len = pickle.load(f)
        
        return model, tokenizer, max_len, "✅ Model loaded successfully!"
        
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

# Load the model
model, tokenizer, max_len, status = load_model_and_files()
st.success(status)

#  MAIN UI 
st.markdown('<h1 class="main-header">✨ NextWord AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent LSTM-powered next word prediction</p>', unsafe_allow_html=True)

# Sidebar Info 
with st.sidebar:
    st.header("📊 Model Information")
    st.write("**Vocabulary Size:**", len(tokenizer.word_index))
    st.write("**Max Sequence Length:**", max_len)
    st.divider()
    st.caption("Built with ❤️ using Streamlit + TensorFlow")

# Main Content
col1, col2 = st.columns([3, 1])

with col1:
    input_text = st.text_area(
        "Type something...",
        placeholder="The future of artificial intelligence is",
        height=150
    )

with col2:
    st.markdown("### ⚙️ Settings")
    num_predictions = st.slider("Number of suggestions", 3, 10, 5)
    temperature = st.slider("Creativity (Temperature)", 0.1, 1.5, 0.7, 0.1)
    
    predict_button = st.button("🔮 Predict Next Words", type="primary", use_container_width=True)

# = PREDICTION LOGIC 
if predict_button and input_text.strip():
    with st.spinner("Thinking..."):
        try:
            sequence = tokenizer.texts_to_sequences([input_text])[0]
            padded = pad_sequences([sequence], maxlen=max_len, padding='pre')
            
            pred = model.predict(padded, verbose=0)[0]
            
            # Apply temperature
            pred = np.log(pred + 1e-8) / temperature
            exp_pred = np.exp(pred)
            pred = exp_pred / np.sum(exp_pred)
            
            # Top predictions
            top_indices = np.argsort(pred)[-num_predictions:][::-1]
            predictions = [(tokenizer.index_word.get(i, "<UNK>"), float(pred[i])) 
                         for i in top_indices if i != 0]
            
            st.session_state.predictions = predictions
            st.session_state.input_text = input_text
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Show Predictions
if 'predictions' in st.session_state:
    st.markdown("### 🎯 Top Next Word Predictions")
    
    for word, prob in st.session_state.predictions:
        prob_percent = prob * 100
        full_sentence = f"{st.session_state.input_text} {word}"
        
        st.markdown(f"""
        <div style="background:#f8f9fa; padding:1.2rem; border-radius:12px; margin:8px 0;">
            <span class="word-chip" onclick="navigator.clipboard.writeText('{full_sentence}')">{word}</span>
            <span style="float:right; margin-top:8px; color:#555;">{prob_percent:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📝 Example Sentences")
    for word, _ in st.session_state.predictions[:3]:
        st.info(f"{st.session_state.input_text} **{word}**...")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#888;'>NextWord AI • Author by Yugal</p>", unsafe_allow_html=True)
