# 🔮 Next Word Prediction — LSTM + Streamlit

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-7c3aed?style=for-the-badge)](LICENSE)

A deep learning–powered Next Word Prediction app built with an **LSTM model** and a sleek **Streamlit** frontend. Type a seed sentence, get top-N word predictions, click to build sentences interactively, or let the model auto-complete for you.

---

<img width="1898" height="862" alt="Screenshot 2026-05-04 162446" src="https://github.com/user-attachments/assets/7b1e29b2-e072-49e1-a009-8540b8811881" />

<img width="1899" height="856" alt="Screenshot 2026-05-04 162520" src="https://github.com/user-attachments/assets/730c0516-201e-4ac7-9e55-30fbfcae18a6" />

<img width="1862" height="843" alt="Screenshot 2026-05-04 162626" src="https://github.com/user-attachments/assets/74f1ab15-3565-4a76-9c27-921de6dc0fc8" />





## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Top-N Predictions** | Adjustable slider — get 1 to 10 candidate next words with confidence % |
| 🖱️ **Interactive Chips** | Click any predicted word to append it and re-predict on the fly |
| ✍️ **Sentence Builder** | Seed text highlighted separately from AI-generated additions |
| ✨ **Auto-Complete Mode** | Generate full sentences automatically from a seed phrase |
| 🎨 **Modern Dark UI** | Custom CSS — glassmorphism, gradient typography, animated buttons |

---

## 📁 Project Structure

```
next-word-prediction/
│
├── app.py                  # Streamlit application (main UI)
├── lstm_model.h5           # Trained LSTM model weights
├── tokenizer.pkl           # Fitted Keras tokenizer
├── max_len.pkl             # Max sequence length used during training
├── requirements.txt        # Python dependencies
├── train.ipynb             # (Optional) Training notebook
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Yugal0708/next-word-prediction.git
cd next-word-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📦 Requirements

```
streamlit>=1.28.0
tensorflow>=2.10.0
numpy>=1.23.0
```

> Save this as `requirements.txt` in your repo root.

---

## 🧠 Model Architecture

```
Input Sequence
      ↓
Embedding Layer
      ↓
LSTM Layer(s)
      ↓
Dense (Softmax)
      ↓
Vocabulary Probabilities
```

- **Model type:** LSTM (Long Short-Term Memory)
- **Input:** Tokenized & padded text sequences
- **Output:** Probability distribution over vocabulary
- **Training data:** Custom text corpus (Shakespeare / custom dataset)

---

## 🖥️ App Preview

```
╔══════════════════════════════════════════════════╗
║  🔮 Next Word Predictor                         ║
║  LSTM · Deep Learning · NLP                     ║
║                                                  ║
║  Seed text: [The quick brown ________]  Top N:5 ║
║  ⚡ Predict Next Word                            ║
║                                                  ║
║  Predicted Words:                               ║
║  [fox 34.2%]  [dog 18.1%]  [cat 12.4%] ...     ║
║                                                  ║
║  Built Sentence:                                ║
║  The quick brown *fox jumps over*               ║
╚══════════════════════════════════════════════════╝
```

---

## 📌 How to Use

1. **Enter seed text** — type any partial sentence in the input box
2. **Set Top-N** — use the slider to choose how many word suggestions to show
3. **Click "⚡ Predict"** — model generates the top word candidates
4. **Click a word chip** — appends it to the sentence and re-predicts automatically
5. **Auto-Complete tab** — enter a seed + word count → full sentence generated

---

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras LSTM
- **Frontend:** Streamlit + Custom CSS
- **NLP:** Keras Tokenizer, Sequence Padding
- **Fonts:** Google Fonts (Syne + Space Mono)

---

## 👤 Author

**Yugal**
- GitHub: [@Yugal0708](https://github.com/Yugal0708)
- Google Student Ambassador 🎓
- IBM SkillBuild AI & ML Intern

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Built with 🔮 by Yugal &nbsp;·&nbsp; LSTM + Streamlit &nbsp;·&nbsp; Nagpur, India
</p>






