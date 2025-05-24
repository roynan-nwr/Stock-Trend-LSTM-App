# 📈 Stock Price Analyzer & LSTM Predictor

A Streamlit web app that fetches historical stock data using Yahoo Finance and predicts future stock prices using an LSTM (Long Short-Term Memory) neural network.

---

## 🚀 Features

* Fetch stock data by ticker and date range
* Normalize and preprocess data using MinMaxScaler
* Train an LSTM model on closing prices
* Visualize real vs predicted prices using Plotly
* Fully interactive Streamlit interface

---

## 🧰 Tech Stack

* **Python**
* **Streamlit**
* **YFinance** for stock data
* **TensorFlow/Keras** for LSTM model
* **scikit-learn** for data preprocessing
* **Plotly** for interactive plotting

---

## 🛠 Installation

1. **Clone the repo:**

```bash
git clone https://github.com/yourusername/stock-lstm-analyzer.git
cd stock-lstm-analyzer
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app locally:**

```bash
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push this code to a public GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click **"New App"** > Select your repo > Choose `app.py` > Click **Deploy**

---

## 📁 File Structure

```
.
├── app.py
├── requirements.txt
└── README.md
```

---

## 📈 Example Outputs

![Example Screenshot](link-to-your-screenshot-if-needed)

---

## ✨ Future Enhancements

* Add Prophet/ARIMA for comparison
* Incorporate technical indicators (MACD, RSI, etc.)
* Forecast future prices beyond the dataset

---

## 📬 Contact

For questions or collaborations: [your-email@example.com](mailto:your-email@example.com)

---

## 📝 License

MIT License. See `LICENSE` file for details.
