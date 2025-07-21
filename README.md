
##  MM-CrowdNet: A Multimodal Crowd Risk Prediction System

MM-CrowdNet is a deep learning framework for real-time crowd risk classification, combining visual analysis of surveillance videos with social sentiment from Reddit. The model uses CNNs, LSTM with attention, and sentiment-aware fusion to predict crowd risk levels (Low, Medium, High, Critical).

---

###  Project Structure

```
MM-CrowdNet/
├── .vscode/
├── data/
├── models/
├── scripts/
├── venv/
├── videos/
├── .env
├── main.py
├── requirements.txt
```

---

###  Quick Start

```bash
git clone https://github.com/yourusername/MM-CrowdNet.git
cd MM-CrowdNet

python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows

pip install -r requirements.txt

python main.py
```

---
## 📦 Dataset Access

The full preprocessed **HAJJv2 dataset** used for training and evaluation is too large for GitHub.

🔗 You can download it here:  
👉 [Download HAJJv2 Processed Dataset (Google Drive)](https://drive.google.com/drive/folders/1yCLG3siw1c0pvIn03XoPkT99JaxutKKL?usp=drive_link)]

###  Core Components

* CNN Feature Extraction: ResNet50 + MobileNetV2
* Temporal Modeling: LSTM with Attention
* Sentiment Analysis: Reddit posts via BERT
* Fusion Logic: Weighted adjustment of video-based predictions using sentiment scores

---

###  Output

* Frame-wise predictions of crowd risk levels
* Adjusted results using public sentiment
* Saved models and evaluation metrics in `/models` and `/data`

---

###  License

MIT License

---


