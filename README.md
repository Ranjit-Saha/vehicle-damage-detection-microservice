# 🚗 Vehicle Damage Detection System (Decoupled Microservice)

<div align="center">

[![Live Web Application](https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://vehicle-damage-detection-microservice.streamlit.app/)
[![API Backend](https://img.shields.io/badge/API_Backend-Hugging_Face_Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](#)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](#)

*A production-grade Computer Vision pipeline utilizing a decoupled architecture. A FastAPI PyTorch backend serves asynchronous inference to a lightweight Streamlit client.*

</div>

---
## 📷 Application Preview

![Vehicle Damage Detection System Live Demo](frontend/assets/demo.png)  
*Figure 1: Reactive UI displaying seamless digital asset upload and remote API classification results.*

---

## 🏗️ System Architecture: True Decoupling

Standard AI portfolio projects often rely on monolithic scripts that crash under concurrent user load. This system is engineered as a robust, distributed microservice, demonstrating industry-standard separation of concerns.

| Component | Responsibility | Technology Stack | Hosting Environment |
| :--- | :--- | :--- | :--- |
| **Client UI (`frontend/`)** | Renders UI, handles asset uploads, and manages network requests. Contains zero machine learning logic. | Streamlit, Requests, python-dotenv | Streamlit Cloud |
| **Inference Engine (`backend/`)** | Caches a fine-tuned ResNet50 model in global memory to asynchronously process tensor matrices. | FastAPI, Uvicorn, PyTorch, PIL | Hugging Face Spaces (Docker) |

### ✨ Engineering Features
- **Deep Learning Backbone:** Transfer-learned PyTorch ResNet50 with a Softmax distribution layer for exact mathematical confidence scoring.
- **Zero-Disk Overhead:** Atomic memory streaming processes binary uploads directly in RAM, eliminating I/O bottlenecks and local storage leakage.
- **Failure-Aware Networking:** The frontend client utilizes strict HTTP timeouts, JSON structure validation, and specific exception handling to fail gracefully during server cold-starts.
- **Runtime Security Configuration:** Hardcoded configurations are stripped from the source code. API URLs are injected dynamically at runtime via `.env` files and Streamlit Secrets.

---

## 📂 Monorepo Structure

```text
vehicle-damage-detection/
│
├── backend/                        # 🧠 Deep Learning API Service
│   ├── model/
│   │   └── saved_model.pth         # Cached PyTorch weights (Ignored in Git)
│   ├── server.py                   # FastAPI Routing Engine
│   ├── model_helper.py             # Tensor pre-processing & Softmax logic
│   ├── Dockerfile                  # Linux Container instructions
│   └── requirements.txt            # Isolated Backend Dependencies
│
├── frontend/                       # 🖥️ User Interface Service
│   ├── assets/                     # Application Preview
│   ├── app.py                      # Streamlit UI & Network Client
│   └── requirements.txt            # Isolated Frontend Dependencies
│
├── .gitignore                      # Security and cache exclusions
└── README.md                       # Master Documentation
```
---
## 💻 Local Developer Setup
To run this microservice architecture locally, you must initialize both nodes in separate terminals and configure your local environment variables.

**1. Initialize the Backend API (Terminal 1)** <br>
Navigate to the backend directory, install dependencies, and boot the Uvicorn server. <br>
*(Note: You must manually download the `saved_model.pth` file from the **[Hugging Face Model Registry](https://huggingface.co/spaces/GeospatialAI/vehicle-damage-detection-system/tree/main)** and place it inside the `backend/model/` directory before booting).*
```bash 
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

# Boot the API server on localhost:8000
uvicorn server:app --host 127.0.0.1 --port 8000
```

**2. Configure Environment Variables** <br>
Inside the `frontend/` directory, create a `.env` file to securely route your network requests.

```bash 
# frontend/.env
API_URL=http://127.0.0.1:8000/predict 
```

(Note: For production deployment, update this URL to your live Hugging Face `.hf.space` endpoint).


**3. Initialize the Frontend UI (Terminal 2)** <br>
Open a new terminal window, navigate to the frontend directory, and boot the Streamlit client:

```bash 
cd frontend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

# Boot the UI client
streamlit run app.py
```
---
## 👨‍💻 Author

**Ranjit Saha** <br> *Geo-Computational Product Engineer* <br>
Building high-performance spatial computation engines and automated risk assessment pipelines.

---
---