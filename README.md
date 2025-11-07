
# Movie Recommender System with Explainability

A movie recommendation system built in Python with machine learning, explainability and an interactive Streamlit interface.

---

## Project Structure

```
movie_recommender
├── main.py               # 🎯 Entry point — orchestrates the entire recommendation pipeline
├── app.py                # ✅ Visual interface with Streamlit
├── core/data_loader.py        # 📦 Loads and prepares rated and unrated movie data
├── core/model.py              # 🧠 Defines, trains, and returns AI models
├── core/prediction.py         # 🔮 Generates rating predictions for unrated movies
├── core/evaluation.py         # 📊 Evaluates model performance using cross-validation and metrics (e.g., MSE, R2)
├── core/recommendation.py     # 🎥 Selects and returns the top movie recommendations
├── core/explainability.py     # 🧩 Explainability tools (KernelExplainer, plots, etc.)
├── core/utils.py              # 🛠️ Utility/helper functions 
├── requirements.txt      # 📦 Python dependencies (for pip install)
└── README.md             # 📖 Project overview and instructions
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/FranciscoOliveira55/MovieRecommendationSystemWithExplainability.git
cd MovieRecommendationSystemWithExplainability
```
---
2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv1
```
On Windows:
```bash
.venv1\Scripts\activate
```
On Linux/macOS:
```bash
.source .venv1/bin/activate
```
---

3. Install Torch with Cuda

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```
---

4. Install dependencies:

```bash
pip install -r requirements.txt
```



## Usage

To run the Streamlit app:

```bash
streamlit run app.py
```

---


## Contact

- Francisco Oliveira — oliveira2000francisco@gmail.com