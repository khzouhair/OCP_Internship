# OCP_Internship# HGL Prediction API

This folder contains the main source code for the Hydraulic Grade Line (HGL) prediction project on the OCP pipeline network.

## Main Features

- **REST API** built with FastAPI for:
  - Training deep learning models (LSTM, GRU, TCN) from CSV files.
  - Automatic model comparison.
  - Generating and analyzing HGL predictions.
  - An intelligent assistant (chatbot) to answer questions about HGL, models, the pipeline, etc.
  - Automatic result delivery to n8n via webhook for automation and reporting.
- **Web interface** (in the `static/` folder) for user interaction.
- **Static file management** (HTML, CSS, JS, CSV template).

## Folder Structure

- `api.py`: Main FastAPI entry point, endpoints for training, comparison, chatbot, and n8n integration.
- `model_utils.py`: Utility functions for data preparation, model building and evaluation, HGL point generation, etc.
- `static/`: Static files for the web interface and CSV template.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn api:app --reload
   ```

3. Open the web interface:
   - Go to [http://localhost:8000](http://localhost:8000) in your browser.

## Main Endpoints

- `/`: Main web interface.
- `/api/health`: API health check.
- `/template.csv`: Download the CSV template.
- `/train_single_model/`: Train a single model on a CSV file.
- `/train_all_models/`: Compare all models on a CSV file.
- `/chat`: Intelligent assistant for HGL, models, etc.

## Automation with n8n

Prediction results are automatically sent to an n8n webhook for automation, reporting, or integration with other industrial tools.

## Requirements

- Python 3.8+
- FastAPI, Uvicorn, Pandas, NumPy, scikit-learn, TensorFlow/Keras, tcn, requests, pydantic

## Example Usage

```bash
uvicorn api:app --reload
```

Then use the web interface or API endpoints to train models, get predictions, or interact with the chatbot.

---

**Authors:**  
[Khadija Zouhair](https://github.com/khzouhair)  
[Chimae Hourri](https://github.com/chourri)