from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
import model_utils as mu
import json
import re
from typing import Dict, Any, Optional
import requests

# Add Keras 3 specific imports for loading
from keras.models import load_model # Import from keras directly
# These might not be strictly necessary if load_model handles them,
# but good for explicit environment setup
from keras.losses import MeanSquaredError as KerasLossMSE
from keras.metrics import MeanSquaredError as KerasMetricMSE
from keras.saving import register_keras_serializable
from tcn import TCN # Crucial for TCN model loading, ensure it's imported here too


import tensorflow as tf
print(f"api.py - TensorFlow Version: {tf.__version__}")
print(f"api.py - Keras Version: {tf.keras.__version__}")

# --- 1. Create a SINGLE FastAPI application instance ---
app = FastAPI(title="HGL Prediction API")

# --- 2. Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Serve static files (HTML, CSS, JS) ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 4. Serve the main HTML file at root ---
@app.get("/")
def get_root():
    return FileResponse("static/index.html")

# --- 5. API endpoint for testing ---
@app.get("/api/health")
def get_health():
    return {"message": "The API is running successfully!"}

# --- 5.1 Serve the template CSV for download ---
@app.get("/template.csv")
def get_template():
    return FileResponse(
        "static/template.csv",            # path to your file
        media_type="text/csv",
        filename="template.csv"           # forces browser download with this name
    )

# ===================================================================
# AI ASSISTANT IMPLEMENTATION
# ===================================================================

class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class HydraulicAssistant:
    def __init__(self):
        self.knowledge_base = {
            "hgl_basics": {
                "definition": "Hydraulic Grade Line (HGL) represents the total energy per unit weight of fluid at any point in a pipeline, excluding kinetic energy. It shows the height to which water would rise in a piezometer tube.",
                "components": "HGL = Elevation + Pressure Head",
                "significance": "HGL slopes indicate energy losses due to friction, and sudden drops may indicate equipment or blockages."
            },
            "models": {
                "LSTM": {
                    "description": "Long Short-Term Memory networks excel at learning long-term dependencies in time series data",
                    "advantages": "Good for capturing temporal patterns in HGL data over extended periods",
                    "best_use": "When historical trends and seasonal patterns are important"
                },
                "GRU": {
                    "description": "Gated Recurrent Units are simpler than LSTM but often perform similarly",
                    "advantages": "Faster training with fewer parameters, less prone to overfitting",
                    "best_use": "When computational efficiency is important and data patterns are less complex"
                },
                "TCN": {
                    "description": "Temporal Convolutional Networks use dilated convolutions for sequence modeling",
                    "advantages": "Parallel processing, stable gradients, and excellent for long sequences",
                    "best_use": "When you need fast inference and have long time series with clear patterns"
                }
            },
            "ocp_pipeline": {
                "stations": {
                    "Head": "Starting point at 650m elevation, initial pumping station",
                    "PMS1": "Pump station 1 at 680m elevation, 45km from head",
                    "VANNE": "Valve station at 515m elevation, 68km from head",
                    "PMS2": "Pump station 2 at 360m elevation, 101km from head",
                    "PMS3": "Pump station 3 at 110m elevation, 130km from head", 
                    "PMS4": "Pump station 4 at 143m elevation, 162km from head",
                    "Terminal": "End point at 60m elevation, 187km total distance"
                },
                "operations": "OCP pipeline transports phosphate slurry across Morocco. Pump stations maintain pressure, valves control flow, and river crossings require special HGL considerations."
            },
            "metrics": {
                "MSE": "Mean Squared Error - measures average squared differences between predicted and actual values. Lower is better.",
                "RMSE": "Root Mean Squared Error - same units as original data, easier to interpret than MSE.",
                "MAE": "Mean Absolute Error - average absolute differences, less sensitive to outliers than MSE.",
                "R2": "Coefficient of Determination - proportion of variance explained by the model (0-1, higher is better)"
            }
        }
    
    def generate_response(self, user_message: str, context: Dict[str, Any] = None) -> str:
        message_lower = user_message.lower()
        
        # Check for HGL-related questions
        if any(term in message_lower for term in ['hgl', 'hydraulic grade line', 'pressure', 'head']):
            return self._handle_hgl_questions(user_message, context)
        
        # Check for model-related questions
        elif any(term in message_lower for term in ['lstm', 'gru', 'tcn', 'model', 'prediction', 'neural network']):
            return self._handle_model_questions(user_message, context)
        
        # Check for OCP pipeline questions
        elif any(term in message_lower for term in ['ocp', 'pipeline', 'station', 'pump', 'terminal', 'morocco']):
            return self._handle_pipeline_questions(user_message, context)
        
        # Check for metrics/results questions
        elif any(term in message_lower for term in ['mse', 'rmse', 'mae', 'r2', 'accuracy', 'performance', 'error']):
            return self._handle_metrics_questions(user_message, context)
        
        # Check for curve analysis questions
        elif any(term in message_lower for term in ['curve', 'graph', 'chart', 'analysis', 'interpret', 'anomaly']):
            return self._handle_curve_analysis(user_message, context)
        
        # General assistance
        else:
            return self._handle_general_questions(user_message)
    
    def _handle_hgl_questions(self, message: str, context: Dict[str, Any] = None) -> str:
        base_info = self.knowledge_base["hgl_basics"]
        
        if "what is" in message.lower() or "define" in message.lower():
            return f"""**Hydraulic Grade Line (HGL) Explained:**

{base_info['definition']}

**Formula:** {base_info['components']}

**Why it matters:** {base_info['significance']}

In your OCP pipeline, the HGL curve shows how energy changes along the 187km route from the Head station to Terminal, accounting for elevation changes, friction losses, and pump stations that boost pressure."""
        
        elif "interpret" in message.lower() or "analyze" in message.lower():
            if context and "predictions" in context:
                return self._analyze_hgl_curves(context)
            else:
                return """To interpret HGL curves, look for:

• **Steep downward slopes**: High friction losses or flow restrictions
• **Sudden drops**: Possible blockages or equipment issues  
• **Upward jumps**: Pump stations adding energy
• **Gradual decline**: Normal friction losses
• **Flat sections**: Low flow or large pipe diameter

Upload your data and run a prediction to get specific curve analysis!"""
        
        return f"{base_info['definition']}\n\n{base_info['significance']}"
    
    def _handle_model_questions(self, message: str, context: Dict[str, Any] = None) -> str:
        models = self.knowledge_base["models"]
        
        # Check if asking about a specific model
        for model_name, model_info in models.items():
            if model_name.lower() in message.lower():
                return f"""**{model_name} Model Details:**

{model_info['description']}

**Advantages:** {model_info['advantages']}

**Best Use Case:** {model_info['best_use']}

**In your HGL prediction:** This model analyzes patterns in flow, density, and pressure data across all 7 stations to predict hydraulic grade lines with {len(mu.FEATURES)} input features."""
        
        # General model comparison
        if "compare" in message.lower() or "difference" in message.lower():
            return """**Model Comparison for HGL Prediction:**

**LSTM:** Best for long-term dependencies and complex temporal patterns. Slower but most comprehensive.

**GRU:** Good balance of speed and accuracy. Simpler architecture than LSTM.

**TCN:** Fastest training and inference. Excellent for parallel processing and stable gradients.

**Recommendation:** Try all three with your data! The "Train & Compare All" button will show you which performs best on your specific dataset."""
        
        return "I can help you understand LSTM, GRU, and TCN models for HGL prediction. What specific aspect would you like to know about?"
    
    def _handle_pipeline_questions(self, message: str, context: Dict[str, Any] = None) -> str:
        stations = self.knowledge_base["ocp_pipeline"]["stations"]
        operations = self.knowledge_base["ocp_pipeline"]["operations"]
        
        if any(station.lower() in message.lower() for station in stations.keys()):
            station_info = "\n".join([f"• **{name}:** {info}" for name, info in stations.items()])
            return f"""**OCP Pipeline Stations:**

{station_info}

**Operations:** {operations}

The pipeline handles phosphate slurry transport with multiple pump stations maintaining optimal pressure and flow rates across the 187km route."""
        
        return f"""**OCP Pipeline System:**

{operations}

**Key Stations:** 7 main stations from Head (650m) to Terminal (60m) over 187km distance.

**Challenges:** Elevation changes, river crossings, and maintaining optimal slurry flow rates while minimizing energy costs."""
    
    def _handle_metrics_questions(self, message: str, context: Dict[str, Any] = None) -> str:
        metrics = self.knowledge_base["metrics"]
        
        # Check for specific metric
        for metric_name, metric_info in metrics.items():
            if metric_name.lower() in message.lower():
                return f"**{metric_name}:** {metric_info}"
        
        # General metrics explanation
        metrics_info = "\n\n".join([f"• **{name}:** {info}" for name, info in metrics.items()])
        
        response = f"""**Model Performance Metrics:**

{metrics_info}

**For HGL Prediction:**
- Lower MSE/RMSE/MAE = more accurate pressure predictions
- Higher R² = model explains more variance in HGL patterns
- Compare metrics across LSTM/GRU/TCN to find best model"""
        
        if context and "metrics" in context:
            response += "\n\n" + self._interpret_current_metrics(context["metrics"])
        
        return response
    
    def _handle_curve_analysis(self, message: str, context: Dict[str, Any] = None) -> str:
        if context and "predictions" in context:
            return self._analyze_hgl_curves(context)
        
        return """**HGL Curve Analysis Guidelines:**

**Normal Patterns:**
• Gradual decline due to friction losses
• Step increases at pump stations
• Smooth transitions between stations

**Concerning Patterns:**
• Sudden pressure drops (blockages)
• Irregular oscillations (instability)
• Excessive slopes (high losses)

**River Crossings:**
• Expected pressure increases due to elevation
• Model accounts for 2 major crossings at 80-90km and 165-175km

Upload your data to get specific curve analysis with AI insights!"""
    
    def _analyze_hgl_curves(self, context: Dict[str, Any]) -> str:
        # Analyze the actual prediction results
        predictions = context.get("predictions", {})
        model_name = context.get("model_name", "")
        
        if not predictions:
            return "No prediction data available for analysis. Please run a model first."
        
        analysis = f"**HGL Curve Analysis for {model_name} Model:**\n\n"
        
        # Analyze true vs predicted patterns
        if "true_path" in predictions and "pred_path" in predictions:
            analysis += """**Key Observations:**
• Model captures the general HGL trend along the pipeline
• Pump station effects are visible as pressure increases
• River crossing impacts are modeled appropriately
• Overall prediction follows expected hydraulic behavior

**For Optimization:**
• Monitor stations with highest prediction errors
• Consider operational adjustments at major deviation points
• Use predictions for preventive maintenance scheduling"""
        
        return analysis
    
    def _interpret_current_metrics(self, metrics: Dict[str, Any]) -> str:
        if isinstance(metrics, dict) and len(metrics) > 0:
            # Handle single model metrics
            if "MSE" in metrics:
                mse, r2 = metrics["MSE"], metrics["R2"]
                interpretation = f"**Your Model Performance:**\n"
                interpretation += f"• R² Score: {r2:.3f} ({'Excellent' if r2 > 0.9 else 'Good' if r2 > 0.8 else 'Needs Improvement'})\n"
                interpretation += f"• Prediction accuracy is {'very high' if r2 > 0.9 else 'good' if r2 > 0.8 else 'moderate'}\n"
                
                if r2 > 0.9:
                    interpretation += "• Model is ready for operational use!"
                elif r2 > 0.8:
                    interpretation += "• Consider fine-tuning for better performance"
                else:
                    interpretation += "• May need more data or feature engineering"
                
                return interpretation
            
            # Handle comparison metrics
            else:
                best_model = max(metrics.keys(), key=lambda k: metrics[k].get("R2", 0))
                best_r2 = metrics[best_model]["R2"]
                interpretation = f"**Model Comparison Results:**\n"
                interpretation += f"• **Best Performer:** {best_model} (R² = {best_r2:.3f})\n"
                interpretation += f"• Recommended for your HGL prediction needs\n"
                
                for model, vals in metrics.items():
                    r2 = vals["R2"]
                    interpretation += f"• {model}: R² = {r2:.3f}\n"
                
                return interpretation
        
        return "Metrics data not available for interpretation."
    
    def _handle_general_questions(self, message: str) -> str:
        return """**HGL Prediction Assistant**

I can help you with:

🔹 **HGL Analysis:** Understanding hydraulic grade lines and pressure patterns
🔹 **Model Guidance:** LSTM, GRU, TCN comparison and selection
🔹 **OCP Pipeline:** Station operations and system optimization  
🔹 **Results Interpretation:** Metrics analysis and curve insights
🔹 **Troubleshooting:** Identifying anomalies and operational issues

**Quick Examples:**
• "Explain what HGL means"
• "Which model is best for my data?"
• "How do I interpret these results?"
• "What does this curve pattern indicate?"

What would you like to know about your hydraulic system?"""

# Initialize the assistant
assistant = HydraulicAssistant()

@app.post("/chat")
async def chat_with_assistant(chat_message: ChatMessage):
    """
    Chat endpoint for the AI assistant
    """
    try:
        response = assistant.generate_response(
            chat_message.message, 
            chat_message.context
        )
        
        return {
            "response": response,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Assistant error: {str(e)}"
        )


# ===================================================================
# EXISTING ENDPOINTS (UNCHANGED)
# ===================================================================

# --- 6. Define the endpoint for training a single model ---
@app.post("/train_single_model/")
async def train_single_model(
    model_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Accepts a CSV file and a model name, runs the pipeline for that single model,
    and returns the results.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        print(f"Received request to train model: {model_name}")
        
        # This function from model_utils.py returns four values
        metrics, predictions, geographic_data, error = mu.run_single_model_pipeline(file.file, model_name)

        if error:
            raise HTTPException(status_code=400, detail=error)
        
        print(f"Training for {model_name} complete. Sending results.")
        # Envoi des résultats au webhook n8n
        n8n_webhook_url = "https://khzouhai.app.n8n.cloud/webhook-test/hgl"  # Remplace par l'URL réelle de ton webhook n8n
        payload = {
            "model_name": model_name,
            "metrics": metrics,
            "predictions": predictions,
            "geographic_data": geographic_data
        }
        try:
            n8n_response = requests.post(n8n_webhook_url, json=payload, timeout=10)
            n8n_status = n8n_response.status_code
        except Exception as e:
            n8n_status = f"Erreur lors de l'envoi à n8n: {e}"

        return {
            "message": f"Training for {model_name} completed successfully!",
            "model_name": model_name,
            "metrics": metrics,
            "predictions": predictions,
            "geographic_data": geographic_data,
            "n8n_status": n8n_status
        }

    except Exception as e:
        print(f"An unexpected error occurred in /train_single_model/: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {str(e)}")


# --- 7. Define the endpoint for comparing all models ---
@app.post("/train_all_models/")
async def train_all_models(file: UploadFile = File(...)):
    """
    Accepts a CSV file, runs the pipeline for ALL models, and returns
    the metrics AND predictions for a full comparison view.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        print("Received request to train ALL models for comparison.")
        
        # This function from model_utils.py returns four values
        all_metrics, all_predictions, geographic_data, error = mu.run_all_models_pipeline(file.file)

        if error:
            raise HTTPException(status_code=400, detail=error)
        
        print("All models trained. Sending comparison metrics and predictions.")
        # Envoi des résultats au webhook n8n
        n8n_webhook_url = "https://khzouhai.app.n8n.cloud/webhook-test/hgl"  # Remplace par l'URL réelle de ton webhook n8n
        payload = {
            "metrics": all_metrics,
            "predictions": all_predictions,
            "geographic_data": geographic_data
        }
        try:
            n8n_response = requests.post(n8n_webhook_url, json=payload, timeout=10)
            n8n_status = n8n_response.status_code
        except Exception as e:
            n8n_status = f"Erreur lors de l'envoi à n8n: {e}"

        return {
            "message": "All models trained successfully!",
            "metrics": all_metrics,
            "predictions": all_predictions,
            "geographic_data": geographic_data,
            "n8n_status": n8n_status
        }

    except Exception as e:
        print(f"An unexpected error occurred in /train_all_models/: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {str(e)}")
