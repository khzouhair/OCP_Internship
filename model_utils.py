import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# REMOVE 'tensorflow as tf' if not strictly needed elsewhere,
# or be very careful how you use it.
# From Keras 3.x, you should explicitly import from 'keras' directly.
# import tensorflow as tf # Keep if you use tf.whatever, but prefer keras.*

# Import Keras 3 modules directly
from keras.models import Sequential, load_model # Import load_model from keras directly
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.losses import MeanSquaredError as KerasLossMSE # Alias to avoid confusion
from keras.metrics import MeanSquaredError as KerasMetricMSE # Alias to avoid confusion
from keras.saving import register_keras_serializable # For custom components if needed

from tcn import TCN

import tensorflow as tf
print(f"model_utils.py - TensorFlow Version: {tf.__version__}")
print(f"model_utils.py - Keras Version: {tf.keras.__version__}")
# ===================================================================
# CONFIGURATION
# ===================================================================
STATIONS = {
    'Head': {'elevation_m': 650, 'distance_km': 0},
    'PMS1': {'elevation_m': 680, 'distance_km': 45},
    'VANNE': {'elevation_m': 515, 'distance_km': 68},
    'PMS2': {'elevation_m': 360, 'distance_km': 101},
    'PMS3': {'elevation_m': 110, 'distance_km': 130},
    'PMS4': {'elevation_m': 143, 'distance_km': 162},
    'Terminal': {'elevation_m': 60, 'distance_km': 187}
}

RIVER_CROSSINGS = [
    {"start_km": 80, "end_km": 90, "rise_height_m": 100},
    {"start_km": 165, "end_km": 175, "rise_height_m": 80}
]

FEATURES = [
    'Flow_HS','Density_HS_Average','Pressure_HS_Average', 'Pressure_PMS1','Flow_PMS1',
    'Density_PMS1', 'Pressure_VANNE','Flow_VANNE','Density_VANNE', 'Pressure_PMS2',
    'Flow_PMS2','Density_PMS2', 'Pressure_PMS3','Flow_PMS3','Density_PMS3',
    'Pressure_PMS4','Flow_PMS4','Density_PMS4', 'Pressure_T','Flow_T','Density_T'
]
TARGETS = ['HGL_Head','HGL_PMS1','HGL_VANNE','HGL_PMS2',
           'HGL_PMS3','HGL_PMS4','HGL_Terminal']

TIME_STEPS = 10
EPOCHS = 50
BATCH_SIZE = 32
TEST_SIZE = 0.2

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def generate_ground_profile(stations, num_points=200):
    ordered_names = [t.replace('HGL_', '') for t in TARGETS]
    distances = np.array([stations[s]['distance_km'] for s in ordered_names])
    elevations = np.array([stations[s]['elevation_m'] for s in ordered_names])
    dist_interp = np.linspace(min(distances), max(distances), num_points)
    elev_interp = np.interp(dist_interp, distances, elevations)
    return [{'x': float(d), 'y': float(e)} for d, e in zip(dist_interp, elev_interp)]

def generate_hgl_path(hgl_values_at_stations, stations, rivers):
    station_names = [t.replace('HGL_', '') for t in TARGETS]
    station_distances = [stations[s]['distance_km'] for s in station_names]
    
    hgl_path_points = []
    
    for i in range(len(station_distances) - 1):
        x0, y0 = station_distances[i], float(hgl_values_at_stations[i])  # Convert to float
        x1, y1 = station_distances[i+1], float(hgl_values_at_stations[i+1])  # Convert to float

        # Add start point of the segment
        hgl_path_points.append({'x': float(x0), 'y': float(y0)})

        # Check for rivers within this segment
        segment_rivers = [r for r in rivers if r["start_km"] >= x0 and r["end_km"] <= x1]
        
        # Sort rivers by distance to process them in order
        segment_rivers.sort(key=lambda r: r["start_km"])

        last_x = x0
        last_y = y0

        for river in segment_rivers:
            # Interpolate to find the HGL level at the start of the river
            base_start = float(np.interp(river["start_km"], [last_x, x1], [last_y, y1]))  # Convert to float
            # Interpolate to find the HGL level at the end of the river
            base_end = float(np.interp(river["end_km"], [last_x, x1], [last_y, y1]))  # Convert to float
            rise = river["rise_height_m"]

            # Add points for the rectangle: line to start, up, across, down
            hgl_path_points.append({'x': float(river["start_km"]), 'y': float(base_start)})
            hgl_path_points.append({'x': float(river["start_km"]), 'y': float(base_start + rise)})
            hgl_path_points.append({'x': float(river["end_km"]), 'y': float(base_end + rise)})
            hgl_path_points.append({'x': float(river["end_km"]), 'y': float(base_end)})

            last_x = river["end_km"]
            last_y = base_end
            
    # Add the final station point
    hgl_path_points.append({'x': float(station_distances[-1]), 'y': float(hgl_values_at_stations[-1])})
    
    return hgl_path_points

def get_geographic_data():
    ground_profile = generate_ground_profile(STATIONS)
    station_distances = [STATIONS[t.replace('HGL_', '')]['distance_km'] for t in TARGETS]
    return { 
        "ground_profile": ground_profile, 
        "station_distances": station_distances, 
        "river_crossings": RIVER_CROSSINGS 
    }

# ===================================================================
# DATA PREPARATION FUNCTION
# ===================================================================
def load_and_prepare_data(uploaded_file, features, targets, time_steps, test_size):
    """Loads data from an in-memory file, scales, creates sequences, and splits it."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, None, None, None, None, f"Error reading CSV file. Please ensure it is a valid, comma-separated CSV. Details: {e}"

    df = df.dropna().reset_index(drop=True)
    
    # --- THIS IS THE CRUCIAL GATEKEEPER LOGIC ---
    required_cols = features + targets
    actual_cols = df.columns
    
    missing_cols = [col for col in required_cols if col not in actual_cols]
    if missing_cols:
        # If any columns are missing, stop immediately and return a specific error message.
        error_message = f"Error: The uploaded CSV is missing the following required columns: {', '.join(missing_cols)}"
        return None, None, None, None, None, error_message

    # If all checks pass, we can safely proceed.
    X = df[features].values
    Y = df[targets].values
    
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_Y = MinMaxScaler()
    Y_scaled = scaler_Y.fit_transform(Y)

    def create_sequences(X_data, Y_data, ts):
        Xs, Ys = [], []
        for i in range(len(X_data) - ts):
            Xs.append(X_data[i:(i + ts)])
            Ys.append(Y_data[i + ts])
        return np.array(Xs), np.array(Ys)

    X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, time_steps)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_seq, Y_seq, test_size=test_size, shuffle=False
    )
    
    return X_train, X_test, Y_train, Y_test, scaler_Y, None

# ===================================================================
# MODEL BUILDING FUNCTIONS
# ===================================================================
def build_lstm_model(input_shape, output_size):
    model = Sequential([LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape), Dropout(0.2),
                        LSTM(64, activation='tanh'), Dense(32, activation='relu'), Dense(output_size)])
    # Use the Keras 3 imports
    model.compile(optimizer='adam', loss=KerasLossMSE(), metrics=[KerasMetricMSE()])
    return model

def build_gru_model(input_shape, output_size):
    model = Sequential([GRU(64, activation='tanh', return_sequences=True, input_shape=input_shape), Dropout(0.2),
                        GRU(64, activation='tanh'), Dense(32, activation='relu'), Dense(output_size)])
    # Use the Keras 3 imports
    model.compile(optimizer='adam', loss=KerasLossMSE(), metrics=[KerasMetricMSE()])
    return model

def build_tcn_model(input_shape, output_size):
    model = Sequential([TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], dropout_rate=0.2, input_shape=input_shape),
                        Dense(32, activation='relu'), Dense(output_size)])
    # Use the Keras 3 imports
    model.compile(optimizer='adam', loss=KerasLossMSE(), metrics=[KerasMetricMSE()])
    return model

# ===================================================================
# MASTER FUNCTION FOR A SINGLE MODEL
# ===================================================================
def run_single_model_pipeline(uploaded_file, model_name):
    X_train, X_test, Y_train, Y_test, scaler_Y, error = load_and_prepare_data(uploaded_file, FEATURES, TARGETS, TIME_STEPS, TEST_SIZE)
    if error: return None, None, None, error

    model_builders = {"LSTM": build_lstm_model, "GRU": build_gru_model, "TCN": build_tcn_model}
    if model_name not in model_builders: return None, None, None, f"Invalid model name '{model_name}'."

    build_func = model_builders[model_name]
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_size = len(TARGETS)
    
    import os
    from tensorflow.keras.models import load_model

    os.makedirs("saved_models", exist_ok=True)
    if model_name in ["LSTM", "GRU"]:
        model_path = f"saved_models/{model_name}_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            model = build_func(input_shape, output_size)
            model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), verbose=0)
            model.save(model_path)
    elif model_name == "TCN":
        model = build_func(input_shape, output_size)
        weights_path = f"saved_models/{model_name}.weights.h5"
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
        else:
            model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), verbose=0)
            model.save_weights(weights_path)

    Y_pred_scaled = model.predict(X_test)
    y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    Y_test_original = scaler_Y.inverse_transform(Y_test)

    metrics = {
        "MSE": float(mean_squared_error(Y_test_original, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(Y_test_original, y_pred))),
        "MAE": float(mean_absolute_error(Y_test_original, y_pred)),
        "R2": float(r2_score(Y_test_original, y_pred))
    }

    predictions = {
        'true_path': generate_hgl_path(Y_test_original[0], STATIONS, RIVER_CROSSINGS),
        'pred_path': generate_hgl_path(y_pred[0], STATIONS, RIVER_CROSSINGS)
    }

    geographic_data = get_geographic_data()

    return convert_numpy_types(metrics), convert_numpy_types(predictions), convert_numpy_types(geographic_data), None

# ===================================================================
# MASTER FUNCTION TO TRAIN ALL MODELS FOR COMPARISON
# ===================================================================
def run_all_models_pipeline(uploaded_file):
    X_train, X_test, Y_train, Y_test, scaler_Y, error = load_and_prepare_data(uploaded_file, FEATURES, TARGETS, TIME_STEPS, TEST_SIZE)
    if error: return None, None, None, error

    model_builders = {"LSTM": build_lstm_model, "GRU": build_gru_model, "TCN": build_tcn_model}
    all_metrics = {}
    all_predictions = {}
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_size = len(TARGETS)
    Y_test_original = scaler_Y.inverse_transform(Y_test)

    import os
    from tensorflow.keras.models import load_model

    os.makedirs("saved_models", exist_ok=True)
    for name, build_func in model_builders.items():
        print(f"--- {name} Model for comparison ---")
        if name in ["LSTM", "GRU"]:
            model_path = f"saved_models/{name}_model.h5"
            if os.path.exists(model_path):
                print(f"Chargement du modèle {name} depuis {model_path}...")
                model = load_model(model_path)
            else:
                print(f"Entraînement du modèle {name}...")
                model = build_func(input_shape, output_size)
                model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), verbose=0)
                model.save(model_path)
                print(f"Modèle {name} sauvegardé dans {model_path}")
        elif name == "TCN":
            weights_path = f"saved_models/{name}.weights.h5"
            model = build_func(input_shape, output_size)
            if os.path.exists(weights_path):
                print(f"Chargement des poids du modèle TCN depuis {weights_path}...")
                model.load_weights(weights_path)
            else:
                print(f"Entraînement du modèle TCN...")
                model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), verbose=0)
                model.save_weights(weights_path)
                print(f"Poids du modèle TCN sauvegardés dans {weights_path}")

        Y_pred_scaled = model.predict(X_test)
        y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
        all_metrics[name] = {
            "MSE": float(mean_squared_error(Y_test_original, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(Y_test_original, y_pred))),
            "MAE": float(mean_absolute_error(Y_test_original, y_pred)),
            "R2": float(r2_score(Y_test_original, y_pred))
        }
        all_predictions[name] = {
            'true_path': generate_hgl_path(Y_test_original[0], STATIONS, RIVER_CROSSINGS),
            'pred_path': generate_hgl_path(y_pred[0], STATIONS, RIVER_CROSSINGS)
        }
        
    geographic_data = get_geographic_data()

    # # ENVOI DES RESULTATS AU WEBHOOK N8N
    # webhook_url = "https://votre_webhook_n8n_url"  # Remplacez par l'URL réelle de votre webhook n8n
    # payload = {
    #     "metrics": all_metrics,
    #     "predictions": all_predictions
    # }
    # try:
    #     response = requests.post(webhook_url, json=payload)
    #     print(f"Webhook n8n status: {response.status_code}")
    # except Exception as e:
    #     print(f"Erreur lors de l'envoi au webhook n8n: {e}")

    return convert_numpy_types(all_metrics), convert_numpy_types(all_predictions), convert_numpy_types(geographic_data), None


