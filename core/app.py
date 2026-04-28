import cv2
import numpy as np
import os
import requests
import datetime
import json
from fastapi import FastAPI, File, UploadFile, Form
from ultralytics import YOLO

app = FastAPI(title="SmokeWatch AI Server - Gold Edition")

# --- CONFIGURAZIONE ---
# Puntiamo al file nella nuova cartella pulita
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
model = YOLO(MODEL_PATH)

# Chiave API (Sarebbe meglio caricarla da file .env)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "bddee4725fdd5c5de177bc5491ca9126")
BASE_OUTPUT_DIR = "runs/detect"

# Assicuriamoci che la cartella base esista
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- FUNZIONI DI SUPPORTO ---

def get_robust_density(image, box):
    """Calcola l'opacità del fumo (Grado Ringelmann 0-5)."""
    x1, y1, x2, y2 = map(int, box)
    roi_smoke = image[y1:y2, x1:x2]
    if roi_smoke.size == 0: return 0.0
    
    # Campionamento cielo sopra il box
    offset = max(10, int((y2 - y1) * 0.15))
    y_sky_start = max(0, y1 - offset)
    roi_sky = image[y_sky_start:y1, x1:x2]
    
    gray_smoke = cv2.cvtColor(roi_smoke, cv2.COLOR_BGR2GRAY)
    smoke_val = np.mean(gray_smoke)
    
    if roi_sky.size == 0:
        sky_ref = 220 
    else:
        gray_sky = cv2.cvtColor(roi_sky, cv2.COLOR_BGR2GRAY)
        sky_ref = np.mean(gray_sky)
    
    # Formula Ringelmann: 1 - (Luminanza Fumo / Luminanza Cielo)
    opacity = max(0, min(1 - (smoke_val / sky_ref), 1))
    return round(opacity * 5, 1)

def identify_polluter(lat, lon):
    """Incrocia GPS con OpenStreetMap (Nominatim)."""
    if lat == 0.0 and lon == 0.0:
        return {"type": "NON FORNITO", "address": "Coordinate GPS assenti"}
        
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        headers = {'User-Agent': 'SmokeWatch_App/1.0'}
        res = requests.get(url, headers=headers, timeout=3).json()
        name = res.get('display_name', '').lower()
        
        is_marine = any(x in name for x in ['porto', 'port', 'mare', 'harbour', 'molo', 'nave', 'sea'])
        return {
            "type": "NAVE/PORTO" if is_marine else "ZONA INDUSTRIALE/FABBRICA",
            "address": res.get('display_name', 'Indirizzo non trovato')
        }
    except:
        return {"type": "SCONOSCIUTO", "address": f"Lat: {lat}, Lon: {lon}"}

def get_weather(lat, lon):
    """Recupera dati qualità dell'aria (OpenWeather)."""
    if lat == 0.0 and lon == 0.0: return None
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
    try:
        data = requests.get(url, timeout=3).json()['list'][0]
        return {
            "aqi": data['main']['aqi'], 
            "no2": data['components']['no2'], 
            "so2": data['components']['so2']
        }
    except:
        return None

# --- ENDPOINT PRINCIPALE ---

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...), 
    lat: float = Form(0.0), 
    lon: float = Form(0.0)
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Pulizia nome file per evitare problemi di path
    safe_filename = file.filename.replace(" ", "_")
    test_name = f"test_{timestamp}_{os.path.splitext(safe_filename)[0]}"
    test_dir = os.path.join(BASE_OUTPUT_DIR, test_name)
    os.makedirs(test_dir, exist_ok=True)

    file_path = os.path.join(test_dir, safe_filename)
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    final_details = []
    is_video = safe_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    try:
        if is_video:
            cap = cv2.VideoCapture(file_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            video_vals = []
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if frame_id % fps == 0:
                    res = model.predict(frame, conf=0.15, verbose=False)
                    for r in res:
                        for b in r.boxes:
                            if model.names[int(b.cls[0])] == "smoke":
                                video_vals.append(get_robust_density(frame, b.xyxy[0].tolist()))
                frame_id += 1
            cap.release()
            max_r = round(np.mean(video_vals), 1) if video_vals else 0.0
            if video_vals: 
                final_details.append({"class": "smoke", "ringelmann_avg": max_r})
        else:
            img = cv2.imread(file_path)
            results = model.predict(img, conf=0.15, verbose=False)
            
            # Salvataggio visivo del rilevamento
            results[0].save(filename=os.path.join(test_dir, f"detected_{safe_filename}"))
            
            for r in results:
                for b in r.boxes:
                    label = model.names[int(b.cls[0])]
                    conf = round(float(b.conf[0]), 2)
                    r_val = get_robust_density(img, b.xyxy[0].tolist()) if label == "smoke" else 0.0
                    final_details.append({"class": label, "ringelmann": r_val, "conf": conf})
            
            all_r = [d.get('ringelmann', 0) for d in final_details if d['class'] == 'smoke']
            max_r = max(all_r) if all_r else 0.0

        # Logica severità
        status = "NESSUN FUMO"
        if max_r > 3.5: status = "CRITICAL"
        elif max_r > 1.0: status = "WARNING"
        elif max_r > 0: status = "OK"

        response_data = {
            "status": "success",
            "timestamp": timestamp,
            "polluter": identify_polluter(lat, lon),
            "environment": get_weather(lat, lon),
            "summary": {"max_ringelmann": max_r, "severity": status},
            "details": final_details
        }

        # Salvataggio Report
        with open(os.path.join(test_dir, "report.json"), "w") as jf:
            json.dump(response_data, jf, indent=4)

        return response_data

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)