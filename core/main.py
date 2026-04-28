from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import os
import datetime
import json

# Import dei tuoi moduli corretti
from vision.smoke_analysis import SmokeAnalyzer
from service.environmental_services import EnvironmentalService

app = FastAPI(title="SmokeWatch Pro API - v1.0")

# Inizializziamo l'analizzatore una sola volta all'avvio per risparmiare memoria
analyzer = SmokeAnalyzer()

# Abilita CORS per il frontend (essenziale se il collega usa React/Vue/HTML locale)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_complete")
async def full_analysis(
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lon: float = Form(0.0),
    heading: float = Form(0.0)
):
    # 1. Creazione cartella per archiviazione (Logica "Organized Results")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"runs/detect/test_{timestamp}_{file.filename.split('.')[0]}"
    os.makedirs(test_dir, exist_ok=True)

    # 2. Lettura Immagine
    contents = await file.read()
    
    # Salviamo l'originale per lo storico
    with open(os.path.join(test_dir, file.filename), "wb") as f:
        f.write(contents)

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 3. Analisi AI (Vision) tramite la classe SmokeAnalyzer
    vision_data = analyzer.analyze(image)

    # 4. Arricchimento Dati Ambientali (solo se viene rilevato fumo)
    report = {
        "status": "success",
        "timestamp": timestamp,
        "metadata": {"lat": lat, "lon": lon, "azimuth": heading},
        "vision": vision_data,
        "environment": None,
        "marine_traffic": None,
        "storage_path": test_dir
    }

    if vision_data["smoke_detected"]:
        # Recupero Vento, Meteo e Qualità Aria
        env_data = EnvironmentalService.get_weather_and_wind(lat, lon)
        if env_data:
            env_data["air_quality"] = EnvironmentalService.get_arpa_data(lat, lon)
            report["environment"] = env_data
        
        # Traffico Navale
        report["marine_traffic"] = EnvironmentalService.get_nearby_ships(lat, lon)

    # 5. Salvataggio del report finale JSON nella cartella del test
    with open(os.path.join(test_dir, "report.json"), "w") as jf:
        json.dump(report, jf, indent=4)

    # 6. Salvataggio immagine annotata per debug visivo
    # (Usiamo OpenCV per disegnare un cerchio o una scritta se vogliamo, 
    # o ci affidiamo al salvataggio interno di YOLO se integrato in analyze)
    
    return report

if __name__ == "__main__":
    # Creiamo la cartella base se non esiste
    os.makedirs("runs/detect", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)