from ultralytics import YOLO
import os

# Carica modello
model = YOLO(r"runs/detect/train-7/weights/best.pt")
input_folder = "immagini_test"

if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    print(f"Cartella '{input_folder}' creata. Aggiungi le foto e riavvia.")
else:
    # Esegue predizione su tutta la cartella
    results = model.predict(source=input_folder, save=True, conf=0.25)
    print(f"Analisi completata. Risultati salvati in runs/detect/predict")
