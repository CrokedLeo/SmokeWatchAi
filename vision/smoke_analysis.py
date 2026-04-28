from ultralytics import YOLO
import cv2
import numpy as np
import os

# Carica il modello definendo un percorso relativo robusto
# Se il file viene chiamato da app.py, models/best.pt sarà corretto
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")

class SmokeAnalyzer:
    def __init__(self, model_path=None):
        self.model = YOLO(model_path or MODEL_PATH)

    def get_ringelmann_value(self, smoke_roi, sky_roi_gray=None):
        """
        Calcola il grado Ringelmann (0-5) basato sulla luminanza.
        """
        gray_smoke = cv2.cvtColor(smoke_roi, cv2.COLOR_BGR2GRAY)
        smoke_val = np.mean(gray_smoke)
        
        # Se non abbiamo un riferimento cielo (es. box troppo in alto), 
        # usiamo un bianco standard (220-255)
        sky_ref = sky_roi_gray if sky_roi_gray is not None else 235
        
        # Formula: Opacità = 1 - (Luminanza Fumo / Luminanza Cielo)
        opacity = max(0, min(1 - (smoke_val / sky_ref), 1))
        
        # Restituisce il grado (0-5)
        return round(opacity * 5, 1)

    def analyze(self, image, conf_threshold=0.25):
        """
        Esegue la detection e l'analisi dell'opacità.
        """
        results = self.model.predict(image, conf=conf_threshold, verbose=False)
        detections = []
        ringelmann_values = []

        for r in results:
            for box in r.boxes:
                # Coordinate e ritaglio
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = image[y1:y2, x1:x2]
                
                if crop.size == 0: continue

                # Campionamento dinamico del cielo (area sopra il fumo)
                # Proviamo a prendere una striscia di 20 pixel sopra il box
                offset = 20
                y_sky_start = max(0, y1 - offset)
                sky_ref_val = None
                
                if y_sky_start < y1:
                    sky_roi = image[y_sky_start:y1, x1:x2]
                    if sky_roi.size > 0:
                        sky_ref_val = np.mean(cv2.cvtColor(sky_roi, cv2.COLOR_BGR2GRAY))

                # Calcolo valore Ringelmann
                ring = self.get_ringelmann_value(crop, sky_ref_val)
                ringelmann_values.append(ring)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "ringelmann": ring,
                    "confidence": round(float(box.conf[0]), 2),
                    "label": self.model.names[int(box.cls[0])]
                })

        avg_ring = round(np.mean(ringelmann_values), 1) if ringelmann_values else 0
        
        return {
            "smoke_detected": len(detections) > 0,
            "ringelmann_avg": avg_ring,
            "detections": detections,
            # Un tocco di competenza chimica per il collega:
            "pollutant_note": "Possibile alta concentrazione di particolato/SOx" if avg_ring > 3 
                              else "Combustione con eccesso d'aria o vapore" if avg_ring < 1 and avg_ring > 0
                              else "Combustione nei limiti normativi"
        }

# Esempio di utilizzo rapido per il collega
if __name__ == "__main__":
    analyzer = SmokeAnalyzer()
    test_img = cv2.imread("immagini_test/nave-inquinamento.jpg")
    if test_img is not None:
        report = analyzer.analyze(test_img)
        print(f"Risultato Test: {report['ringelmann_avg']} Ringelmann")