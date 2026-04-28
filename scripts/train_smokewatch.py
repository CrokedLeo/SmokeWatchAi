from ultralytics import YOLO

def train():
    # Carica il modello
    model = YOLO("yolov8n.pt")

    # Avvia l'addestramento
    results = model.train(
        data="data.yaml",
        epochs=30,
        imgsz=640,
        device=0,       # Finalmente usiamo la tua RTX 4050!
        workers=4,      # Puoi mettere 4 o 8 per velocizzare il caricamento
        batch=16        # Con 6GB di VRAM, 16 dovrebbe andare bene
    )

if __name__ == '__main__':
    train()