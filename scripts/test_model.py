from ultralytics import YOLO
import cv2

# 1. Carica il modello che hai appena addestrato
# NB: Assicurati che il percorso sia corretto (train-7 o l'ultimo creato)
model = YOLO(r"C:\Users\lauda\Desktop\smokewatch\runs\detect\train-7\weights\best.pt")

# 2. Percorso di un'immagine di prova (mettine una nella cartella o usa un URL)
img_path = "test_nave.jpg" 

# 3. Esegui il riconoscimento
results = model.predict(source=img_path, conf=0.25, save=True)

# 4. Mostra i risultati a video
for r in results:
    im_array = r.plot()  # Disegna i rettangoli e le etichette
    cv2.imshow("SmokeWatch Test", im_array)
    cv2.waitKey(0) # Premi un tasto per chiudere la finestra

cv2.destroyAllWindows()