import requests
import os
import json

URL = "http://127.0.0.1:8000/analyze"
FOLDER_PATH = "immagini_test"

def run_test():
    if not os.path.exists(FOLDER_PATH):
        print(f"Cartella {FOLDER_PATH} non trovata!")
        return

    files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    print(f"🚀 Analisi di {len(files)} immagini in corso...\n")

    report = []

    for filename in files:
        path = os.path.join(FOLDER_PATH, filename)
        # Specifichiamo il file correttamente come un tupla (nome, contenuto)
        with open(path, 'rb') as f:
            files_payload = {'file': (filename, f, 'image/jpeg')} # Forza il tipo immagine
            data_payload = {'lat': '40.83', 'lon': '14.24'}
            
            try:
                response = requests.post(URL, files=files_payload, data=data_payload)
                
                if response.status_code == 200:
                    res = response.json()
                    # Gestiamo il caso in cui il server risponda con "status": "error"
                    if res.get("status") == "error":
                        print(f"❌ Errore logica su {filename}: {res.get('message')}")
                        continue

                    summary = res.get('summary', {})
                    r_val = summary.get('max_ringelmann', 0.0)
                    sev = summary.get('severity', 'N/D')
                    
                    print(f"✅ {filename:40} | Ringelmann: {r_val} | Stato: {sev}")
                    report.append({"file": filename, "result": res})
                else:
                    print(f"❌ Errore HTTP {response.status_code} su {filename}")
            except Exception as e:
                print(f"💥 Errore connessione su {filename}: {e}")

    with open("risultati_finali.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n📊 Test concluso.")

if __name__ == "__main__":
    run_test()