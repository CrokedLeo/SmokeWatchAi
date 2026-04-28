import requests
import logging

# Configurazione logging per monitorare fallimenti API senza bloccare il server
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentalService:
    @staticmethod
    def get_weather_and_wind(lat, lon):
        """Recupera dati vento da Open-Meteo (Gratis, senza API Key)"""
        if lat == 0 or lon == 0:
            return None
            
        try:
            # Aggiunto current_weather=true per una risposta più compatta
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            res = requests.get(url, timeout=5).json()
            
            if "current_weather" in res:
                cw = res["current_weather"]
                return {
                    "wind_speed": cw.get("windspeed"),
                    "wind_direction": cw.get("winddirection"),
                    "temperature": cw.get("temperature"),
                    "unit_speed": "km/h",
                    "unit_temp": "°C"
                }
            return None
        except Exception as e:
            logger.error(f"Errore Open-Meteo: {e}")
            return None

    @staticmethod
    def get_arpa_data(lat, lon):
        """Recupera dati qualità aria da OpenAQ"""
        if lat == 0 or lon == 0:
            return "Coordinate non fornite"
            
        try:
            # Radius aumentato a 25km per trovare più facilmente stazioni in zone portuali
            url = f"https://api.openaq.org/v2/locations?coordinates={lat},{lon}&radius=25000"
            res = requests.get(url, timeout=5).json()
            
            if res.get('results'):
                station = res['results'][0]
                # Pulizia dei parametri per restituire solo i valori correnti
                measurements = {p['parameter']: f"{p['lastValue']} {p['unit']}" for p in station.get('parameters', [])}
                
                return {
                    "station_name": station.get('name', 'Stazione Anonima'),
                    "measurements": measurements,
                    "source": "OpenAQ"
                }
            return "Nessuna stazione di monitoraggio rilevata nel raggio di 25km"
        except Exception as e:
            logger.error(f"Errore OpenAQ: {e}")
            return "Dati qualità aria non disponibili"

    @staticmethod
    def get_nearby_ships(lat, lon):
        """
        Simulazione integrazione AIS. 
        Nota per il collega: Richiede abbonamento a servizi come AISHub o Spire.
        """
        # In una fase futura, qui implementeremo la chiamata API reale.
        # Per ora restituiamo dati simulati coerenti con una zona portuale.
        if lat == 0 or lon == 0:
            return []
            
        return [
            {"name": "Cargo-Alpha", "type": "Portacontainer", "dist": "1.2nm", "status": "In manovra"},
            {"name": "Ferry-Beta", "type": "Traghetto", "dist": "0.5nm", "status": "Attraccato"}
        ]

# Test rapido se eseguito come script principale
if __name__ == "__main__":
    service = EnvironmentalService()
    # Esempio su Genova (Zona Porto)
    print("Test Meteo:", service.get_weather_and_wind(44.40, 8.92))
    print("Test Qualità Aria:", service.get_arpa_data(44.40, 8.92))