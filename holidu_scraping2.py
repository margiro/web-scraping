"""Scrape listings from Holidu using their public API."""



import uuid
import time
import requests
import pandas as pd
from typing import Tuple, Dict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import WebDriverException

from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import re
from datetime import date
# --------------------------------------------------------------------------- #
# CONFIGURACIÓN BÁSICA                                                        #
# --------------------------------------------------------------------------- #
API_BASE = "https://api.holidu.com/old/rest/v6/search/offers"
HEADERS = {
    # Copiadas de tu cURL.  Cambia UA o accept-language si lo deseas.
    "accept": "*/*",
    "accept-language": "ca-ES,ca;q=0.9",
    "cache-control": "no-cache, no-store, must-revalidate",
    "content-type": "application/json",
    "origin": "https://www.holidu.es",
    "pragma": "no-cache",
    "referer": "https://www.holidu.es/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
    ),
    # cabeceras opcionales:
    # "userid": str(uuid.uuid4()),
    # "uuid":   "mKgy2N4TBo",
}
PAGE_SIZE      = 30          # No puede ser mayor: Holidu devuelve máx. 30 objetos
SLEEP_SECS     = 1        # cortesía entre peticiones
MIN_TILE_DEG   = 0.002       # tamaño mínimo de quadtree (~200 m) para evitar bucles


ROOT_SW = (39.10129556224702, 2.3739429649583883)
ROOT_NE = (40.17457797140084, 3.4572007960993005)

# --------------------------------------------------------------------------- #
# Llamada a la API                                                            #
# --------------------------------------------------------------------------- #
session = requests.Session()
session.headers.update(HEADERS)

def holidu_query(sw: Tuple[float, float],
                 ne: Tuple[float, float]) -> Dict:
    """Pide una página de 30 ofertas y devuelve el JSON."""
    params = {
        "topicId":      uuid.uuid4(),            # vale cualquiera
        "searchId":     uuid.uuid4(),
        "subscriberId": uuid.uuid4(),
        "searchTerm":   "Mallorca, Islas Baleares, España",
        "swLat": sw[0], "swLng": sw[1],
        "neLat": ne[0], "neLng": ne[1],
        "pageSize": PAGE_SIZE,
        "subscription": "map",
        "domainId": 361,
        "locale": "es-ES",
        "currency": "EUR",
    }
    r = session.get(API_BASE, params=params, timeout=15, verify=False)
    r.raise_for_status()
    return r.json()

# --------------------------------------------------------------------------- #
# Recorrido tipo quadtree                                                     #
# --------------------------------------------------------------------------- #
results: Dict[str, dict] = {}    # clave: offer["id"] → deduplicado

def process_tile(sw: Tuple[float, float],
                 ne: Tuple[float, float],
                 depth: int = 0) -> None:
    """Evalúa el tile, decide procesar, subdividir o saltar."""
    lat_min, lon_min = sw
    lat_max, lon_max = ne
    # guarda coordenadas para log
    box = f"[{lat_min:.4f},{lon_min:.4f}] – [{lat_max:.4f},{lon_max:.4f}]"
    data = holidu_query(sw, ne)
    total = data["metaData"]["cursor"]["totalCount"]

    indent = "-" * depth
    if total == 0:
        print(f"{indent}{box}: vacío (0)")
        return

    if total <= PAGE_SIZE:
        print(f"{indent}{box}: procesar ({total} ofertas)")
        for offer in data.get("offers", []):
            oid = str(offer["id"])
            if oid not in results:                 # deduplicar
                results[oid] = {
                    "id":                oid,
                    "lat":               offer["location"]["lat"],
                    "lng":               offer["location"]["lng"],
                    "nombre":              offer["details"]["name"],
                    "habitaciones":     offer["details"]["bedroomsCount"],
                    "capacidad":       offer["details"]["guestsCount"],
                    "tipo":              offer["details"]["apartmentTypeTitle"]
                }
        return

    # ─── total > 30: subdividir ──────────────────────────────────────────────
    print(f"{indent}{box}: subdividir ({total} > 30)")
    # evita recursión infinita si el tile ya es muy pequeño
    if (lat_max - lat_min <= MIN_TILE_DEG) or (lon_max - lon_min <= MIN_TILE_DEG):
        print(f"{indent}Tamaño mínimo alcanzado → IGNORADO para no recursar infinito")
        return

    lat_mid = (lat_min + lat_max) / 2
    lon_mid = (lon_min + lon_max) / 2

    # cuatro sub-tiles
    sub_tiles = [
        ((lat_min, lon_min), (lat_mid, lon_mid)),  # SW
        ((lat_min, lon_mid), (lat_mid, lon_max)),  # SE
        ((lat_mid, lon_min), (lat_max, lon_mid)),  # NW
        ((lat_mid, lon_mid), (lat_max, lon_max)),  # NE
    ]
    time.sleep(SLEEP_SECS)
    for sub_sw, sub_ne in sub_tiles:
        process_tile(sub_sw, sub_ne, depth + 1)
        time.sleep(SLEEP_SECS)



def ad_query(id):
    return "https://www.holidu.es/d/"+str(id)



def extract_host(html):
    """
    Extrae el nombre del anfitrión a partir del HTML:
      - Si aparece "Gestionado por Holidu", devuelve "Holidu".
      - En caso contrario, busca "Nuestro socio verificado [nombre] gestionará tu reserva"
        y devuelve el nombre capturado.
      - Si no se encuentra ningún patrón, devuelve None.
    """
    # 1) ¿Está gestionado por Holidu?
    if re.search(r'Gestionado por Holidu', html):
        return "Holidu"

    # 2) Buscar "Nuestro socio verificado [nombre] gestionará tu reserva"
    match = re.search(
        r'Nuestro socio verificado\s+(.+?)\s+gestionará tu reserva',
        html,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    # 3) No encontrado
    return None

def get_license_host(driver, url, timeout=10, pause=0.5):
    """Return (license_number, host) for a Holidu listing."""

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 3)
    except:
        return None

    try:
        ver_mas = wait.until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="detailDescription"]/div[2]/div')))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", ver_mas)
        wait.until(EC.element_to_be_clickable(
            (By.XPATH, '//*[@id="detailDescription"]/div[2]/div'))).click()
    except TimeoutException:
        pass  # Si no aparece el botón, continuamos
    last_h = driver.execute_script("return document.body.scrollHeight")
    end_time = time.time() + timeout
    while time.time() < end_time:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_h = driver.execute_script("return document.body.scrollHeight")
        if new_h == last_h:
            break
        last_h = new_h
    html = driver.execute_script("return document.body.innerText;")
    host=extract_host(html)
    numero_de_licencia=None
    match = re.search(r'Número de licencia:\s*([^\n<]+)', html, re.IGNORECASE)
    if match:
        numero_de_licencia = match.group(1).strip()
    
    return numero_de_licencia, host



# --------------------------------------------------------------------------- #
# MAIN                                                                        #
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    ISLAND_BBOXES = {
        "mallorca": ((39.10129556224702, 2.3739429649583883),
                     (40.17457797140084, 3.4572007960993005)),
        "menorca": ((39.73202837261271, 3.8166048322976565),
                    (40.183968810121, 4.306760466171312)),
        "ibiza": ((38.76877967614272, 1.19828131814009),
                  (39.164966472991914, 1.6218922749454237)),
        "formentera": ((38.579720345013094, 1.3865663541037065),
                       (38.77151823526658, 1.5908027388515222)),
    }

    all_dfs = []
    for island, (SW, NE) in ISLAND_BBOXES.items():
        print(f"▶️ Iniciando procesamiento inicial para {island}...")
        results = {}
        process_tile(SW, NE)
        df = pd.DataFrame(results.values())
        driver = webdriver.Chrome()
        driver.set_page_load_timeout(30)
        licencias = []
        hosts = []
        urls = []
        failed_licenses = []
        for id in df['id']:
            url = ad_query(id)
            try:
                license, host = get_license_host(driver, url)
                licencias.append(license)
                hosts.append(host)
            except:
                failed_licenses.append(url)
                licencias.append(None)
                hosts.append(None)
            urls.append(url)

        df['licencia'] = licencias
        df['anfitrion'] = hosts
        df['url'] = urls
        df['fecha'] = str(date.today())
        df['isla'] = island
        print(df)
        print('han fallado '+ str(len(failed_licenses))+ 'licencias')
        driver.quit()
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv("holidu_islas_"+str(date.today())+".csv", index=False)
    print(f"\n✅ Guardado {len(final_df)} alojamientos")
