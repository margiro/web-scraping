"""Scrape property data from VRBO using their GraphQL API."""
import uuid
import requests
from typing import Tuple, Dict, Any, List
import json, textwrap
import time
import math
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
import re
import random
from pathlib import Path
from http.cookies import SimpleCookie
from webdriver_manager.chrome import ChromeDriverManager
from datetime import date
from urllib3.exceptions import ReadTimeoutError
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

URL  = "https://www.vrbo.com/graphql?locale=es_ES&siteId=9003009"
HASH = "2ddc0a496b417b852a4f27f426e346051bbd0eb36a2f0db184469c0e4ee3cc0e"
OPERATION = "PropertyListingBoundingBoxQuery"
SEARCH_ID = str(uuid.uuid4())                # uno por sesión

# 1) ––––– guarda base_vars_bbox una SÓLA vez (si ya lo tienes, omite)
raw_payload = """
{"variables":{"context":{"siteId":9003009,"locale":"es_ES","eapid":9,"tpid":9003,
"currency":"EUR","device":{"type":"DESKTOP"},"identity":{"duaid":"REPLACE_ME",
"authState":"ANONYMOUS"},"privacyTrackingState":"CAN_TRACK","debugContext":{"abacusOverrides":[]}},
"criteria":{"primary":{"dateRange": null,
"destination":{"regionName":null,"regionId":null,
"coordinates":{"latitude":39.54162,"longitude":2.8909},
"pinnedPropertyId":null,"propertyIds":null,
"mapBounds":[{"latitude":39.43487,"longitude":2.80287},
             {"latitude":39.6482,"longitude":2.97893}]},
"rooms":[{"adults":2,"children":[]}]},"secondary":{"counts":
[{"id":"resultsStartingIndex","value":0},{"id":"resultsSize","value":50}],
"booleans":[],"selections":[{"id":"privacyTrackingState","value":"CAN_TRACK"},
{"id":"searchId","value":"REPLACE_ID"},{"id":"sort","value":"RECOMMENDED"}],
"ranges":[]}},"destination":{"regionName":null,"regionId":null,
"coordinates":null,"pinnedPropertyId":null,"propertyIds":null,"mapBounds":null},
"shoppingContext":{"multiItem":null},"returnPropertyType":false,"includeDynamicMap":true}}
"""
with open("base_vars_bbox.json","w",encoding="utf-8") as f:
    json.dump(json.loads(textwrap.dedent(raw_payload))["variables"],
              f, indent=2, ensure_ascii=False)
    
# 2) ––––– consigue cookies automáticas
driver = webdriver.Chrome()
driver.get("https://www.vrbo.com"); time.sleep(3)
session = requests.Session()
for c in driver.get_cookies():
    session.cookies.set(c["name"], c["value"], domain=c["domain"])
    session.cookies.set(c["name"], c["value"], domain="www.vrbo.com")
driver.quit()

HEADERS = {
    "accept": "*/*",
    "accept-language": "es-ES,es;q=0.9",
    "content-type": "application/json",
    "origin": "https://www.vrbo.com",
    "referer": "https://www.vrbo.com",
    "x-enable-apq": "true",
    "client-info": "shopping-pwa,,us-east-1",
    "x-page-id": "page.Hotel-Search,H,20",
    "x-shopping-product-line": "lodging",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/136.0.0.0 Safari/537.36"
}

BASE_VARS = json.load(open("base_vars_bbox.json"))

# ① si sigue el literal “REPLACE_ME”, cámbialo por un uuid válido
if BASE_VARS["context"]["identity"]["duaid"].startswith("REPLACE"):
    BASE_VARS["context"]["identity"]["duaid"] = str(uuid.uuid4())

def build_payload(sw, ne):
    v = json.loads(json.dumps(BASE_VARS))
    v["criteria"]["primary"]["destination"]["coordinates"] = {
        "latitude":  (sw[0]+ne[0])/2,
        "longitude": (sw[1]+ne[1])/2
    }
    v["criteria"]["primary"]["destination"]["mapBounds"] = [
        {"latitude": sw[0], "longitude": sw[1]},
        {"latitude": ne[0], "longitude": ne[1]}
    ]
    for sel in v["criteria"]["secondary"]["selections"]:
        if sel["id"] == "searchId":
            sel["value"] = SEARCH_ID
    return {
        "operationName": OPERATION,
        "variables": v,
        "extensions": {"persistedQuery": {"sha256Hash": HASH, "version": 1}}
    }

def vrbo_query(sw, ne):
    payload = build_payload(sw, ne)
    time.sleep(2 + random.uniform(0, .4))            # ritmo
    r = session.post(URL, headers=HEADERS, json=payload, timeout=20, verify=False)
    r.raise_for_status()
    return r.json()


def parse_heading(text):
    # Ejemplo: "Villa · 8 huéspedes · 4 habitaciones · 2 baños"
    tipo=None
    huespedes=None
    habitaciones=None
    if text:
        parts = [p.strip() for p in text.split("·")]
        
        tipo = parts[0] if parts else None
        huespedes = None
        habitaciones = None

        for part in parts:
            if re.search(r"huésped(?:es)?", part, re.IGNORECASE):
                m = re.search(r"(\d+)", part)
                if m: huespedes = int(m.group(1))

            # habitaci[oó]n(?:es)?  →  habitación  ó  habitaciones
            if re.search(r"habitaci[oó]n(?:es)?", part, re.IGNORECASE):
                m = re.search(r"(\d+)", part)
                if m: habitaciones = int(m.group(1))

    return {
        "tipo": tipo,
        "capacidad": huespedes,
        "habitaciones": habitaciones
    }



PAGE_SIZE=50
results: Dict[str, dict] = {}    

def process_tile(sw: Tuple[float, float],
                 ne: Tuple[float, float],
                 depth: int = 0) -> None:
    #Evalúa el tile, decide procesar, subdividir o saltar.
    lat_min, lon_min = sw
    lat_max, lon_max = ne
    # guarda coordenadas para log
    box = f"[{lat_min:.4f},{lon_min:.4f}] – [{lat_max:.4f},{lon_max:.4f}]"
    data = vrbo_query(sw, ne)
    total = len(data["data"]["propertySearch"]["dynamicMap"]["map"]["markers"])

    indent = "-" * depth
    if total == 0:
        print(f"{indent}{box}: vacío (0)")
        return

    if total < PAGE_SIZE:
        print(f"{indent}{box}: procesar ({total} ofertas)")
        marker_list = data["data"]["propertySearch"]["dynamicMap"]["map"]["markers"]
        markers = {str(m["id"]): {
            "lat": m["markerPosition"]["latitude"],
            "lng": m["markerPosition"]["longitude"]
        } for m in marker_list}

        # 2. Listings (alojamientos y otras cosas)
        for listing in data["data"]["propertySearch"]["propertySearchListings"]:
            try:
                prop_id = str(listing["id"])
                coordinates=markers[prop_id]

            except KeyError:
                continue
            heading=None
            resource=None
            url=None
            name=None

            try:
                heading = listing["headingSection"]["messages"][0]["text"]
                resource = listing["cardLink"]["resource"]
                url=resource["value"]
                name=listing["headingSection"]["heading"]
            except (IndexError, KeyError):
                print("Error con "+ prop_id)
                if resource:
                   url=resource["value"]
                   print(url)
                
            if prop_id not in results:

                results[prop_id] = {
                    **markers[prop_id],
                    "id": prop_id,
                    "url": url,
                    'nombre': name,
                    **parse_heading(heading)
                }
            
        return

    # ─── total > 50: subdividir ──────────────────────────────────────────────
    print(f"{indent}{box}: subdividir ({total} > 50)")

    lat_mid = (lat_min + lat_max) / 2
    lon_mid = (lon_min + lon_max) / 2

    # cuatro sub-tiles
    sub_tiles = [
        ((lat_min, lon_min), (lat_mid, lon_mid)),  # SW
        ((lat_min, lon_mid), (lat_mid, lon_max)),  # SE
        ((lat_mid, lon_min), (lat_max, lon_mid)),  # NW
        ((lat_mid, lon_mid), (lat_max, lon_max)),  # NE
    ]
    time.sleep(0.5)
    for sub_sw, sub_ne in sub_tiles:
        process_tile(sub_sw, sub_ne, depth + 1)
        time.sleep(0.5)

def extract_vrbo_host(html):
    """
    Extrae el nombre del anfitrión de VRBO a partir del HTML.
    Busca el patrón "Propietario/a: [nombre]" y devuelve el nombre,
    o None si no lo encuentra.
    """
    match = re.search(
        r'Propietario/a:\s*([^\n<]+)',
        html,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return None

def extract_license_host(url, driver,timeout=7, pause=0.8):
    """Return (license_number, host) from a VRBO property page."""

    driver.get(url)

    # Scroll hasta el fondo para cargar todo el contenido dinámico
    last_h = driver.execute_script("return document.body.scrollHeight")
    end_time = time.time() + timeout
    while time.time() < end_time:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_h = driver.execute_script("return document.body.scrollHeight")
        if new_h == last_h:
            break
        last_h = new_h

    html=driver.execute_script("return document.body.innerText;")
    host=extract_vrbo_host(html)
    numero_de_licencia=None
    match=re.search(r"Número de registro de la propiedad\s*([^\n<]+)", html, re.IGNORECASE)
    if match:
        numero_de_licencia = match.group(1).strip()
    else:

        match2=re.search(r"Licencia turística:\s*([^\n<]+)", html, re.IGNORECASE)
        if match2:
            numero_de_licencia = match2.group(1).strip()
    return numero_de_licencia, host

    
    

if __name__ == "__main__":
    ROOT_SW = (39.10129556224702, 2.3739429649583883)
    ROOT_NE = (40.17457797140084, 3.4572007960993005)
    process_tile(ROOT_SW, ROOT_NE)
    df = pd.DataFrame(results.values())
    print(df)
    df.to_csv("vrbo_mallorca0.csv", index=False)
    
    driver.quit()
    driver = webdriver.Chrome()
    driver.set_page_load_timeout(30)
    licencias = []
    hosts=[]
    base_urls=[]

    for url in df['url']:
        base_url = url.split("?", 1)[0]
        base_urls.append(base_url)
        failed_licenses=[]
        try:
            licencia, host = extract_license_host(base_url, driver)
        except:
            licencia = None
            host=None
            failed_licenses.append(base_url)
        licencias.append(licencia)
        hosts.append(host)

    driver.quit()
    df['licencia'] = licencias
    df['anfitrion']=hosts
    df['url']=base_urls
    df['fecha']=str(date.today())


    df.to_csv('vrbo_mallorca_'+str(date.today())+'.csv')
    print(f"\nGuardado {len(df)} alojamientos")
    print('han fallado '+ str(len(failed_licenses))+ ' licencias')

