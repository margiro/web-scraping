"""Airbnb scraper using Selenium and the public API."""

import pandas as pd
import requests
import base64
import re
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import json
import urllib.parse
import math
import time
from urllib3.exceptions import ReadTimeoutError
from selenium.common.exceptions import TimeoutException
from datetime import date
from typing import Any, Optional

SLEEP_SECS  = 0.5

driver = webdriver.Chrome()
driver.set_page_load_timeout(30)
cursors = []
for i in range(16):
    items_offset = i * 18  # Suponiendo 18 resultados por página
    data = {"section_offset": 0, "items_offset": items_offset, "version": 1}

    # Convertir a JSON sin espacios innecesarios
    json_str = json.dumps(data, separators=(',', ':'))

    # Codificar a Base64
    b64_str = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")

    # URL encodear el string Base64
    cursor_encoded = urllib.parse.quote(b64_str)

    cursors.append(cursor_encoded)

print(cursors)

def obtener_numero_alojamientos(driver):
    """Return the number of listings shown in the results heading."""
    try:
        # Caso 1 y 2: Buscar el elemento con el data-testid correspondiente
        span_resultados = driver.find_element(By.CSS_SELECTOR, 'span[data-testid="stays-page-heading"]')
        texto = span_resultados.text.strip()

        if "Más de" in texto:
            return 1000  # Suponemos que es 1000 como mínimo
        else:
            # Extraer número con expresión regular (como 286 en "286 alojamientos...")
            match = re.search(r"(\d[\d\.]*)", texto)
            if match:
                return int(match.group(1).replace(".", "").replace(" ", ""))  # Eliminar puntos o espacios finos
            else:
                return 0

    except NoSuchElementException:
        try:
            # Caso 3: Buscar el mensaje de "No hay coincidencias exactas"
            driver.find_element(By.XPATH, '//h1[contains(text(), "No hay coincidencias exactas")]')
            return 0
        except NoSuchElementException:
            # Si no encontramos ninguno, devolvemos -1 como error
            return -1
        
def construir_url_con_bbox(bbox):
    """Build the Airbnb search URL for the given bounding box."""
    lat_min, lng_min, lat_max, lng_max = bbox

    base_url = "https://www.airbnb.es/s/Mallorca--España/homes"
    params = {
        "refinement_paths[]": "/homes",
        "query": "Mallorca, España",
        "flexible_trip_lengths[]": "one_week",
        "monthly_start_date": "2025-05-01",
        "monthly_length": "3",
        "monthly_end_date": "2025-08-01",
        "price_filter_input_type": "0",
        "channel": "EXPLORE",
        "search_type": "user_map_move",
        "price_filter_num_nights": "5",
        "zoom_level": "10",  # puedes ajustar el zoom si quieres
        "date_picker_type": "calendar",
        "source": "structured_search_input_header",
        "place_id": "ChIJKcEGZna4lxIRwOzSAv-b67c",  # place_id de Mallorca
        "search_mode": "regular_search",
        "ne_lat": lat_max,
        "ne_lng": lng_max,
        "sw_lat": lat_min,
        "sw_lng": lng_min,
        "zoom": "13",
        "search_by_map": "true"

    }

    from urllib.parse import urlencode
    url = f"{base_url}?{urlencode(params)}"
    return url

def subdividir_bbox(bbox):
    """Split the bounding box into four smaller quadrants."""
    lat_min, lng_min, lat_max, lng_max = bbox
    lat_mid = (lat_min + lat_max) / 2
    lng_mid = (lng_min + lng_max) / 2
    # Generar los 4 cuadrantes
    cuadrante1 = (lat_min, lng_min, lat_mid, lng_mid)
    cuadrante2 = (lat_min, lng_mid, lat_mid, lng_max)
    cuadrante3 = (lat_mid, lng_min, lat_max, lng_mid)
    cuadrante4 = (lat_mid, lng_mid, lat_max, lng_max)
    return [cuadrante1, cuadrante2, cuadrante3, cuadrante4]


def procesar_bbox(driver, bbox, cursors, datos, habitaciones, failed_urls, fecha=date.today()):
    """Recursively scrape all listings inside the bounding box."""
    wait = WebDriverWait(driver, 5)  # Espera máxima de 5 segundos

    if habitaciones:
        url_inicial = construir_url_con_bbox(bbox)  + "&selected_filter_order%5B%5D=room_types%3APrivate%20room&update_selected_filters=true&pagination_search=true&room_types%5B%5D=Private%20room&" + "federated_search_session_id=98e14e57-13ef-4bde-824b-05d141c38284&pagination_search=true&cursor=" + str(cursors[0])
        print(url_inicial)
    else:
        url_inicial = construir_url_con_bbox(bbox) + "&selected_filter_order%5B%5D=room_types%3AEntire%20home%2Fapt&update_selected_filters=true&pagination_search=true&room_types%5B%5D=Entire%20home%2Fapt&" + "federated_search_session_id=98e14e57-13ef-4bde-824b-05d141c38284&pagination_search=true&cursor=" + str(cursors[0])
        print(url_inicial)

    # Cargar la página inicial
    try:
        driver.get(url_inicial)
    except:
                    print("Timeout reached. Retrying...")
                    driver.refresh()
    try:
        # Espera hasta que aparezca el span que indica que la página se ha cargado
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'span[data-testid="stays-page-heading"]')))
    except Exception as e:
        print(f"Error al cargar la página inicial: {e}")

    num_alojamientos = obtener_numero_alojamientos(driver)
    print(f"{bbox} — Total alojamientos: {num_alojamientos} alojamientos procesados: {len(datos)}")
    
    if num_alojamientos == 0:
        pass

    # Si hay 270 o menos, procesamos la región
    elif num_alojamientos <= 270:
        num_paginas = math.ceil(num_alojamientos // 18)
        if habitaciones:
            # Iteramos por cada página para habitaciones
            for i in range(num_paginas + 1):
                skip=False
                url = construir_url_con_bbox(bbox)
                url += "&selected_filter_order%5B%5D=room_types%3APrivate%20room&update_selected_filters=true&pagination_search=true&room_types%5B%5D=Private%20room&"
                url += "&federated_search_session_id=98e14e57-13ef-4bde-824b-05d141c38284"
                url += "&pagination_search=true&cursor=" + str(cursors[i])
                try:
                    driver.get(url)
                except:
                    print("Timeout reached. Retrying...")
                    skip=True
                if skip:
                    failed_urls.append(url)
                    pass
                    
                try:
                    # Espera hasta que carguen los contenedores de alojamientos
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='card-container']")))
                except Exception as e:
                    print(f"Error al cargar la página {i} para habitaciones: {e}")
                alojamientos = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='card-container']")
                for alojamiento in alojamientos:
                    try:
                        nombre = alojamiento.find_element(By.CSS_SELECTOR, "div[data-testid='listing-card-title']").text
                        enlace = alojamiento.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                        precio = alojamiento.find_element(By.XPATH, ".//span[contains(text(), '€')]").text
                        fechas = alojamiento.find_element(By.XPATH, ".//span[contains(text(), '–')]").text
                        tipo_de_alojamiento = 'habitación'
                        datos.append({
                            'Nombre': nombre,
                            'url': enlace,
                            'Precio': precio,
                            'Fechas': fechas,
                            'alojamiento': tipo_de_alojamiento,
                            'fecha_scraping': fecha
                            
                        })
                    except Exception as e:
                        print(f"Error procesando un alojamiento: {e}")
        else:
            # Iteramos por cada página para alojamiento entero
            
            for i in range(num_paginas + 1):
                skip=False
                url = construir_url_con_bbox(bbox)
                url += "&selected_filter_order%5B%5D=room_types%3AEntire%20home%2Fapt&update_selected_filters=true&pagination_search=true&room_types%5B%5D=Entire%20home%2Fapt&"
                url += "&federated_search_session_id=98e14e57-13ef-4bde-824b-05d141c38284"
                url += "&pagination_search=true&cursor=" + str(cursors[i])
                try:
                    driver.get(url)
                except:
                    print("Timeout reached.")
                    skip=True
                if skip:
                    failed_urls.append(url)
                    pass
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='card-container']")))
                except Exception as e:
                    print(f"Error al cargar la página {i} para alojamiento entero: {e}")
                alojamientos = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='card-container']")
                for alojamiento in alojamientos:
                    try:
                        nombre = alojamiento.find_element(By.CSS_SELECTOR, "div[data-testid='listing-card-title']").text
                        enlace = alojamiento.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                        precio = alojamiento.find_element(By.XPATH, ".//span[contains(text(), '€')]").text
                        fechas = alojamiento.find_element(By.XPATH, ".//span[contains(text(), '–')]").text
                        tipo_de_alojamiento = 'alojamiento entero'
                        datos.append({
                            'Nombre': nombre,
                            'url': enlace,
                            'Precio': precio,
                            'Fechas': fechas,
                            'alojamiento': tipo_de_alojamiento,
                            'fecha_scraping': fecha
                        })
                    except Exception as e:
                        print(f"Error procesando un alojamiento: {e}")
    else:
        # Si hay más de 270 alojamientos, subdividimos la región en 4 sub-bbox
        sub_bboxes = subdividir_bbox(bbox)
        for i in range(len(sub_bboxes)):
            print(f"Procesando subregión {i+1} de {len(sub_bboxes)}")
            procesar_bbox(driver, sub_bboxes[i], cursors, datos, habitaciones, failed_urls)
    
    df = pd.DataFrame(datos)
    return df

#https://www.airbnb.es/s/Mallorca--Espa%C3%B1a/homes?refinement_paths%5B%5D=%2Fhomes&query=Mallorca%2C%20Espa%C3%B1a&flexible_trip_lengths%5B%5D=one_week&monthly_start_date=2025-05-01&monthly_length=3&monthly_end_date=2025-08-01&price_filter_input_type=0&channel=EXPLORE&search_type=user_map_move&price_filter_num_nights=5&zoom_level=13.987858887077024&date_picker_type=calendar&source=structured_search_input_header&place_id=ChIJKcEGZna4lxIRwOzSAv-b67c&search_mode=regular_search&ne_lat=39.59494291361849&ne_lng=2.6871615778142655&sw_lat=39.54163115210418&sw_lng=2.6334102699946698&zoom=13.987858887077024&search_by_map=true&update_selected_filters=true
bbox_mallorca=(39.10129556224702,
               2.3739429649583883,
               40.17457797140084,
               3.4572007960993005)
bbox_menorca=(39.73202837261271,
              3.8166048322976565,
              40.183968810121,
              4.306760466171312)
bbox_ibiza=(38.76877967614272,
            1.19828131814009,
            39.164966472991914,
            1.6218922749454237)
bbox_formentera=(38.579720345013094,
                 1.3865663541037065,
                 38.77151823526658,
                 1.5908027388515222)
ISLAND_BBOXES={
    "mallorca":bbox_mallorca,
    "menorca":bbox_menorca,
    "ibiza":bbox_ibiza,
    "formentera":bbox_formentera
}

def procesar_failed_urls(driver, failed_urls, habitacion, datos, fecha=date.today()):
    """Retry scraping URLs that failed during the first pass."""
    wait = WebDriverWait(driver, 5) 
    if habitacion:
        for url in failed_urls:
            skip=False
            try:
                driver.get(url)
            except:
                    print("Timeout reached")
                    skip=True
                

            if skip:
                pass
            else:

                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='card-container']")))
                except Exception as e:
                    print(f"Error al cargar la página {i} para habitaciones: {e}")
                alojamientos = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='card-container']")
                for alojamiento in alojamientos:
                    try:
                        nombre = alojamiento.find_element(By.CSS_SELECTOR, "div[data-testid='listing-card-title']").text
                        enlace = alojamiento.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                        precio = alojamiento.find_element(By.XPATH, ".//span[contains(text(), '€')]").text
                        fechas = alojamiento.find_element(By.XPATH, ".//span[contains(text(), '–')]").text
                        tipo_de_alojamiento = 'habitación'
                        datos.append({
                            'Nombre': nombre,
                            'url': enlace,
                            'Precio': precio,
                            'Fechas': fechas,
                            'alojamiento': tipo_de_alojamiento,
                            'fecha_scraping': fecha
                        })
                    except Exception as e:
                        print(f"Error procesando un alojamiento: {e}")
    else:
            # Iteramos por cada página para alojamiento entero
            
            for url in failed_urls:
                skip=False
            try:
                driver.get(url)
            except:
                    print("Timeout reached")
                    skip=True
                

            if skip:
                pass
            else:

                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='card-container']")))
                except Exception as e:
                    print(f"Error al cargar la página {i} para habitaciones: {e}")
                alojamientos = driver.find_elements(By.CSS_SELECTOR, "div[data-testid='card-container']")
                for alojamiento in alojamientos:
                    try:
                        nombre = alojamiento.find_element(By.CSS_SELECTOR, "div[data-testid='listing-card-title']").text
                        enlace = alojamiento.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                        precio = alojamiento.find_element(By.XPATH, ".//span[contains(text(), '€')]").text
                        fechas = alojamiento.find_element(By.XPATH, ".//span[contains(text(), '–')]").text
                        tipo_de_alojamiento = 'alojamiento entero'
                        datos.append({
                            'Nombre': nombre,
                            'url': enlace,
                            'Precio': precio,
                            'Fechas': fechas,
                            'alojamiento': tipo_de_alojamiento,
                            'fecha_scraping': fecha
                        })
                    except Exception as e:
                        print(f"Error procesando un alojamiento: {e}")
    df = pd.DataFrame(datos)
    return df


all_aloj=[]
all_hab=[]
for isla,bbox in ISLAND_BBOXES.items():
    datos=[]
    failed_urls=[]
    alojamientos=procesar_bbox(driver, bbox, cursors, datos, False, failed_urls)
    alojamientos=procesar_failed_urls(driver,failed_urls, False, datos )
    alojamientos=alojamientos.drop_duplicates()
    alojamientos['isla']=isla
    print(alojamientos)
    all_aloj.append(alojamientos)

    datos=[]
    failed_urls=[]
    habitaciones=procesar_bbox(driver, bbox, cursors, datos, True, failed_urls)
    habitaciones=procesar_failed_urls(driver,failed_urls, True, datos)
    habitaciones=habitaciones.drop_duplicates()
    habitaciones['isla']=isla
    print(habitaciones.head())
    all_hab.append(habitaciones)

driver.quit()

def build_query_url(listing_id: str,
                    check_in: str | None = None,
                    check_out: str | None = None,
                    locale: str = "es",
                    currency: str = "EUR") -> str:
    """
    Devuelve la URL completa para la query StaysPdpSections con el hash
    cd2327… (el mismo que trae tu cURL) y con la sección del host incluida.
    """
    # Airbnb codifica “StayListing:<id>” en base-64
    encoded_id = base64.b64encode(f"StayListing:{listing_id}".encode()).decode()

    variables = {
        "id": encoded_id,
        "pdpSectionsRequest": {
            "adults": "1",
            "children": "0",
            "infants": "0",
            "pets": 0,
            "checkIn": check_in,
            "checkOut": check_out,
            # añadir explícitamente la sección del anfitrión evita que falte
            "sectionIds": ["HOST_PROFILE_DEFAULT"],
            "layouts": ["SIDEBAR", "SINGLE_COLUMN"],
        },
        "wishlistTenantIntegrationEnabled": False,
    }

    # Hash persistido que aparece en tu cURL
    sha = "cd23276083d6ed97dd8eadfa25571df09e59d9ad8fa9ed1271d8411889173ac8"
    vars_enc = urllib.parse.quote(json.dumps(variables, separators=(",", ":")))

    return (
        "https://www.airbnb.es/api/v3/StaysPdpSections"
        f"?operationName=StaysPdpSections"
        f"&locale={locale}&currency={currency}"
        f"&variables={vars_enc}"
        f"&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1"
        f"%2C%22sha256Hash%22%3A%22{sha}%22%7D%7D"
    )


def fetch_pdp(url: str) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json",
        "Accept-Language": "es",
        "X-Airbnb-API-Key": "d306zoyjsyarp7ifhu67rjxn52tv0t20",
    }
    return requests.get(url, headers=headers, timeout=20, verify=False).json()


TITLE_RX = re.compile(r"\b(?:Anfitri[oó]n|Host)\b\s*:\s*(.+)", re.I)

def host_name(resp: dict[str, Any]) -> Optional[str]:
    """
    Devuelve el nombre del anfitrión a partir del JSON de StaysPdpSections.
    Si no lo encuentra, devuelve None.
    """

    # 1) Ruta directa más frecuente
    try:
        return resp["data"]["presentation"]["stayProductDetailPage"]["listing"][
            "primaryHost"
        ]["firstName"]
    except (KeyError, TypeError):
        pass

    # 2) Búsqueda recursiva de un título tipo "Anfitrión: Eva"
    def walk(node: Any) -> Optional[str]:
        if isinstance(node, dict):
            title = node.get("title")
            if title:
                m = TITLE_RX.search(title)
                if m:
                    return m.group(1).strip()

            # seguir bajando
            for v in node.values():
                found = walk(v)
                if found:
                    return found

        elif isinstance(node, list):
            for item in node:
                found = walk(item)
                if found:
                    return found
        return None

    return walk(resp)


def airbnb_df(df):
    latitudes = []
    longitudes = []
    capacities = []
    registrations = []
    bedrooms=[]
    titles=[]
    bathrooms=[]
    titulos=[]
    hosts=[]
    procesadas=0

    for url in df['url']:
        procesadas+=1
        if procesadas%10==0:
            print(procesadas)

        try:
            clean_url = url.split('?')[0]  # Elimina parámetros
            listing_id = clean_url.split('/')[-1]

            to_encode = 'StayListing:'+listing_id
            encoded = base64.b64encode(bytes(to_encode,'utf-8')).decode('ascii').replace('=','%3D')

            headers = {
                'accept-language': 'en-US',
                'content-type': 'application/json',
                'user-agent': 'Mozilla/5.0',
                'x-airbnb-api-key': 'd306zoyjsyarp7ifhu67rjxn52tv0t20'
            }

            query_url = "https://www.airbnb.ca/api/v3/StaysPdpSections?operationName=StaysPdpSections&locale=en-CA¤cy=CAD&_cb=1987xzg1yzv9ed124yrix00ybtwv&variables=%7B%22id%22%3A%22"+encoded+"%22%2C%22pdpSectionsRequest%22%3A%7B%22adults%22%3A%221%22%2C%22bypassTargetings%22%3Afalse%2C%22categoryTag%22%3Anull%2C%22causeId%22%3Anull%2C%22children%22%3Anull%2C%22disasterId%22%3Anull%2C%22discountedGuestFeeVersion%22%3Anull%2C%22displayExtensions%22%3Anull%2C%22federatedSearchId%22%3Anull%2C%22forceBoostPriorityMessageType%22%3Anull%2C%22infants%22%3Anull%2C%22interactionType%22%3Anull%2C%22layouts%22%3A%5B%22SIDEBAR%22%2C%22SINGLE_COLUMN%22%5D%2C%22pets%22%3A0%2C%22pdpTypeOverride%22%3Anull%2C%22preview%22%3Afalse%2C%22previousStateCheckIn%22%3Anull%2C%22previousStateCheckOut%22%3Anull%2C%22priceDropSource%22%3Anull%2C%22privateBooking%22%3Afalse%2C%22promotionUuid%22%3Anull%2C%22relaxedAmenityIds%22%3Anull%2C%22searchId%22%3Anull%2C%22selectedCancellationPolicyId%22%3Anull%2C%22selectedRatePlanId%22%3Anull%2C%22staysBookingMigrationEnabled%22%3Afalse%2C%22translateUgc%22%3Anull%2C%22useNewSectionWrapperApi%22%3Afalse%2C%22sectionIds%22%3Anull%2C%22checkIn%22%3Anull%2C%22checkOut%22%3Anull%7D%7D&extensions=%7B%22persistedQuery%22%3A%7B%22version%22%3A1%2C%22sha256Hash%22%3A%222e71de979aa92574a9e7e83d9192b3bb6bb184b8c446380b12c3160cfc8a9cbc%22%7D%7D"
            
            time.sleep(SLEEP_SECS)

            resp = requests.get(query_url, headers=headers, verify=False).json()

            time.sleep(SLEEP_SECS)

            # Lat & Lng
            lat = resp['data']['presentation']['stayProductDetailPage']['sections']['metadata']['loggingContext']['eventDataLogging']['listingLat']
            lng = resp['data']['presentation']['stayProductDetailPage']['sections']['metadata']['loggingContext']['eventDataLogging']['listingLng']

            # personCapacity
            sections_list = resp['data']['presentation']['stayProductDetailPage']['sections']['sections']
            person_capacity = None
            for section in sections_list:
                try:
                    pc = section['section']['shareSave']['embedData']['personCapacity']
                    if pc is not None:
                        person_capacity = pc
                        break
                except (KeyError, TypeError):
                    pass

            title = None
            for section in sections_list:
                try:
                    title = section['section']['shareSave']['sharingConfig']['title']
                    if title is not None:
                        break
                except (KeyError, TypeError):
                    pass
            
            titulo = None
            if title:
                    parts = [p.strip() for p in title.split("·")]

                    titulo = parts[0] if parts else None
                    
                    
                    habitaciones = None
                    baños=None

                    for part in parts:
                        if re.search(r"bath(?:s)?", part, re.IGNORECASE):
                            m = re.search(r"(\d+)", part)
                            if m: baños = int(m.group(1))

                        if re.search(r"bedroom(?:s)?", part, re.IGNORECASE):
                            m = re.search(r"(\d+)", part)
                            if m: habitaciones = int(m.group(1))
            else:
                habitaciones = None
                baños=None

            


            og_title=None
            for section in sections_list:
                try:
                    og_title = section['section']['listingTitle']
                    if og_title is not None:
                        break
                except (KeyError, TypeError):
                    pass
            

            # registrationNumber
            registration_number = None
            for section in sections_list:
                try:
                    html_text = section['section']['htmlDescription']['htmlText']
                    match = re.search(r'Registration Details\s*(?:</b>)?(?:<br\s*/?>)*\s*([A-Za-z0-9./: _,\\-]+)', html_text, re.IGNORECASE)
                    if match:
                        registration_number = match.group(1).strip()
                        break
                except (KeyError, TypeError):
                    pass

            host=None
            try:
                query_url = build_query_url(
                listing_id,
                check_in=None,
                check_out=None,
                )
                resp = fetch_pdp(query_url)
                host=host_name(resp)
            except:
                pass

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            lat = None
            lng = None
            person_capacity = None
            registration_number = None
            habitaciones=None
            og_title=None
            baños=None
            titulo=None
            host=None

        latitudes.append(lat)
        longitudes.append(lng)
        capacities.append(person_capacity)
        registrations.append(registration_number)
        bedrooms.append(habitaciones)
        titles.append(og_title)  
        bathrooms.append(baños)
        titulos.append(titulo)
        hosts.append(host)



    # Añadimos las columnas al DataFrame original
    df['lat'] = latitudes
    df['lng'] = longitudes
    df['capacidad'] = capacities
    df['licencia'] = registrations
    df['habitaciones']=bedrooms
    df['baños']=bathrooms
    df['titulo']=titles
    df['anfitrion']=hosts
   

    return df

print('hello world')

df_all_aloj=airbnb_df(pd.concat(all_aloj, ignore_index=True))
df_all_hab=airbnb_df(pd.concat(all_hab, ignore_index=True))
df_islas=pd.concat([df_all_aloj, df_all_hab], ignore_index=True)
df_islas['tipo'] = df_islas['Nombre'].apply(lambda x: x.split()[0] if isinstance(x, str) and x.split() else '')
df_islas['tipo'] = df_islas['tipo'].replace('Casa', 'Casa rural')
df_islas['tipo'] = df_islas['tipo'].replace('Apto.', 'Apartamento')
print(df_islas.head())
df_islas.to_csv('airbnb_islas_'+str(date.today())+'.csv', index=False)
