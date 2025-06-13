"""Scrape listings from Booking.com using the GraphQL API and Selenium."""

import uuid
import requests
from typing import Tuple, Dict, Any, List
import json
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support import expected_conditions as EC
import re
from datetime import date

CSRF_TOKEN = "eyJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJjb250ZXh0LWVucmljaG1lbnQtYXBpIiwic3ViIjoiY3NyZi10b2tlbiIsImlhdCI6MTc0OTc5ODc5OCwiZXhwIjoxNzQ5ODg1MTk4fQ.72i3T1jHyPsj0pjMJxJx3p0I28Jt0I4_D8XUIC0jMRxpj747gDN3_pkP1tafKghYRdLgvQQdPPlBjqWmruxv_g"
COOKIE     = "pcm_personalization_disabled=0; bkng_sso_auth=CAIQ0+WGHxpm/qOmw32tpexXbMqsjXWHRSXWgoLTVcRoxg/VhxufCB2FAPANpS/XiuVh0fdK//FyuYYZ6h4WM9i2WCmkY6YEeR8kv1GWJwt2dnjUfEsFDRd4XI8Np2i2G0TJe7CH89NSj1nb8Ohk; pcm_consent=analytical%3Dfalse%26countryCode%3DES%26consentId%3D01d9d2a6-9288-4e69-9afc-84d52bc46c0e%26consentedAt%3D2025-06-13T07%3A12%3A55.853Z%26expiresAt%3D2025-12-10T07%3A12%3A55.853Z%26implicit%3Dtrue%26marketing%3Dfalse%26regionCode%3DIB%26regulation%3Dgdpr%26legacyRegulation%3Dgdpr; cors_js=1; bkng_sso_session=e30; bkng_sso_ses=e30; BJS=-; OptanonConsent=implicitConsentCountry=GDPR&implicitConsentDate=1749798789495&isGpcEnabled=0&datestamp=Fri+Jun+13+2025+09%3A13%3A35+GMT%2B0200+(hora+de+verano+de+Europa+central)&version=202501.2.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=b35100b7-bec5-478b-99cc-6c2fc9ca92df&interactionCount=0&isAnonUser=1&landingPath=https%3A%2F%2Fwww.booking.com%2Fsearchresults.es.html%3Fss%3Dmallorca&groups=C0001%3A1%2CC0002%3A0%2CC0004%3A0; bkng=11UmFuZG9tSVYkc2RlIyh9Yaa29%2F3xUOLbof7CEiNviT9%2BFb1dHmWwa0MrwvsDXU2jDrqXO2FOEbToVg3xNpND6ppDrtUjcC%2Bm53KTCpNiwNy1Zhs2Q0NLxlrWjl7CNA304RIrwk%2BQ8d6iDoIERAQwC3bePAtlxKS4SGRnaC6ihKTw2ROfohMMm4DUQ2lpOp7dRRJCVbWV4PA%3D; aws-waf-token=edcbe00d-36f7-41ff-86e8-ba26cca3ee69:HQoAlCwxyImFAgAA:MPp5cUIEQDHc8yh0xkxTWVpEMSYgy1Mz28vOT/y53HR5sAT/0MlHo3XSwdWqRm5Pbs4qRTqFwkmhqfXtx37rIJwlj2y4LAu/yqrS1DkHX2QrL9rVUQk0t5q4Y30AxNkCqFml+s1GBgekj8c6SOGSTbYFDD2+O6WmMttuG55dAeSGTNk6JO+uDd6MkZ0P3cVfYBQLkgIYT6nyBGJbIoLZK7ITWborB24YTaO4bfQk9fl52SnWn0t8kalhSrkxtE/BeZI=; lastSeen=0"


PAGEVIEW_ID = str(uuid.uuid4().hex)[:16]
CLIENT_VERSION = "NEISQFGF"              # copia literal del header original

PAGE_SIZE      = 100
SLEEP_SECS     = 2        # cortesía entre peticiones
MIN_TILE_DEG   = 0.002       # tamaño mínimo de quadtree (~200 m) para evitar bucles

HEADERS = {
    "accept": "*/*",
    "content-type": "application/json",
    "apollographql-client-name": "b-search-web-searchresults_rust",
    "apollographql-client-version": CLIENT_VERSION,
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
    ),
    "x-booking-context-action": "markers_on_map-search_results",
    "x-booking-context-action-name": "searchresults_irene",
    "x-booking-context-aid": "356980",
    "x-booking-csrf-token": CSRF_TOKEN,
    "x-booking-dml-cluster": "rust",
    "x-booking-original-sli-identifier-suffix": "SearchResults",
    "x-booking-pageview-id": PAGEVIEW_ID,
    "x-booking-site-type-id": "1",
    "x-booking-timeout-budget-ms": "200000",
    "x-booking-timeout-ms": "200000",
    "x-booking-topic":
        "capla_browser_b-search-web-searchresults, markers_on_map-search_results",
    "x-kl-kes-ajax-request": "Ajax_Request",
    "origin": "https://www.booking.com",
    "referer": (
        "https://www.booking.com/searchresults.es.html?"
        "ss=Mallorca&dest_type=region&dest_id=767"
    ),
    # cookies completas en un único header
    "cookie": COOKIE,
}

BOOKING_URL = (
    "https://www.booking.com/dml/graphql"
    "?ss=Mallorca&dest_id=767&dest_type=region&group_adults=2"
    "&no_rooms=1&group_children=0"
)

# ── ❷  QUERY COMPLETA (tal cual) ── #
Raw_QUERY = r"""
"query MapMarkersDesktop($input: SearchQueryInput!, $markersInput: MarkersInput!, $airportsInput: AirportMarkersInput!, $citiesInput: CityMarkersInput!, $landmarksInput: LandmarkMarkersInput!, $beachesInput: BeachMarkersInput!, $skiResortsInput: SkiResortMarkersInput!, $includeBeachMarkers: Boolean!, $includeSkiMarkers: Boolean!, $includeBundle: Boolean!, $includeCityMarkers: Boolean!, $includeAirportMarkers: Boolean!, $includeLandmarkMarkers: Boolean!) {\n  searchQueries {\n    search(input: $input) {\n      ...PropertyMarkers\n      __typename\n    }\n    __typename\n  }\n  mapMarkers {\n    ...NonPropertyMarkers\n    __typename\n  }\n}\n\nfragment PropertyMarkers on SearchQueryOutput {\n  results {\n    acceptsWalletCredit\n    basicPropertyData {\n      accommodationTypeId\n      id\n      location {\n        countryCode\n        latitude\n        longitude\n        address\n        formattedAddressShort\n        __typename\n      }\n      pageName\n      photos {\n        main {\n          lowResUrl {\n            relativeUrl\n            __typename\n          }\n          highResUrl {\n            relativeUrl\n            __typename\n          }\n          __typename\n        }\n        __typename\n      }\n      reviewScore: reviews {\n        score: totalScore\n        reviewCount: reviewsCount\n        totalScoreTextTag {\n          translation\n          __typename\n        }\n        secondaryScore\n        secondaryTextTag {\n          translation\n          __typename\n        }\n        showSecondaryScore\n        __typename\n      }\n      externalReviewScore: externalReviews {\n        score: totalScore\n        reviewCount: reviewsCount\n        totalScoreTextTag {\n          translation\n          __typename\n        }\n        __typename\n      }\n      starRating {\n        value\n        symbol\n        __typename\n      }\n      ufi\n      __typename\n    }\n    displayName {\n      text\n      __typename\n    }\n    generatedPropertyTitle\n    geniusInfo {\n      geniusBenefitsData {\n        hotelCardHasFreeRoomUpgrade\n        __typename\n      }\n      showGeniusRateBadge\n      __typename\n    }\n    location {\n      beachDistance\n      mainDistance\n      mainDistanceDescription\n      nearbyBeachNames\n      locationScore\n      locationFormattedScore\n      locationTextTag {\n        translation\n        __typename\n      }\n      displayLocation\n      __typename\n    }\n    persuasion {\n      preferred\n      preferredPlus\n      highlighted\n      showNativeAdLabel\n      nativeAdId\n      nativeAdsCpc\n      nativeAdsTracking\n      sponsoredAdsData {\n        isDsaCompliant\n        legalEntityName\n        designType\n        __typename\n      }\n      __typename\n    }\n    policies {\n      showFreeCancellation\n      showNoPrepayment\n      showPetsAllowedForFree\n      __typename\n    }\n    recommendedDate {\n      checkin\n      checkout\n      __typename\n    }\n    mealPlanIncluded {\n      mealPlanType\n      text\n      __typename\n    }\n    nbWishlists\n    soldOutInfo {\n      isSoldOut\n      alternativeDatesMessages {\n        text\n        __typename\n      }\n      __typename\n    }\n    isNewlyOpened\n    isTpiExclusiveProperty\n    seoThemes {\n      caption\n      __typename\n    }\n    ...MatchingUnitConfigurations\n    ...PropertyBlocks\n    ...BookerExperienceData\n    priceDisplayInfoIrene {\n      ...PriceDisplayInfoIrene\n      __typename\n    }\n    __typename\n  }\n  __typename\n}\n\nfragment MatchingUnitConfigurations on SearchResultProperty {\n  matchingUnitConfigurations {\n    commonConfiguration {\n      name\n      unitId\n      bedConfigurations {\n        beds {\n          count\n          type\n          __typename\n        }\n        nbAllBeds\n        __typename\n      }\n      nbAllBeds\n      nbBathrooms\n      nbBedrooms\n      nbKitchens\n      nbLivingrooms\n      nbUnits\n      unitTypeNames {\n        translation\n        __typename\n      }\n      localizedArea {\n        localizedArea\n        unit\n        __typename\n      }\n      __typename\n    }\n    unitConfigurations {\n      name\n      unitId\n      bedConfigurations {\n        beds {\n          count\n          type\n          __typename\n        }\n        nbAllBeds\n        __typename\n      }\n      apartmentRooms {\n        config {\n          roomId: id\n          roomType\n          bedTypeId\n          bedCount: count\n          __typename\n        }\n        roomName: tag {\n          tag\n          translation\n          __typename\n        }\n        __typename\n      }\n      nbAllBeds\n      nbBathrooms\n      nbBedrooms\n      nbKitchens\n      nbLivingrooms\n      nbUnits\n      unitTypeNames {\n        translation\n        __typename\n      }\n      localizedArea {\n        localizedArea\n        unit\n        __typename\n      }\n      unitTypeId\n      __typename\n    }\n    __typename\n  }\n  __typename\n}\n\nfragment PropertyBlocks on SearchResultProperty {\n  blocks {\n    blockId {\n      roomId\n      occupancy\n      policyGroupId\n      packageId\n      mealPlanId\n      bundleId\n      __typename\n    }\n    finalPrice {\n      amount\n      currency\n      __typename\n    }\n    originalPrice {\n      amount\n      currency\n      __typename\n    }\n    onlyXLeftMessage {\n      tag\n      variables {\n        key\n        value\n        __typename\n      }\n      translation\n      __typename\n    }\n    freeCancellationUntil\n    hasCrib\n    blockMatchTags {\n      childStaysForFree\n      freeStayChildrenAges\n      __typename\n    }\n    thirdPartyInventoryContext {\n      isTpiBlock\n      __typename\n    }\n    bundle @include(if: $includeBundle) {\n      highlightedText\n      __typename\n    }\n    __typename\n  }\n  __typename\n}\n\nfragment BookerExperienceData on SearchResultProperty {\n  bookerExperienceContentUIComponentProps {\n    ... on BookerExperienceContentLoyaltyBadgeListProps {\n      badges {\n        amount\n        variant\n        key\n        title\n        hidePopover\n        popover\n        tncMessage\n        tncUrl\n        logoSrc\n        logoAlt\n        __typename\n      }\n      __typename\n    }\n    ... on BookerExperienceContentFinancialBadgeProps {\n      paymentMethod\n      backgroundColor\n      hideAccepted\n      __typename\n    }\n    __typename\n  }\n  __typename\n}\n\nfragment PriceDisplayInfoIrene on PriceDisplayInfoIrene {\n  badges {\n    name {\n      translation\n      __typename\n    }\n    tooltip {\n      translation\n      __typename\n    }\n    style\n    identifier\n    __typename\n  }\n  chargesInfo {\n    translation\n    __typename\n  }\n  displayPrice {\n    copy {\n      translation\n      __typename\n    }\n    amountPerStay {\n      amount\n      amountRounded\n      amountUnformatted\n      currency\n      __typename\n    }\n    __typename\n  }\n  averagePricePerNight {\n    amount\n    amountRounded\n    amountUnformatted\n    currency\n    __typename\n  }\n  priceBeforeDiscount {\n    copy {\n      translation\n      __typename\n    }\n    amountPerStay {\n      amount\n      amountRounded\n      amountUnformatted\n      currency\n      __typename\n    }\n    __typename\n  }\n  rewards {\n    rewardsList {\n      termsAndConditions\n      amountPerStay {\n        amount\n        amountRounded\n        amountUnformatted\n        currency\n        __typename\n      }\n      breakdown {\n        productType\n        amountPerStay {\n          amount\n          amountRounded\n          amountUnformatted\n          currency\n          __typename\n        }\n        __typename\n      }\n      __typename\n    }\n    rewardsAggregated {\n      amountPerStay {\n        amount\n        amountRounded\n        amountUnformatted\n        currency\n        __typename\n      }\n      copy {\n        translation\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n  useRoundedAmount\n  discounts {\n    amount {\n      amount\n      amountRounded\n      amountUnformatted\n      currency\n      __typename\n    }\n    name {\n      translation\n      __typename\n    }\n    description {\n      translation\n      __typename\n    }\n    itemType\n    productId\n    __typename\n  }\n  excludedCharges {\n    excludeChargesAggregated {\n      copy {\n        translation\n        __typename\n      }\n      amountPerStay {\n        amount\n        amountRounded\n        amountUnformatted\n        currency\n        __typename\n      }\n      __typename\n    }\n    excludeChargesList {\n      chargeMode\n      chargeInclusion\n      chargeType\n      amountPerStay {\n        amount\n        amountRounded\n        amountUnformatted\n        currency\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n  taxExceptions {\n    shortDescription {\n      translation\n      __typename\n    }\n    longDescription {\n      translation\n      __typename\n    }\n    __typename\n  }\n  displayConfig {\n    key\n    value\n    __typename\n  }\n  serverTranslations {\n    key\n    value\n    __typename\n  }\n  __typename\n}\n\nfragment NonPropertyMarkers on MapMarkersQueries {\n  airports @include(if: $includeAirportMarkers) {\n    markers(airportsInput: $airportsInput, markersInput: $markersInput) {\n      id\n      translation {\n        name\n        id\n        __typename\n      }\n      location {\n        latitude\n        longitude\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n  cities @include(if: $includeCityMarkers) {\n    markers(citiesInput: $citiesInput, markersInput: $markersInput) {\n      id\n      translation {\n        name\n        id\n        __typename\n      }\n      location {\n        latitude\n        longitude\n        __typename\n      }\n      photos {\n        url\n        height\n        width\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n  landmarks @include(if: $includeLandmarkMarkers) {\n    markers(landmarksInput: $landmarksInput, markersInput: $markersInput) {\n      id\n      translation {\n        name\n        id\n        __typename\n      }\n      location {\n        latitude\n        longitude\n        __typename\n      }\n      photos {\n        url\n        height\n        width\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n  beaches @include(if: $includeBeachMarkers) {\n    markers(beachesInput: $beachesInput, markersInput: $markersInput) {\n      id\n      translation {\n        name\n        id\n        description\n        __typename\n      }\n      location {\n        latitude\n        longitude\n        __typename\n      }\n      photos {\n        url\n        height\n        width\n        __typename\n      }\n      review {\n        count\n        score\n        __typename\n      }\n      geometry\n      __typename\n    }\n    __typename\n  }\n  ski @include(if: $includeSkiMarkers) {\n    resortMarkers(skiResortsInput: $skiResortsInput, markersInput: $markersInput) {\n      id\n      translation {\n        name\n        id\n        description\n        __typename\n      }\n      location {\n        latitude\n        longitude\n        __typename\n      }\n      lifts {\n        id\n        translation {\n          name\n          id\n          description\n          __typename\n        }\n        location {\n          longitude\n          latitude\n          __typename\n        }\n        liftTypes\n        __typename\n      }\n      __typename\n    }\n    translations {\n      liftLabel\n      __typename\n    }\n    __typename\n  }\n  __typename\n}\n"
"""  #  ← NO toques nada dentro de estas triple comillas
GRAPHQL_QUERY = json.loads(Raw_QUERY)
GRAPHQL_QUERY = GRAPHQL_QUERY.lstrip("\ufeff")

# 2) Elimina \r ocultos
GRAPHQL_QUERY = GRAPHQL_QUERY.replace("\r", "")

# 3) Asegúrate de quitar el salto inicial vacío
if GRAPHQL_QUERY.startswith("\n"):
    GRAPHQL_QUERY = GRAPHQL_QUERY[1:]
GRAPHQL_QUERY = json.loads(Raw_QUERY)


# ───────────────────── FUNCIÓN ÚNICA ───────────────────── #
failed_tiles: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
results: Dict[str, dict] = {}

def booking_query(sw: Tuple[float, float],
                  ne: Tuple[float, float]) -> Dict[str, Any] | None:
    """Devuelve el JSON tal cual lo devuelve Booking. Devuelve None si falla."""
    sw_lat, sw_lon = sw
    ne_lat, ne_lon = ne

    variables = {
        "input": {
            "location": {
                "destType": "BOUNDING_BOX",
                "boundingBox": {
                    "swLat": sw_lat,
                    "swLon": sw_lon,
                    "neLat": ne_lat,
                    "neLon": ne_lon,
                    "precision": 1
                },
                "initialDestination": {
                    "destType": "REGION",
                    "destId": 767
                },
                "hotelIds": []
            },
            "nbRooms": 1, "nbAdults": 2, "nbChildren": 0,
            "filters": {},
            "pagination": { "rowsPerPage": 100, "offset": 0 },
            "seoThemeIds": [],
        },
        "markersInput":   { "actionType": "SEARCH_RESULTS" },
        "airportsInput":  { "count": 0, "searchStrategy": {} },
        "citiesInput":    { "count": 0, "searchStrategy": {} },
        "landmarksInput": { "count": 0, "searchStrategy": {} },
        "beachesInput":   { "count": 0, "searchStrategy": {} },
        "skiResortsInput":{ "count": 0, "searchStrategy": {} },
        "includeBeachMarkers":   False,
        "includeSkiMarkers":     False,
        "includeBundle":         False,
        "includeCityMarkers":    False,
        "includeAirportMarkers": False,
        "includeLandmarkMarkers":False,
    }

    payload = {
        "operationName": "MapMarkersDesktop",
        "variables": variables,
        "query": GRAPHQL_QUERY
    }

    try:
        r = requests.post(BOOKING_URL, json=payload, headers=HEADERS, timeout=25, verify=False)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f" ERROR {sw}–{ne}: {e}")
        failed_tiles.append((sw, ne))
        return None

def process_tile(sw: Tuple[float, float],
                 ne: Tuple[float, float],
                 depth: int = 0) -> None:
    """Evalúa el tile, decide procesar, subdividir o saltar."""
    lat_min, lon_min = sw
    lat_max, lon_max = ne
    box = f"[{lat_min:.4f},{lon_min:.4f}] – [{lat_max:.4f},{lon_max:.4f}]"
    indent = "-" * depth

    data = booking_query(sw, ne)
    if data is None:
        print(f"{indent}{box}: ❌ Fallo al obtener datos")
        return

    results_list = data['data']['searchQueries']['search']['results']
    total = len(results_list)

    if total == 0:
        print(f"{indent}{box}: vacío (0)")
        return

    if total < PAGE_SIZE and total != 30:
        print(f"{indent}{box}: procesar ({total} ofertas)")
        for offer in results_list:
            name = str(offer['basicPropertyData']['pageName'])
            if name not in results:
                try:
                    results[name] = {
                        "name": name,
                        "lat":       offer['basicPropertyData']["location"]["latitude"],
                        "lng":       offer['basicPropertyData']["location"]["longitude"], 
                        "DisplayName": offer['displayName']['text'],
                    }
                except:
                     results[name] = {
                        "name": name,
                        "lat":       offer['basicPropertyData']["location"]["latitude"],
                        "lng":       offer['basicPropertyData']["location"]["longitude"], 
                        "DisplayName": None
                    }

        return

    print(f"{indent}{box}: subdividir (más de 100)")
    if (lat_max - lat_min <= MIN_TILE_DEG) or (lon_max - lon_min <= MIN_TILE_DEG):
        print(f"{indent}Tamaño mínimo alcanzado → IGNORADO para no recursar infinito")
        return

    lat_mid = (lat_min + lat_max) / 2
    lon_mid = (lon_min + lon_max) / 2
    sub_tiles = [
        ((lat_min, lon_min), (lat_mid, lon_mid)),
        ((lat_min, lon_mid), (lat_mid, lon_max)),
        ((lat_mid, lon_min), (lat_max, lon_mid)),
        ((lat_mid, lon_mid), (lat_max, lon_max)),
    ]
    time.sleep(SLEEP_SECS)
    for sub_sw, sub_ne in sub_tiles:
        process_tile(sub_sw, sub_ne, depth + 1)
        time.sleep(SLEEP_SECS)
def ad_query(pageName):
    return "https://www.booking.com/hotel/es/"+pageName+".es.html"

def extract_booking(html):
    """
    Extrae el nombre del anfitrión de VRBO a partir del HTML.
    Busca el patrón "Propietario/a: [nombre]" y devuelve el nombre,
    o None si no lo encuentra.
    """
    m = re.search(r'"HotelierInfo","name"\s*:\s*"([^"]+)"', html)
    if m:
        host = m.group(1)
        return host
    return None
def extract_booking_host_cadena(text):
    """
    Dado el texto plano de Booking (`body.innerText`), busca la línea
    “Marca/cadena de hotel” y devuelve la línea justo anterior, que
    corresponde al nombre del anfitrión/marca. Si no encuentra el patrón,
    devuelve None.
    """
    # (?m) = multiline, ^ y $ se aplican por línea
    # (?i) = ignore case
    patrón = re.compile(r'(?mi)^\s*(.+?)\s*$\r?\n\s*Marca/cadena de hotel\b')
    match = patrón.search(text)
    if match:
        return match.group(1).strip()
    return None

def details(driver,url,  timeout=10, pause=0.5):
    """Return extra booking data such as rooms, license number and host."""
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 3)
    except:
        return None, None, None, None, None, None
    # Espera unos segundos para que cargue la página, ajustar el tiempo según la conexión o condiciones
    time.sleep(5)
    n_adultos=None
    n_niños=None

    try:
        html_source=driver.page_source
        
        match_capn = re.findall(r'aria-label="(\d+)\s+adultos?(?:,\s*(\d+)\s+niños?)?', html_source)
        if match_capn:
          if len(match_capn)==1:
            match_cap = re.search(r'aria-label="(\d+)\s+adultos?(?:,\s*(\d+)\s+niños?)?', html_source)
            n_adultos = int(match_cap.group(1))
            n_niños = int(match_cap.group(2)) if match_cap.group(2) else 0
        html = driver.execute_script("return document.body.innerText;")
        
        matches=re.findall(r'dormitorio\s*(\d)', html, flags=re.IGNORECASE)
        dormitorios = [int(num) for num in matches]
        num_dorm=None
        if dormitorios and len(match_capn)==1:
            num_dorm=max(dormitorios)

        numero_de_licencia=None
        match = re.search(r'Número de licencia:\s*([^\n<]+)', html_source, re.IGNORECASE)
        if match:
            numero_de_licencia = match.group(1).strip()

        tipo_alojamiento=None
        match_tipo = re.search(r"Ofertas en .*?\(([^)]+)\)",  html, re.IGNORECASE)
        if match_tipo:
            tipo_alojamiento = match_tipo.group(1)

        host=extract_booking(html_source)
        if host is None:
            host=extract_booking_host_cadena(html)
        if host is None:
            
            try:
                btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, 'button[data-testid="trader-information-modal-button"]')
                )
            )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                btn.click()
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
            pat = r'^\s*Datos de la empresa\s*\n\s*([^\n\r]+)'
            m   = re.search(pat, html, flags=re.IGNORECASE | re.MULTILINE)


            if m:
                host = m.group(1).strip()

        if host is not None:
            if len(host)>1000:
                host=None
            
        return num_dorm, numero_de_licencia, n_adultos, n_niños, tipo_alojamiento, host   
    except NoSuchElementException:
        return None, None, None, None, None, None
# ──────────────── MAIN ───────────────── #
if __name__ == "__main__":
    SW = (39.10129556224702, 2.3739429649583883)
    NE = (40.17457797140084, 3.4572007960993005)

    print("▶️ Iniciando procesamiento inicial...")
    process_tile(SW, NE)

    if failed_tiles:
        print(f"\n🔁 Reintentando {len(failed_tiles)} tiles fallidos...")
        time.sleep(120)
        for sw, ne in failed_tiles[:]:
            process_tile(sw, ne)
            time.sleep(SLEEP_SECS)

    df = pd.DataFrame(results.values())
    print(df)
    df.to_csv("booking_mallorca_1.csv", index=False)
    driver = webdriver.Chrome()
    driver.set_page_load_timeout(30)
    num_dorms=[]
    numero_de_licencias=[]
    adultos=[]
    niños=[]
    tipo_alojamientos=[]
    hosts=[]
    urls=[]

    for name in df['name']:
        url=ad_query(name)
        urls.append(url)
        try:
            num_dorm, numero_de_licencia, n_adultos, n_niños, tipo_alojamiento, host=details(driver, url)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            num_dorm = None
            numero_de_licencia = None
            n_adultos = None
            n_niños = None
            tipo_alojamiento=None
            host=None
        num_dorms.append(num_dorm)
        numero_de_licencias.append(numero_de_licencia)
        adultos.append(n_adultos)
        niños.append(n_niños)
        tipo_alojamientos.append(tipo_alojamiento)
        hosts.append(host)

    df['habitaciones'] = num_dorms
    df['licencia'] = numero_de_licencias
    df['capacidad'] = adultos
    df['niños'] = niños
    df['tipo de alojamiento']= tipo_alojamientos
    df['anfitrion']=hosts
    df['url']=urls
    df['fecha']=str(date.today())
    driver.quit()

    print(df)
    df.to_csv("booking_mallorca_"+str(date.today())+".csv", index=False)

    print(f"\n✅ Guardado {len(df)} alojamientos")

    #data = booking_query(SW, NE)
    #print(data['data']['searchQueries']['search']['results'][20]['basicPropertyData']['pageName'])
   
