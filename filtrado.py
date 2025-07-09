import pandas as pd
import re
import numpy as np
from sklearn.neighbors import NearestNeighbors
from difflib import SequenceMatcher
from datetime import date

df_airbnb=pd.read_csv('C:/Users/u155043/airbnb_Baleares_2025-06-25.csv')
df_airbnb['nombre']=df_airbnb['Nombre']
df_booking=pd.read_csv('C:/Users/u155043/booking_Baleares_2025-06-24.csv')
df_vrbo=pd.read_csv('C:/Users/u155043/vrbo_Baleares_2025-06-27.csv')
df_holidu=pd.read_csv('C:/Users/u155043/holidu_Baleares_2025-06-25.csv')
dfs={'airbnb':df_airbnb,
     'booking': df_booking,
      'vrbo': df_vrbo,
       'holidu': df_holidu}


df_fulls = {} 
def normalize_license(text):
    

    
    if pd.isna(text):
        return ''
    
    # Convertir a string y mayúsculas
    s = str(text).upper()
    
    # Eliminar espacios iniciales y finales
    s = s.strip()
    
    # Eliminar años (patrones como /2017, ,2017, etc.)
    s = re.sub(r'[,/]\s*(?:19|20)\d{2}', '', s)
    
    # Caso 1: Patrones ESFCTU seguido de dígitos y luego cualquier prefijo de letras
    esfctu_match = re.search(r'ESFCTU\d+([A-Z]+)[/\s]*(\d+)', s)
    if esfctu_match:
        prefix = esfctu_match.group(1)  # Cualquier secuencia de letras
        number = esfctu_match.group(2)  # números
        return f'{prefix}{number}'
    
    # Caso 2: Buscar patrón L seguido de números con posibles letras intermedias
    # Esto captura casos como L95E138, L12S4282, etc.
    l_match = re.search(r'L(\d+[A-Z]?\d*)', s)
    if l_match:
        license_part = l_match.group(1)  # Parte después de L
        
        # Extraer solo los números finales más largos
        # Para L95E138 -> buscar 138 (números al final)
        # Para L12S4282 -> buscar 4282 (números al final)
        numbers_match = re.search(r'(\d+)$', license_part)
        if numbers_match:
            final_numbers = numbers_match.group(1)
            return f'L{final_numbers}'
        
        # Si no hay números al final, extraer todos los números
        all_numbers = re.sub(r'[^0-9]', '', license_part)
        if all_numbers:
            return f'L{all_numbers}'
    
    # Fallback: aplicar la lógica original
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s

def canonical_license(text: str) -> str:
    """
    - Normaliza la cadena.

    """
    s_clean = normalize_license(text)
    if not s_clean:
        return ''

    # Extrae primer bloque Letras+Dígitos o Dígitos+Letras
    m = re.search(r'([A-Z]+)(\d+)', s_clean)
    if m:
        letters, num = m.groups()
    else:
        m2 = re.search(r'(\d+)([A-Z]+)', s_clean)
        if m2:
            num, letters = m2.groups()


    # Clasificación:
    #  • si hay 'E' → ETV
    #  • elif hay 'V' y 'T' → VT
    #  • else → ETV
    if m:
      s_clean=f'{letters}{num}'
      if 'A'in letters or 'H' in letters:
        return s_clean
      if 'E' in letters:
          suf = 'ETV'
      elif 'V' in letters and 'T' in letters:
          suf = 'VT'
      else:
        return s_clean

      return f'{suf}{num}'
    elif m2:
      s_clean=f'{letters}{num}'
      if 'A'in letters or 'H' in letters:
        return s_clean
      if 'E' in letters:
          suf = 'ETV'
      elif 'V' in letters and 'T' in letters:
          suf = 'VT'
      else:
        return s_clean

      return f'{suf}{num}'

    else:
        return s_clean
    
def normalize_title(title):
    if pd.isna(title): return ''
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s]', ' ', title)
    return re.sub(r'\s+', ' ', title).strip()

for plataforma, df in dfs.items():
    if plataforma=='holidu':
        df['licencia']=df['licencia'].replace('No proporcionado', '')
        df = df[~df['anfitrion'].str.contains('Vrbo|Booking.com', na=False)]
    host_col = 'anfitrion'

    df_no_host = df[df['anfitrion'].isna()]
    df_no_host['_license_norm']=df_no_host['licencia'].apply(canonical_license)
    df_no_host['group_id']=[i for i in range(len(df_no_host))]
    df_no_host['_host_norm']='NA'
    df_no_host['group_id']=df_no_host['group_id'].astype(str)+'No_host'

    missing_mask = df[host_col].isna() | (df[host_col].astype(str).str.strip() == '')
    n_missing = missing_mask.sum()
    n_total = len(df)

    summary_df = pd.DataFrame({
        'Metric': ['Total listings', 'Listings without host'],
        'Count': [n_total, n_missing],
        'Share (%)': [100.0, round(n_missing / n_total * 100, 2)]
    })

    print(summary_df)

    missing_host_mask = df[host_col].isna() | (df[host_col].astype(str).str.strip() == '')
    df_no_host = df[missing_host_mask]
    reg_col='licencia'
    license_present_mask = ~(df_no_host[reg_col].isna() | (df_no_host[reg_col].astype(str).str.strip() == ''))
    n_license_in_nohost = license_present_mask.sum()
    n_nohost_total = len(df_no_host)

    summary_df = pd.DataFrame({
        'Metric': ['Listings without host', '…con número de licencia'],
        'Count': [n_nohost_total, n_license_in_nohost],
        'Share (%)': [round(100 * n_nohost_total / len(df), 2),
                    round(100 * n_license_in_nohost / n_nohost_total, 2)]
    })
    print(summary_df)
    mask_host = ~(df[host_col].isna() | (df[host_col].astype(str).str.strip() == ''))
    mask_license = ~(df[reg_col].isna() | (df[reg_col].astype(str).str.strip() == ''))
    df_filtered = df[mask_host & mask_license]

    summary_df = pd.DataFrame({
        'Metric': ['Total listings', 'Listings with host + license'],
        'Count': [len(df), len(df_filtered)],
        'Share (%)': [100.0, round(100 * len(df_filtered) / len(df), 2)]
    })

    print(summary_df)


    df_filtered['_license_norm'] = df_filtered[reg_col].apply(canonical_license)
    df_filtered['_host_norm'] = df_filtered[host_col].astype(str).str.strip().str.lower()

    group_ids = [''] * len(df_filtered)                       # ⬅️  inicializamos el vector

    for lic, grp in df_filtered.groupby('_license_norm'):
        counts = grp['_host_norm'].value_counts()
        max_rep = counts.max()

        # ─────────────────────────────
        # ‣ CASO 1 · ningún host repetido  (max_rep == 1)
        # ─────────────────────────────
        if max_rep == 1:
            if len(counts) > 1:
                # ⇒ hay ≥2 anfitriones distintos: ¡agrupar TODOS juntos!
                anchor_gid = f'LIC_{lic}_ALL'
                for idx in grp.index:
                    group_ids[df_filtered.index.get_loc(idx)] = anchor_gid
            else:
                # ⇒ sólo 1 anfitrión (licencia usada por un único host)
                #    -> cada anuncio queda en su propio grupo
                for idx in grp.index:
                    group_ids[df_filtered.index.get_loc(idx)] = f'ROW_{idx}_{lic}'
            continue   # ¡pasa al siguiente lic!

        # ─────────────────────────────
        # ‣ CASO 2 · hay un anfitrión dominante  (max_rep ≥ 2)
        # ─────────────────────────────
        dom_host = counts[counts == max_rep].index[0]        # el que más repite
        dom_idx  = grp[grp['_host_norm'] == dom_host].index  # anuncios de ese host
        oth_idx  = grp[grp['_host_norm'] != dom_host].index  # resto de anfitriones

        # → 2.A  Anuncio-ancla del host dominante
        anchor_idx  = dom_idx[0]
        anchor_gid  = f'LIC_{lic}_ANCH'
        group_ids[df_filtered.index.get_loc(anchor_idx)] = anchor_gid

        # → 2.B  Todos los *otros* anfitriones van con el ancla
        for idx in oth_idx:
            group_ids[df_filtered.index.get_loc(idx)] = anchor_gid

        # → 2.C  Resto de anuncios del host dominante quedan separados
        for idx in dom_idx[1:]:
            group_ids[df_filtered.index.get_loc(idx)] = f'ROW_{idx}_{lic}'

    # añadimos la columna final
    df_filtered['group_id'] = group_ids
    df_unique_groups_licenses = df_filtered.drop_duplicates(subset=['group_id'], keep='first')


    df = df.dropna(subset=['lat', 'lng'])
    
    df = df.reset_index(drop=True)
    
    df['title_norm'] = df['nombre'].astype(str).apply(normalize_title)
    coords_rad = np.radians(df[['lat', 'lng']].values)
    kms_per_radian = 6371.0088
    radius_m = 200
    radius_rad = radius_m / 1000.0 / kms_per_radian

    # Creamos el "buscador" de vecinos
    # Usamos 'ball_tree' y 'haversine', que es perfecto para datos geográficos
    nn_search = NearestNeighbors(radius=radius_rad, algorithm='ball_tree', metric='haversine')
    nn_search.fit(coords_rad)
    print("Índice espacial construido.")


    # --- PASO 2: Union-Find (como antes) ---
    print("\nPaso 2: Inicializando estructura Union-Find...")
    # ... (código de Union-Find: parent, rank, find, union, diff_ok) ...
    parent = {i: i for i in df.index}
    rank   = {i: 0 for i in df.index}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry: return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[ry] < rank[rx]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    def diff_ok(a, b, max_diff):
        return abs(a - b) <= max_diff

    # --- PASO 3: Bucle principal iterativo ---
    print("\nPaso 3: Fusionando duplicados con búsqueda de vecinos...")

    SIM_THRESHOLD = 0.7 

    # Usamos un array booleano para marcar los nodos ya procesados como "semilla"
    processed_as_seed = np.zeros(len(df), dtype=bool)


    # Iteramos sobre cada alojamiento
    for i in range(len(df)):
        print(f"  Procesando semilla {i+1}/{len(df)}...", end='\r')

        # Si este nodo ya fue agrupado como parte del vecindario de otra semilla, no necesita ser semilla
        # (Esto es una optimización, puedes quitarla si quieres que todos sean semilla)
        if find(i) != i:
            continue

        # Consultamos el índice para encontrar los vecinos de `i` en un radio de 500m
        # `query_radius` devuelve una lista de arrays. Tomamos el primer elemento.
        # Los resultados incluyen al propio punto `i`.
        neighbor_indices = nn_search.radius_neighbors([coords_rad[i]], return_distance=False)[0]

        # Comparamos la semilla `i` con cada uno de sus vecinos
        for j in neighbor_indices:
            # No comparar un punto consigo mismo, y solo procesar pares (i, j) donde i < j
            if i >= j:
                continue

            # --- Aplicamos la misma lógica de filtros que antes ---

            # Filtro 1: Capacidad y habitaciones
            if not (diff_ok(df.at[i,'capacidad'], df.at[j,'capacidad'], 0) and
                    diff_ok(df.at[i,'habitaciones'], df.at[j,'habitaciones'], 0)):
                continue

           
            if not SequenceMatcher(None, df.at[i,'title_norm'], df.at[j,'title_norm']).ratio() >= SIM_THRESHOLD:
                continue
            union(i, j) # Si pasa los filtros, fusionar

    print("\nProcesamiento completado.")

    # --- PASO 4: Generar group IDs finales (como antes) ---
    print("\nPaso 4: Generando los group_id finales...")
    df['dup_group_id'] = df.index.to_series().apply(find)

    host_col = 'anfitrion'
    reg_col='dup_group_id'

    mask_host = ~(df[host_col].isna() | (df[host_col].astype(str).str.strip() == ''))
    df_filtered = df[mask_host]
    df=df_filtered

    df['_license_norm'] = df[reg_col].astype(str).str.strip().str.upper()
    df['_host_norm'] = df[host_col].astype(str).str.strip().str.lower()
    group_ids = [''] * len(df)
    for lic, grp in df.groupby('_license_norm'):
        counts = grp['_host_norm'].value_counts()
        max_rep = counts.max()

        # ─────────────────────────────
        # ‣ CASO 1 · ningún host repetido  (max_rep == 1)
        # ─────────────────────────────
        if max_rep == 1:
            if len(counts) > 1:
                # ⇒ hay ≥2 anfitriones distintos: ¡agrupar TODOS juntos!
                anchor_gid = f'LIC_{lic}_ALL'
                for idx in grp.index:
                    group_ids[df.index.get_loc(idx)] = anchor_gid
            else:
                # ⇒ sólo 1 anfitrión (licencia usada por un único host)
                #    -> cada anuncio queda en su propio grupo
                for idx in grp.index:
                    group_ids[df.index.get_loc(idx)] = f'ROW_{idx}'
            continue   # ¡pasa al siguiente lic!

        # ─────────────────────────────
        # ‣ CASO 2 · hay un anfitrión dominante  (max_rep ≥ 2)
        # ─────────────────────────────
        dom_host = counts[counts == max_rep].index[0]        # el que más repite
        dom_idx  = grp[grp['_host_norm'] == dom_host].index  # anuncios de ese host
        oth_idx  = grp[grp['_host_norm'] != dom_host].index  # resto de anfitriones

        # → 2.A  Anuncio-ancla del host dominante
        anchor_idx  = dom_idx[0]
        anchor_gid  = f'LIC_{lic}_ANCH'
        group_ids[df.index.get_loc(anchor_idx)] = anchor_gid

        # → 2.B  Todos los *otros* anfitriones van con el ancla
        for idx in oth_idx:
            group_ids[df.index.get_loc(idx)] = anchor_gid

        # → 2.C  Resto de anuncios del host dominante quedan separados
        for idx in dom_idx[1:]:
            group_ids[df.index.get_loc(idx)] = f'ROW_{idx}'

    # añadimos la columna final
    df['group_id'] = group_ids


    reg_col='licencia'
    # 1) Marca quién SÍ tiene licencia
    df['has_license'] = (
        df[reg_col].notna()
        & df[reg_col].astype(str).str.strip().ne('')
    )

    # 2) Por cada grupo, calcula:
    #    • group_has_license: si hay al menos una licencia
    #    • no_license_count: cuántas filas NO tienen licencia
    df['group_has_license'] = df.groupby('group_id')['has_license'].transform('any')
    df['no_license_count']  = df.groupby('group_id')['has_license'] \
                                .transform(lambda x: (~x).sum())

    # 3) Filtra eliminando *solo* esas filas sin licencia
    #    cuando el grupo tiene licencia Y sólo hay 1 fila sin licencia
    mask_drop = (
        (~df['has_license'])                      # fila sin licencia
        & df['group_has_license']                 # grupo con al menos 1 licencia
        & (df['no_license_count'] == 1)           # exactamente 1 sin licencia en el grupo
    )
    df_filtered = df[~mask_drop].copy()

    # 4) (Opcional) Quita columnas auxiliares
    df_filtered.drop(
        columns=['has_license','group_has_license','no_license_count'],
        inplace=True
    )

    df = df_filtered[df_filtered['licencia'].isna()].copy()
    df_unique_groups_unlicenses = df.drop_duplicates(subset=['group_id'], keep='first')
    df_unique_groups_unlicenses['group_id']=df_unique_groups_unlicenses['group_id']+'Na'
    df_full = pd.concat([
        df_unique_groups_licenses,
        df_unique_groups_unlicenses,
        df_no_host
    ])
    df_full['plataforma']=plataforma
    df_fulls[plataforma]=df_full 



df=pd.concat([df_fulls['airbnb'], df_fulls['booking'], df_fulls['vrbo'], df_fulls['holidu']])

df['_license_norm']=df['licencia'].apply(canonical_license)

df['_license_norm'] = df['_license_norm'].astype('string')
df['licencia_letras']  = df['_license_norm'].str.extract(r'^([A-Za-z]+)',  expand=False)
df['licencia_numeros'] = df['_license_norm'].str.extract(r'(\d+)$',        expand=False)
# Lista de valores a eliminar en licencia_letras
blacklist = ['AG','AT','A','H','HA','HT', 'APM','PM','TI','HPM']

# Máscara de filas a eliminar:
#   - licencia_letras está en la lista blacklist
#   - o licencia_letras contiene la palabra "EDIFICIO" (case-insensitive)
mask_remove = (
    df['licencia_letras'].isin(blacklist)
    | df['licencia_letras'].str.contains('EDIFICI', case=False, na=False)
)

# Opción 1: quedarte sólo con las filas que NO cumplen la condición
df_filtrado = df[~mask_remove].copy()

# Lista de valores a eliminar en licencia_letras
blacklist = ['Hotel','Apartahotel','Agroturismo','Hostal o pensión','Apartotel','Complejo turístico','Albergue','Resort','Suite', 'Aparthotel', 'Camping']

# Máscara de filas a eliminar:
#   - licencia_letras está en la lista blacklist
#   - o licencia_letras contiene la palabra "EDIFICIO" (case-insensitive)
mask_remove = (
    df_filtrado['tipo'].isin(blacklist)
)

# Opción 1: quedarte sólo con las filas que NO cumplen la condición
df_filtrado = df_filtrado[~mask_remove].copy()
df_filtrado.dropna(subset=['capacidad'], inplace=True)

df_filtrado.to_csv('df_filtrado_'+str(date.today())+'.csv', index=False)