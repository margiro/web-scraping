import pandas as pd
import re
import numpy as np
from sklearn.neighbors import NearestNeighbors
from difflib import SequenceMatcher
from datetime import date

def haversine(lon1, lat1, lon2, lat2):
        """
        Calcula la distancia entre dos puntos geográficos usando la fórmula de Haversine.
        Parámetros:
        - lon1, lat1: coordenadas del primer punto en grados
        - lon2, lat2: coordenadas del segundo punto en grados
        Devuelve:
        - Distancia en kilómetros (float)
        """
        # Convertir grados a radianes
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0  # Radio de la Tierra en km
        return R * c 


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


df_full_raw=pd.read('C:/Users/u155043/Dedup_ML_2025-07-09.csv')
df=df_full_raw
host_col = 'anfitrion'

df_no_host = df[df['anfitrion'].isna()]
df_no_host['_license_norm']=df_no_host['licencia'].apply(canonical_license)
df_no_host['group_id']=[i for i in range(len(df_no_host))]
df_no_host['_host_norm']='NA'
df_no_host['group_id']=df_no_host['group_id'].astype(str)+'No_host'

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
df.dropna(subset=['licencia'], inplace=True)

reg_col  = 'licencia'     # identificador de licencia turística
host_col = 'plataforma'   # nombre de la plataforma (booking, vrbo…)
lat_col  = 'lat'
lng_col  = 'lng'

mask_host = ~(df[host_col].isna() | (df[host_col].astype(str).str.strip() == ''))
mask_license = ~(df[reg_col].isna() | (df[reg_col].astype(str).str.strip() == ''))
df_filtered = df_full_raw[mask_host & mask_license]
df=df_filtered

# ----------------------------------------------
# 2. PRIMERA PASADA: lógica original de agrupación
# ----------------------------------------------
df['_license_norm'] = df[reg_col].apply(canonical_license)
df['_host_norm']    = df[host_col].astype(str).str.strip().str.lower()

group_ids = [''] * len(df)  # lista que rellenaremos con el group_id final

for lic, grp in df.groupby('_license_norm'):
    counts   = grp['_host_norm'].value_counts()
    max_rep  = counts.max()

    # ── CASO 1: ningún host repetido ──────────────────────────────
    if max_rep == 1:
        if len(counts) > 1:  # ≥2 plataformas distintas
            anchor_gid = f'LIC_{lic}_ALL'
            for idx in grp.index:
                group_ids[df.index.get_loc(idx)] = anchor_gid
        else:  # sólo 1 plataforma o una única fila
            for idx in grp.index:
                group_ids[df.index.get_loc(idx)] = f'ROW_{idx}'
        continue

    # ── CASO 2: host dominante ───────────────────────────────────
    dom_host = counts[counts == max_rep].index[0]
    dom_idx  = grp[grp['_host_norm'] == dom_host].index
    oth_idx  = grp[grp['_host_norm'] != dom_host].index

    # --- En lugar de tomar dom_idx[0], buscamos el "medoid" geográfico ---
    # Para cada index en dom_idx, calculamos la suma de distancias
    # hacia todas las filas de 'grp' (excluyéndose a sí mismo).
    # Quedamos con el que tenga la mínima suma de distancias.
    mejor_idx = None
    mejor_dist_sum = None

    # Pre-extraemos las coordenadas de todo el grupo de licencia
    coords_grp = grp[['lat', 'lng']].to_dict('index')
    # coords_grp[idx] -> {'lat': ..., 'lng': ...}

    for idx_candidato in dom_idx:
        lat_cand = coords_grp[idx_candidato]['lat']
        lng_cand = coords_grp[idx_candidato]['lng']

        # Sumar distancias a cada otra fila de grp (incluyendo otras de dom_idx y las de otros hosts)
        dist_sum = 0.0
        for idx_otro in grp.index:
            if idx_otro == idx_candidato:
                continue
            lat_otro = coords_grp[idx_otro]['lat']
            lng_otro = coords_grp[idx_otro]['lng']
            dist_sum += haversine(lng_cand, lat_cand, lng_otro, lat_otro)

        # Si es el primero o mejora la suma total, lo guardamos
        if (mejor_dist_sum is None) or (dist_sum < mejor_dist_sum):
            mejor_dist_sum = dist_sum
            mejor_idx = idx_candidato
        # En caso de empate exacto en la suma de distancias, como iteramos en el orden de dom_idx,
        # se conservará el primero (no hace falta lógica adicional).

    anchor_idx = mejor_idx
    anchor_gid = f'LIC_{lic}_ANCH'

    # 2.A) Asignamos al ancla
    group_ids[df.index.get_loc(anchor_idx)] = anchor_gid

    # 2.B) A todos los que no son del host dominante (oth_idx) también les damos el mismo anchor_gid
    for idx in oth_idx:
        group_ids[df.index.get_loc(idx)] = anchor_gid

    # 2.C) A las filas adicionales del host dominante (salvo la que elegimos como anchor) las marcamos como ROW_<idx>
    for idx in dom_idx:
        if idx == anchor_idx:
            continue
        group_ids[df.index.get_loc(idx)] = f'ROW_{idx}'

# Finalmente añadimos la columna resultante al DataFrame
df['group_id'] = group_ids

# ── PÁSO INTERMEDIO: unir numerico ↔ numerico+letra ─────────────────────
merge_threshold_km = 1.0

# Nos apoyamos en la columna 'licencia_numeros' y en si 'licencia_letras' está vacía o no.
# Para cada grupo de mismo número, buscamos pares (num_only, num+let)
for num, grp_num in df.groupby('licencia_numeros'):
    # índices de los que NO tienen letra y de los que SÍ
    idx_num_only = grp_num[ grp_num['licencia_letras'].isna() ].index
    idx_num_let  = grp_num[ grp_num['licencia_letras'].notna() ].index

    # si no hay ambos tipos, nos saltamos
    if len(idx_num_only)==0 or len(idx_num_let)==0:
        continue

    # extraemos coords en dict para no llamar pandas cada vez
    coords = grp_num[['lat','lng']].to_dict('index')

    # para cada par posible, medimos distancia
    for i in idx_num_only:
        lat1, lon1 = coords[i]['lat'], coords[i]['lng']
        if pd.isna(lat1) or pd.isna(lon1):
            continue
        for j in idx_num_let:
            lat2, lon2 = coords[j]['lat'], coords[j]['lng']
            if pd.isna(lat2) or pd.isna(lon2):
                continue

            # si están a ≤ 1 km, los unimos en el mismo grupo
            if haversine(lon1, lat1, lon2, lat2) <= merge_threshold_km:
                # escogemos como "anchor" el que tiene letra (j)
                gid_anchor = df.at[j, 'group_id']
                df.at[i, 'group_id'] = gid_anchor
# ----------------------------------------------
# 3. SEGUNDA PASADA: dividir grupos por distancia
#
# ----------------------------------------------


dist_threshold_km = 2
new_group_ids = df['group_id'].tolist()

for gid, grp in df.groupby('group_id'):
    if len(grp) <= 1:
        continue  # nada que partir

    # Clustering por proximidad ≤ 1 km (BFS sencillo)
    clusters = []        # lista de listas de índices
    for idx in grp.index:
        if any(idx in c for c in clusters):
            continue  # ya asignado a un cluster

        # Si no hay coordenadas -> su propio cluster
        if pd.isna(grp.at[idx, lat_col]) or pd.isna(grp.at[idx, lng_col]):
            clusters.append([idx])
            continue

        # BFS: agrupar todos los vecinos a ≤2 km
        current_cluster = [idx]
        queue = [idx]
        while queue:
            base = queue.pop()
            lat1, lon1 = grp.at[base, lat_col], grp.at[base, lng_col]
            for other in grp.index.difference(current_cluster):
                lat2, lon2 = grp.at[other, lat_col], grp.at[other, lng_col]
                if pd.isna(lat2) or pd.isna(lon2):
                    continue
                if haversine( lon1,lat1, lon2, lat2) <= dist_threshold_km:
                    current_cluster.append(other)
                    queue.append(other)
        clusters.append(current_cluster)

    # Si sólo hay 1 cluster, no hace falta cambiar los IDs
    if len(clusters) == 1:
        continue

    # Renombrar los clusters: _LOC1, _LOC2, ...
    for cnum, cluster in enumerate(clusters, start=1):
        suffix = f'_LOC{cnum}'
        for idx in cluster:
            pos = df.index.get_loc(idx)
            new_group_ids[pos] = f'{gid}{suffix}'

df['group_id'] = new_group_ids
df_unique_groups_licenses = df.drop_duplicates(subset=['group_id'], keep='first')
df_no_host=df_no_host[~df_no_host['_license_norm'].isin(df_unique_groups_licenses['_license_norm'])]

df_full=pd.concat([df_unique_groups_licenses, df_unique_groups_unlicenses, df_no_host ])
df_full_raw.drop_duplicates()
df_full_raw.to_csv('df_final_'+str(date.today())+'.csv', index=False)
