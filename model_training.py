import pandas as pd, random, itertools
import math
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import re
import unicodedata
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import requests
from huggingface_hub import configure_http_backend, snapshot_download
from sentence_transformers import SentenceTransformer
import torch
import joblib, os
from sklearn.model_selection import GroupShuffleSplit
import torch.nn as nn, torch.nn.functional as F
from pytorch_metric_learning.losses import ContrastiveLoss
from sklearn.metrics import  precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from datetime import date
from rapidfuzz.distance import Levenshtein  

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, X):
        self.pairs = pairs          # ndarray Nx3  (i, j, label)
        self.X     = X              # matriz de características
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        i, j, y = self.pairs[idx]
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.X[j]),
            torch.tensor(y, dtype=torch.float32)
        )
def main():

    df_full_raw=pd.read_csv('C:/Users/u155043/df_filtrado_2025-06-30.csv')
    df=df_full_raw
    df.dropna(subset=['licencia'], inplace=True)

    reg_col  = 'licencia'     # identificador de licencia turística
    host_col = 'plataforma'   # nombre de la plataforma (booking, vrbo…)
    lat_col  = 'lat'
    lng_col  = 'lng'

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



    # ----------------------------------------------
    # 1. FUNCIÓN AUXILIAR: distancia Haversine en km
    # ----------------------------------------------
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

    df_training = df[~df['group_id'].astype(str).str.contains('ANCH', na=False)]


    df=df_training

    dummy_row = pd.DataFrame([{
        "nombre": "dummy",
        "habitaciones": np.nan,
        "anfitrion": np.nan,
        "tipo": np.nan,
        "plataforma": np.nan,
        "capacidad": np.nan,
        "latitud": np.nan,
        "longitud": np.nan,
        
    }])
    df = pd.concat([df, dummy_row], ignore_index=True)

    def normalize(txt) -> str:
         txt = unicodedata.normalize("NFKD", str(txt).lower())
         txt = txt.encode("ascii", "ignore").decode()
         return re.sub(r"\s+", " ", txt).strip()

    df["title_norm"] = df["nombre"].fillna("").astype(str).apply(normalize)


    df["habitaciones"] = df["habitaciones"].fillna(-1).astype(int)
    df["anfitrion"]    = df["anfitrion"].fillna("host_desconocido").astype(str)
    df["tipo"]         = df["tipo"].fillna("tipo_desconocido").astype(str)
    df["plataforma"]   = df["plataforma"].fillna("platform_desconocido").astype(str)
    df["latitud"] = df["lat"].fillna(0).astype(float)
    df["longitud"] = df["lng"].fillna(0).astype(float)

    le_plat = LabelEncoder().fit(df["plataforma"])
    le_tipo = LabelEncoder().fit(df["tipo"])
    le_host = LabelEncoder().fit(df["anfitrion"])

    df["plat_id"] = le_plat.transform(df["plataforma"])
    df["tipo_id"] = le_tipo.transform(df["tipo"])
    df["host_id"] = le_host.transform(df["anfitrion"])

    num_scaler = MinMaxScaler()

    cols_to_scale = ["capacidad", "habitaciones", "lat", "lng"]
    scaled_cols = ["cap_scaled", "hab_scaled", "lat_scaled", "lon_scaled"]

    df[scaled_cols] = num_scaler.fit_transform(df[cols_to_scale].astype(float))


    def insecure_session() -> requests.Session:
        s = requests.Session()
        s.verify = False          # <─ aquí bypass SSL
        return s

    configure_http_backend(backend_factory=insecure_session)

    # ───────────────────────────────────────────────────────
    # 1) DESCARGAR *TODO* EL REPO
    #    snapshot_download clona la carpeta entera (≈50 MB)
    # ───────────────────────────────────────────────────────
    MODEL_REPO = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    LOCAL_DIR  = Path("C:/Users/u155043/hf_models/paraphrase-MiniLM-L6-v2")

    snapshot_download(
        repo_id               = MODEL_REPO,
        local_dir             = LOCAL_DIR,
        local_dir_use_symlinks= False,   # copias reales → más portable en Windows
        revision              = None     # última versión
    )

    # ───────────────────────────────────────────────────────
    # 2) CARGAR EL MODELO DESDE DISCO
    # ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert  = SentenceTransformer(str(LOCAL_DIR), device=device)

    embeddings = sbert.encode(
        df["title_norm"].tolist(),
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    X = np.hstack([
        embeddings,
        df[["plat_id", "tipo_id", "host_id",
            "cap_scaled", "hab_scaled",
            "lat_scaled", "lon_scaled"]].values  # <-- AÑADIMOS LAS NUEVAS COLUMNAS
    ])

    # La forma de X ahora será (n, 391) en lugar de (n, 389)
    print("Shape de X:", X.shape)



    df.to_csv("df_preprocessed.csv", index=False)
    np.save("X_preprocessed.npy", X)
    joblib.dump({
        "scaler": num_scaler,
        "le_plat": le_plat,
        "le_tipo": le_tipo,
        "le_host": le_host
    }, "encoders.pkl")

    df  = pd.read_csv("df_preprocessed.csv")
    X   = np.load("X_preprocessed.npy").astype("float32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    def haversine_km(lat1, lon1, lat2, lon2):
        """Calcula la distancia en km entre dos puntos lat-lon."""
        R = 6371  # Radio de la Tierra en km

        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) * math.sin(dlon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance

    def make_pairs(df, max_pos_per_gid=10, neg_ratio=3, num_geo_negatives=70000, dist_threshold_km=0.5):
        pos, neg = [], []

        # 1. Positivos (mismo group_id) - Sin cambios
        for gid, sub in df.groupby("group_id"):
            idx = sub.index.tolist()
            if len(idx) < 2: continue
            pairs = list(itertools.combinations(idx, 2))
            random.shuffle(pairs)
            for i, j in pairs[:max_pos_per_gid]:
                pos.append((i, j, 1))

        # 2. Negativos "difíciles" (misma config, diferente group_id) - Sin cambios
        for _, sub in df.groupby(["capacidad", "habitaciones"]):
            idx = sub.index.tolist()
            if len(idx) < 2: continue
            for i, j in itertools.combinations(idx, 2):
                if df.at[i, "group_id"] != df.at[j, "group_id"]:
                    neg.append((i, j, 0))

        # 3. NUEVO: Negativos "fáciles" geográficos
        # Genera pares aleatorios y si están lejos, son negativos seguros.
        all_indices = df.index.tolist()
        for _ in range(num_geo_negatives):
            i, j = random.sample(all_indices, 2)

            # Evitar comparar un par que ya sabemos que es positivo
            if df.at[i, "group_id"] == df.at[j, "group_id"]:
                continue

            lat1, lon1 = df.at[i, "lat"], df.at[i, "lng"]
            lat2, lon2 = df.at[j, "lat"], df.at[j, "lng"]

            if haversine_km(lat1, lon1, lat2, lon2) > dist_threshold_km:
                neg.append((i, j, 0))

        # Combinar y muestrear
        # Quitamos duplicados por si acaso
        neg = list(set(neg))
        neg = random.sample(neg, k=min(len(neg), neg_ratio * len(pos)))

        pairs = pos + neg
        random.shuffle(pairs)
        return pairs

    # Luego, al llamar a la función, ya usará la nueva lógica:
    pairs = make_pairs(df)
    print(f"Pares: {len(pairs)}  (positivos {sum(l for *_,l in pairs)})")

    pairs_df = pd.DataFrame(pairs, columns=["i", "j", "label"])
    df["group_id"] = df["group_id"].fillna("no_gid")
    groups   = df["group_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(pairs_df, groups=pairs_df["i"].map(groups)))

    train_pairs = pairs_df.iloc[train_idx].to_numpy()
    val_pairs   = pairs_df.iloc[val_idx].to_numpy()
    train_set = PairDataset(train_pairs, X)
    val_set   = PairDataset(val_pairs,   X)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=512, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=512, shuffle=False, num_workers=2
    )



    class EmbNet(nn.Module):
        def __init__(self, in_dim=391, out_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim)
            )
        def forward(self, x):
            return F.normalize(self.net(x), p=2, dim=1)   # ℓ2-norm

    model = EmbNet(in_dim=X.shape[1], out_dim=128).to(device)

    # ▶ CosineEmbeddingLoss: espera target = 1 (positivo) ó –1 (negativo)
    criterion = nn.CosineEmbeddingLoss(margin=0.6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    EPOCHS = 20
    best_f1, best_state = 0, None

    def evaluate(loader):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xa, xb, y in loader:
                xa, xb = xa.to(device), xb.to(device)
                ya, yb = model(xa), model(xb)
                # similitud coseno ∈ [-1,1]  → distancia = 1-sim
                sim = F.cosine_similarity(ya, yb)
                y_true.extend(y.numpy())
                y_pred.extend((sim.cpu().numpy() > 0.75).astype(int))  # umbral 0.75≈margin
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score   (y_true, y_pred, zero_division=0)
        print(f"Precision: {prec:.4f}  Recall: {rec:.4f}")

        return f1_score(y_true, y_pred)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xa, xb, y in train_loader:
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)
            ya, yb = model(xa), model(xb)
            target = 2*y - 1                     # 1→+1   0→-1
            loss = criterion(ya, yb, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        f1_val = evaluate(val_loader)
        print(f"epoc {epoch:02d}  |  F1 valid = {f1_val:.4f}")
        if f1_val > best_f1:
            best_f1, best_state = f1_val, model.state_dict()

    print("Mejor F1:", best_f1)
    torch.save(best_state, "siamese_model.pt")

    # --- PASO 1: Cargar todos los artefactos y el nuevo dataset ---
    print("Paso 1: Cargando modelos, codificadores y el nuevo dataset...")

    # Cargar el nuevo dataset completo
    df_full = df_full_raw
    # Añadir una columna de ID único 

    df_full['id'] = range(len(df_full))


    # Cargar los codificadores y el escalador guardados
    encoders = joblib.load("encoders.pkl")
    num_scaler = encoders["scaler"]
    le_plat = encoders["le_plat"]
    le_tipo = encoders["le_tipo"]
    le_host = encoders["le_host"]

    # Cargar tu modelo EmbNet entrenado
    # (Asegúrate de que la definición de la clase EmbNet está disponible)
    # class EmbNet(nn.Module): ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbNet(in_dim=391) # Asegúrate que in_dim es el correcto para tu modelo
    model.load_state_dict(torch.load("siamese_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Cargar el modelo SBERT
    sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)


    # --- PASO 2: Pipeline de Preprocesamiento para nuevos datos ---



    def handle_unknown_categories(series, encoder):
        """Mapea valores no vistos en el entrenamiento a un valor conocido ('unknown')."""
        known_classes = set(encoder.classes_)
        # Reemplaza cualquier valor no conocido por el valor de 'desconocido' que usaste en el entrenamiento
        # (ej. 'platform_desconocido') si existe, si no, lo ignora.
        # Esta es una forma simple. La más robusta es tener una categoría 'unknown' explícita.
        unknown_token_plat = "platform_desconocido"
        unknown_token_tipo = "tipo_desconocido"
        unknown_token_host = "host_desconocido"

        # Asumimos que los tokens '..._desconocido' están en las clases del encoder
        # y los usamos como fallback.
        if encoder is le_plat:
            return series.apply(lambda x: x if x in known_classes else unknown_token_plat)
        if encoder is le_tipo:
            return series.apply(lambda x: x if x in known_classes else unknown_token_tipo)
        if encoder is le_host:
            return series.apply(lambda x: x if x in known_classes else unknown_token_host)
        return series


    def preprocess_new_data(df, sbert, scaler, encoders_dict):
        """Aplica la pipeline de preprocesamiento a un nuevo DataFrame."""
        print("  Iniciando preprocesamiento del nuevo dataset...")
        df_proc = df.copy()

        # 1. Normalizar texto
        df_proc["title_norm"] = df_proc["nombre"].fillna("").apply(normalize)

        # 2. Rellenar nulos
        df_proc["habitaciones"] = df_proc["habitaciones"].fillna(-1)
        df_proc["capacidad"] = df_proc["capacidad"].fillna(-1)
        df_proc["lat"] = df_proc["lat"].fillna(0)
        df_proc["lng"] = df_proc["lng"].fillna(0)
        df_proc["anfitrion"] = df_proc["anfitrion"].fillna("host_desconocido")
        df_proc["tipo"] = df_proc["tipo"].fillna("tipo_desconocido")
        df_proc["plataforma"] = df_proc["plataforma"].fillna("platform_desconocido")

        # 3. Codificar categorías usando los encoders YA ENTRENADOS
        le_p, le_t, le_h = encoders_dict["le_plat"], encoders_dict["le_tipo"], encoders_dict["le_host"]

        # Manejar categorías no vistas durante el entrenamiento
        df_proc["plataforma"] = handle_unknown_categories(df_proc["plataforma"], le_p)
        df_proc["tipo"] = handle_unknown_categories(df_proc["tipo"], le_t)
        df_proc["anfitrion"] = handle_unknown_categories(df_proc["anfitrion"], le_h)

        # ¡¡USAR .transform()!!
        df_proc["plat_id"] = le_p.transform(df_proc["plataforma"])
        df_proc["tipo_id"] = le_t.transform(df_proc["tipo"])
        df_proc["host_id"] = le_h.transform(df_proc["anfitrion"])

        # 4. Escalar numéricos usando el scaler YA ENTRENADO
        cols_to_scale = ["capacidad", "habitaciones", "lat", "lng"]
        scaled_cols = ["cap_scaled", "hab_scaled", "lat_scaled", "lon_scaled"]
        # ¡¡USAR .transform()!!
        df_proc[scaled_cols] = scaler.transform(df_proc[cols_to_scale])

        # 5. Generar embeddings de texto
        print("  Generando embeddings de texto (SBERT)...")
        embeddings = sbert.encode(
            df_proc["title_norm"].tolist(),
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # 6. Ensamblar la matriz de características final X
        feature_cols = ["plat_id", "tipo_id", "host_id", "cap_scaled", "hab_scaled", "lat_scaled", "lon_scaled"]
        X_final = np.hstack([embeddings, df_proc[feature_cols].values]).astype("float32")

        print(f"  Preprocesamiento completado. Forma de la matriz X: {X_final.shape}")
        return df_proc, X_final

    # Ejecutar la pipeline
    df_full_processed, X_full = preprocess_new_data(df_full, sbert_model, num_scaler, encoders)

    # --- PASO 0: Pre-calcular embeddings (como antes) ---
    print("Paso 0: Pre-calculando embeddings...")
    # ... (código para obtener all_embeddings) ...
    model.eval()
    with torch.no_grad():
        all_embeddings = model(torch.from_numpy(X_full).to(device))
    all_embeddings = all_embeddings.cpu()
    # --- PASO 1: Construir el índice espacial para búsqueda de vecinos ---
    print("\nPaso 1: Construyendo índice espacial (Ball Tree) para búsqueda de vecinos rápida...")
    df_full_processed = df_full_processed.reset_index(drop=True)
    coords_rad = np.radians(df_full_processed[['lat', 'lng']].values)
    kms_per_radian = 6371.0088
    radius_m = 500
    radius_rad = radius_m / 1000.0 / kms_per_radian

    # Creamos el "buscador" de vecinos
    # Usamos 'ball_tree' y 'haversine', que es perfecto para datos geográficos
    nn_search = NearestNeighbors(radius=radius_rad, algorithm='ball_tree', metric='haversine')
    nn_search.fit(coords_rad)
    print("Índice espacial construido.")


    # --- PASO 2: Union-Find (como antes) ---
    print("\nPaso 2: Inicializando estructura Union-Find...")
    # ... (código de Union-Find: parent, rank, find, union, diff_ok) ...
    parent = {i: i for i in df_full_processed.index}
    rank   = {i: 0 for i in df_full_processed.index}

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

    MODEL_SIM_THRESHOLD = 0.8 # O tu umbral elegido

    # Usamos un array booleano para marcar los nodos ya procesados como "semilla"
    processed_as_seed = np.zeros(len(df_full_processed), dtype=bool)

    # Iteramos sobre cada alojamiento
    for i in range(len(df_full_processed)):
        print(f"  Procesando semilla {i+1}/{len(df_full_processed)}...", end='\r')

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
            if not (diff_ok(df_full_processed.at[i,'capacidad'], df_full_processed.at[j,'capacidad'], 0) and
                    diff_ok(df_full_processed.at[i,'habitaciones'], df_full_processed.at[j,'habitaciones'], 0)):
                continue

            # Filtro 2: Similitud del modelo
            emb_a = all_embeddings[i].unsqueeze(0)
            emb_b = all_embeddings[j].unsqueeze(0)
            similarity = F.cosine_similarity(emb_a, emb_b).item()

            if similarity >= MODEL_SIM_THRESHOLD:
                union(i, j) # Si pasa los filtros, fusionar

    print("\nProcesamiento completado.")

    # --- PASO 4: Generar group IDs finales (como antes) ---
    print("\nPaso 4: Generando los group_id finales...")
    df_full_processed['dup_group_id_neighbor_search'] = df_full_processed.index.to_series().apply(find)

    df_full_processed.to_csv('Dedup_ML_'+str(date.today())+'.csv', index=False)

if __name__ == "__main__":
    main()