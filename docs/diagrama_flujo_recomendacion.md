# Diagrama de flujo: recomendación de libros

Este diagrama documenta el flujo implementado: catálogo → PCA → clústeres → perfiles de usuario → ranking → API/UI.

```mermaid
flowchart TD
    %% ── 1. Datos de libros ────────────────────────────────────────────────
    subgraph A["1. Catálogo curado y espacio de ítems"]
        A1["books_curated.parquet<br/>por cada categoría"] --> A2["merge_master"]
        A2 --> A3["books_master.parquet<br/>1 fila = 1 book_id<br/>flags multi-género"]

        A3 --> N["Bloque numérico<br/>• average_rating<br/>• log1p(ratings_count)<br/>• log1p(text_reviews_count)<br/>• num_pages + indicador missing<br/>• publication_year + indicador missing<br/>• author_count<br/>• genre_count"]
        A3 --> B["Bloque binario/categórico<br/>• series<br/>• 5 flags de género<br/>• one-hot language_code<br/>• bucket language_code_other"]
        A3 --> E0["Texto del libro<br/>description → title → '[no description]'"]
        E0 --> E["Embeddings semánticos<br/>embeddinggemma-300m<br/>256 dimensiones"]
    end

    %% ── 2. PCA ───────────────────────────────────────────────────────────
    subgraph P["2. Construcción del PCA"]
        N --> P1["StandardScaler independiente<br/>por bloque"]
        B --> P1
        E --> P1
        P1 --> P2["Balance por dimensión<br/>cada bloque × 1 / sqrt(n_features)<br/><br/>Evita que las 256 dimensiones<br/>de embeddings dominen la representación"]
        P2 --> P3["Concatenar bloques ponderados"]
        P3 --> P4["PCA completo<br/>n_components = varianza acumulada 95%"]
        P4 --> P5["master_feature_matrix.parquet<br/>book_id + pc_0 ... pc_N<br/><br/>Cada libro queda como vector PCA"]
        P4 --> P6["master_pca_model.joblib<br/>scalers, medianas, categorías,<br/>pesos y PCA para transformar libros nuevos"]
        P4 --> P7["master_pca_meta.json<br/>varianza explicada, diagnósticos<br/>y chequeo de dominancia embedding"]
    end

    %% ── 3. Clusters ──────────────────────────────────────────────────────
    subgraph C["3. Vecindarios de libros: clustering"]
        P5 --> C1["Matriz X = pc_0 ... pc_N<br/>1 fila = 1 libro"]
        C1 --> C2["KMeans<br/>se comparan K=50 y K=100<br/>random_state=42"]
        C2 --> C3["Diagnóstico<br/>inercia, silhouette muestreado,<br/>tamaños y cohesión de clusters"]
        C3 --> C4["Selección productiva: K=100"]
        C4 --> C5["book_clusters_k100.parquet<br/>book_id → fine_cluster"]
        C4 --> C6["Centroides KMeans<br/>100 centroides PCA"]
        C6 --> C7["Ward hierarchical clustering<br/>sobre los 100 centroides"]
        C7 --> C8["10 macro-clusters<br/>agrupan vecindarios finos relacionados"]
        C8 --> C9["macro_cluster_assignments_k100.csv<br/>fine_cluster → macro_cluster"]
    end

    %% ── 4. Datos y perfiles de usuario ───────────────────────────────────
    subgraph U["4. Perfil de gusto del usuario"]
        U1["Interacciones Goodreads globales"] --> U2["Curación global<br/>deduplicación keep-best<br/>se excluyen libros fuera del catálogo"]
        U2 --> U3["interactions_curated.parquet"]
        U3 --> U4{"Positivo de gusto?"}
        U4 -->|"is_read = True<br/>y rating_clean ≥ 4"| U5["Tomar vector PCA<br/>del libro leído"]
        U4 -->|"No"| U6["No entra en la geometría<br/>del perfil"]
        U5 --> U7["user_matrix<br/>media simple de vectores positivos<br/>1 vector PCA por usuario"]
        U5 --> U8["user_centroids<br/>modos de lectura del usuario"]
        U8 --> U9["Si tiene menos de 6 positivos: m=1<br/>Si no: KMeans intrausuario<br/>m = min(4, positivos / 3)"]
        U9 --> U10["Cada modo = media de libros<br/>en ese subgrupo"]
        U10 --> U11["centroid_weight<br/>compromiso relativo del modo:<br/>(rating - 3) × bonus reseña × bonus duración"]
    end

    %% ── 5. Consulta y retrieve ───────────────────────────────────────────
    subgraph R["5. Recomendación para una persona"]
        Q1["Entrada<br/>user_id existente<br/>o libros semilla seleccionados"] --> Q2{"¿Hay perfil?"}
        Q2 -->|"Sí"| Q3["Usar modos de gusto<br/>user_centroids o user_matrix"]
        Q2 -->|"No, pero hay semillas"| Q4["Media de PCA de libros semilla"]
        Q2 -->|"No hay historial ni semillas"| CS["Cold start<br/>1 libro accesible por macro-cluster<br/>sin ordenar por popularidad"]
        Q3 --> Q5["Subespacio de gusto<br/>se excluyen pc_0 ... pc_5<br/>para reducir influencia tabular:<br/>popularidad, idioma, missingness"]
        Q4 --> Q5
        Q5 --> Q6["Normalización L2"]
        Q6 --> Q7["Similitud coseno<br/>entre cada modo y los centroides<br/>de los 100 clusters"]
        Q7 --> Q8["Retrieve<br/>por defecto: 5 clusters finos más cercanos<br/>alternativa: clusters por cada modo"]
        Q8 --> Q9["Candidatos elegibles<br/>id, título, PCA y cluster válidos<br/>menos libros ya consumidos"]
    end

    %% ── 6. Score y diversidad ────────────────────────────────────────────
    subgraph S["6. Ranking content-only: interés + diversidad + exploración"]
        Q9 --> S1["Score de interés<br/>max_c [peso_modo_c × cos(modo_c, libro)]"]
        S1 --> S2["Bonus suave de accesibilidad<br/>prefiere menos páginas solo<br/>cuando la afinidad es comparable"]
        S2 --> S3["MMR greedy<br/>0.7 × relevancia<br/>− 0.3 × redundancia semántica<br/>− penalización por repetir género"]
        S3 --> S4["Slots de interés<br/>por defecto: 8 de top-10"]
        Q9 --> S5["Segmentos de popularidad<br/>calculados dinámicamente<br/>sobre ratings_count"]
        S5 --> S6["tail: ≤ percentil 25<br/>mid: entre p25 y p90<br/>head: ≥ percentil 90"]
        S4 --> S7["Exploración controlada"]
        S6 --> S7
        S7 --> S8{"¿Candidato fuera de<br/>macro-clusters recuperados,<br/>tail/mid y afinidad ≥75%<br/>de la mejor afinidad?"}
        S8 -->|"Sí"| S9["Hasta 2 slots exploration<br/>prioridad: tail y luego similitud"]
        S8 -->|"No"| S10["Completar con resultados<br/>normales de interés"]
        S9 --> OUT["Top-k explicado"]
        S10 --> OUT
    end

    %% ── 7. Ranking híbrido opcional ──────────────────────────────────────
    subgraph H["7. Variante híbrida V1.2: percentiles por pool de candidatos"]
        Q9 --> H1["Unir candidatos de:<br/>• clusters por contenido<br/>• popularidad global histórica<br/>• popularidad en géneros leídos<br/>• co-ocurrencia PMI<br/>• user-kNN"]
        H1 --> H2["Calcular señales crudas<br/>content, global_popularity,<br/>genre_popularity, cooccurrence, user_knn"]
        H2 --> H3["Convertir cada señal a percentil<br/>dentro del pool de ese usuario<br/><br/>percentil = (rank_promedio - 1) / (n - 1)<br/>empates reciben rango promedio<br/>señal constante = 0.5"]
        H3 --> H4["Score híbrido<br/>0.35 content<br/>+ 0.30 popularidad global<br/>+ 0.20 popularidad por género<br/>+ 0.10 co-ocurrencia<br/>+ 0.05 user-kNN"]
        H4 --> H5["Eliminar duplicados de título<br/>y devolver top-k"]
    end

    %% ── 8. Presentación ──────────────────────────────────────────────────
    subgraph V["8. Presentación al usuario"]
        OUT --> V1["Respuesta API /recommendations"]
        H5 --> V1
        V1 --> V2["Cada tarjeta incluye<br/>• posición/rank<br/>• título y descripción<br/>• géneros<br/>• slot: interest, exploration o cold_start<br/>• segmento tail/mid/head<br/>• cluster fino y macro-cluster<br/>• rating, ratings_count y páginas"]
        V2 --> V3["UI Next.js<br/>flujo por usuario Goodreads<br/>o búsqueda de libros semilla"]
        V2 --> V4["Explicación posible<br/>'afín a tu modo de lectura'<br/>+ vecindario/cluster<br/>+ etiqueta de descubrimiento"]
    end
```

## Notas de lectura

- Los clústeres se forman con el espacio PCA completo porque representan vecindarios eficientes de libros. La afinidad de ranking excluye `pc_0` a `pc_5` para reducir la influencia directa de popularidad, idioma y metadatos faltantes.
- Los percentiles se usan de dos formas: `p25` y `p90` segmentan popularidad para exploración (`tail`, `mid`, `head`); y V1.2 convierte cada señal de su pool de candidatos a percentil antes de combinarla.
- La popularidad no excluye libros ni ordena los slots de interés del ranking content-only. Solo rige los slots de exploración y puede actuar como señal calibrada en el ranking híbrido.
