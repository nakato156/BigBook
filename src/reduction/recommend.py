"""Ranking layer (``retrieve -> score -> diversify -> explain``) with mitigations
for the four design contradictions C1–C4 (A1–A4).

This is the layer the docs declared pending. Its **only** job here is to resolve the
four contradictions from ``docs/alcance_y_limitaciones.md`` §2; split/cohorts/offline
evaluation live elsewhere and are deliberately out of scope.

How each fix lands in the code:

- **A1 (C1) — reduce early tabular influence.** Interest similarity is the cosine over a
  **taste subspace** that drops the tabular early axes ``pc_0..pc_5`` (popularity, language,
  metadata missingness and coarse genre separation — see README *Interpreting Early
  Components* and ``master_pca_meta.json`` block shares). Because the user vector is a mean
  of book vectors, it inherited ``pc_0`` too; slicing reduces those early axes on both
  sides. PCA and the fitted clusters can retain residual popularity signal, so this remains
  an empirical mitigation. See :data:`RankingConfig.tabular_pcs` and
  :func:`taste_pc_indices`.
- **A2 (C2) — technical eligibility, not a popularity gate.** A book is eligible when its
  id, title, PCA vector and cluster assignment are valid. ``ratings_count`` never excludes
  or orders books; it is retained only to measure exposure. See :func:`eligibility_mask`.
- **A3 (C3) — controlled long-tail exploration.** The top-k reserves ``explore_slots`` for
  relevant books outside the user's retrieved macro-neighbourhood, preferring catalog
  tail/mid segments computed from the current ``ratings_count`` distribution. Exploration
  must pass a relevance floor; otherwise normal interest results fill the list. See
  :func:`popularity_segments` and :func:`select_exploration_rows`.
- **A4 (C4) — cold-start without popularity.** Users with no/too-little history get one
  *accessible* book per macro-cluster (diversity sampler), never a bestseller list. See
  :meth:`Recommender.recommend_cold_start`.

Invoke as a module (writes a small sample, does not batch the whole user base)::

    env/bin/python -m src.reduction.recommend
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    BOOKS_MASTER_PATH,
    INTERACTIONS_CURATED_PATH,
    MASTER_FEATURE_MATRIX_PATH,
    PROJECT_ROOT,
    USER_CENTROIDS_PATH,
    USER_MATRIX_PATH,
    USER_META_PATH,
)

CLUSTERING_DIR = PROJECT_ROOT / "data" / "outputs" / "clustering"
BOOK_CLUSTERS_PATH = CLUSTERING_DIR / "book_clusters_k100.parquet"
MACRO_ASSIGN_PATH = CLUSTERING_DIR / "macro_cluster_assignments_k100.csv"
CENTROIDS_PATH = CLUSTERING_DIR / "kmeans_centroids_k100.npy"
RECS_SAMPLE_PATH = PROJECT_ROOT / "data" / "outputs" / "recommendations" / "recommendations_v1_sample.csv"

GENRE_COLUMNS = ["genre_fantasy", "genre_mystery", "genre_history", "genre_ya", "genre_romance"]


@dataclass(frozen=True)
class RankingConfig:
    """Tunables for the four fixes. Defaults are the v1 decisions in the alcance doc."""

    # A1: tabular early axes excluded from the interest cosine (popularity/lang/missingness).
    tabular_pcs: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    k: int = 10
    # A3: slots reserved for relevant tail/mid books outside the retrieved macros.
    explore_slots: int = 2
    popularity_tail_quantile: float = 0.25
    popularity_head_quantile: float = 0.90
    # Exploration candidates must retain this fraction of the best interest similarity.
    explore_min_relevance_ratio: float = 0.75
    # Diversity/relevance trade-off for MMR (1.0 = pure relevance, 0.0 = pure diversity).
    mmr_lambda: float = 0.7
    # Explicit penalty when a candidate repeats genres already present in the list.
    genre_diversity_weight: float = 0.15
    # Soft tie-break toward shorter eligible books in the normal interest ranking.
    accessibility_weight: float = 0.05
    # How many nearest fine clusters to pull candidates from (retrieve breadth).
    n_clusters_retrieve: int = 5
    # A4: accessibility floor so the cold-start sampler does not surface pamphlets.
    min_pages_accessible: int = 50
    # Users with 1-2 positives keep their profile, shrunk toward the nearest catalog cluster.
    sparse_profile_weight: float = 0.5


# --------------------------------------------------------------------------- #
# Pure, testable helpers
# --------------------------------------------------------------------------- #
def pc_columns(frame: pd.DataFrame) -> list[str]:
    """``pc_*`` columns sorted by integer suffix (matches the item/user matrices)."""
    cols = [c for c in frame.columns if c.startswith("pc_")]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def taste_pc_indices(pc_cols: Sequence[str], tabular_pcs: Iterable[int]) -> np.ndarray:
    """A1: positional indices of the **taste subspace** (``pc_*`` minus the tabular axes).

    Returned indices address columns of a ``pc``-ordered array, so they can slice both the
    book matrix and the user matrix (same column order by construction).
    """
    excluded = set(int(i) for i in tabular_pcs)
    return np.array(
        [i for i, c in enumerate(pc_cols) if int(c.split("_", 1)[1]) not in excluded],
        dtype=np.int64,
    )


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows stay zero (cosine with them is 0, not NaN)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def eligibility_mask(
    book_ids: np.ndarray,
    titles: np.ndarray,
    book_pc: np.ndarray,
    book_cluster: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """A2: technical catalog eligibility without using popularity.

    Eligible books have a non-empty id/title, finite PCA coordinates and a valid cluster.
    """
    ids = np.char.strip(np.asarray(book_ids, dtype=str))
    names = np.asarray(titles, dtype=object)
    clean_names = np.char.strip(np.asarray(names, dtype=str))
    title_ok = pd.notna(names) & (np.char.str_len(clean_names) > 0)
    vector_ok = np.isfinite(np.asarray(book_pc)).all(axis=1)
    clusters = np.asarray(book_cluster)
    cluster_ok = np.isfinite(clusters) & (clusters >= 0) & (clusters < n_clusters)
    return (np.char.str_len(ids) > 0) & title_ok & vector_ok & cluster_ok


def popularity_segments(
    ratings_count: np.ndarray,
    tail_quantile: float = 0.25,
    head_quantile: float = 0.90,
) -> tuple[np.ndarray, float, float]:
    """Label books ``tail``/``mid``/``head`` using current-catalog quantiles."""
    counts = np.asarray(ratings_count, dtype=np.float64)
    finite = counts[np.isfinite(counts)]
    if not len(finite):
        return np.full(len(counts), "unknown", dtype=object), np.nan, np.nan
    tail_cut, head_cut = np.quantile(finite, [tail_quantile, head_quantile])
    labels = np.full(len(counts), "mid", dtype=object)
    labels[counts <= tail_cut] = "tail"
    labels[counts >= head_cut] = "head"
    labels[~np.isfinite(counts)] = "unknown"
    return labels, float(tail_cut), float(head_cut)


def mmr_select(
    cand_norm: np.ndarray,
    relevance: np.ndarray,
    k: int,
    lam: float,
    candidate_genres: np.ndarray | None = None,
    genre_weight: float = 0.0,
) -> list[int]:
    """Maximal Marginal Relevance over the candidate set (diversity vs. relevance).

    ``cand_norm`` are L2-normalized candidate vectors in the **taste subspace**, so the
    redundancy penalty is itself popularity-free (A1). Greedy, ``O(k·n·d)``; no popularity
    enters here, only interest similarity, pairwise redundancy and optional genre overlap.
    """
    n = len(relevance)
    if n == 0 or k <= 0:
        return []
    selected: list[int] = []
    remaining = list(range(n))
    max_sim = np.zeros(n, dtype=np.float64)
    while remaining and len(selected) < k:
        if not selected:
            pick = remaining[int(np.argmax(relevance[remaining]))]
        else:
            mmr = lam * relevance[remaining] - (1.0 - lam) * max_sim[remaining]
            if candidate_genres is not None and genre_weight > 0:
                selected_genres = candidate_genres[selected].max(axis=0)
                overlap = candidate_genres[remaining] @ selected_genres
                denom = np.maximum(candidate_genres[remaining].sum(axis=1), 1.0)
                mmr -= genre_weight * (overlap / denom)
            pick = remaining[int(np.argmax(mmr))]
        selected.append(pick)
        remaining.remove(pick)
        sims = cand_norm @ cand_norm[pick]
        max_sim = np.maximum(max_sim, sims)
    return selected


def accessibility_scores(num_pages: np.ndarray, min_pages: int) -> np.ndarray:
    """Soft accessibility score: shorter valid books rank higher; missing/too short get zero."""
    pages = np.asarray(num_pages, dtype=np.float64)
    valid = np.isfinite(pages) & (pages >= min_pages)
    scores = np.zeros(len(pages), dtype=np.float64)
    if not valid.any():
        return scores
    log_pages = np.log1p(pages[valid])
    low, high = float(log_pages.min()), float(log_pages.max())
    scores[valid] = 1.0 if high == low else 1.0 - ((log_pages - low) / (high - low))
    return scores


def nearest_clusters(user_taste_norm: np.ndarray, centroids_taste_norm: np.ndarray) -> np.ndarray:
    """Cluster ids ordered by cosine closeness of their centroid to the user (retrieve)."""
    sims = centroids_taste_norm @ user_taste_norm
    return np.argsort(-sims)


def retrieve_top_clusters(
    modes_taste_norm: np.ndarray,
    weights: np.ndarray,
    centroids_taste_norm: np.ndarray,
    n_clusters_retrieve: int,
) -> np.ndarray:
    """Top-N cluster ids by max-pooled weighted cosine across taste modes (today's retrieve)."""
    cluster_mode_sim = modes_taste_norm @ centroids_taste_norm.T
    cluster_relevance = (weights[:, None] * cluster_mode_sim).max(axis=0)
    ranked_clusters = np.argsort(-cluster_relevance)
    return ranked_clusters[:n_clusters_retrieve]


def select_exploration_rows(
    rows: np.ndarray,
    relevance: np.ndarray,
    popularity_segment: np.ndarray,
    k: int,
    best_relevance: float,
    min_relevance_ratio: float,
) -> np.ndarray:
    """A3: select relevant exploration books from tail/mid only.

    Head books are never eligible for exploration. Tail is preferred over mid among
    candidates that pass the relevance floor. Returned values are catalog row ids.
    """
    if not len(rows) or k <= 0:
        return np.array([], dtype=np.int64)
    floor = best_relevance * min_relevance_ratio if best_relevance > 0 else best_relevance
    segments = popularity_segment[rows]
    keep = (relevance >= floor) & np.isin(segments, ["tail", "mid"])
    if not keep.any():
        return np.array([], dtype=np.int64)
    eligible_rows = rows[keep]
    eligible_rel = relevance[keep]
    priority = {"tail": 0, "mid": 1}
    segment_priority = np.array(
        [priority.get(str(popularity_segment[row]), 3) for row in eligible_rows]
    )
    order = np.lexsort((-eligible_rel, segment_priority))
    return eligible_rows[order[:k]].astype(np.int64)


def consumed_books_for_users(
    interactions_path: Path,
    user_ids: Sequence[str],
) -> dict[str, set[str]]:
    """Load completed books for a small user set using parquet predicate pushdown."""
    users = [str(user_id) for user_id in user_ids]
    consumed = {user_id: set() for user_id in users}
    if not users:
        return consumed

    table = pq.read_table(
        interactions_path,
        columns=["user_id", "book_id", "is_read"],
        filters=[("user_id", "in", users), ("is_read", "=", True)],
    )
    frame = table.to_pandas()
    if frame.empty:
        return consumed

    frame["user_id"] = frame["user_id"].astype(str)
    frame["book_id"] = frame["book_id"].astype(str)
    for user_id, group in frame.groupby("user_id", sort=False):
        consumed[user_id] = set(group["book_id"])
    return consumed


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #
@dataclass
class Recommender:
    """Marshals artifacts into the pure helpers and assembles an explained top-k.

    Construct from disk with :meth:`from_artifacts`, or pass frames directly (tests).
    """

    book_ids: np.ndarray            # (n_books,) str, item-matrix order
    book_pc: np.ndarray             # (n_books, n_pc) float32, full pc space
    ratings_count: np.ndarray       # (n_books,) int
    num_pages: np.ndarray           # (n_books,) float (may be NaN)
    genres: pd.DataFrame            # book_id-indexed genre flags + title
    book_cluster: np.ndarray        # (n_books,) int fine-cluster id
    centroids: np.ndarray           # (n_clusters, n_pc) float
    macro_of_cluster: np.ndarray    # (n_clusters,) int macro id
    user_ids: np.ndarray            # (n_users,) str (user_matrix order)
    user_pc: np.ndarray             # (n_users, n_pc) float
    positive_count_by_user: dict[str, int]
    centroid_user_ids: np.ndarray
    user_centroid_pc: np.ndarray
    user_centroid_weight: np.ndarray
    pc_cols: list[str]
    config: RankingConfig = field(default_factory=RankingConfig)

    def __post_init__(self) -> None:
        self.taste_idx = taste_pc_indices(self.pc_cols, self.config.tabular_pcs)
        self.book_taste_norm = l2_normalize_rows(self.book_pc[:, self.taste_idx].astype(np.float64))
        self.centroids_taste_norm = l2_normalize_rows(
            self.centroids[:, self.taste_idx].astype(np.float64)
        )
        self.n_macro = int(self.macro_of_cluster.max()) + 1
        self._book_row = {bid: i for i, bid in enumerate(self.book_ids)}
        self._user_row = {uid: i for i, uid in enumerate(self.user_ids)}
        self._centroid_rows: dict[str, np.ndarray] = {}
        if len(self.centroid_user_ids):
            centroid_frame = pd.DataFrame(
                {"user_id": self.centroid_user_ids.astype(str), "row": np.arange(len(self.centroid_user_ids))}
            )
            self._centroid_rows = {
                str(uid): group["row"].to_numpy(dtype=np.int64)
                for uid, group in centroid_frame.groupby("user_id", sort=False)
            }
        titles = self.genres.reindex(self.book_ids)["title"].to_numpy()
        self.genre_matrix = (
            self.genres.reindex(self.book_ids)[GENRE_COLUMNS].fillna(0).to_numpy(dtype=np.float64)
        )
        self.eligible_mask = eligibility_mask(
            self.book_ids,
            titles,
            self.book_pc,
            self.book_cluster,
            self.centroids.shape[0],
        )
        (
            self.popularity_segment,
            self.popularity_tail_cut,
            self.popularity_head_cut,
        ) = popularity_segments(
            self.ratings_count,
            self.config.popularity_tail_quantile,
            self.config.popularity_head_quantile,
        )

    # -- assembly ---------------------------------------------------------- #
    def _candidate_rows(self, cluster_ids: Sequence[int], exclude: set[str]) -> np.ndarray:
        """Technically eligible, not-yet-read rows in the given fine clusters."""
        in_clusters = np.isin(self.book_cluster, np.asarray(cluster_ids))
        keep = in_clusters & self.eligible_mask
        rows = np.nonzero(keep)[0]
        if exclude:
            rows = np.array([r for r in rows if self.book_ids[r] not in exclude], dtype=np.int64)
        return rows

    def retrieved_candidate_rows(
        self,
        modes_pc: np.ndarray,
        mode_weights: np.ndarray,
        exclude: set[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cluster ids retrieved for these modes, and the technically-eligible candidate rows in them."""
        modes_taste_norm = l2_normalize_rows(modes_pc[:, self.taste_idx].astype(np.float64))
        weights = np.asarray(mode_weights, dtype=np.float64)
        weights = weights / (weights.sum() or 1.0)
        near = retrieve_top_clusters(
            modes_taste_norm, weights, self.centroids_taste_norm, self.config.n_clusters_retrieve
        )
        rows = self._candidate_rows(near, exclude)
        return near, rows

    def _explain(self, row: int, slot: str) -> dict:
        bid = self.book_ids[row]
        grow = self.genres.loc[bid] if bid in self.genres.index else None
        active = (
            [g.replace("genre_", "") for g in GENRE_COLUMNS if grow is not None and grow.get(g, 0) == 1]
            if grow is not None
            else []
        )
        return {
            "book_id": bid,
            "title": grow["title"] if grow is not None else "",
            "slot": slot,  # "interest" or "exploration"
            "fine_cluster": int(self.book_cluster[row]),
            "macro_cluster": int(self.macro_of_cluster[self.book_cluster[row]]),
            "genres": "|".join(active),
            "ratings_count": int(self.ratings_count[row]),
            "popularity_segment": str(self.popularity_segment[row]),
            "num_pages": float(self.num_pages[row]) if np.isfinite(self.num_pages[row]) else np.nan,
        }

    def _profile_modes(self, user_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Return one or more taste modes for a known user."""
        row = self._user_row.get(user_id)
        if row is None:
            return None
        positive_count = self.positive_count_by_user.get(user_id, 0)
        if positive_count < 3:
            user_vec = self.user_pc[row].astype(np.float64)
            taste = user_vec[self.taste_idx]
            taste_norm = taste / (np.linalg.norm(taste) or 1.0)
            nearest = nearest_clusters(taste_norm, self.centroids_taste_norm)[0]
            shrunk = (
                self.config.sparse_profile_weight * user_vec
                + (1.0 - self.config.sparse_profile_weight) * self.centroids[nearest]
            )
            return shrunk[None, :], np.array([1.0], dtype=np.float64)

        centroid_rows = self._centroid_rows.get(user_id)
        if centroid_rows is not None and len(centroid_rows):
            return (
                self.user_centroid_pc[centroid_rows].astype(np.float64),
                self.user_centroid_weight[centroid_rows].astype(np.float64),
            )
        return self.user_pc[row][None, :].astype(np.float64), np.array([1.0], dtype=np.float64)

    def _seed_modes(self, seed_book_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray] | None:
        rows = [self._book_row.get(str(book_id), -1) for book_id in seed_book_ids]
        rows = [row for row in rows if row >= 0 and self.eligible_mask[row]]
        if not rows:
            return None
        seed_vector = self.book_pc[np.asarray(rows)].mean(axis=0, keepdims=True)
        return seed_vector.astype(np.float64), np.array([1.0], dtype=np.float64)

    def modes_from_history(
        self,
        positive_book_ids: Sequence[str],
        engagement_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Build leakage-free user modes from positive books in a training window."""
        mapped = np.asarray(
            [self._book_row.get(str(book_id), -1) for book_id in positive_book_ids], dtype=np.int64
        )
        present = mapped >= 0
        present[present] &= self.eligible_mask[mapped[present]]
        rows = mapped[present]
        if not len(rows):
            return None
        vecs = self.book_pc[rows].astype(np.float32)
        if len(rows) < 3:
            mean = vecs.mean(axis=0).astype(np.float64)
            taste = mean[self.taste_idx]
            taste_norm = taste / (np.linalg.norm(taste) or 1.0)
            nearest = nearest_clusters(taste_norm, self.centroids_taste_norm)[0]
            shrunk = (
                self.config.sparse_profile_weight * mean
                + (1.0 - self.config.sparse_profile_weight) * self.centroids[nearest]
            )
            return shrunk[None, :], np.array([1.0], dtype=np.float64)

        from src.reduction.build_user_centroids import _user_centroids

        weights = (
            np.ones(len(rows), dtype=np.float32)
            if engagement_weights is None
            else np.asarray(engagement_weights, dtype=np.float32)[present]
        )
        modes, _, _, centroid_weight = _user_centroids(vecs, weights)
        return modes.astype(np.float64), centroid_weight.astype(np.float64)

    def recommend(
        self,
        user_id: str,
        exclude_book_ids: set[str],
        seed_book_ids: Sequence[str] = (),
    ) -> pd.DataFrame:
        """Top-k for one user; consumed books must be supplied and are always excluded."""
        exclude = {str(book_id) for book_id in exclude_book_ids}
        modes = self._profile_modes(user_id)
        if modes is None:
            modes = self._seed_modes(seed_book_ids)
            if modes is not None:
                exclude.update(str(book_id) for book_id in seed_book_ids)
        if modes is None:
            return self.recommend_cold_start(user_id, exclude)
        return self.recommend_from_modes(user_id, modes[0], modes[1], exclude)

    def recommend_from_profile(
        self, user_id: str, user_pc: np.ndarray, exclude_book_ids: set[str]
    ) -> pd.DataFrame:
        """Recommend from an externally built profile, used by temporal evaluation."""
        return self.recommend_from_modes(
            user_id,
            np.asarray(user_pc, dtype=np.float64).reshape(1, -1),
            np.array([1.0], dtype=np.float64),
            {str(book_id) for book_id in exclude_book_ids},
        )

    def recommend_from_modes(
        self,
        user_id: str,
        modes_pc: np.ndarray,
        mode_weights: np.ndarray,
        exclude: set[str],
    ) -> pd.DataFrame:
        """Rank with one or more user taste modes in the shared PCA space."""

        cfg = self.config
        modes_taste_norm = l2_normalize_rows(modes_pc[:, self.taste_idx].astype(np.float64))
        weights = np.asarray(mode_weights, dtype=np.float64)
        weights = weights / (weights.sum() or 1.0)

        # RETRIEVE: use the strongest weighted taste mode for each cluster.
        near, rows = self.retrieved_candidate_rows(modes_pc, mode_weights, exclude)

        records: list[dict] = []
        interest_order: list[int] = []
        best_relevance = 0.0
        if len(rows):
            # SCORE: interest cosine in taste subspace; popularity does not order or filter.
            mode_sim = modes_taste_norm @ self.book_taste_norm[rows].T
            relevance = (weights[:, None] * mode_sim).max(axis=0)
            best_relevance = float(relevance.max())
            ranking_relevance = relevance + (
                cfg.accessibility_weight
                * accessibility_scores(self.num_pages[rows], cfg.min_pages_accessible)
            )
            # Select a full interest list first; exploration replaces only available slots.
            interest_order = mmr_select(
                self.book_taste_norm[rows],
                ranking_relevance,
                cfg.k,
                cfg.mmr_lambda,
                candidate_genres=self.genre_matrix[rows],
                genre_weight=cfg.genre_diversity_weight,
            )
            n_interest = max(cfg.k - cfg.explore_slots, 0)
            records = [self._explain(int(rows[i]), "interest") for i in interest_order[:n_interest]]

        # EXPLORE (A3): fill remaining slots from a macro the user does not occupy.
        occupied = {int(self.macro_of_cluster[c]) for c in near}
        already = {r["book_id"] for r in records} | exclude
        if cfg.explore_slots > 0 and len(occupied) < self.n_macro and len(records) < cfg.k:
            exploration_clusters = np.nonzero(
                ~np.isin(self.macro_of_cluster, np.asarray(sorted(occupied)))
            )[0]
            erows = self._candidate_rows(exploration_clusters, already)
            if len(erows):
                mode_sim = modes_taste_norm @ self.book_taste_norm[erows].T
                erel = (weights[:, None] * mode_sim).max(axis=0)
                etop = select_exploration_rows(
                    erows,
                    erel,
                    self.popularity_segment,
                    cfg.k - len(records),
                    best_relevance,
                    cfg.explore_min_relevance_ratio,
                )
                records += [self._explain(int(r), "exploration") for r in etop]

        # If exploration cannot meet the relevance floor, preserve a complete top-k.
        already = {r["book_id"] for r in records} | exclude
        for i in interest_order:
            row = int(rows[i])
            if len(records) >= cfg.k:
                break
            if self.book_ids[row] not in already:
                records.append(self._explain(row, "interest"))
                already.add(self.book_ids[row])

        out = pd.DataFrame(records)
        out.insert(0, "user_id", user_id)
        out.insert(1, "rank", range(1, len(out) + 1))
        return out

    def recommend_cold_start(self, user_id: str, exclude_book_ids: set[str]) -> pd.DataFrame:
        """A4: one **accessible** book per macro-cluster — diversity sampler, no popularity.

        Accessibility is the shortest eligible book above ``min_pages_accessible`` in each macro
        (a deliberately thin proxy; see Cubo B). Popularity never orders the list.
        """
        cfg = self.config
        exclude = {str(book_id) for book_id in exclude_book_ids}
        records: list[dict] = []
        for macro in range(self.n_macro):
            clusters_in_macro = np.nonzero(self.macro_of_cluster == macro)[0]
            rows = self._candidate_rows(clusters_in_macro, exclude)
            if not len(rows):
                continue
            pages = self.num_pages[rows]
            ok = np.isfinite(pages) & (pages >= cfg.min_pages_accessible)
            pool = rows[ok] if ok.any() else rows
            pick = pool[int(np.argmin(self.num_pages[pool]))] if ok.any() else pool[0]
            records.append(self._explain(int(pick), "cold_start"))
        out = pd.DataFrame(records)
        out.insert(0, "user_id", user_id)
        out.insert(1, "rank", range(1, len(out) + 1))
        return out

    # -- loading ----------------------------------------------------------- #
    @classmethod
    def from_artifacts(cls, config: RankingConfig | None = None) -> "Recommender":
        config = config or RankingConfig()
        fm = pd.read_parquet(MASTER_FEATURE_MATRIX_PATH)
        pc_cols = pc_columns(fm)
        master = pd.read_parquet(
            BOOKS_MASTER_PATH,
            columns=["book_id", "title", "ratings_count", "num_pages", *GENRE_COLUMNS],
        )
        master["book_id"] = master["book_id"].astype(str)
        fm["book_id"] = fm["book_id"].astype(str)
        master = master.set_index("book_id").reindex(fm["book_id"]).reset_index()

        clusters = pd.read_parquet(BOOK_CLUSTERS_PATH)
        clusters["book_id"] = clusters["book_id"].astype(str)
        cluster_of = clusters.set_index("book_id").reindex(fm["book_id"])["cluster"].to_numpy()

        macro_df = pd.read_csv(MACRO_ASSIGN_PATH).sort_values("cluster")
        macro_of_cluster = macro_df.set_index("cluster")["macro_cluster"].to_numpy()
        centroids = np.load(CENTROIDS_PATH)

        um = pd.read_parquet(USER_MATRIX_PATH)
        um["user_id"] = um["user_id"].astype(str)
        meta = pd.read_parquet(USER_META_PATH, columns=["user_id", "positive_count"])
        positive_count_by_user = dict(
            zip(meta["user_id"].astype(str), meta["positive_count"].astype(int), strict=False)
        )
        uc = pd.read_parquet(USER_CENTROIDS_PATH)
        uc["user_id"] = uc["user_id"].astype(str)

        genres = master.set_index("book_id")[["title", *GENRE_COLUMNS]]
        return cls(
            book_ids=fm["book_id"].to_numpy().astype(str),
            book_pc=fm[pc_cols].to_numpy(dtype=np.float32),
            ratings_count=master["ratings_count"].to_numpy(),
            num_pages=master["num_pages"].to_numpy(dtype=np.float64),
            genres=genres,
            book_cluster=cluster_of.astype(np.int64),
            centroids=centroids,
            macro_of_cluster=macro_of_cluster.astype(np.int64),
            user_ids=um["user_id"].to_numpy().astype(str),
            user_pc=um[pc_cols].to_numpy(dtype=np.float32),
            positive_count_by_user=positive_count_by_user,
            centroid_user_ids=uc["user_id"].to_numpy().astype(str),
            user_centroid_pc=uc[pc_cols].to_numpy(dtype=np.float32),
            user_centroid_weight=uc["centroid_weight"].to_numpy(dtype=np.float32),
            pc_cols=pc_cols,
            config=config,
        )


def build_recommendation_sample(
    recommender: Recommender,
    sample_users: Sequence[str],
    consumed_by_user: dict[str, set[str]],
) -> pd.DataFrame:
    """Build the demo sample while enforcing consumed-book exclusions."""
    frames = [
        recommender.recommend(
            str(user_id),
            consumed_by_user.get(str(user_id), set()),
        )
        for user_id in sample_users
    ]
    frames.append(recommender.recommend_cold_start("__cold_start_demo__", set()))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    rec = Recommender.from_artifacts()
    print(f"Loaded {len(rec.book_ids):,} books, {len(rec.user_ids):,} users.")
    print(f"A1 taste subspace: {len(rec.taste_idx)}/{len(rec.pc_cols)} pcs "
          f"(dropped pc_{list(rec.config.tabular_pcs)}).")
    print(f"A2 technical eligibility keeps {int(rec.eligible_mask.sum()):,}/"
          f"{len(rec.book_ids):,} books (ratings_count is not a filter).")
    print(f"Popularity segments: tail <= {rec.popularity_tail_cut:.0f}, "
          f"head >= {rec.popularity_head_cut:.0f} ratings.")

    # A small, honest sample: a few real users + one synthetic cold-start.
    sample_users = list(rec.user_ids[:4])
    consumed_by_user = consumed_books_for_users(INTERACTIONS_CURATED_PATH, sample_users)
    out = build_recommendation_sample(rec, sample_users, consumed_by_user)

    RECS_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RECS_SAMPLE_PATH, index=False)
    n_explore = int((out["slot"] == "exploration").sum())
    print(f"A3 exploration slots in sample: {n_explore} rows tagged 'exploration'.")
    print(f"Wrote {len(out)} sample recommendation rows to {RECS_SAMPLE_PATH}")


if __name__ == "__main__":
    main()
