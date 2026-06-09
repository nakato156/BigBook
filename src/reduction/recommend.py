"""Ranking layer (``retrieve -> score -> diversify -> explain``) with the four
design contradictions C1–C4 resolved *by construction* (fixes A1–A4).

This is the layer the docs declared pending. Its **only** job here is to resolve the
four contradictions from ``docs/alcance_y_limitaciones.md`` §2; split/cohorts/offline
evaluation live elsewhere and are deliberately out of scope.

How each fix lands in the code:

- **A1 (C1) — popularity out of the geometry.** Interest similarity is the cosine over a
  **taste subspace** that drops the tabular early axes ``pc_0..pc_5`` (popularity, language,
  metadata missingness and coarse genre separation — see README *Interpreting Early
  Components* and ``master_pca_meta.json`` block shares). Because the user vector is a mean
  of book vectors, it inherited ``pc_0`` too; slicing the columns removes the bias from
  *both* sides at once. See :data:`RankingConfig.tabular_pcs` and :func:`taste_pc_indices`.
- **A2 (C2) — popularity is a gate, not a multiplier.** ``ratings_count`` is used only to
  **drop** low-evidence books (a data-quality floor); it never enters the score. The order
  is pure interest cosine + diversity, so the model no longer embeds the B1 popularity
  baseline as a factor. See :func:`quality_gate_mask`.
- **A3 (C3) — exploration is generated, not just measured.** The top-k reserves
  ``explore_slots`` for books pulled from a macro-cluster the user's neighbourhood does
  **not** occupy, so the catalog's unexplored regions can surface. See
  :func:`pick_exploration_macro`.
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

from src.config import (
    BOOKS_MASTER_PATH,
    FEATURES_DIR,
    MASTER_FEATURE_MATRIX_PATH,
    PROJECT_ROOT,
    USER_MATRIX_PATH,
    USER_META_PATH,
)
from src.utils.io import safe_write_parquet  # noqa: F401  (kept for parity / future batch)

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
    # A2: data-quality floor. Books below this many ratings are dropped (gate, never scored).
    min_ratings_gate: int = 5
    k: int = 10
    # A3: of the k slots, how many are reserved for exploration outside the user's macros.
    explore_slots: int = 2
    # Diversity/relevance trade-off for MMR (1.0 = pure relevance, 0.0 = pure diversity).
    mmr_lambda: float = 0.7
    # How many nearest fine clusters to pull candidates from (retrieve breadth).
    n_clusters_retrieve: int = 5
    # A4: accessibility floor so the cold-start sampler does not surface pamphlets.
    min_pages_accessible: int = 50


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


def quality_gate_mask(ratings_count: np.ndarray, min_ratings: int) -> np.ndarray:
    """A2: boolean keep-mask. Popularity is used **only** to drop low-evidence books."""
    return np.asarray(ratings_count) >= min_ratings


def mmr_select(
    cand_norm: np.ndarray, relevance: np.ndarray, k: int, lam: float
) -> list[int]:
    """Maximal Marginal Relevance over the candidate set (diversity vs. relevance).

    ``cand_norm`` are L2-normalized candidate vectors in the **taste subspace**, so the
    redundancy penalty is itself popularity-free (A1). Greedy, ``O(k·n·d)``; no popularity
    enters here, only interest similarity and pairwise redundancy.
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
            pick = remaining[int(np.argmax(mmr))]
        selected.append(pick)
        remaining.remove(pick)
        sims = cand_norm @ cand_norm[pick]
        max_sim = np.maximum(max_sim, sims)
    return selected


def nearest_clusters(user_taste_norm: np.ndarray, centroids_taste_norm: np.ndarray) -> np.ndarray:
    """Cluster ids ordered by cosine closeness of their centroid to the user (retrieve)."""
    sims = centroids_taste_norm @ user_taste_norm
    return np.argsort(-sims)


def pick_exploration_macro(
    occupied_macros: set[int],
    macro_centroids_taste_norm: np.ndarray,
    user_taste_norm: np.ndarray,
) -> int | None:
    """A3: choose the **nearest macro-cluster the user does not occupy**.

    "Nearest non-occupied" is the gentlest possible exploration: it surfaces a region the
    user's neighbourhood never reaches (raising catalog coverage) while staying the least
    jarring jump available. Returns ``None`` if the user already spans every macro-cluster.
    """
    sims = macro_centroids_taste_norm @ user_taste_norm
    order = np.argsort(-sims)
    for macro in order:
        if int(macro) not in occupied_macros:
            return int(macro)
    return None


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
    cold_start_users: set[str]      # is_cold_start == True (from user_meta)
    pc_cols: list[str]
    config: RankingConfig = field(default_factory=RankingConfig)

    def __post_init__(self) -> None:
        self.taste_idx = taste_pc_indices(self.pc_cols, self.config.tabular_pcs)
        self.book_taste_norm = l2_normalize_rows(self.book_pc[:, self.taste_idx].astype(np.float64))
        self.centroids_taste_norm = l2_normalize_rows(
            self.centroids[:, self.taste_idx].astype(np.float64)
        )
        # Macro centroid = mean of its fine centroids, in the taste subspace (A3 exploration).
        n_macro = int(self.macro_of_cluster.max()) + 1
        macro_taste = np.zeros((n_macro, len(self.taste_idx)), dtype=np.float64)
        counts = np.zeros(n_macro, dtype=np.int64)
        cen_taste = self.centroids[:, self.taste_idx].astype(np.float64)
        for c in range(self.centroids.shape[0]):
            m = int(self.macro_of_cluster[c])
            macro_taste[m] += cen_taste[c]
            counts[m] += 1
        macro_taste /= np.maximum(counts, 1)[:, None]
        self.macro_centroids_taste_norm = l2_normalize_rows(macro_taste)
        self._book_row = {bid: i for i, bid in enumerate(self.book_ids)}
        self._user_row = {uid: i for i, uid in enumerate(self.user_ids)}
        # Gate mask (A2): precompute once, reused for every user.
        self.gate_mask = quality_gate_mask(self.ratings_count, self.config.min_ratings_gate)

    # -- assembly ---------------------------------------------------------- #
    def _candidate_rows(self, cluster_ids: Sequence[int], exclude: set[str]) -> np.ndarray:
        """Gated, not-yet-read book rows belonging to the given fine clusters."""
        in_clusters = np.isin(self.book_cluster, np.asarray(cluster_ids))
        keep = in_clusters & self.gate_mask
        rows = np.nonzero(keep)[0]
        if exclude:
            rows = np.array([r for r in rows if self.book_ids[r] not in exclude], dtype=np.int64)
        return rows

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
            "num_pages": float(self.num_pages[row]) if np.isfinite(self.num_pages[row]) else np.nan,
        }

    def recommend(self, user_id: str, exclude_book_ids: set[str] | None = None) -> pd.DataFrame:
        """Top-k for one user. ``exclude_book_ids`` is the already-read set (eval-time;
        building it via temporal split is out of scope here)."""
        exclude = exclude_book_ids or set()
        if user_id not in self._user_row or user_id in self.cold_start_users:
            return self.recommend_cold_start(user_id)

        cfg = self.config
        user_taste = self.user_pc[self._user_row[user_id], self.taste_idx].astype(np.float64)
        user_taste_norm = user_taste / (np.linalg.norm(user_taste) or 1.0)

        # RETRIEVE: nearest fine clusters (in taste subspace).
        ranked_clusters = nearest_clusters(user_taste_norm, self.centroids_taste_norm)
        near = ranked_clusters[: cfg.n_clusters_retrieve]
        rows = self._candidate_rows(near, exclude)

        records: list[dict] = []
        if len(rows):
            # SCORE: pure interest cosine in taste subspace (A1); no popularity (A2).
            relevance = self.book_taste_norm[rows] @ user_taste_norm
            # DIVERSIFY: MMR to fill the interest slots.
            n_interest = max(cfg.k - cfg.explore_slots, 0)
            order = mmr_select(self.book_taste_norm[rows], relevance, n_interest, cfg.mmr_lambda)
            records = [self._explain(int(rows[i]), "interest") for i in order]

        # EXPLORE (A3): fill remaining slots from a macro the user does not occupy.
        occupied = {int(self.macro_of_cluster[c]) for c in near}
        explore_macro = pick_exploration_macro(
            occupied, self.macro_centroids_taste_norm, user_taste_norm
        )
        already = {r["book_id"] for r in records} | exclude
        if explore_macro is not None and len(records) < cfg.k:
            clusters_in_macro = np.nonzero(self.macro_of_cluster == explore_macro)[0]
            erows = self._candidate_rows(clusters_in_macro, already)
            if len(erows):
                erel = self.book_taste_norm[erows] @ user_taste_norm
                etop = erows[np.argsort(-erel)[: cfg.k - len(records)]]
                records += [self._explain(int(r), "exploration") for r in etop]

        out = pd.DataFrame(records)
        out.insert(0, "user_id", user_id)
        out.insert(1, "rank", range(1, len(out) + 1))
        return out

    def recommend_cold_start(self, user_id: str) -> pd.DataFrame:
        """A4: one **accessible** book per macro-cluster — diversity sampler, no popularity.

        Accessibility is the shortest gated book above ``min_pages_accessible`` in each macro
        (a deliberately thin proxy; see Cubo B). Popularity never orders the list.
        """
        cfg = self.config
        records: list[dict] = []
        for macro in range(self.macro_centroids_taste_norm.shape[0]):
            clusters_in_macro = np.nonzero(self.macro_of_cluster == macro)[0]
            rows = self._candidate_rows(clusters_in_macro, set())
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
        meta = pd.read_parquet(USER_META_PATH, columns=["user_id", "is_cold_start"])
        cold = set(meta.loc[meta["is_cold_start"], "user_id"].astype(str))

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
            cold_start_users=cold,
            pc_cols=pc_cols,
            config=config,
        )


def main() -> None:
    rec = Recommender.from_artifacts()
    print(f"Loaded {len(rec.book_ids):,} books, {len(rec.user_ids):,} users.")
    print(f"A1 taste subspace: {len(rec.taste_idx)}/{len(rec.pc_cols)} pcs "
          f"(dropped pc_{list(rec.config.tabular_pcs)}).")
    print(f"A2 quality gate keeps {int(rec.gate_mask.sum()):,}/{len(rec.book_ids):,} books "
          f"(ratings_count >= {rec.config.min_ratings_gate}).")

    # A small, honest sample: a few real users + one synthetic cold-start.
    sample_users = list(rec.user_ids[:4])
    frames = [rec.recommend(uid) for uid in sample_users]
    frames.append(rec.recommend_cold_start("__cold_start_demo__"))
    out = pd.concat(frames, ignore_index=True)

    RECS_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RECS_SAMPLE_PATH, index=False)
    n_explore = int((out["slot"] == "exploration").sum())
    print(f"A3 exploration slots in sample: {n_explore} rows tagged 'exploration'.")
    print(f"Wrote {len(out)} sample recommendation rows to {RECS_SAMPLE_PATH}")


if __name__ == "__main__":
    main()
