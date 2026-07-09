"use client";

import {
  AlertCircle,
  BookOpen,
  Loader2,
  Search,
  Sparkles,
  UserRound,
  X
} from "lucide-react";
import React from "react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import type { BookSearchResult, RecommendationItem, RecommendationResponse, SampleUser } from "../lib/types";

type Mode = "seed" | "user";

const SLOT_LABELS: Record<string, string> = {
  interest: "Taste match",
  exploration: "Discovery",
  cold_start: "Starter pick",
  hybrid_v12: "Hybrid match"
};

function hashBookId(bookId: string): number {
  let hash = 0;
  for (const char of bookId) {
    hash = (hash * 31 + char.charCodeAt(0)) % 360;
  }
  return hash;
}

function formatCount(value?: number | null): string {
  if (value == null) return "0";
  return new Intl.NumberFormat("en", { notation: "compact" }).format(value);
}

async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : "Request failed.";
    throw new Error(detail);
  }
  return payload as T;
}

export function RecommenderApp() {
  const [mode, setMode] = useState<Mode>("seed");
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<BookSearchResult[]>([]);
  const [selectedBooks, setSelectedBooks] = useState<BookSearchResult[]>([]);
  const [userId, setUserId] = useState("");
  const [k, setK] = useState(10);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [responseMode, setResponseMode] = useState<RecommendationResponse["mode"] | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sampleUsers, setSampleUsers] = useState<SampleUser[]>([]);
  const [samplesLoading, setSamplesLoading] = useState(false);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setSearchResults([]);
      setIsSearching(false);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setIsSearching(true);
      try {
        const response = await fetch(`/api/books/search?q=${encodeURIComponent(trimmed)}&limit=8`, {
          signal: controller.signal
        });
        setSearchResults(await readJson<BookSearchResult[]>(response));
      } catch (searchError) {
        if (!controller.signal.aborted) {
          setSearchResults([]);
          setError(searchError instanceof Error ? searchError.message : "Search failed.");
        }
      } finally {
        if (!controller.signal.aborted) setIsSearching(false);
      }
    }, 220);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [query]);

  const canSubmit = mode === "seed" || userId.trim().length > 0;
  const selectedIds = useMemo(() => new Set(selectedBooks.map((book) => book.book_id)), [selectedBooks]);

  function addBook(book: BookSearchResult) {
    if (selectedIds.has(book.book_id) || selectedBooks.length >= 5) return;
    setSelectedBooks((current) => [...current, book]);
    setQuery("");
    setSearchResults([]);
    setError(null);
  }

  function removeBook(bookId: string) {
    setSelectedBooks((current) => current.filter((book) => book.book_id !== bookId));
  }

  async function loadSampleUsers() {
    setSamplesLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/users/sample?limit=6");
      setSampleUsers(await readJson<SampleUser[]>(response));
    } catch (sampleError) {
      setError(sampleError instanceof Error ? sampleError.message : "Could not load sample users.");
    } finally {
      setSamplesLoading(false);
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) return;
    setIsLoading(true);
    setError(null);
    setRecommendations([]);
    setResponseMode(null);
    const payload =
      mode === "seed"
        ? { seed_book_ids: selectedBooks.map((book) => book.book_id), k }
        : { user_id: userId.trim(), k };
    try {
      const response = await fetch("/api/recommendations", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload)
      });
      const body = await readJson<RecommendationResponse>(response);
      setRecommendations(body.recommendations);
      setResponseMode(body.mode);
    } catch (recommendError) {
      setError(recommendError instanceof Error ? recommendError.message : "Recommendation failed.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace" aria-label="BigBook recommender">
        <header className="topbar">
          <div>
            <p className="eyebrow">BigBook</p>
            <h1>Recomendador de libros</h1>
          </div>
          <div className="status-pill">
            <Sparkles size={16} />
            API powered
          </div>
        </header>

        <div className="layout-grid">
          <form className="control-panel" onSubmit={submit}>
            <div className="segmented" aria-label="Recommendation mode">
              <button
                type="button"
                className={mode === "seed" ? "active" : ""}
                onClick={() => setMode("seed")}
              >
                <BookOpen size={16} />
                Books
              </button>
              <button
                type="button"
                className={mode === "user" ? "active" : ""}
                onClick={() => setMode("user")}
              >
                <UserRound size={16} />
                User
              </button>
            </div>

            {mode === "seed" ? (
              <div className="field-stack">
                <label htmlFor="book-search">Liked books</label>
                <div className="search-box">
                  <Search size={18} />
                  <input
                    id="book-search"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Search a title"
                    autoComplete="off"
                  />
                  {isSearching ? <Loader2 className="spin" size={18} /> : null}
                </div>
                {searchResults.length > 0 ? (
                  <div className="results-list">
                    {searchResults.map((book) => (
                      <button type="button" key={book.book_id} onClick={() => addBook(book)}>
                        <span>{book.title}</span>
                        <small>{book.genres.slice(0, 2).join(" / ") || "Catalog"}</small>
                      </button>
                    ))}
                  </div>
                ) : null}
                <div className="chips" aria-label="Selected books">
                  {selectedBooks.map((book) => (
                    <span className="chip" key={book.book_id}>
                      <span className="chip-label">{book.title}</span>
                      <button type="button" onClick={() => removeBook(book.book_id)} aria-label={`Remove ${book.title}`}>
                        <X size={14} />
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <div className="field-stack">
                <label htmlFor="user-id">Goodreads user_id</label>
                <div className="search-box">
                  <UserRound size={18} />
                  <input
                    id="user-id"
                    value={userId}
                    onChange={(event) => setUserId(event.target.value)}
                    placeholder="Paste a dataset user_id"
                    autoComplete="off"
                  />
                </div>
                <button className="secondary-button" type="button" onClick={loadSampleUsers}>
                  {samplesLoading ? <Loader2 className="spin" size={16} /> : <Search size={16} />}
                  Load samples
                </button>
                {sampleUsers.length > 0 ? (
                  <div className="sample-users">
                    {sampleUsers.map((sample) => (
                      <button type="button" key={sample.user_id} onClick={() => setUserId(sample.user_id)}>
                        {sample.user_id}
                        <small>{sample.positive_count ?? 0} positives</small>
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            )}

            <div className="slider-row">
              <label htmlFor="k">Results</label>
              <output>{k}</output>
            </div>
            <input id="k" type="range" min="1" max="20" value={k} onChange={(event) => setK(Number(event.target.value))} />

            <button className="primary-button" type="submit" disabled={!canSubmit || isLoading}>
              {isLoading ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
              Recommend
            </button>

            {error ? (
              <div className="error-box" role="alert">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            ) : null}
          </form>

          <section className="recommendation-panel" aria-live="polite">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">{responseMode ? responseMode.replace("_", " ") : "Ready"}</p>
                <h2>Recommendations</h2>
              </div>
              <span>{recommendations.length} books</span>
            </div>
            {isLoading ? <LoadingRows /> : null}
            {!isLoading && recommendations.length === 0 ? <EmptyState /> : null}
            {!isLoading && recommendations.length > 0 ? (
              <div className="book-grid">
                {recommendations.map((book) => (
                  <BookCard key={`${book.rank}-${book.book_id}`} book={book} />
                ))}
              </div>
            ) : null}
          </section>
        </div>
      </section>
    </main>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <BookOpen size={36} />
      <p>Select books, paste a user, or run a cold-start recommendation.</p>
    </div>
  );
}

function LoadingRows() {
  return (
    <div className="loading-stack">
      {Array.from({ length: 4 }).map((_, index) => (
        <div className="skeleton" key={index} />
      ))}
    </div>
  );
}

function BookCard({ book }: { book: RecommendationItem }) {
  const hue = hashBookId(book.book_id);
  const coverStyle = {
    "--cover-hue": `${hue}deg`,
    "--cover-accent": `${(hue + 68) % 360}deg`
  } as React.CSSProperties;
  return (
    <article className="book-card">
      <div className="cover" style={coverStyle}>
        <span>{book.rank}</span>
        <BookOpen size={34} />
      </div>
      <div className="book-copy">
        <div className="book-title-row">
          <h3>{book.title}</h3>
          <span className="slot">{SLOT_LABELS[book.slot] ?? book.slot}</span>
        </div>
        <p>{book.description ?? "No description available."}</p>
        <div className="genre-row">
          {book.genres.slice(0, 3).map((genre) => (
            <span key={genre}>{genre}</span>
          ))}
        </div>
        <dl className="meta-grid">
          <div>
            <dt>Rating</dt>
            <dd>{book.average_rating?.toFixed(2) ?? "n/a"}</dd>
          </div>
          <div>
            <dt>Readers</dt>
            <dd>{formatCount(book.ratings_count)}</dd>
          </div>
          <div>
            <dt>Pages</dt>
            <dd>{book.num_pages ? Math.round(book.num_pages) : "n/a"}</dd>
          </div>
          <div>
            <dt>Cluster</dt>
            <dd>{book.macro_cluster}.{book.fine_cluster}</dd>
          </div>
        </dl>
      </div>
    </article>
  );
}
