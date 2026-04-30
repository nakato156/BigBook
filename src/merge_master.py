import pandas as pd
from pathlib import Path
from src.config import CATEGORIES, PROCESSED_DIR


LEGACY_PROCESSED_DIRS = {
    "fantasy_paranormal": [PROCESSED_DIR / "fantasy"],
    "history_biography": [PROCESSED_DIR / "history"],
}


def resolve_curated_books_path(category_key: str) -> Path | None:
    cfg = CATEGORIES[category_key]
    candidate_dirs = [cfg.processed_dir, *LEGACY_PROCESSED_DIRS.get(category_key, [])]
    for processed_dir in candidate_dirs:
        path = processed_dir / "books_curated.parquet"
        if path.exists():
            return path
    return None

def main():
    # Merge happens after curation, so we consume curated books artifacts only.
    genre_paths = {
        "fantasy": resolve_curated_books_path("fantasy_paranormal"),
        "mystery": resolve_curated_books_path("mystery_thriller_crime"),
        "history": resolve_curated_books_path("history_biography"),
        "ya": resolve_curated_books_path("young_adult"),
        "romance": resolve_curated_books_path("romance"),
    }
    
    genre_cols = [f"genre_{g}" for g in genre_paths.keys()]
    dfs = []
    
    # 1. Load each genre parquet and add boolean flags for all 5 genres
    # This ensures consistent columns across all dataframes before concatenation
    print("Loading and tagging genre datasets...")
    for genre_key, path in genre_paths.items():
        if path is None or not path.exists():
            print(f"Warning: {path} not found. Skipping.")
            continue
            
        df = pd.read_parquet(path)
        
        # Add boolean flag columns; set True only for the current genre
        for g in genre_paths.keys():
            df[f"genre_{g}"] = (g == genre_key)
            
        dfs.append(df)
        
    if not dfs:
        print("Error: No data loaded. Check file paths.")
        return

    # 2. Concatenate all 5 dataframes into one master table
    print("Concatenating all genre dataframes...")
    master_df = pd.concat(dfs, ignore_index=True)
    
    # 3. Deduplicate by book_id using groupby().agg()
    # For genre flags, keep True if the book appeared in that genre (max)
    # For all other columns, take the first non-null value found (first)
    print("Deduplicating by book_id...")
    agg_dict = {}
    for col in master_df.columns:
        if col == "book_id":
            continue
        if col in genre_cols:
            agg_dict[col] = "max"
        else:
            agg_dict[col] = "first"
            
    master_df = master_df.groupby("book_id").agg(agg_dict).reset_index()
    
    # 4. Add derived column genre_count (how many genres each book belongs to)
    print("Calculating genre_count...")
    master_df["genre_count"] = master_df[genre_cols].sum(axis=1)
    
    # 5. Drop columns entirely that are not needed for the feature matrix
    print("Dropping unnecessary columns...")
    # These were explicitly listed to be dropped or ignored
    cols_to_drop = [
        "authors",
        "popular_shelves",
        "primary_author_id_role_filtered", 
        "author_fallback_id", 
        "work_id", 
        "top_shelves", 
        "top_shelves_json",
        "to_read_count",
        "shelf_count",
        "genre_theme_count"
    ]
    # Also drop any genre-specific theme columns (starting with 'theme_')
    theme_cols = [c for c in master_df.columns if c.startswith("theme_")]
    cols_to_drop.extend(theme_cols)
    
    # Drop only if they exist in the dataframe to avoid errors
    master_df = master_df.drop(columns=[c for c in cols_to_drop if c in master_df.columns])
    
    # 6. Keep and clean columns with specific rules
    print("Cleaning columns and enforcing types...")
    
    # average_rating: float, drop rows where this is null
    master_df["average_rating"] = pd.to_numeric(master_df["average_rating"], errors="coerce")
    master_df = master_df.dropna(subset=["average_rating"])
    
    # ratings_count: int, fill nulls with 0
    master_df["ratings_count"] = pd.to_numeric(master_df["ratings_count"], errors="coerce").fillna(0).astype(int)
    
    # text_reviews_count: int, fill nulls with 0
    master_df["text_reviews_count"] = pd.to_numeric(master_df["text_reviews_count"], errors="coerce").fillna(0).astype(int)
    
    # num_pages: float, keep nulls as NaN
    master_df["num_pages"] = pd.to_numeric(master_df["num_pages"], errors="coerce")
    
    # publication_year: float, keep nulls
    master_df["publication_year"] = pd.to_numeric(master_df["publication_year"], errors="coerce")
    
    # author_count: int, fill nulls with 1
    master_df["author_count"] = pd.to_numeric(master_df["author_count"], errors="coerce").fillna(1).astype(int)
    
    # series: curated notebooks expose this as boolean is_in_series; map to int
    master_df["series"] = master_df["is_in_series"].fillna(False).astype(int)
    
    # language_code: keep as string, fill nulls with 'unknown'
    master_df["language_code"] = master_df["language_code"].fillna("unknown").astype(str)

    # description: keep as string, fill nulls with empty text
    if "description" in master_df.columns:
        master_df["description"] = master_df["description"].fillna("").astype(str)
    
    # title: string, drop rows where null
    master_df = master_df.dropna(subset=["title"])
    master_df["title"] = master_df["title"].astype(str)
    
    # genre flag columns and genre_count: convert to int (0/1)
    for col in genre_cols + ["genre_count"]:
        master_df[col] = master_df[col].astype(int)
        
    # 7. Reset index and save to processed/books_master.parquet
    print("Saving final master table...")
    # Select only the columns we want to keep in the final table
    final_columns = [
        "book_id", "title", "description", "series", "language_code", "average_rating", 
        "ratings_count", "text_reviews_count", "num_pages", "publication_year", 
        "author_count", "genre_fantasy", "genre_mystery", "genre_history", 
        "genre_ya", "genre_romance", "genre_count"
    ]
    master_df = master_df[final_columns].reset_index(drop=True)
    
    output_path = PROCESSED_DIR / "books_master.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_parquet(output_path, index=False)
    
    # 8. Print final summary statistics and diagnostics
    print("\n" + "="*40)
    print("BOOKS MASTER BUILD COMPLETE")
    print("="*40)
    print(f"Final shape of the table: {master_df.shape}")
    
    print("\nCount of books per genre:")
    for col in genre_cols:
        print(f"  {col}: {master_df[col].sum()}")
        
    multi_genre_count = (master_df["genre_count"] > 1).sum()
    print(f"\nCount of books in more than one genre: {multi_genre_count}")
    
    print("\nNull counts for every column:")
    print(master_df.isnull().sum())
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nFile size on disk: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
