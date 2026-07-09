export type BookSearchResult = {
  book_id: string;
  title: string;
  description?: string | null;
  genres: string[];
  average_rating?: number | null;
  ratings_count?: number | null;
  num_pages?: number | null;
  publication_year?: number | null;
};

export type RecommendationItem = {
  rank: number;
  book_id: string;
  title: string;
  description?: string | null;
  genres: string[];
  slot: string;
  fine_cluster: number;
  macro_cluster: number;
  popularity_segment: string;
  ratings_count: number;
  average_rating?: number | null;
  num_pages?: number | null;
  publication_year?: number | null;
};

export type RecommendationResponse = {
  mode: "seed" | "user" | "cold_start";
  user_id: string;
  k: number;
  recommendations: RecommendationItem[];
};

export type SampleUser = {
  user_id: string;
  positive_count?: number | null;
};
