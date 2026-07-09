import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { RecommenderApp } from "./RecommenderApp";

describe("RecommenderApp", () => {
  beforeEach(() => {
    global.fetch = jest.fn(async () => ({
      ok: true,
      json: async () => []
    })) as jest.Mock;
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  it("renders the recommendation workspace", () => {
    render(<RecommenderApp />);
    expect(screen.getByRole("heading", { name: /recomendador de libros/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /recommend/i })).toBeInTheDocument();
  });

  it("switches to user mode", async () => {
    render(<RecommenderApp />);
    await userEvent.click(screen.getByRole("button", { name: /^user$/i }));
    expect(screen.getByLabelText(/goodreads user_id/i)).toBeInTheDocument();
  });
});
