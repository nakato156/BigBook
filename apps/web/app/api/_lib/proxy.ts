const API_BASE_URL = process.env.FASTAPI_BASE_URL ?? "http://127.0.0.1:8000";

export async function proxyToApi(path: string, init?: RequestInit): Promise<Response> {
  try {
    const upstream = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        "content-type": "application/json",
        ...(init?.headers ?? {})
      },
      cache: "no-store"
    });
    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") ?? "application/json"
      }
    });
  } catch (error) {
    return Response.json(
      {
        detail:
          error instanceof Error
            ? `BigBook API is unreachable: ${error.message}`
            : "BigBook API is unreachable."
      },
      { status: 503 }
    );
  }
}
