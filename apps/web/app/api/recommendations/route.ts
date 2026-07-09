import { type NextRequest } from "next/server";
import { proxyToApi } from "../_lib/proxy";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  const body = await request.text();
  return proxyToApi("/recommendations", {
    method: "POST",
    body
  });
}
