import { type NextRequest } from "next/server";
import { proxyToApi } from "../../_lib/proxy";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  return proxyToApi(`/users/sample${request.nextUrl.search}`);
}
