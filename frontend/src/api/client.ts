// The only file that talks to the backend. Pages call these functions
// and never use fetch directly.
import type { Crop, OutbreakResponse, PlantDetail, PlantSummary, ScanResult, Session } from "./types";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const SESSION_KEY = "cdm.session";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

export function loadSession(): Session | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export function storeSession(session: Session | null): void {
  try {
    if (session) localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    else localStorage.removeItem(SESSION_KEY);
  } catch {
    // Private browsing: the session just lasts until the tab closes.
  }
}

// Set by AuthContext so an expired session sends the user back to the login page.
let onSessionExpired: () => void = () => {};
export function setSessionExpiredHandler(handler: () => void): void {
  onSessionExpired = handler;
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const session = loadSession();
  const headers = new Headers(init.headers);
  if (session) headers.set("Authorization", `Bearer ${session.token}`);

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "offline", "We could not reach the server. Check your connection and try again.");
  }
  if (res.ok) return res;

  let code = "server_error";
  let message = "Something went wrong on our side. Please try again.";
  try {
    const body = await res.json();
    if (typeof body.code === "string") code = body.code;
    if (typeof body.message === "string") message = body.message;
  } catch {
    // Keep the generic message.
  }
  if (res.status === 401 && session) onSessionExpired();
  throw new ApiError(res.status, code, message);
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await request(path, init);
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

function post<T>(path: string, body: unknown): Promise<T> {
  return json<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  signup: (email: string, password: string) => post<Session>("/auth/signup", { email, password }),
  login: (email: string, password: string) => post<Session>("/auth/login", { email, password }),
  logout: () => json<void>("/auth/logout", { method: "POST" }),

  createScan(crop: Crop, photos: Blob[], plantId?: string): Promise<ScanResult> {
    const form = new FormData();
    form.set("crop", crop);
    if (plantId) form.set("plant_id", plantId);
    photos.forEach((photo, i) => form.append("photos", photo, `photo-${i + 1}.jpg`));
    return json<ScanResult>("/scans", { method: "POST", body: form });
  },
  getScan: (id: string) => json<ScanResult>(`/scans/${encodeURIComponent(id)}`),
  async getScanPhoto(scanId: string, index: number): Promise<Blob> {
    return (await request(`/scans/${encodeURIComponent(scanId)}/photos/${index}`)).blob();
  },

  savePlant: (name: string, scanId: string) => post<PlantSummary>("/plants", { name, scan_id: scanId }),
  listPlants: () => json<PlantSummary[]>("/plants"),
  getPlant: (id: string) => json<PlantDetail>(`/plants/${encodeURIComponent(id)}`),
  deletePlant: (id: string) => json<void>(`/plants/${encodeURIComponent(id)}`, { method: "DELETE" }),

  outbreaks: (postalCode: string) =>
    json<OutbreakResponse>(`/outbreaks?postal_code=${encodeURIComponent(postalCode)}`),
};
