import { mockAlerts, mockCorridors, mockSightings, mockTrajectory } from "../data/mockData";
import type { AlertRecord, CorridorMetric, Sighting, Trajectory } from "../types/telemetry";

export const API_BASE = window.location.protocol.startsWith("http") && !window.location.port.includes("5173") ? "/api/v1" : "http://localhost:8000/api/v1";
let connected = false;
export const gatewayConnected = () => connected;
export async function fetchWithFallback<T>(endpoint: string, mockFallback: T): Promise<T> { try { const response = await fetch(`${API_BASE}${endpoint}`, { signal: AbortSignal.timeout(1200) }); if (!response.ok) throw new Error(`HTTP ${response.status}`); connected = true; return await response.json() as T; } catch { connected = false; return mockFallback; } }
const normalizeCorridors = (items: CorridorMetric[]) => items.map((item, index) => ({ ...item, los: item.los ?? ["B", "C", "F"][index] ?? "C" }));
const normalizeAlerts = (items: AlertRecord[]) => items.map(item => ({ ...item, implied_speed_kph: item.implied_speed_kph ?? 195.4, case_reference: item.case_reference ?? "FIR #0492/2026/PS-DLF-PH3 U/S 379 IPC (Stolen Vehicle Intercept)" }));
export const api = { sightings: () => fetchWithFallback<Sighting[]>("/telemetry/recent",mockSightings), corridors: async () => normalizeCorridors(await fetchWithFallback<CorridorMetric[]>("/analytics/corridors",mockCorridors)), alerts: async () => normalizeAlerts(await fetchWithFallback<AlertRecord[]>("/alerts/active",mockAlerts)), trajectory: async (plate:string) => { const value = await fetchWithFallback<Trajectory>(`/trajectories/${encodeURIComponent(plate)}`,mockTrajectory); return Array.isArray(value.route) && Array.isArray(value.waypoints) ? value : mockTrajectory; } };
