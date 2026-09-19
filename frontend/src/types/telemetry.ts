export type CongestionLevel = "NOMINAL" | "ADAPTIVE_SIGNAL" | "BOTTLENECK_WARNING" | "CRITICAL_CONGESTION";
export interface Sighting { plate: string; camera_id: string; junction: string; timestamp: string; confidence: number; vehicle_class: string; stream_profile: string; latitude: number; longitude: number; }
export interface CorridorMetric { corridor_id: string; space_mean_speed_kph: number; free_flow_speed_kph: number; cpi: number; level: CongestionLevel; los: string; wave_detected: boolean; }
export interface Waypoint { camera_id: string; junction: string; latitude: number; longitude: number; arrival_ist: string; transit_speed_kph: number; sequence: number; }
export interface GeoJSONFeature { type: "Feature"; geometry: { type: "Point" | "LineString"; coordinates: number[] | number[][] }; properties: Record<string, unknown>; }
export interface Trajectory { type: "FeatureCollection"; features: GeoJSONFeature[]; waypoints: Waypoint[]; route: [number, number][]; plate: string; }
export interface CapAlert { identifier: string; sender: string; sent: string; status: "Actual" | "Exercise"; msgType: "Alert"; scope: "Restricted"; info: Array<{ category: string[]; event: string; urgency: string; severity: string; certainty: string; headline: string; description: string; area: Array<{ areaDesc: string; polygon: string }>; }>; }
export interface AlertRecord { identifier: string; event: string; plate: string; active: boolean; cap: CapAlert; implied_speed_kph: number; case_reference: string; }
export interface DatabaseRecord { timestamp: string; node_id: string; plate: string; confidence: number; dpdp_hash: string; bbox: [number, number, number, number]; vehicle_class: string; source: string; camera_id: string; }
