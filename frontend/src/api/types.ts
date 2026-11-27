// Mirrors backend/app/schemas.py. Change both files together
// (the source of truth is contracts/openapi.json).

export type Crop = "tomato" | "apple" | "corn";
export type Category = "disease" | "pest" | "healthy";
export type ScanStatus = "diagnosed" | "not_sure" | "rejected";

export interface Session {
  token: string;
  user_id: string;
  email: string;
}

export interface PhotoIssue {
  photo_index: number;
  code: string;
  message: string;
  tip: string;
}

export interface Candidate {
  label: string;
  display_name: string;
  category: Category;
  confidence: number;
}

export interface CareStep {
  title: string;
  detail: string;
  kind: "practice" | "product";
  source: string;
}

export interface CarePlan {
  label: string;
  display_name: string;
  summary: string;
  steps: CareStep[];
  review_status: "draft" | "reviewed";
  disclaimer: string;
}

export type SeverityLevel = "none" | "mild" | "moderate" | "severe";

export interface SpotBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Severity {
  leaf_pct: number;
  level: SeverityLevel;
  photo_index: number;
  spots: SpotBox[];
  method: "colour_estimate";
}

export interface Attention {
  photo_index: number;
  grid: number[][];
  method: "occlusion";
}

export interface LeafLook {
  leaf: Crop;
  lesion: string;
  where: "lower" | "upper" | "all";
}

export interface DiseaseVisuals {
  label: string;
  look: LeafLook;
  examples: string[];
  stages: string[];
  source: string;
}

export interface ScanResult {
  id: string;
  status: ScanStatus;
  crop: Crop;
  created_at: string;
  photo_count: number;
  issues: PhotoIssue[];
  top: Candidate | null;
  candidates: Candidate[];
  care_plan: CarePlan | null;
  model_version: string | null;
  plant_id: string | null;
  changed_from: string | null;
  severity: Severity | null;
  visuals: DiseaseVisuals | null;
  attention: Attention | null;
}

export interface PlantSummary {
  id: string;
  name: string;
  crop: Crop;
  created_at: string;
  scan_count: number;
  latest: Candidate | null;
  latest_scan_at: string | null;
}

export interface PlantDetail extends PlantSummary {
  care_plan: CarePlan | null;
  scans: ScanResult[];
}

export interface OutbreakAlert {
  id: string;
  crop: Crop;
  label: string;
  display_name: string;
  postal_prefix: string;
  reported_on: string;
  report_count: number;
  note: string;
}

export interface OutbreakResponse {
  postal_prefix: string;
  alerts: OutbreakAlert[];
  is_sample_data: boolean;
}
