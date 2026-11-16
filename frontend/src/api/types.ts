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
