// Small shared pieces used on several pages.
import { useEffect, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { Candidate, CarePlan, Category } from "../api/types";
import { formatConfidence } from "../lib/validation";

export function Notice({ kind, children }: { kind: "error" | "warning" | "info" | "success"; children: ReactNode }) {
  return (
    <div className={`notice notice-${kind}`} role={kind === "error" ? "alert" : "status"}>
      {children}
    </div>
  );
}

export function LoadState({ loading, error, retry }: { loading: boolean; error: string | null; retry: () => void }) {
  if (loading) return <p className="muted" aria-live="polite">Loading…</p>;
  if (error) {
    return (
      <Notice kind="error">
        <p>{error}</p>
        <button className="btn btn-secondary" onClick={retry}>Try again</button>
      </Notice>
    );
  }
  return null;
}

const CATEGORY_TEXT: Record<Category, string> = { disease: "Disease", pest: "Pest", healthy: "Healthy" };

export function CategoryBadge({ category }: { category: Category }) {
  return <span className={`badge badge-${category}`}>{CATEGORY_TEXT[category]}</span>;
}

export function ConfidenceBar({ candidate, label }: { candidate: Candidate; label?: string }) {
  const percent = formatConfidence(candidate.confidence);
  return (
    <div className="confidence">
      <div className="confidence-row">
        <span className="confidence-name">{label ?? candidate.display_name}</span>
        <span>{percent}</span>
      </div>
      <div className="bar" role="img" aria-label={`${candidate.display_name}: ${percent} confidence`}>
        <div className="bar-fill" style={{ width: percent }} />
      </div>
    </div>
  );
}

export function CarePlanCard({ plan }: { plan: CarePlan }) {
  return (
    <section className="card">
      <h2>Care plan</h2>
      <p>{plan.summary}</p>
      <ol className="steps">
        {plan.steps.map((step) => (
          <li key={step.title}>
            <strong>{step.title}</strong>
            <span>{step.detail}</span>
          </li>
        ))}
      </ol>
      <Notice kind={plan.review_status === "draft" ? "warning" : "info"}>{plan.disclaimer}</Notice>
    </section>
  );
}

/** Photos are private, so they are fetched with the user's token instead of a plain <img src>. */
export function useScanPhotoUrl(scanId: string, index = 0): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;
    api
      .getScanPhoto(scanId, index)
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [scanId, index]);
  return url;
}

export function ScanPhoto({ scanId, index = 0, alt }: { scanId: string; index?: number; alt: string }) {
  const url = useScanPhotoUrl(scanId, index);
  return url ? <img className="thumb" src={url} alt={alt} /> : <div className="thumb thumb-empty" aria-hidden="true" />;
}

const STEPS = ["Photo", "Check", "Analysis", "Results", "Care plan"];

/** Shows where the user is in the disease detection flow. */
export function Stepper({ current }: { current: number }) {
  return (
    <ol className="stepper" aria-label="Progress">
      {STEPS.map((name, i) => (
        <li key={name} className={i < current ? "done" : i === current ? "current" : ""} aria-current={i === current ? "step" : undefined}>
          <span className="dot">{i < current ? "✓" : i + 1}</span>
          <span className="step-name">{name}</span>
        </li>
      ))}
    </ol>
  );
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-CA", { year: "numeric", month: "short", day: "numeric" });
}
