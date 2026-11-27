// Disease detection flow, steps 4 and 5: Results (diagnosis + confidence) -> Care plan -> Save plant.
import { useState, type FormEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { ScanResult } from "../api/types";
import { ExamplesCard, SeverityCard, SpotPhoto, StageStrip } from "../components/DiseaseVisuals";
import { CarePlanCard, CategoryBadge, ConfidenceBar, LoadState, Notice, ScanPhoto, Stepper } from "../components/ui";
import { useLoad } from "../lib/useLoad";
import { validatePlantName } from "../lib/validation";

function SavePlant({ scan }: { scan: ScanResult }) {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const check = validatePlantName(name);
    if (!check.ok) return setError(check.message);
    setBusy(true);
    try {
      const plant = await api.savePlant(name.trim(), scan.id);
      navigate(`/plants/${plant.id}`);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not save. Please try again.");
      setBusy(false);
    }
  }

  return (
    <form className="card form" onSubmit={submit} noValidate>
      <h2>Save this plant</h2>
      <p className="muted">Keep this result and add new photos later to track how the plant is doing.</p>
      <label htmlFor="plant-name">Plant name</label>
      <input id="plant-name" value={name} maxLength={60} placeholder="For example: Balcony tomato"
        onChange={(e) => { setName(e.target.value); setError(null); }}
        aria-invalid={!!error} aria-describedby="plant-name-error" />
      <p id="plant-name-error" className="field-error">{error ?? ""}</p>
      <button className="btn btn-primary" disabled={busy}>{busy ? "Saving…" : "Save to My plants"}</button>
    </form>
  );
}

export function ResultPage() {
  const { id = "" } = useParams();
  const { data: scan, ...state } = useLoad(() => api.getScan(id), [id]);
  if (!scan) return <LoadState {...state} />;

  const retakeLink = scan.plant_id ? `/scan?plant=${scan.plant_id}` : "/scan";
  const others = scan.candidates.slice(1);

  return (
    <>
      <Stepper current={scan.status === "diagnosed" ? 4 : 3} />
      {scan.status === "diagnosed" && scan.severity
        ? <SpotPhoto scanId={scan.id} severity={scan.severity} attention={scan.attention} />
        : <ScanPhoto scanId={scan.id} alt="The photo you scanned" />}

      {scan.status === "diagnosed" && scan.top && (
        <>
          <section className="card">
            <p className="muted small">Most likely</p>
            <h1>{scan.top.display_name} <CategoryBadge category={scan.top.category} /></h1>
            <ConfidenceBar candidate={scan.top} label="Confidence" />
            {scan.changed_from && (
              <Notice kind="warning">The condition has changed since the last photo (was: {scan.changed_from}). The care plan below is updated.</Notice>
            )}
            {others.length > 0 && (
              <details>
                <summary>Other possibilities</summary>
                {others.map((c) => <ConfidenceBar key={c.label} candidate={c} />)}
              </details>
            )}
          </section>
          {scan.severity && <SeverityCard scanId={scan.id} severity={scan.severity} isHealthy={scan.top.category === "healthy"} />}
          {scan.visuals && <ExamplesCard visuals={scan.visuals} name={scan.top.display_name} />}
          {scan.visuals && scan.severity && scan.top.category !== "healthy" && <StageStrip visuals={scan.visuals} level={scan.severity.level} />}
          {scan.care_plan && <CarePlanCard plan={scan.care_plan} />}
        </>
      )}

      {scan.status === "not_sure" && (
        <section className="card">
          <h1>We are not sure</h1>
          <p>The photo did not match one problem clearly enough, so we are not giving a diagnosis or a care plan. These are the closest matches:</p>
          {scan.candidates.map((c) => <ConfidenceBar key={c.label} candidate={c} />)}
          <h2>What you can do</h2>
          <ul className="tips">
            <li>Retake the photo closer to one affected leaf, in daylight.</li>
            <li>Add a second photo of a different affected leaf.</li>
            <li>Show the leaf to a local garden centre or plant expert.</li>
          </ul>
          <Link className="btn btn-primary" to={retakeLink}>Retake photo</Link>
        </section>
      )}

      {scan.plant_id
        ? <Link className="btn btn-secondary" to={`/plants/${scan.plant_id}`}>Go to this plant's page</Link>
        : <SavePlant scan={scan} />}
      {scan.status === "diagnosed" && <Link className="link-btn center" to={retakeLink}>Scan again</Link>}
      <p className="muted small center">This is an AI estimate, not a professional diagnosis.</p>
    </>
  );
}
