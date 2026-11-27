// Severity calculator: two quick questions, combined with the photo estimate.
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { SeverityMeter } from "../components/DiseaseVisuals";
import { LeafDrawing } from "../components/LeafDrawing";
import { LoadState, Notice } from "../components/ui";
import { combine, LEVEL_ADVICE, LEVEL_NAME, REFERENCE_PCTS, SPREAD_TEXT, type Spread } from "../lib/severity";
import { useLoad } from "../lib/useLoad";

const SPREADS = Object.keys(SPREAD_TEXT) as Spread[];

function pctText(pct: number) {
  return pct >= 50 ? "50% or more" : `${pct}%`;
}

export function SeverityPage() {
  const { id = "" } = useParams();
  const { data: scan, ...state } = useLoad(() => api.getScan(id), [id]);
  const [pct, setPct] = useState<number | null>(null);
  const [spread, setSpread] = useState<Spread | null>(null);
  if (!scan) return <LoadState {...state} />;

  const back = <Link className="btn btn-secondary" to={`/scans/${scan.id}`}>Back to the result</Link>;
  if (!scan.severity || !scan.visuals || !scan.top || scan.top.category === "healthy") {
    return (
      <>
        <h1>Severity calculator</h1>
        <Notice kind="info">The calculator is for scans that found a disease or pest.</Notice>
        {back}
      </>
    );
  }

  const photoPct = scan.severity.leaf_pct;
  const result = pct !== null && spread !== null ? combine(photoPct, pct, spread) : null;

  return (
    <>
      <p className="muted small">{scan.top.display_name}</p>
      <h1>Severity calculator</h1>
      <p className="muted">Two quick questions. We combine your answers with what we measured in the photo.</p>

      <fieldset className="card stack-sm">
        <legend>1. Which leaf looks most like your worst leaf?</legend>
        <p className="muted small">The share of the leaf covered by spots or yellowing.</p>
        <div className="ref-leaves">
          {REFERENCE_PCTS.map((p) => (
            <button key={p} type="button" className={`ref-leaf${pct === p ? " selected" : ""}`} aria-pressed={pct === p} onClick={() => setPct(p)}>
              <LeafDrawing look={scan.visuals!.look} amount={p / 100} yellowing={Math.min(0.8, p / 60)} width={44} seed={3} label="" />
              <span>{pctText(p)}</span>
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className="card stack-sm">
        <legend>2. How many leaves have spots?</legend>
        <div className="stack-sm">
          {SPREADS.map((s) => (
            <button key={s} type="button" className={`choice choice-wide${spread === s ? " selected" : ""}`} aria-pressed={spread === s} onClick={() => setSpread(s)}>
              {SPREAD_TEXT[s]}
            </button>
          ))}
        </div>
      </fieldset>

      <section className="card stack-sm result-box" aria-live="polite">
        <p className="muted small">Result</p>
        {result ? (
          <>
            <p className={`result-level level-${result.level}`}>{LEVEL_NAME[result.level]}</p>
            <SeverityMeter level={result.level} />
            <p>{LEVEL_ADVICE[result.level]}</p>
            <div className="explain small">
              <span>Photo measured: <strong>{photoPct < 1 ? "under 1" : Math.round(photoPct)}%</strong> of the leaf</span>
              <span>You chose: <strong>{pctText(pct!)}</strong>, {SPREAD_TEXT[spread!].toLowerCase()}</span>
              <span className="muted">
                {result.usedPhoto ? "We used the photo value because it was higher." : "We used your answer because it was the same or higher."}{" "}
                When unsure we round up, so a problem is not missed.
              </span>
            </div>
          </>
        ) : (
          <p>Answer both questions to see the result. The photo alone says <strong>{LEVEL_NAME[scan.severity.level]}</strong>.</p>
        )}
      </section>
      {back}
      <p className="muted small center">Reference leaves are drawings, not photos. This is an estimate, not a professional diagnosis.</p>
    </>
  );
}
