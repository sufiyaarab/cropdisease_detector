// Result page pictures: spots circled on the photo, severity meter,
// example pictures of the disease, and the stage strip.
import { useState } from "react";
import { Link } from "react-router-dom";
import type { Attention, DiseaseVisuals, Severity, SeverityLevel } from "../api/types";
import { LEVEL_ADVICE, LEVEL_NAME, LEVELS, stageIndex } from "../lib/severity";
import { LeafDrawing, LesionCloseup, PlantDrawing } from "./LeafDrawing";
import { useScanPhotoUrl } from "./ui";

type PhotoView = "spots" | "attention" | "photo";

/**
 * The user's photo with two overlays: dashed circles where spots were found, and
 * a heatmap of where the AI looked (the squares that changed its answer most).
 */
export function SpotPhoto({ scanId, severity, attention }: { scanId: string; severity: Severity; attention: Attention | null }) {
  const url = useScanPhotoUrl(scanId, severity.photo_index);
  const [view, setView] = useState<PhotoView>("spots");
  const count = severity.spots.length;
  const views: [PhotoView, string][] = [["spots", "Spots"], ...(attention ? [["attention", "AI focus"] as [PhotoView, string]] : []), ["photo", "Plain photo"]];
  return (
    <section className="card spot-photo">
      <div className="spot-frame">
        {url ? <img src={url} alt="The photo you scanned" /> : <div className="thumb thumb-empty" aria-hidden="true" />}
        {url && view === "spots" && severity.spots.map((s, i) => (
          <span
            key={i}
            className="spot-ring"
            aria-hidden="true"
            style={{ left: `${(s.x - s.w * 0.25) * 100}%`, top: `${(s.y - s.h * 0.25) * 100}%`, width: `${s.w * 150}%`, height: `${s.h * 150}%` }}
          />
        ))}
        {url && view === "attention" && attention && <Heatmap grid={attention.grid} />}
      </div>
      <div className="view-switch" role="group" aria-label="What to show on the photo">
        {views.map(([v, name]) => (
          <button key={v} type="button" className={`choice${view === v ? " selected" : ""}`} aria-pressed={view === v} onClick={() => setView(v)}>{name}</button>
        ))}
      </div>
      <div className="spot-bar small">
        {view === "spots" && (count > 0
          ? <><span className="spot-key" aria-hidden="true" /><span>{count === 1 ? "1 spotted area found" : `${count} spotted areas found`}</span></>
          : <span className="muted">No clear spots found in the photo.</span>)}
        {view === "attention" && (
          <div className="stack-xs grow">
            <div className="heat-legend" aria-hidden="true"><span>Less</span><span className="heat-scale" /><span>More</span></div>
            <span className="muted">Stronger colour = parts of the photo that mattered most for the answer. It shows what the AI used, not how much of the leaf is damaged.</span>
          </div>
        )}
        {view === "photo" && <span className="muted">Your photo, nothing drawn on it.</span>}
      </div>
    </section>
  );
}

function Heatmap({ grid }: { grid: number[][] }) {
  const n = grid.length;
  return (
    <div className="heatmap" aria-hidden="true" style={{ gridTemplateColumns: `repeat(${n}, 1fr)`, gridTemplateRows: `repeat(${n}, 1fr)` }}>
      {grid.flatMap((row, r) => row.map((v, c) => <span key={`${r}-${c}`} style={{ background: `rgba(214, 69, 0, ${(v * 0.7).toFixed(2)})` }} />))}
    </div>
  );
}

export function SeverityMeter({ level }: { level: SeverityLevel }) {
  return (
    <div className="meter" role="img" aria-label={`Severity: ${LEVEL_NAME[level]}`}>
      <div className="meter-bars">
        {LEVELS.map((l) => <span key={l} className={`meter-${l}${l === level ? " on" : ""}`} />)}
      </div>
      <div className="meter-labels" aria-hidden="true">
        {LEVELS.map((l) => <span key={l} className={l === level ? "on" : ""}>{LEVEL_NAME[l]}</span>)}
      </div>
    </div>
  );
}

/** "How bad is it?" card with the photo estimate and a link to the calculator. */
export function SeverityCard({ scanId, severity, isHealthy }: { scanId: string; severity: Severity; isHealthy: boolean }) {
  if (isHealthy) {
    // Missing a problem costs more than a false alarm, so say so when the photo shows spots anyway.
    return severity.leaf_pct >= 5 ? (
      <div className="notice notice-warning" role="status">
        <p>About {Math.round(severity.leaf_pct)}% of the leaf has brown or yellow patches. If you can see spots, take a closer photo of one spotted leaf.</p>
      </div>
    ) : null;
  }
  return (
    <section className="card stack-sm">
      <div className="card-head">
        <h2>How bad is it?</h2>
        <span className="muted small">Estimate</span>
      </div>
      <p className="severity-line">
        <strong className={`level-${severity.level}`}>{LEVEL_NAME[severity.level]}</strong>
        {" · "}about {severity.leaf_pct < 1 ? "under 1" : Math.round(severity.leaf_pct)}% of the leaf has spots
      </p>
      <SeverityMeter level={severity.level} />
      <p className="small">{LEVEL_ADVICE[severity.level]}</p>
      <Link className="link-btn" to={`/scans/${scanId}/severity`}>Check it yourself with the severity calculator</Link>
    </section>
  );
}

/** Three drawings: one spot close up, a whole leaf, and where on the plant it starts. */
export function ExamplesCard({ visuals, name }: { visuals: DiseaseVisuals; name: string }) {
  const [closeup, leaf, plant] = visuals.examples;
  const healthy = visuals.look.lesion === "none";
  return (
    <section className="card stack-sm">
      <h2>{healthy ? "What a healthy leaf looks like" : `What ${name.toLowerCase()} looks like`}</h2>
      <p className="muted small">Compare your leaf with these pictures.</p>
      <div className="examples">
        <figure>
          <div className="example-box"><LesionCloseup lesion={visuals.look.lesion} label={closeup} /></div>
          <figcaption>{closeup}</figcaption>
        </figure>
        <figure>
          <div className="example-box"><LeafDrawing look={visuals.look} amount={healthy ? 0 : 0.12} yellowing={healthy ? 0 : 0.25} width={56} label={leaf} /></div>
          <figcaption>{leaf}</figcaption>
        </figure>
        <figure>
          <div className="example-box"><PlantDrawing look={visuals.look} label={plant} /></div>
          <figcaption>{plant}</figcaption>
        </figure>
      </div>
      <p className="muted small">{visuals.source}</p>
    </section>
  );
}

const STAGE_NAMES = ["Early", "Spreading", "Severe"];
const STAGE_AMOUNT = [0.03, 0.14, 0.4];

/** Early / spreading / severe, with the user's stage marked. */
export function StageStrip({ visuals, level }: { visuals: DiseaseVisuals; level: SeverityLevel }) {
  if (visuals.stages.length !== 3) return null;
  const current = stageIndex(level);
  return (
    <section className="card stack-sm">
      <h2>Where it is now</h2>
      <ol className="stages">
        {visuals.stages.map((text, i) => (
          <li key={i} className={i === current ? "current" : ""} aria-current={i === current ? "step" : undefined}>
            <LeafDrawing look={visuals.look} amount={STAGE_AMOUNT[i]} yellowing={i * 0.4} width={44} seed={11 + i} label={`${STAGE_NAMES[i]}: ${text}`} />
            <strong>{i + 1}. {STAGE_NAMES[i]}</strong>
            {i === current && <span className="you-are-here">You are here</span>}
            <span className="small">{text}</span>
          </li>
        ))}
      </ol>
      <p className="muted small">Based on the severity estimate. Check it with the calculator if it looks wrong.</p>
    </section>
  );
}
