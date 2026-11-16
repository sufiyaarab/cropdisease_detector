// Local outbreaks flow: Check local outbreaks (enter postal code) -> View alerts (nearby disease reports).
import { useState, type FormEvent } from "react";
import { api, ApiError } from "../api/client";
import type { OutbreakResponse } from "../api/types";
import { formatDate, Notice } from "../components/ui";
import { parsePostalCode } from "../lib/validation";

export function OutbreaksPage() {
  const [postalCode, setPostalCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<OutbreakResponse | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    const parsed = parsePostalCode(postalCode);
    if (!parsed.ok) return setError(parsed.message);
    setBusy(true);
    setError(null);
    try {
      setResult(await api.outbreaks(parsed.prefix));
    } catch (e) {
      setResult(null);
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <h1>Local outbreaks</h1>
      <form className="card form" onSubmit={submit} noValidate>
        <label htmlFor="postal">Your postal code</label>
        <input id="postal" value={postalCode} maxLength={7} autoComplete="postal-code" autoCapitalize="characters"
          placeholder="L1H 7K4" onChange={(e) => { setPostalCode(e.target.value); setError(null); }}
          aria-invalid={!!error} aria-describedby="postal-help" />
        <p id="postal-help" className={error ? "field-error" : "muted small"}>
          {error ?? "We only use the first three characters, so your exact address stays private."}
        </p>
        <button className="btn btn-primary" disabled={busy}>{busy ? "Checking…" : "Check my area"}</button>
      </form>

      {result && (
        <section aria-live="polite">
          <h2>Reports near {result.postal_prefix}</h2>
          {result.is_sample_data && <Notice kind="warning">These are sample alerts for development, not real reports.</Notice>}
          {result.alerts.length === 0 && <Notice kind="success">No disease reports in your area right now.</Notice>}
          <div className="stack">
            {result.alerts.map((alert) => (
              <article key={alert.id} className="card">
                <strong>{alert.display_name}</strong>
                <p className="capitalize">{alert.crop} · {alert.report_count} {alert.report_count === 1 ? "report" : "reports"}</p>
                <p className="muted small">Last reported {formatDate(alert.reported_on + "T12:00:00")}</p>
              </article>
            ))}
          </div>
        </section>
      )}
    </>
  );
}
