// My Plants flow, steps 2 and 3: plant page (photo history and results) -> add new photo / updated care plan.
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { CarePlanCard, CategoryBadge, formatDate, LoadState, Notice, ScanPhoto } from "../components/ui";
import { useLoad } from "../lib/useLoad";
import { formatConfidence } from "../lib/validation";

export function PlantPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const { data: plant, ...state } = useLoad(() => api.getPlant(id), [id]);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!plant) return <LoadState {...state} />;

  async function remove() {
    try {
      await api.deletePlant(id);
      navigate("/plants", { replace: true });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not delete. Please try again.");
    }
  }

  return (
    <>
      <Link className="link-btn" to="/plants">‹ My plants</Link>
      <h1>{plant.name}</h1>
      <p className="muted capitalize">{plant.crop} · saved {formatDate(plant.created_at)}</p>

      <Link className="btn btn-primary" to={`/scan?plant=${plant.id}`}>Add new photo</Link>
      <p className="muted small center">Add a photo every week or so to monitor progress. The care plan updates if the condition changes.</p>

      {plant.care_plan
        ? <CarePlanCard plan={plant.care_plan} />
        : <Notice kind="info">No care plan yet, because no photo has given a clear diagnosis. Try adding a new photo.</Notice>}

      <section className="card">
        <h2>Photo history</h2>
        <ul className="history">
          {plant.scans.map((scan) => (
            <li key={scan.id}>
              <ScanPhoto scanId={scan.id} alt={`Photo from ${formatDate(scan.created_at)}`} />
              <Link to={`/scans/${scan.id}`} className="history-text">
                <span className="muted small">{formatDate(scan.created_at)}</span>
                {scan.status === "diagnosed" && scan.top ? (
                  <>
                    <strong>{scan.top.display_name}</strong>
                    <span><CategoryBadge category={scan.top.category} /> {formatConfidence(scan.top.confidence)} confidence</span>
                  </>
                ) : (
                  <strong>Not sure</strong>
                )}
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {error && <Notice kind="error">{error}</Notice>}
      {confirming ? (
        <Notice kind="warning">
          <p>Delete {plant.name} from My plants? This cannot be undone.</p>
          <div className="row">
            <button className="btn btn-danger" onClick={remove}>Yes, delete</button>
            <button className="btn btn-secondary" onClick={() => setConfirming(false)}>Keep</button>
          </div>
        </Notice>
      ) : (
        <button className="link-btn center danger" onClick={() => setConfirming(true)}>Delete this plant</button>
      )}
    </>
  );
}
