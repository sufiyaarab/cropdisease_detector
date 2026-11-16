// My Plants flow, step 1: the list of saved plants.
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { CategoryBadge, formatDate, LoadState } from "../components/ui";
import { useLoad } from "../lib/useLoad";

export function PlantsPage() {
  const { data: plants, ...state } = useLoad(() => api.listPlants(), []);
  return (
    <>
      <h1>My plants</h1>
      <LoadState {...state} />
      {plants && plants.length === 0 && (
        <section className="card center">
          <p>You have not saved any plants yet.</p>
          <p className="muted">Scan a plant, then choose "Save to My plants" on the results page.</p>
          <Link className="btn btn-primary" to="/scan">Scan a plant</Link>
        </section>
      )}
      <div className="stack">
        {plants?.map((plant) => (
          <Link key={plant.id} to={`/plants/${plant.id}`} className="card tile">
            <strong>{plant.name}</strong>
            <span className="capitalize">{plant.crop} · {plant.scan_count} {plant.scan_count === 1 ? "photo check" : "photo checks"}</span>
            {plant.latest
              ? <span>{plant.latest.display_name} <CategoryBadge category={plant.latest.category} /></span>
              : <span className="muted">No clear diagnosis yet</span>}
            {plant.latest_scan_at && <span className="muted small">Last checked {formatDate(plant.latest_scan_at)}</span>}
          </Link>
        ))}
      </div>
    </>
  );
}
