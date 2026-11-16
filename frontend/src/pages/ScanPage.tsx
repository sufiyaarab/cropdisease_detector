// Disease detection flow, steps 1 to 3: Scan plant -> Validate photo and crop -> AI analysis.
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { Crop } from "../api/types";
import { Notice, Stepper } from "../components/ui";
import { preparePhoto } from "../lib/image";
import { MAX_PHOTOS } from "../lib/validation";

const CROPS: { value: Crop; label: string }[] = [
  { value: "tomato", label: "Tomato" },
  { value: "apple", label: "Apple" },
  { value: "corn", label: "Corn" },
];

interface Photo {
  key: number;
  name: string;
  state: "checking" | "good" | "bad";
  blob: Blob | null;
  previewUrl: string | null;
  problems: { message: string; tip: string }[];
}

export function ScanPage() {
  const navigate = useNavigate();
  const plantId = useSearchParams()[0].get("plant") ?? undefined;
  const [plantName, setPlantName] = useState<string | null>(null);
  const [crop, setCrop] = useState<Crop | null>(null);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextKey = useRef(0);
  const cameraInput = useRef<HTMLInputElement>(null);
  const libraryInput = useRef<HTMLInputElement>(null);

  // Adding a photo to a saved plant: the crop is already known, so lock it.
  useEffect(() => {
    if (!plantId) return;
    api.getPlant(plantId)
      .then((plant) => { setCrop(plant.crop); setPlantName(plant.name); })
      .catch((e) => setError(e instanceof ApiError ? e.message : "We could not load that plant."));
  }, [plantId]);

  async function addFiles(files: FileList | null) {
    if (!files) return;
    setError(null);
    const room = MAX_PHOTOS - photos.length;
    const chosen = Array.from(files).slice(0, room);
    if (files.length > room) setError(`You can add up to ${MAX_PHOTOS} photos. Extra photos were skipped.`);
    for (const file of chosen) {
      const key = nextKey.current++;
      setPhotos((list) => [...list, { key, name: file.name, state: "checking", blob: null, previewUrl: null, problems: [] }]);
      const prepared = await preparePhoto(file);
      setPhotos((list) => list.map((p) => p.key !== key ? p : {
        ...p,
        state: prepared.blob ? "good" : "bad",
        blob: prepared.blob,
        previewUrl: prepared.previewUrl,
        problems: prepared.issues,
      }));
    }
  }

  function remove(key: number) {
    setPhotos((list) => {
      const gone = list.find((p) => p.key === key);
      if (gone?.previewUrl) URL.revokeObjectURL(gone.previewUrl);
      return list.filter((p) => p.key !== key);
    });
  }

  const good = photos.filter((p) => p.state === "good");
  const checking = photos.some((p) => p.state === "checking");
  const hasBad = photos.some((p) => p.state === "bad");
  const ready = crop !== null && good.length > 0 && !checking && !hasBad && !busy;

  // Tells the user the one thing still missing, instead of a silently disabled button.
  const hint = busy ? "" : !crop ? "Choose the crop first." : photos.length === 0 ? "Add at least one photo."
    : checking ? "Checking your photo…" : hasBad ? "Remove or retake the photos marked with a problem." : "";

  async function submit() {
    if (!ready || !crop) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.createScan(crop, good.map((p) => p.blob!), plantId);
      if (result.status === "rejected") {
        // The server found a problem the phone check missed. Mark those photos so the user can retake them.
        setPhotos((list) => list.map((p) => {
          const index = good.findIndex((g) => g.key === p.key);
          const issues = result.issues.filter((issue) => issue.photo_index === index);
          return issues.length ? { ...p, state: "bad", blob: null, problems: issues } : p;
        }));
        setError("We could not use one of your photos. See the tip under the photo and retake it.");
      } else {
        navigate(`/scans/${result.id}`);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  const step = busy ? 2 : photos.length > 0 ? 1 : 0;

  return (
    <>
      <Stepper current={step} />
      <h1>{plantName ? `New photo for ${plantName}` : "Scan a plant"}</h1>

      <section className="card">
        <h2>1. Which crop is it?</h2>
        <div className="choices" role="radiogroup" aria-label="Crop">
          {CROPS.map((option) => (
            <button key={option.value} type="button" role="radio" aria-checked={crop === option.value}
              className={crop === option.value ? "choice selected" : "choice"}
              disabled={!!plantId && crop !== option.value}
              onClick={() => setCrop(option.value)}>
              {option.label}
            </button>
          ))}
        </div>
        {!plantId && <p className="muted small">Growing something else? Other crops are not supported yet.</p>}
      </section>

      <section className="card">
        <h2>2. Add 1 to {MAX_PHOTOS} photos</h2>
        <ul className="tips">
          <li>Fill the frame with one leaf that shows the problem.</li>
          <li>Use daylight, without flash or direct sun.</li>
          <li>Hold steady and tap the leaf to focus.</li>
        </ul>

        <ul className="photo-list">
          {photos.map((photo, i) => (
            <li key={photo.key} className={`photo photo-${photo.state}`}>
              {photo.previewUrl ? <img className="thumb" src={photo.previewUrl} alt={`Photo ${i + 1}`} /> : <div className="thumb thumb-empty" aria-hidden="true" />}
              <div className="photo-info" aria-live="polite">
                {photo.state === "checking" && <span>Checking photo…</span>}
                {photo.state === "good" && <span className="ok">✓ Photo looks good</span>}
                {photo.state === "bad" && photo.problems.map((problem) => (
                  <p key={problem.message}><strong>{problem.message}</strong> {problem.tip}</p>
                ))}
              </div>
              <button type="button" className="link-btn" onClick={() => remove(photo.key)} aria-label={`Remove photo ${i + 1}`}>
                {photo.state === "bad" ? "Remove and retake" : "Remove"}
              </button>
            </li>
          ))}
        </ul>

        {photos.length < MAX_PHOTOS && (
          <div className="row">
            <button type="button" className="btn btn-secondary" onClick={() => cameraInput.current?.click()}>Take photo</button>
            <button type="button" className="btn btn-secondary" onClick={() => libraryInput.current?.click()}>Upload photo</button>
          </div>
        )}
        {/* Hidden inputs: one opens the camera, one opens the photo library. The value is cleared so the same file can be picked again. */}
        <input ref={cameraInput} type="file" accept="image/*" capture="environment" hidden
          onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
        <input ref={libraryInput} type="file" accept="image/jpeg,image/png,image/webp" multiple hidden
          onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
      </section>

      {error && <Notice kind="error">{error}</Notice>}
      <button className="btn btn-primary" disabled={!ready} onClick={submit}>
        {busy ? "Analysing your photo…" : "Check my plant"}
      </button>
      {hint && <p className="muted center" aria-live="polite">{hint}</p>}
      {plantId && <Link className="link-btn center" to={`/plants/${plantId}`}>Cancel</Link>}
    </>
  );
}
