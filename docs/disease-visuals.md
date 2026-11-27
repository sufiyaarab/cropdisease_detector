# Disease pictures and severity

Added 2026-11-27 from the supervisor feedback asking for more visuals in the interface.
This build covers the result page and the severity calculator. The disease guide page
(better vs worse, how it spreads) and the severity chart on My plants are planned next.

## What the user sees

| Where | What | Code |
| --- | --- | --- |
| Result page | Their photo with dashed circles around the spotted areas found | `SpotPhoto` in `frontend/src/components/DiseaseVisuals.tsx` |
| Result page | "How bad is it?": None / Mild / Moderate / Severe, the % of leaf with spots, what to do | `SeverityCard` |
| Result page | 3 pictures: one spot close up, a whole leaf, where on the plant it starts | `ExamplesCard` |
| Result page | Early / Spreading / Severe strip with "You are here" | `StageStrip` |
| `/scans/:id/severity` | Severity calculator: pick the closest reference leaf, say how many leaves have spots | `frontend/src/pages/SeverityPage.tsx` |

Healthy results get the "what a healthy leaf looks like" pictures and no severity card. If a
healthy result still has 5% or more brown or yellow area in the photo, a warning asks the user
to take a closer photo, because missing a problem costs more than a false alarm.
Nothing is shown on "not sure" results.

## Severity

- Backend: `backend/app/pipeline/severity.py`. Leaf severity (%) = spot pixels / leaf pixels x 100,
  counted on a 96 x 96 copy of the photo. Spots are brown, tan, yellow or dark-brown patches with
  green around them, so soil, shadows and grey backgrounds are ignored. With several photos the one
  with the most spots is used.
- Levels: under 5% Mild, 5 to 25% Moderate, 25% and over Severe. A diagnosed disease is never "None".
  Same cut-offs in `frontend/src/lib/severity.ts`.
- Calculator: keeps the higher of the photo value and the user's pick; "several lower leaves" makes it
  at least Moderate; "lower and upper leaves" raises it one level.
- It is a colour count, not a trained model, so it is always labelled "Estimate". Before the demo,
  check it against about 30 hand-marked photos and report the average error. In Tier 2 the leaf
  detector can replace it.

## Pictures and licences

- The pictures are drawings (`frontend/src/components/LeafDrawing.tsx`), driven by
  `backend/app/data/visuals.json` (leaf shape, spot type, where on the plant, captions, stage text).
- Captions and stage text are drafts: check them against OMAFRA guidance before release.
- Real photos replace drawings only when their licence allows reuse in an app. Credit each one
  under the photo and list it in an `ATTRIBUTION.md`. Check every photo's licence yourself:
  PlantVillage (CC BY-SA 3.0), PlantDoc (CC BY 4.0), Bugwood / IPM Images (mostly CC BY, some
  non-commercial), Wikimedia Commons (per file). Extension sites and the Plant Pathology 2021
  Kaggle set are not reusable. No dataset labels stages, so stage photos need an expert check.

## API

`ScanResult` gains `severity` (leaf_pct, level, photo_index, spots as 0..1 boxes, method) and
`visuals` (label, look, examples, stages, source) on diagnosed scans. See `contracts/openapi.json`.

## "AI focus" heatmap (where the model looked)

- The result photo has three views: Spots (dashed circles), AI focus (heatmap), Plain photo.
- Backend: `backend/app/pipeline/explain.py`. It is an **occlusion map**, not Grad-CAM: each of 7 x 7 squares
  of the photo is covered in turn, and the drop in the model's confidence for its answer is the square's
  value (0 to 1). Only for disease and pest results. Costs 49 extra model runs per scan; turn it off with
  `SHOW_ATTENTION=false`.
- With the stub model the map means nothing. It becomes useful once a trained model is loaded.
- It shows what drove the answer, not how much leaf is damaged. Severity stays the spot count above.
  A heatmap on the background (not the leaf) is a warning sign that the model learned the wrong cue.
- Real Grad-CAM later: the exported `model.tflite` only outputs scores. For this model (global average
  pooling, then one dense layer per head) Grad-CAM equals a class activation map, so the export only needs
  one more output, the last feature map (7 x 7 x 1280), plus the head's dense weights. Then the backend can
  replace the occlusion map with a single model run. Do this after the current training changes in `ml/` land.
