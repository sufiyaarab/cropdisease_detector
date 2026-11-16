# Flow-to-code map

Each box in the user-flow diagram, and the files that implement it.

| Flow | Box in the diagram | Screen (frontend/src/pages) | API call | Backend file |
| --- | --- | --- | --- | --- |
| Start | Login / Sign up | `LoginPage.tsx` | `POST /auth/signup`, `POST /auth/login` | `api/auth.py` |
| Start | Home page | `HomePage.tsx` | none | none |
| Disease detection | Scan plant (take / upload picture) | `ScanPage.tsx` | none yet | none |
| Disease detection | Validate photo and crop | `ScanPage.tsx` + `lib/image.ts`, `lib/validation.ts` | `POST /scans` | `pipeline/validate.py` |
| Disease detection | AI analysis (disease, pest or healthy) | `ScanPage.tsx` (waiting state) | `POST /scans` | `pipeline/model.py`, `pipeline/diagnose.py` |
| Disease detection | Results (diagnosis + confidence) | `ResultPage.tsx` | `GET /scans/{id}` | `api/scans.py` |
| Disease detection | Care plan (treatment and Ontario guidelines) | `ResultPage.tsx` | part of the scan result | `treatment/engine.py` |
| My plants | My plants (saved plants) | `PlantsPage.tsx` | `GET /plants` | `api/plants.py` |
| My plants | Plant page (photo history and results) | `PlantPage.tsx` | `GET /plants/{id}`, `GET /scans/{id}/photos/{n}` | `api/plants.py`, `api/scans.py` |
| My plants | Add new photo (monitor progress) | `ScanPage.tsx?plant=ID` | `POST /scans` with `plant_id` | `api/scans.py` |
| My plants | Update care plan (if condition changes) | `ResultPage.tsx`, `PlantPage.tsx` | `changed_from` in the scan result | `api/scans.py`, `api/plants.py` |
| Local outbreaks | Check local outbreaks (enter postal code) | `OutbreaksPage.tsx` | `GET /outbreaks?postal_code=` | `api/outbreaks.py` |
| Local outbreaks | View alerts (nearby disease reports) | `OutbreaksPage.tsx` | same | same |

Taken from the second flow diagram:

- **Unable to identify, retake photo**: a photo that fails a check is marked with the reason and a tip, with a "Remove and retake" button. A low-confidence result shows a "We are not sure" screen with a "Retake photo" button.
- **Multiple results by confidence**: the results screen lists the top three matches with their confidence.
- **Save plant after the care plan**, which leads to My plants.

# Validation rules

Every check runs twice: on the phone for instant feedback, and on the server because the phone cannot be trusted.

| What | Rule | Message code |
| --- | --- | --- |
| Email | Looks like an email address | `invalid_email` |
| Password | 8 to 128 characters | `invalid_request` |
| Crop | tomato, apple or corn | `unsupported_crop` |
| Photo count | 1 to 3 | `wrong_photo_count` |
| Photo type | JPEG, PNG or WebP, and the file really is an image | `unsupported_type`, `unreadable_image` |
| Photo size | Up to 8 MB on the server (the phone shrinks to about 1 MB first) | `file_too_large` |
| Photo resolution | Shortest side at least 224 px | `too_small` |
| Brightness | Average brightness between 40 and 225 (0 to 255 scale) | `too_dark`, `too_bright` |
| Sharpness | Variance of Laplacian at least 40 | `too_blurry` |
| Leaf present | At least 12% plant-coloured pixels (placeholder) | `not_a_plant` |
| Confidence | Top match at least 0.70, otherwise "not sure" and no care plan | status `not_sure` |
| Saved plant | Name 1 to 60 characters; a new photo must be the same crop | `name_required`, `crop_mismatch` |
| Postal code | Canadian format, Ontario only (K, L, M, N, P); only the first 3 characters are used | `invalid_postal_code`, `outside_ontario` |
| Privacy | Users can only read their own scans, photos and plants | 404 |

Thresholds live in `backend/app/config.py` and `frontend/src/lib/validation.ts`. Keep them the same.
