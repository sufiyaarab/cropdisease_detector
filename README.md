# Crop Disease Monitoring Platform (G24-1)

Base code for the capstone app. It follows the team's user-flow diagram:

```
Login / Sign up -> Home -> Disease detection:  Scan plant -> Validate photo and crop -> AI analysis -> Results -> Care plan -> Save plant
                        -> My plants:          Saved plants -> Plant page (photo history) -> Add new photo / updated care plan
                        -> Local outbreaks:    Enter postal code -> View alerts
```

Everything runs on your laptop with no accounts or keys. The AI model, the database and
the login system are stand-ins for now (see "What is real and what is a stand-in").

## Run it (about 5 minutes)

You need Python 3.11+ and Node 20+.

**Terminal 1: backend**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload    # http://localhost:8000  (API docs at /docs)
```

**Terminal 2: frontend**
```bash
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

Open http://localhost:5173, create an account (any email, 8+ character password) and scan a leaf photo.
To test on a phone, run `npm run dev -- --host` and set `VITE_API_BASE_URL` and `ALLOWED_ORIGINS` to your laptop's network address.

## Run the tests

```bash
cd backend && pytest -q && ruff check .
cd frontend && npm test && npm run build
```

## Where things are

| Folder | What it holds | Owner |
| --- | --- | --- |
| `frontend/` | React + TypeScript app (Vite). One file per screen in `src/pages/` | Frontend |
| `backend/app/api/` | One file per flow: `auth`, `scans`, `plants`, `outbreaks` | Backend |
| `backend/app/pipeline/` | Photo validation, model interface, diagnosis logic | Backend + ML |
| `backend/app/treatment/` | Care plan engine and its data | Treatment |
| `backend/app/store.py` | In-memory data store (to be replaced by Supabase) | Backend |
| `contracts/` | `openapi.json` (API contract) and `model_io.md` (model contract) | Lead |
| `db/migrations/` | Draft Supabase schema | Backend |
| `ml/` | Empty: training notebooks and code go here | ML |
| `docs/` | Flow-to-code map, validation rules | Everyone |

## What is real and what is a stand-in

| Part | Status | Replace with |
| --- | --- | --- |
| Screens, navigation, form checks | Real | n/a |
| Photo checks (type, size, blur, brightness) on phone and server | Real, thresholds need tuning on field photos | n/a |
| Photo resize to about 1 MB and GPS removal | Real | n/a |
| "Is this a leaf?" check | Placeholder (counts green pixels) | Trained crop gate, then YOLOv8n detector (Tier 2) |
| AI model | **Stub: gives made-up answers** | EfficientNet-B0 from Hugging Face Hub (`pipeline/model.py`) |
| Confidence threshold (0.70) and "not sure" screen | Real | n/a |
| Care plans | Draft text, practices only, not reviewed | Reviewed Ontario treatment table |
| Login and data storage | In memory, wiped on restart | Supabase Auth + Postgres + private storage |
| Outbreak alerts | Sample data | Opt-in scan reports, admin reviewed |

## Rules of the road

- The API shape lives in three places that must change together: `backend/app/schemas.py`, `frontend/src/api/types.ts`, `contracts/openapi.json` (regenerate with `scripts/export_openapi.sh`).
- Never commit keys. Copy `.env.example` to `.env` (backend) or `.env.local` (frontend).
- The team plan for branches, reviews and milestones is in the "G24-1 GitHub base plan" doc.
