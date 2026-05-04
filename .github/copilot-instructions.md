# Stitch Companion – Copilot Instructions

## Project Overview

**Stitch Companion** is a web application that converts uploaded images into cross-stitch patterns. Users upload a photo, select an image type and parameters, and receive a downloadable cross-stitch pattern matched to embroidery thread colors (e.g., DMC).

---

## Tech Stack

| Layer              | Technology                               |
| ------------------ | ---------------------------------------- |
| Frontend           | React 18, Axios, Create React App        |
| Backend            | Python 3.12, Flask, Flask-CORS           |
| Image Processing   | Pillow (PIL), OpenCV (cv2)               |
| Pattern Clustering | scikit-learn KMeans                      |
| Color Data         | `dmc_colors.csv` — DMC thread RGB values |
| Containerization   | Docker + Docker Compose                  |

---

## How It Runs

### Docker (recommended)

```bash
docker-compose up   # frontend → http://localhost:3000, backend → http://localhost:5001
docker-compose down
```

### Local (no Docker)

**Backend:**

```bash
cd backend
pip install -r requirements.txt
export FLASK_APP=app/routes.py
flask run --host=0.0.0.0 --port=5001
```

**Frontend:**

```bash
cd frontend
npm install
npm start   # http://localhost:3000
```

---

## Architecture & Data Flow

```
User (browser)
  ↓  selects image + parameters
React Frontend (port 3000)
  ↓  POST /upload  multipart/form-data (image, image_type, color_count, thread_brand)
Flask Backend (port 5001)
  ↓  process_image()   — opens PIL image, converts colorspace per image_type
  ↓  generate_pattern() — applies KMeans / edge detection / illustration logic
  ↓  saves PNG to backend/app/generated_patterns/output_pattern.png
  ↓  returns PNG blob
React Frontend
  ↓  URL.createObjectURL(blob) → displays preview image
```

### Key Files

| File                                        | Role                                                                                   |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| `backend/app/routes.py`                     | Flask app, `/upload`, `/preview`, `/download` endpoints                                |
| `backend/app/image_processing.py`           | `process_image()` — opens + colorspace-converts image                                  |
| `backend/app/pattern_generation.py`         | `generate_pattern()` — KMeans, edge detection, illustration stubs                      |
| `frontend/src/components/ImageUploader.js`  | Upload form + POST to backend                                                          |
| `frontend/src/components/PatternPreview.js` | Renders pattern image (currently unused in App.js)                                     |
| `frontend/src/App.js`                       | Root component — wires uploader and preview                                            |
| `dmc_colors.csv`                            | DMC thread color RGB reference data                                                    |
| `scripts/`                                  | Standalone R&D scripts; contains more advanced color-matching logic not yet integrated |

---

## Image Type Modes

| Mode           | Strategy                                         | Status                   |
| -------------- | ------------------------------------------------ | ------------------------ |
| `painting`     | KMeans clustering to `color_count` colors        | Working                  |
| `icon`         | Canny edge detection for single-color silhouette | Bug (see below)          |
| `illustration` | Background removal + content-aware crop          | Stub only (returns None) |

---

## Outstanding MVP Items

### Critical (blocks MVP)

1. **`/preview` and `/download` endpoints** — both are stubs (`pass`). `/download` should serve the saved pattern file; `/preview` can return the current saved pattern.
2. **`generate_illustration_pattern()` returns `None`** — calling `pattern.save()` on `None` crashes the server if the user selects "illustration" mode. Must be implemented or return a meaningful error.
3. **Thread color matching not integrated** — `thread_brand` is accepted as a parameter but ignored. `dmc_colors.csv` and the `colormath`-based matching logic in `scripts/pattern_generator_rgb_2.py` must be ported into `pattern_generation.py` to map KMeans cluster centers to nearest DMC thread colors.
4. **Parameters hardcoded in frontend** — `image_type`, `color_count`, and `thread_brand` are hardcoded in `ImageUploader.js`. A `ParameterForm` component (dropdowns + slider per the architecture spec) must be built and wired to state in `App.js`.
5. **`PatternPreview` component unused** — `App.js` renders the image directly; it should use `<PatternPreview pattern={imageSrc} />` instead.

### Important (spec requirements)

6. **`DownloadButton` component** — not yet created; needed for `/download` endpoint integration.
7. **URL/link-based image input** — architecture requires accepting an image URL in addition to file upload; neither the frontend input nor the backend URL-fetch logic (`requests`) exist in the current app code (only imported, not used).
8. **Cross-stitch grid overlay** — the current output is just a color-quantized image, not a cross-stitch pattern. The pattern should include a pixel/grid representation with stitch counts and a color legend (symbol-per-color key).

### Should Have (post-MVP)

9. User authentication for saving patterns.
10. Real-time parameter adjustment with live re-generation.
11. Fine-tuning / manual color-area editing.

---

## Known Bugs

1. **`generate_icon_pattern()` – wrong channel conversion**
   - `image_processing.py` converts icon images to grayscale (`Image.convert('L')`), producing a 2D numpy array.
   - `pattern_generation.py` then calls `cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)` on a 2D array, which raises a `cv2` error (expects 3-channel input).
   - **Fix**: Remove the `cvtColor` call; pass the grayscale array directly to `cv2.Canny`.

2. **`generate_illustration_pattern()` returns `None`**
   - Route calls `pattern.save(pattern_file)` unconditionally; `None.save()` raises `AttributeError`.
   - **Fix**: Implement the function or have the route guard against `None` with a 501 response.

3. **Docker Compose volume mount wipes `node_modules`**
   - `./frontend:/app` overwrites the container's `node_modules` installed during the build.
   - **Fix**: Add a named volume for `node_modules` in `docker-compose.yml`:
     ```yaml
     volumes:
       - ./frontend:/app
       - /app/node_modules
     ```

4. **No file size limit on uploads**
   - Large files can cause memory exhaustion or slow responses (OWASP DoS risk).
   - **Fix**: Add `app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024` (16 MB) in `routes.py`.

5. **`output_pattern.png` is overwritten on every request**
   - Concurrent requests corrupt each other's output.
   - **Fix**: Use a unique filename per request (e.g., `uuid4()`).

---

## Conventions & Patterns to Follow

- Backend uses Flask with `flask_cors` CORS enabled globally — keep this pattern.
- `process_image()` handles only colorspace conversion; all algorithmic logic lives in `pattern_generation.py`.
- Frontend posts to `http://localhost:5001` — use an environment variable (`REACT_APP_API_URL`) instead of hardcoding for Docker portability.
- The `scripts/` directory contains standalone R&D scripts — do not import from them directly; port logic into `backend/app/` before use.
- `dmc_colors.csv` is the source of truth for thread color data; load it once and cache it.
