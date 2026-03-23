README

# UTI Predict: Full-Stack Clinical Decision Support Tool

> XGBoost-powered UTI prediction — trained on clinical urinalysis data, deployed on AWS Lambda, accessible in a browser.

🔗 **[Live Demo](https://d30wt5pj3v1788.cloudfront.net)** · 📓 **[Notebook](notebook/UTI.ipynb)**

---

### The Story Behind the Project

Urinary Tract Infections are among the most commonly misdiagnosed conditions in clinical settings — and most ML projects that tackle healthcare problems never leave a Jupyter notebook. This project set out to change that: a fully deployed, browser-accessible prediction tool built on real clinical data, trained with XGBoost, served via AWS Lambda, and accessible to any clinician with a URL.

Built during ALX Data Science training. The model was the first challenge. The deployment was the real education.

---

### Architecture

```
Browser → S3 (frontend HTML)
              ↓
     API Gateway /predict
              ↓
         AWS Lambda
       (XGBoost inference)
              ↓
    S3 (private model bucket)
              ↓
       CloudWatch (logs)

CI/CD: GitHub → CodeBuild → ECR → Lambda update
```

---

### Model Performance

| Model | Evaluation |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Ensemble — feature importance used for selection |
| XGBoost (tuned) | AUC 0.91 — best params: `learning_rate=0.1`, `max_depth=5`, `n_estimators=300` |

Evaluated across ROC-AUC, Precision-Recall AUC, calibration curves, and confusion matrices on held-out test set.

---

### Key Learnings — Model

* **Challenge:** Too many clinical features with unclear predictive value.
  → **Solution:** Ran Random Forest importance and Chi-Square selection in parallel, cross-referenced results, and built a defensible final feature set from the overlap.

* **Challenge:** A urinalysis variable assumed to be the strongest predictor ranked surprisingly low.
  → **Solution:** Trusted the feature selection math over intuition — updated the feature set and model reliability improved.

* **Challenge:** Comparing three models fairly without cherry-picking metrics.
  → **Solution:** Built a reusable `evaluate_model()` function outputting ROC-AUC, PR-AUC, calibration curves, and confusion matrices consistently across all models.

---

### Key Learnings — Deployment

* **Challenge:** Docker build failed at 100s with `gcc not found` — Lambda base image has no C compiler, `scipy 1.17.1` tried to compile from source.
  → **Solution:** Pinned `scipy==1.13.1` (pre-built wheel available) and added `--prefer-binary` to pip. Added `gcc gcc-c++` to Dockerfile as a compile fallback.

* **Challenge:** Lambda rejected the ECR image with `media type not supported`.
  → **Solution:** Set `DOCKER_BUILDKIT=0` before rebuilding. BuildKit's default manifest format is incompatible with Lambda's image loader.

* **Challenge:** PowerShell mangling JSON payloads three different ways (curl alias, UTF-8 BOM, quote stripping).
  → **Solution:** Write all payloads with `[System.IO.File]::WriteAllText()` and invoke via `file://` — bypasses shell interpretation entirely.

* **Challenge:** Frontend showing `Cannot read properties of undefined (reading 'probability')` after successful Lambda invocation.
  → **Solution:** API Gateway double-encodes the Lambda response body. Added defensive unwrapping handling plain object, JSON string, and double-encoded JSON string.

---

### Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost · scikit-learn · Python 3.11 |
| Container | Docker · AWS ECR |
| Compute | AWS Lambda (container image) |
| API | AWS API Gateway (REST + CORS) |
| Storage | AWS S3 — model bucket (private) + frontend bucket |
| Monitoring | AWS CloudWatch · IAM (least-privilege roles) |
| Frontend | HTML · CSS · JavaScript — confidence gauge + feature importance |

---

### Repo Structure

```
uti-clinical-predictor/
├── notebook/
│   └── UTI.ipynb              # Full EDA, feature selection, model training
├── lambda/
│   ├── app/
│   │   └── handler.py         # Lambda prediction handler
│   ├── Dockerfile             # Container definition
│   └── requirements.txt       # Pinned dependencies
├── frontend/
│   └── index.html             # Clinical UI
└── README.md
```

---

### Running Locally

```bash
# Build the container
docker build --platform linux/amd64 --tag uti-predictor:latest ./lambda

# Run with local model (mount your .pkl file)
docker run --rm -p 9000:8080 \
  -v $(pwd)/xgb_model.pkl:/tmp/xgb_model.pkl \
  --env MODEL_BUCKET=local-test \
  --env MODEL_KEY=xgb_model.pkl \
  uti-predictor:latest

# Test
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -H "Content-Type: application/json" \
  -d '{"httpMethod":"POST","body":"{\"age\":45,\"ua_bacteria\":3,\"ua_nitrite\":1}"}'
```

---

*Feedback on architecture decisions, model calibration, or the deployment approach is very welcome — open an issue or reach out.*
