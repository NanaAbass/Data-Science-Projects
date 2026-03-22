import json
import boto3
import pickle
import os
import numpy as np
import logging

# ── Step 1: Logger setup ──────────────────────────────────────────────────────
# Initialise a logger so all activity (requests, errors, predictions)
# is captured automatically by CloudWatch Logs.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Step 2: S3 client + env config ───────────────────────────────────────────
# Boto3 picks up IAM credentials automatically from the Lambda execution role —
# no hardcoded keys. Bucket name and model path come from environment variables
# set in the Lambda console, so nothing sensitive lives in source code.
s3 = boto3.client("s3")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "uti-model-bucket")
MODEL_KEY    = os.environ.get("MODEL_KEY",    "models/xgb_model.pkl")
TMP_MODEL    = "/tmp/xgb_model.pkl"

# ── Step 3: Cold-start model loader ──────────────────────────────────────────
# Lambda reuses the same execution environment across warm invocations.
# By loading the model once at module level (outside the handler function),
# subsequent requests skip the S3 download entirely — cold starts pay the
# cost (~300–500 ms), warm calls do not.
model = None

def load_model():
    global model
    if model is None:
        logger.info("Cold start: downloading model from S3...")
        s3.download_file(MODEL_BUCKET, MODEL_KEY, TMP_MODEL)
        with open(TMP_MODEL, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")
    return model


# ── Step 4: Feature extractor ─────────────────────────────────────────────────
# Pulls the exact features the XGBoost model was trained on from the
# incoming JSON body. Missing fields default to 0 rather than crashing.
# Order must match the column order used during model.fit() in the notebook.
FEATURE_ORDER = [
    "ua_wbc", "ua_bacteria", "ua_leuk", "ua_nitrite",
    "ua_clarity", "ua_protein", "ua_ph", "ua_color",
    "age", "gender", "prior_uti", "catheter",
    "ua_rbc", "ua_glucose", "ua_ketone", "ua_bilirubin",
    "ua_urobilinogen", "ua_specific_gravity",
    "wbc", "neutrophil_pct", "lymphocyte_pct",
]

def extract_features(body: dict) -> np.ndarray:
    features = [float(body.get(feat, 0)) for feat in FEATURE_ORDER]
    return np.array(features).reshape(1, -1)


# ── Step 5: CORS headers ──────────────────────────────────────────────────────
# The frontend is served from a different origin (CloudFront / S3) than the
# API Gateway URL, so every response must include these headers or the browser
# will block the response under the same-origin policy.
CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type":                 "application/json",
}


def build_response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers":    CORS_HEADERS,
        "body":       json.dumps(body),
    }


# ── Step 6: Main Lambda handler ───────────────────────────────────────────────
# AWS invokes this function for every HTTP request routed through API Gateway.
# `event`   — the full HTTP request (method, headers, body, path params).
# `context` — Lambda runtime metadata (timeout remaining, request ID, etc.).
def lambda_handler(event: dict, context) -> dict:

    # ── 6a: Preflight OPTIONS request (CORS handshake) ────────────────────────
    # Before sending the actual POST, browsers send a preflight OPTIONS request
    # to check whether the server accepts cross-origin calls.
    # We return 200 immediately with the CORS headers — no model needed.
    if event.get("httpMethod") == "OPTIONS":
        return build_response(200, {"message": "CORS preflight OK"})

    # ── 6b: Parse request body ────────────────────────────────────────────────
    # API Gateway passes the HTTP body as a JSON string inside event["body"].
    # We parse it here and bail early with a 400 if it is malformed.
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON body: {e}")
        return build_response(400, {"error": "Invalid JSON in request body."})

    # ── 6c: Input validation ──────────────────────────────────────────────────
    # Reject requests that arrive with no fields at all. A more thorough
    # implementation would validate each field's range here (e.g. 0 ≤ age ≤ 120).
    if not body:
        return build_response(400, {"error": "Request body is empty."})

    # ── 6d: Load model + run inference ───────────────────────────────────────
    # Delegates to load_model() — cached after first call.
    # predict_proba returns [[prob_negative, prob_positive]]; we take index [1].
    try:
        clf        = load_model()
        features   = extract_features(body)
        probability = float(clf.predict_proba(features)[0][1])
        prediction  = int(probability >= 0.5)

        logger.info(
            f"Prediction complete | "
            f"result={'POSITIVE' if prediction else 'NEGATIVE'} | "
            f"confidence={probability:.3f}"
        )

        # ── 6e: Build and return prediction response ──────────────────────────
        # `prediction`  — 1 (UTI likely) or 0 (UTI unlikely).
        # `probability` — raw model confidence score (0.0 – 1.0).
        # `label`       — human-readable string for the frontend to display.
        return build_response(200, {
            "prediction":  prediction,
            "probability": round(probability, 4),
            "label":       "UTI Likely" if prediction else "UTI Unlikely",
            "confidence":  f"{round(probability * 100, 1)}%",
        })

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return build_response(500, {"error": "Model inference failed. Check CloudWatch logs."})
