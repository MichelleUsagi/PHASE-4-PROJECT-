import os
import shap
import joblib
import pandas as pd
import numpy as np

# Load the trained pipeline
pipeline = joblib.load(
    os.path.join(os.getcwd(), "models", "logreg_pipeline.pkl")
)

# Prepare a small background dataset for SHAP
background_texts = pd.DataFrame({
    "text": [
        "I feel okay.",
        "It's been a rough day.",
        "Life feels stable.",
        "I am tired."
    ]
})

# Transform background texts into feature space
X_bg = pipeline.named_steps['tfidf'].transform(background_texts["text"])

# Initialize SHAP explainer with the classifier and background
explainer = shap.Explainer(
    pipeline.named_steps['clf'],
    X_bg
)

def explain_text(text: str, top_k: int = 5):
    """
    Returns the top_k features with highest absolute SHAP values
    for a single input text.
    """
    # 1) Vectorize the input text
    X_input = pipeline.named_steps['tfidf'].transform([text])

    # 2) Compute SHAP values
    shap_values = explainer(X_input)

    # 3) Get feature names from the tfidf step
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()

    # 4) Pair each feature name with its SHAP value
    values = shap_values.values[0] \
        if isinstance(shap_values.values, np.ndarray) \
        else shap_values.values

    impact = list(zip(feature_names, values))

    # 5) Sort by absolute impact and return the top_k
    top_features = sorted(
        impact,
        key=lambda pair: abs(pair[1]),
        reverse=True
    )[:top_k]

    # 6) Format output
    return [
        {"feature": feat, "shap_value": round(val, 4)}
        for feat, val in top_features
    ]
