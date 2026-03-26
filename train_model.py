import os
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "college_student_placement_dataset.csv")

NUMERIC_FEATURES = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed",
]

CATEGORICAL_FEATURES = ["Internship_Experience"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset() -> pd.DataFrame:
    """Load the college student placement dataset."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please keep college_student_placement_dataset.csv in the project root."
        )
    df = pd.read_csv(DATASET_PATH)
    df["career_ready"] = df["Placement"].map({"Yes": 1, "No": 0})
    return df


def train_model():
    """Train the Random Forest model on the dataset."""
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["career_ready"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    joblib.dump({"model": pipeline, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return pipeline, FEATURE_COLUMNS


def load_or_train_model():
    """Load existing model or train a new one if not found."""
    if os.path.exists(MODEL_PATH):
        artifact = joblib.load(MODEL_PATH)
        return artifact["model"], artifact["feature_columns"]
    return train_model()


def prepare_input_features(student_data: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    """Convert student form data into a DataFrame for ML prediction."""
    technical = int(student_data.get("technical_skill_rating") or 0)
    soft = int(student_data.get("soft_skill_rating") or 0)
    cgpa = float(student_data.get("cgpa") or 0.0)
    academic_year = int(student_data.get("academic_year") or 1)
    projects = int(student_data.get("num_projects") or 0)
    internships = int(student_data.get("internship_experience") or 0)
    hours = int(student_data.get("weekly_upskilling_hours") or 0)

    row = {
        "IQ": max(60, min(technical * 10, 160)),
        "Prev_Sem_Result": cgpa,
        "CGPA": cgpa,
        "Academic_Performance": max(1, min(10, academic_year * 2 + hours / 2)),
        "Extra_Curricular_Score": max(0, min(10, soft)),
        "Communication_Skills": max(0, min(10, soft)),
        "Projects_Completed": projects,
        "Internship_Experience": "Yes" if internships > 0 else "No",
    }
    return pd.DataFrame([row], columns=feature_columns)


if __name__ == "__main__":
    # Train model when run directly
    train_model()
