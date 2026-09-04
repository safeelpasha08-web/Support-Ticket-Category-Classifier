
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = os.path.join(
    "data",
    "tickets.csv"
)

MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "customer_support_ticket_classifier.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("CUSTOMER SUPPORT TICKET CLASSIFIER - MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully.")
print(f"Total records: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["text", "category"]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' is missing from dataset."
        )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=["text", "category"]
)

df["text"] = (
    df["text"]
    .astype(str)
    .str.strip()
)

df["category"] = (
    df["category"]
    .astype(str)
    .str.strip()
)

df = df[
    (df["text"] != "") &
    (df["category"] != "")
]

print(f"Records after cleaning: {len(df)}")


# ============================================================
# DISPLAY CATEGORIES
# ============================================================

print(
    f"\nNumber of categories: {df['category'].nunique()}"
)

print("\nCategories:")

for category in sorted(
    df["category"].unique()
):

    print(f" - {category}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df["text"]
y = df["category"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=100000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    f"TF-IDF training shape: {X_train_tfidf.shape}"
)


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training completed.")


# ============================================================
# PREDICTION
# ============================================================

print("\nEvaluating model...")

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy:      {accuracy * 100:.2f}%"
)

print(
    f"Weighted F1:   {weighted_f1 * 100:.2f}%"
)

print(
    f"Macro F1:      {macro_f1 * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nModel saved to:\n{MODEL_PATH}"
)

print(
    f"\nVectorizer saved to:\n{VECTORIZER_PATH}"
)

print("\nYou can now run the Streamlit application with:")

print("\nstreamlit run app.py")
