"""
classifier.py

Trains and evaluates a spam classifier on the parsed email dataset.
 
Pipeline:
  TF-IDF vectoriser (ta kanei convert se numerical values) →  Multinomial Naive Bayes  
 
Also trains an SVM for comparison.
"""
 
#dependencies/imports
import pandas as pd
import numpy as np
from pathlib import Path
 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
#evaluation metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

#visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
 
#config
DATA_DIR  = Path("data")    #folder containing dataset
MODEL_DIR = Path("models")    #folder where models & plots will be saved
MODEL_DIR.mkdir(exist_ok=True)
 
DATASET   = DATA_DIR / "emails.csv"
TEST_SIZE = 0.2       #80% train, 20% test (allakste to an den sas aresei)
RANDOM_STATE = 42       #orisa to random state gia reproducibility
 
#load data function
def load_data():                                    #ensures file exists
    if not DATASET.exists():                        #removes rows with missing text or labels
        raise FileNotFoundError(                    #prints dataset distribution (spam vs ham)
            f"Dataset not found at {DATASET}.\n"
            "Run  python src/parse_dataset.py  first."
        )
    df = pd.read_csv(DATASET)
    df = df.dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} emails  (ham={( df.label==0).sum()}, spam={(df.label==1).sum()})")
    return df
 

#function for building pipelines for the models 
def build_pipelines() -> dict[str, Pipeline]:
    vectoriser = dict(              #TF-IDF converts text to numerical features based on word frequency (tf) and inverse document frequency (idf)
        analyzer="word",
        ngram_range=(1, 2),   
        max_features=30_000,     #limit vocabulary size for performance (can change this if we don't care)
        sublinear_tf=True,    #reduces dominance of very frequent words
        min_df=2,             #ignore rare words (appear only once means likely noise)
    )
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**vectoriser)),
            ("clf",   MultinomialNB(alpha=0.1)),    #works well for word count based features
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**vectoriser)),
            ("clf",   LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),   
        ]),
    }
 
#evaluation helpers, matrix etc
def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = MODEL_DIR / f"confusion_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved confusion matrix in {path}")                        #plots gia confusion matrix klp
    plt.close()
 
 
def evaluate(pipeline: Pipeline, X_test, y_test, model_name: str):
    y_pred = pipeline.predict(X_test)
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    plot_confusion_matrix(y_test, y_pred, model_name)
 

#main function
def main():
    #load the data
    df = load_data()
    X = df["text"].astype(str)
    y = df["label"].astype(int)
 
    #split it into training and testing sets (change if want validation dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
 
    #training and evaluating each model
    pipelines = build_pipelines()
    best_name, best_pipeline, best_acc = None, None, 0.0
 
    for name, pipeline in pipelines.items():
        #5fold cross validation on training set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
        print(f"\n[{name}] Cross-val F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
 
        #final fit on full training set
        pipeline.fit(X_train, y_train)
        evaluate(pipeline, X_test, y_test, name)
 
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        if acc > best_acc:
            best_acc, best_name, best_pipeline = acc, name, pipeline
 
    #save best model
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")
    print(f"Saved to {model_path}")
 
 
if __name__ == "__main__":
    main()