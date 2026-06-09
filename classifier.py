"""
classifier.py

G: Loads the preprocessed emails.csv and trains two spam classifiers.

Pipeline role (L2 - separation of concerns):
  parse_dataset.py  → raw .txt → preprocessing → emails.csv (clean text)
  classifier.py     → emails.csv → TF-IDF features → model → evaluation

Models trained:
  1. Multinomial Naive Bayes  - fast, strong baseline for text (L3: supervised)
  2. Linear SVM               - better boundary separation (L3: supervised, SVM)

Evaluation (L5 apo zarra):
  Cross-validation F1  → unbiased estimate on training set
  Test Precision/Recall/F1 → final performance on unseen data
  Confusion Matrix     → see exactly where the model makes mistakes

Best model selected by F1 (not accuracy) - correct for imbalanced classes (L5).

# G: Task 2 (assignment brief): "extract message structure features"
    # G: TESTED — structural features do NOT discriminate Nigerian Prince spam:
    #
    #    Feature         Ham    Spam   Expected   Result
    #    caps_ratio      0.079  0.066  spam>ham   ❌ reversed
    #    exclamations    2.698  0.726  spam>ham   ❌ reversed
    #    dollar_signs    0.940  1.345  spam>ham   ✅ only useful one
    #    url_count       3.088  2.222  spam>ham   ❌ reversed
    #
    # G: WHY: Nigerian Prince emails mimic formal correspondence (L5 Zarras:
    # G: "Attackers adapt — evasion: changing features to look legitimate")
    # G: DECISION: structural features omitted — would confuse classifier.
    # G: Limitation noted for report (Task 8: limitations & future work).


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
        f1_score,          # G: added - needed for best model selection (L5)
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
def load_data():
    #G ensures dataset exists before attempting to load
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET}.\n"
            "Run  python src/parse_dataset.py  first."
        )
    df = pd.read_csv(DATASET)
    #G removes rows with missing text or labels (data quality check)
    df = df.dropna(subset=["text", "label"])
    #G prints dataset distribution (spam vs ham) for verification
    print(f"Loaded {len(df)} emails  (ham={(df.label==0).sum()}, spam={(df.label==1).sum()})")
    return df
 

#function for building pipelines for the models 

#G: H build_pipelines eixe thema!!! prin: to best model epilegetai me accuracy anti gia F1 

def build_pipelines() -> dict[str, Pipeline]:
    #G L2 apo zarra: Feature extraction ->TF-IDF converts raw text to numerical
    #G vectors based on word frequency (TF) and inverse document frequency (IDF).
    #G IDF meiwnei thn epirroh leksewn pou emfanizontai pantou (the, and ..)
    #G kai enisxuei xarakthristikes lekseis


    vectoriser = dict(
        analyzer="word",
        ngram_range=(1, 2),      # L5: unigrams + bigrams "bank transfer", "million dollars"
        max_features=56_000,     # # covers full vocabulary after min_df=2 filtering
        sublinear_tf=True,       # log(TF) αντί raw count — μειώνει dominance πολυσύχναστων λέξεων
        min_df=2,                # Agnoise tis lekseis pou emfanizontai mono mia fora (noise)
    )

    #G EDW EINAI SHMANTIKO TO CHECK GIA TOUS ARITHMOUS - STIS DIAFANEIES LEEI O ZARRAS -> Tuning: choose hyperparameters based on data, not guesswork


    return {
        #G L3 Zarras: Supervised learning kai ta 2 einai supervised classifiers. labeled data (ham=0, spam=1)

        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**vectoriser)),
            #GMultinomialNB: κατάλληλο για word count features (L3: text classification)
            #G alpha=0.1: Laplace smoothing — αποτρέπει zero probability για άγνωστες λέξεις
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**vectoriser)),
            #G LinearSVC: βρίσκει hyperplane που μεγιστοποιεί margin μεταξύ ham/spam (L3: SVM)
            #G C=1.0: regularization — ισορροπία μεταξύ margin και misclassification
            #G ⚠ L2 ΣΗΜΕΙΩΣΕΩΝ: SVM χρειάζεται scaling — το TF-IDF το κάνει ήδη αυτό εδώ
            ("clf",   LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
    }
 

""" 

before
    vectoriser = dict(
        analyzer="word",
        ngram_range=(1, 2),      # L5: unigrams + bigrams "bank transfer", "million dollars"
        max_features=30_000,     # limit vocabulary για performance
        sublinear_tf=True,       # log(TF) αντί raw count — μειώνει dominance πολυσύχναστων λέξεων
        min_df=2,                # Agnoise tis lekseis pou emfanizontai mono mia fora (noise)
    )

"""
"""
python -c "
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('data/emails.csv')
vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2)
vec.fit(df['text'])
print('Πραγματικό vocabulary size:', len(vec.vocabulary_))
"
Unigrams mono:        48,919
Unigrams + bigrams:  214,177  ← pragmatiko vocabulary
Με min_df=2:          56,184  ← auto xrhsimopoiei o kvdikas
max_features:         30,000  ← to orio pou exoyme

ti thema uphrxe prin:

min_df=2 meiwnei to voc apo 214,177 se 56,184 lekseis
ALLA to max_features=30,000 kovei ta epipleon  26,184 pio spania features
ARA KOVOUME TO 47% TOU XRHSIMOU VOC

ara vasei twn pragmatikwn dedomenwn: max_features=56_000

"""

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
    ax.set_title(f"Confusion Matrix - {model_name}")
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
    #G best_name, best_pipeline, best_acc = None, None, 0.0
    best_name, best_pipeline, best_f1 = None, None, 0.0

    for name, pipeline in pipelines.items():
        #5fold cross validation on training set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
        print(f"\n[{name}] Cross-val F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
 
        #final fit on full training set
        pipeline.fit(X_train, y_train)
        evaluate(pipeline, X_test, y_test, name)
 
 #G
 #acc = accuracy_score(y_test, pipeline.predict(X_test))
  #      if acc > best_acc:
   #         best_acc, best_name, best_pipeline = acc, name, pipeline
 

    # G: L5 (Zarras): επιλογή best model με F1, όχι accuracy
    # G: Accuracy παραπλανά σε imbalanced data — F1 συνεπές με cross-val
        f1 = f1_score(y_test, pipeline.predict(X_test))
        if f1 > best_f1:
            best_f1, best_name, best_pipeline = f1, name, pipeline
        
    #save best model
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    #G print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")
    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")
    print(f"Saved to {model_path}")
 
 
if __name__ == "__main__":
    main()


    # G: ΣΥΓΚΡΙΣΗ ΠΡΙΝ και ΜΕΤΑ ΤΙΣ ΑΛΛΑΓΕΣ:
# G: Naive Bayes  - πριν F1: 0.97  | μετα F1: 0.97  (ιδιο)
# G: Linear SVM   - πριν F1: 0.99  | μετα F1: 0.9913 (καλυτερο)
# G: Βελτιωση λογω: (1) max_features=56_000 αντι 30_000 (δινουμε πληρες vocabulary)
# G:                (2) spaCy entities (ENT_MONEY, ENT_PERCENT) - διακριτικη ικανοτητα
# G: Best model τωρα επιλεγεται με F1 (οχι accuracy) - συμφωνα με L5 του ζαρρα