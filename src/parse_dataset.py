"""
parse_dataset.py

Reads the raw .txt email files and produces a clean labeled dataframe
 
Labels:
0 = ham  
1 = spam 

G: Pipeline role (L2 - separation of concerns):
G:   parse_dataset.py  → raw .txt → preprocessing → emails.csv (clean text)
G:   classifier.py     → emails.csv → TF-IDF features → model → evaluation
 
"""
 
#dependencies/imports
import pandas as pd
import re                    #G NEW ADDITION
import spacy                 #G NEW ADDITION: named entity extraction (Task 2)
from pathlib import Path
from collections import Counter

#G: load spaCy English model - used for named entity recognition
nlp = spacy.load("en_core_web_sm") 

#config
 
DATA_DIR = Path("data")
DELIMITER = "-> END OF EMAIL <-"        #auto pou orisate
 
#G: Dataset composition - emphasis on Nigerian Prince scam (Task 1, assignment brief):
#G: - ham500.txt             → 500 legitimate emails (label=0)
#G: - spam100.txt            → 100 generic spam emails (label=1)
#G: - nigerian_format400.txt → 400 Nigerian Prince advance-fee fraud (label=1)
#G: - nigerian_format80.txt  →  80 shorter Nigerian Prince variants (label=1)
#G: Total: 500 ham vs 580 spam - roughly balanced, no oversampling needed
#G: NOTES: Spam is intentionally Nigerian-heavy per assignment brief.
#G:       Limitation: model may not generalise well to other spam types.

FILES = {
    #file name              labels
    "ham500.txt":             0,
    "spam100.txt":            1,
    "nigerian_format400.txt": 1,
    "nigerian_format80.txt":  1,
}

#G: Ti leei h theoria: to dataset exei 500 ham vs 580 spam, pou einai sxetika balanced, opote de xreiazetai oversampling.
#G: An htan 500 vs 50, tha eixame to provlhma imbalance, opws anaferetai sto Lecture 5.
 

#G  NEW ADDITION preprocessing
#G Στο L2 leei o zarras (Foundations of AI and ML):Text data → χρειάζεται tokenization, normalization, feature extraction 

def clean_email(text: str) -> str: #G: To vazw edw kai oxi ston epomeno kwdika giati einai pio swsto sa logikh to preprocessing na bainei edw
    """
    Basic text preprocessing before feature extraction.
    Keeps semantic content, removes noise.
    (L2: preprocessing pipeline separates raw data from feature extraction)
    """
    text = text.lower()                        # normalize case: "FREE" == "free"
    text = re.sub(r'http\S+', 'URL', text)     # replace URLs with token
    text = re.sub(r'\S+@\S+', 'EMAIL', text)   # replace emails with token
    text = re.sub(r'\s+', ' ', text)           # normalize whitespace
    return text.strip()

#G  extract_entities

"""
pws ginotan prin kai giati htan lathos:

prin to TF-IDF ekane auto:

"CONTACT DR. WILLIAMS FROM NIGERIA FOR $10,000,000"
        ↓ TF-IDF
"contact" → 0.3
"williams" → 0.2  
"nigeria" → 0.4
"10" → 0.1
"000" → 0.1

to TF-IDF den kserei oti to "dr williams" einai proswpo px.
apla metraei lekseis.

twra me to spacy:

"CONTACT DR. WILLIAMS FROM NIGERIA FOR $10,000,000"
        ↓ extract_entities()
"contact dr. williams from nigeria for $10,000,000 
 ENT_PERSON ENT_GPE ENT_MONEY"

 ara ti ginetai
to modelo vlepei epipleon tokens pou tou lene: se auto to mail uparxei proswpo, xwra kai xrhmatiko poso
exei nohma giati ayto einai to pattern twn nigerian prince emails

"""


def extract_entities(text: str) -> str:
    """
    Extracts named entities from email text using spaCy.
    Task 2 (assignment brief): 'extract useful information such as
    keywords, sentence patterns, named entities, or message structure'
    Zarras: named entities are key features for spam/phishing detection.

    Examples of what this catches in Nigerian Prince emails:
    - MONEY: '$10,000,000', 'TEN MILLION DOLLARS'
    - PERSON: 'DR. WILLIAMS', 'PRINCE ABUBAKAR'
    - GPE: 'NIGERIA', 'BANK OF AFRICA'

    """
    doc = nlp(text[:100_000])  #G: limit length to avoid memory issues on very long emails
    #G: Safety limit: mean email = ~3,600 chars, only 2/1080 emails exceed 100k.
    #G: Truncation affects <0.2% of dataset - acceptable tradeoff for memory safety.

    #G: pws dialeksame to 100k :

    """
    
    python -c "
    import pandas as pd
    df = pd.read_csv('data/emails.csv')
    print('Max email length:', df['text'].str.len().max())
    print('Mean email length:', df['text'].str.len().mean().round(0))
    print('Emails over 100k chars:', (df['text'].str.len() > 100000).sum())
    "
    -> 2 mails ksepernane to orio kai kovontai! 
    
    Max email length:         166911   chars
    Mean email length:          3598   
    Emails over 100k chars:         2             ayta kovontai
    
    ti shmainei auto? oti pithanws auta ta 2 mails na exoyn attached content h einai malformed.
    den aksizei na allaksoume to orio gia 2 emails
    
    """

    #G: collect entity labels as extra tokens appended to the text
    #G: e.g. "contact dr williams" → "contact dr williams ENT_PERSON ENT_GPE"
    
    entity_tokens = [f"ENT_{ent.label_}" for ent in doc.ents]
    
    return text + " " + " ".join(entity_tokens) if entity_tokens else text


#G: Verified on dataset - Nigerian Prince emails show:
#G: ENT_MONEY (x3), ENT_PERCENT (x4), ENT_PERSON, ENT_GPE, ENT_ORG
#G: These are exactly the patterns described in L5 (Zarras) for fraud detection.
#G: TF-IDF alone cannot capture semantic entity types — spaCy fills this gap.


#parsing
 
#G: 

def parse_file(filepath: Path, label: int) -> list[dict]:

    #read file safely (replace invalid characters instead of crashing)  h malakia crashare kai to vrhka meta apo mish wra :(
    text = filepath.read_text(encoding="utf-8", errors="replace")

    #split file into individual emails
    #filter removes empty entries (e.g. trailing delimiter)
    emails = [e.strip() for e in text.split(DELIMITER) if e.strip()]

    #convert into structured records        
    #return [{"text": email, "label": label} for email in emails] ειναι για να βγει!!
    #G το αλλαξα για να γινεται και η κληση της clean_email KAI extract_entities:
    return [{"text": extract_entities(clean_email(email)), "label": label} for email in emails]

 
def build_dataset() -> pd.DataFrame:
    records = []
    for filename, label in FILES.items():
        path = DATA_DIR / filename

        #skip missing files instead of crashing
        if not path.exists():
            print(f"  WARNING: File not found, skipping: {path}")
            continue
        emails = parse_file(path, label)
        print(f"  Loaded {len(emails):>4} emails  (label={label})  ← {filename}")
        #accumulate all emails into one list
        records.extend(emails)
    
    #convert to DataFrame
    #shuffle dataset to prevent model learning file-order patterns
    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df
 
 
#main
if __name__ == "__main__":
    print("Parsing dataset...")
    df = build_dataset()
 
    print(f"\nTotal emails : {len(df)}")
    print(f"Ham  (0)     : {(df.label == 0).sum()}")
    print(f"Spam (1)     : {(df.label == 1).sum()}")
    print(f"\nSample:\n{df.head(3)}")
 
    out = DATA_DIR / "emails.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # === Entity Verification (L4: verify features are reproducible) ===
    # G: Results confirm entities discriminate spam from ham:
    # G: ENT_PERCENT : +930% in spam (Nigerian % promises)
    # G: ENT_MONEY   : +55%  in spam (large sum promises)  
    # G: ENT_GPE     : +102% in spam (Nigeria, Africa references)
    # G: ENT_DATE    : much higher in ham (newsletters, legitimate scheduling)
    # G: Coverage: ~97% both classes — entities present in almost all emails
    
    #import re
    #from collections import Counter

    ham_text  = " ".join(df[df.label==0]["text"])
    spam_text = " ".join(df[df.label==1]["text"])

    ham_ents  = re.findall(r'ENT_\w+', ham_text)
    spam_ents = re.findall(r'ENT_\w+', spam_text)

    ham_coverage  = df[df.label==0]["text"].str.contains("ENT_").mean()*100
    spam_coverage = df[df.label==1]["text"].str.contains("ENT_").mean()*100

    print(f"\n=== Entity Coverage ===")
    print(f"Ham  with entities: {ham_coverage:.1f}%")
    print(f"Spam with entities: {spam_coverage:.1f}%")

    print(f"\n=== Top Entities in SPAM ===")
    for ent, count in Counter(spam_ents).most_common(10):
        print(f"  {ent:<20} {count}")

    print(f"\n=== Top Entities in HAM ===")
    for ent, count in Counter(ham_ents).most_common(10):
        print(f"  {ent:<20} {count}")


    # G Shmantiko eurhma: to ent_percent einai 10x pio suxno sto spam. o zarras to leei sto L5: Ta Nigerian Prince emails uposxontai pososta kerdous
   
    # G mini anakefalaiwsh: 

    # ti kseroume twra: 
    # to spacy entopizei entities swsta
    # ta entities diaferoun metaky spam/ham eg ENT_PERCENT 
    # Coverage 97% sxedon ola ta emails exoun entities

    # ti den kseroume akoma: 
    # an o classifier xrhsimopoiei ta entities apotelesmatika 
    # an to  F1/Precision/Recall veltiwthhke me ta entities
    #  an to modelo genikeyei swsta
    # 
    # -> before entities: F1 = 0.97 (Naive Bayes) / 0.99 (SVM)