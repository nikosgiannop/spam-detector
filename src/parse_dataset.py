"""
parse_dataset.py

Reads the raw .txt email files and produces a clean labeled dataframe
 
Labels:
0 = ham  
1 = spam 
 
"""
 
#dependencies/imports
import pandas as pd
from pathlib import Path
 
#config
 
DATA_DIR = Path("data")
DELIMITER = "-> END OF EMAIL <-"        #auto pou orisate
 
FILES = {
    #file name              label
    "ham500.txt":             0,
    "spam100.txt":            1,
    "nigerian_format400.txt": 1,
    "nigerian_format80.txt":  1,
}
 
#parsing
 
def parse_file(filepath: Path, label: int) -> list[dict]:

    #read file safely (replace invalid characters instead of crashing)  h malakia crashare kai to vrhka meta apo mish wra :(
    text = filepath.read_text(encoding="utf-8", errors="replace")

    #split file into individual emails
    #filter removes empty entries (e.g. trailing delimiter)
    emails = [e.strip() for e in text.split(DELIMITER) if e.strip()]

    #convert into structured records        
    return [{"text": email, "label": label} for email in emails]
 
 
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