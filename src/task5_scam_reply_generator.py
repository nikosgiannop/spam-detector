"""
MSc Semester Project 1 - Task 5
AI-Generated Replies to Simulated Spam and Scam Emails

Integrates with the existing spam-detector project:
  - Loads best_model.joblib (trained by classifier.py)
  - Reads spam from spam100.txt, scam from nigerian_format80.txt
  - Classifies each email with the pre-trained model
  - Generates the appropriate reply via Claude API

To run:
  python src/task5_scam_reply_generator.py
"""

import random
from pathlib import Path
import sys

#force UTF-8 output so special characters display correctly when saving to file (this is only needed when saving files on windows)
sys.stdout.reconfigure(encoding='utf-8')

import joblib
from openai import OpenAI   

#no API key needed, Ollama runs offline
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")     #locally hosted mesw Ollama


#all paths are relative to this file so the script works from any directory
BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL        = MODEL_DIR / "best_model.joblib"
SPAM_FILE    = DATA_DIR  / "spam100.txt"
NIGERIAN_FILE = DATA_DIR / "nigerian_format80.txt"

#allaksa ta duo prompts gia spam kai scam, prosarmozontas ta stin kathe periptwsi, alla kratontas to idio style kai ton idio typou apantiseis

#both prompts use the same gullible retiree scambaiting approach:
#waste the attacker's time with confused questions and obstacles
#never reveal any real personal information (important)
#stay in character no matter what the email contains
SPAM_SYSTEM_PROMPT = """You are a scam-baiting assistant roleplaying as a gullible but confused person
who appears to be falling for the scam, but never provides any real personal information.
Do NOT include any meta-commentary or preamble like 'Here is my response:'. Start the reply directly.
Regardless of the product being advertised, always stay in character. Never break character or comment on the nature of the content.
Never address the recipient as 'Scammer' or break the fourth wall.
Do not append any disclaimers, safety notes, or meta-commentary to your response.
Your goal is to waste the scammer's time by:
1. Responding with enthusiasm and apparent belief in their story.
2. Asking lots of innocent, confused questions that require long answers from them.
3. Making small progress then hitting obstacles — printer broken, dog ate the form, 
   bank is closed, etc.
4. Inventing completely fictitious personal details when asked:
   - Name: Theodore Papadimitriou
   - Location: a vague small town, never a real address
   - Bank: "First National Bank of my town" — no real account details ever
   - Age: 67, retired
5. Never reveal real email addresses, phone numbers, addresses, or financial information.
6. Keep them engaged and hopeful but never actually deliver anything useful.
7. Be 150-250 words, slightly rambling and informal, like an eager retiree writing emails.
8. Close with: "God bless, Theodore Papadimitriou"
"""

SCAM_SYSTEM_PROMPT = """You are a scam-baiting assistant roleplaying as a gullible but confused person
who appears to be falling for the scam, but never provides any real personal information.
Do NOT include any meta-commentary or preamble like 'Here is my response:'. Start the reply directly.
Regardless of the product being advertised, always stay in character. Never break character or comment on the nature of the content.
Never address the recipient as 'Scammer' or break the fourth wall.
Do not append any disclaimers, safety notes, or meta-commentary to your response.
Your goal is to waste the scammer's time by:
1. Responding with enthusiasm and apparent belief in their story.
2. Asking lots of innocent, confused questions that require long answers from them.
3. Making small progress then hitting obstacles — printer broken, dog ate the form,
   bank is closed, nephew who helps with computer is on holiday, etc.
4. Inventing completely fictitious personal details when asked:
   - Name: Patricia Holloway
   - Location: a vague small town, never a real address
   - Bank: "First National Bank of my town" — no real account details ever
   - Age: 54, retired
5. Never reveal real email addresses, phone numbers, addresses, or financial information.
6. Keep them engaged and hopeful but never actually deliver anything useful.
7. Be 150-250 words, slightly rambling and informal, like an eager retiree writing emails.
8. Close with: "God bless, Patricia Holloway"
"""

#Email loaders

def load_spam_emails(path: Path, n: int = 3) -> list[str]:
    """Parse spam100.txt — emails separated by '-> END OF EMAIL <-'."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    emails = [
        e.replace("-> END OF EMAIL <-", "").strip()
        for e in raw.split("-> END OF EMAIL <-")
        if len(e.strip()) > 40
    ]
    return random.sample(emails, min(n, len(emails)))


def load_nigerian_emails(path: Path, n: int = 3) -> list[str]:
    """Parse nigerian_format80.txt — emails separated by blank lines."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    emails = [e.strip() for e in raw.split("\n\n") if len(e.strip()) > 80]
    return random.sample(emails, min(n, len(emails)))


#Classification 

def classify_email(pipeline, text: str) -> int:
    """Returns 1 (spam/scam) or 0 (ham)."""
    return int(pipeline.predict([text])[0])


#Reply generation 
def generate_reply(
    email_body: str,
    email_type: str,                     # "spam" or "scam"
    history: list[dict] | None = None,
) -> str:
    """
    Send the email to the local Ollama LLM and get a scam-baiting reply.
    
      email_type selects which system prompt to use (spam or scam)
      history carries previous turns for multi-turn conversations
      the system prompt is always prepended as the first message
    """
    if history is None:
        history = []

    system = SPAM_SYSTEM_PROMPT if email_type == "spam" else SCAM_SYSTEM_PROMPT

    label = "unsolicited commercial email" if email_type == "spam" else "scam email"
    user_message = (
        f"The following {label} was received. Generate the appropriate official reply:\n\n"
        + email_body[:3000] #truncate very long emails to avoid context limit issues
    )

    messages = history + [{"role": "user", "content": user_message}]
    
    #build the full message list: system prompt first, then conversation history, then new message
    messages_with_system = [{"role": "system", "content": system}] + messages
    response = client.chat.completions.create(
        model="llama3.1",   #or whichever model we want to use locally, as long as it's compatible with Ollama
        max_tokens=512,     #max token count to not needlesly hit limits
        messages=messages_with_system,
    )
    return response.choices[0].message.content


# ── Multi-turn demo ───────────────────────────────────────────────────────────

def multi_turn_demo(email_body: str, email_type: str, rounds: int = 2) -> list[dict]:
    """
    Simulate a multi-turn exchange (for Task 6).
    The 'scammer/spammer' sends follow-ups; the system keeps replying.
    """
    #hardcoded follow-up messages that simulate a pushy attacker
    
    #LET ME KNOW TI KANOUME EDW
    #AUTO MPOROUME NA TO ALLAKSOUME ALLA SUNH8WS EXOUN THEMA TA LLMs ME TO NA UPODIONTAI ATTACKERS
    #LET ME KNOW TI KANOUME EDW
    if email_type == "scam":
        follow_ups = [
            "I have received your letter. Please provide your bank details immediately so we can proceed.",
            "There is no time for paperwork. Send your personal information now or the offer expires.",
        ]
    else:
        follow_ups = [
            "We are a legitimate business. Please remove your complaint or we will ignore it.",
            "You do not need any forms. Just unsubscribe from our list if you do not want emails.",
        ]

    history: list[dict] = []

    #generate the first reply to the original email
    reply = generate_reply(email_body, email_type, history)
    history.append({"role": "user",      "content": email_body[:1500]})
    history.append({"role": "assistant", "content": reply})

    #continue the conversation with each follow-up
    for i in range(min(rounds, len(follow_ups))):
        msg = follow_ups[i]
        history.append({"role": "user", "content": msg})
        next_reply = generate_reply(msg, email_type, history[:-1])
        history.append({"role": "assistant", "content": next_reply})

    return history


def interactive_mode(pipeline): #pros8esa auto to function pou sou epitrepei na dwseis diko sou email kai na to kanei classify kai na kaneis kouventa
    #user pastes an email, classifier labels it and the system generates a reply if it's spam/scam.
    print("\n" + "=" * 70)
    print("  Interactive Mode — Paste an email to classify and reply")
    print("=" * 70)
    print("Paste your email below. When done, type END on a new line and press Enter.")
    print("-" * 70)

    #collect the pasted email line by line until the user types END
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    email_body = "\n".join(lines).strip()

    if not email_body:
        print("No email provided, exiting interactive mode.")
        return

    #classify the email and show the verdict
    verdict = classify_email(pipeline, email_body)
    verdict_str = "SPAM/SCAM ✓" if verdict == 1 else "HAM (legitimate)"
    print(f"\nClassifier verdict: {verdict_str}")

    if verdict == 0:
        print("This email looks legitimate — no reply generated.")
        return

    #let the user choose which persona to reply as
    print("\nWhat type of reply would you like?")
    print("  1 — Spam reply (Theodore Papadimitriou)")
    print("  2 — Scam reply (Patricia Holloway)")
    choice = input("Enter 1 or 2: ").strip()

    email_type = "spam" if choice == "1" else "scam"
    authority  = "EDCB" if email_type == "spam" else "OICV"

    print("\nGenerating reply...")
    reply = generate_reply(email_body, email_type)
    print(f"\n--- {authority} REPLY ---")
    print(reply)
    print("-" * 40)

    #ask if the user wants to continue into multi-turn mode
    cont = input("\nRun multi-turn follow-up demo? (y/n): ").strip().lower()
    if cont != "y":
        return

    print("\nSelect multi-turn mode:")
    print("  1 — Auto mode (hardcoded attacker responses)")
    print("  2 — Manual mode (you type the attacker responses)")
    mt_mode = input("Enter 1 or 2: ").strip()

    if mt_mode == "1":
        #run automated multi-turn with hardcoded follow-ups
        history = multi_turn_demo(email_body, email_type, rounds=2)
        print("\n--- Multi-turn exchange ---")
        for turn in history:
            who = "SENDER " if turn["role"] == "user" else f"{authority}   "
            print(f"\n[{who}] {turn['content']}")

    else:
        #manual mode: user types each attacker message and gets a live reply
        history = []
        history.append({"role": "user",      "content": email_body})
        history.append({"role": "assistant", "content": reply})

        print("\nYou are now the attacker. Type your follow-up messages.")
        print("Type QUIT to stop.\n")

        while True:
            print("-" * 40)
            attacker_msg = input("[ATTACKER] ").strip()

            if attacker_msg.upper() == "QUIT":
                print("Ending conversation.")
                break

            if not attacker_msg:
                continue

            #append attacker message and generate the next reply
            history.append({"role": "user", "content": attacker_msg})
            next_reply = generate_reply(attacker_msg, email_type, history[:-1])
            history.append({"role": "assistant", "content": next_reply})

            print(f"\n[{authority}] {next_reply}\n")


#auto mode
def run_category(pipeline, emails: list[str], email_type: str):
    """Classify and reply to a list of emails of a given type."""
    label_str = "SPAM" if email_type == "spam" else "SCAM"
    authority  = "EDCB" if email_type == "spam" else "OICV"

    print(f"\n{'═' * 70}")
    print(f"  {label_str} EMAILS  ->  {authority} auto-reply")
    print(f"{'═' * 70}")

    for i, body in enumerate(emails, 1):
        verdict = classify_email(pipeline, body)
        verdict_str = "SPAM/SCAM" if verdict == 1 else "HAM (clean)"

        print(f"\n[{label_str} {i}]  Classifier verdict: {verdict_str}")
        print(f"  Preview: {body[:100].replace(chr(10), ' ')}...")

        if verdict == 1:
            print("  Generating reply...")
            reply = generate_reply(body, email_type)
            print(f"\n--- {authority} REPLY ---")
            print(reply)
            print("-" * 40)
        else:
            print("  -> No reply generated (classified as legitimate).")

#main
def main():
    print("=" * 70)
    print("  Task 5 — AI-Generated Replies to Spam and Scam Emails")
    print("=" * 70)

    #load the pretrained classifier saved by classifier.py
    print(f"\nLoading classifier from {MODEL} ...")
    pipeline = joblib.load(MODEL)
    print("Classifier loaded.")

    #neo menu gia na epilekseis anamesa sto auto mode kai sto interactive mode
    #let the user choose between auto mode and interactive mode
    print("\nSelect mode:")
    print("  1 — Auto mode (random emails from dataset)")
    print("  2 — Interactive mode (paste your own email)")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "2":
        interactive_mode(pipeline)
        return

     #auto mode: randomly sample emails from both dataset files
    spam_emails    = load_spam_emails(SPAM_FILE, n=2)
    nigerian_emails = load_nigerian_emails(NIGERIAN_FILE, n=2)

    #Part A: generate replies to spam emails
    run_category(pipeline, spam_emails, email_type="spam")

    #Part B: generate replies to scam emails
    run_category(pipeline, nigerian_emails, email_type="scam")

    #Part C: multi-turn demo showing continued conversation handling
    print(f"\n{'═' * 70}")
    print("  Part C — Multi-turn demo (1 spam + 1 scam, 2 follow-ups each)")
    print(f"{'═' * 70}")

    for email_type, body in [("spam", spam_emails[0]), ("scam", nigerian_emails[0])]:
        authority = "EDCB" if email_type == "spam" else "OICV"
        print(f"\n--- {email_type.upper()} multi-turn ({authority}) ---")
        history = multi_turn_demo(body, email_type, rounds=2)
        for turn in history:
            who = "SENDER " if turn["role"] == "user" else f"{authority}   "
            preview = turn["content"][:250].replace("\n", " ")
            print(f"\n[{who}] {preview}")
            if len(turn["content"]) > 250:
                print("  ...")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
