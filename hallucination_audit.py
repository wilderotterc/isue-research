from dotenv import load_dotenv
import os
import openai
import pandas as pd
 
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
questions = [
    # S — Single source
    {"id": "S-01", "category": "S",  "question": "What is the maximum Pell Grant award for the 2025-26 award year?"},
    {"id": "S-02", "category": "S",  "question": "What is the deadline to complete the FAFSA for the 2025-26 award year?"},
 
    # M2 — Two source
    {"id": "M2-01", "category": "M2", "question": "If I take out a subsidized loan, do I accrue interest while I'm enrolled in school?"},
    {"id": "M2-02", "category": "M2", "question": "Can I receive both a Pell Grant and a Federal Work-Study award at the same time?"},
 
    # M3 — Three or more sources
    {"id": "M3-01", "category": "M3", "question": "Can I receive a Pell Grant, work-study, and a subsidized loan simultaneously, and is there a total aid cap?"},
    {"id": "M3-02", "category": "M3", "question": "What happens to my financial aid if I drop below half-time enrollment and I have both federal loans and a university scholarship?"},
 
    # C — Contradictory source
    # C-01: Federal SAP pace is 67%. University of Washington requires 80%.
    {"id": "C-01", "category": "C",  "question": "Does the federal government require undergraduate students to complete 80% of their attempted credits each year to keep financial aid?"},
    # C-02: Federal SAP GPA is 2.0. Pace University School of Education requires 2.5.
    {"id": "C-02", "category": "C",  "question": "Is a 2.5 GPA required by the federal government for education students to maintain financial aid eligibility?"},
 
    # T — Temporal change
    {"id": "T-01", "category": "T",  "question": "How is my financial need calculated — what formula does the government use?"},
    {"id": "T-02", "category": "T",  "question": "What information from my tax returns is used to determine my financial aid eligibility?"},
]
 
results = []
 
for q in questions:
    print(f"Running question {q['id']}...")
 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=400,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant answering questions about US student financial aid. Answer as accurately and completely as you can."
            },
            {
                "role": "user",
                "content": q["question"]
            }
        ]
    )
 
    answer = response.choices[0].message.content
 
    results.append({
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "llm_answer": answer,
        "correct_answer": "",
        "source": "",
        "classification": ""
    })
 
    print(f"  Answer preview: {answer[:100]}...")
    print()
 
df = pd.DataFrame(results)
df.to_csv("hallucination_audit.csv", index=False)
print("Saved to hallucination_audit.csv")
print("Fill in correct_answer, source, and classification for each row.")