from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

phrases = [
    # Group 1 - Pell Grant related
    "Pell Grant eligibility",
    "federal grant for undergraduates",
    "need-based grant for college students",
    
    # Group 2 - Loan related
    "subsidized loan interest",
    "federal loan while enrolled in school",
    "student loan accrual during enrollment",
    
    # Group 3 - Unrelated
    "satisfactory academic progress GPA requirement",
    "cost of attendance calculation",
    
    # Group 4 - EFC vs SAI (temporal confusion test)
    "expected family contribution formula",
    "student aid index calculation",
]

embeddings = model.encode(phrases)

print("Pairwise Cosine Similarities")
print("=" * 60)

for i in range(len(phrases)):
    for j in range(i+1, len(phrases)):
        # Compute cosine similarity
        a = embeddings[i]
        b = embeddings[j]
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        print(f"{similarity:.3f} | {phrases[i][:35]:<35} <-> {phrases[j][:35]}")