from bert_score import score

# Example texts
candidates = ["Your generated text goes here."]
references = ["Your reference text goes here."]

# Calculate BERTScore
P, R, F1 = score(candidates, references, lang='en', verbose=True)

# Print scores
print(f"Precision: {P.mean()}")
print(f"Recall: {R.mean()}")
print(f"F1 Score: {F1.mean()}")

