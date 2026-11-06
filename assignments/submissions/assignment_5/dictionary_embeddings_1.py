# Dictionary embeddings, part 1
# 1. Load the 100-dimensional GloVe model
# Run the script from the MAN7916 repository root. 
import csv
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from docx import Document
from gensim.models import KeyedVectors

repo = Path.cwd()
output_dir = repo / "assignments" / "submissions" / "assignment_5"
glove_path = repo / "local_data" / "glove.6B.100d.txt"
article_path = repo / "assignments" / "materials" / "week_4" / "article_preprint.txt"

model = KeyedVectors.load_word2vec_format(
    str(glove_path), binary=False, no_header=True
)
# model["creative"] looks up the 100 numbers that represent creative. 
# no_header=True tells Gensim that the raw text file starts with a word and its vector.
output_dir.mkdir(parents=True, exist_ok=True)
report = Document()
report.add_heading("Word Embeddings Results", level=0)

# compare every pair of root words
root_words = ["entrepreneurial", "creative", "innovative", "trailblazing"]
root_vectors = {word: model[word] for word in root_words}

report.add_heading("Pairwise cosine similarities", level=1)

for word_a, word_b in combinations(root_words, 2):
    score = model.similarity(word_a, word_b)
    result = f"{word_a} and {word_b}: {score:.4f}"
    print(result)
    report.add_paragraph(result)
#combinations(root_words, 2) gives six unique pairs. 
# Cosine similarity measures how closely two vectors point in the same direction; 
# Higher values indicate greater similarity. 
# :.4f displays four decimal places.
word_to_drop = input("Which root word will you drop? ").strip().lower()
remaining_roots = [word for word in root_words if word != word_to_drop]

# Drop the root words that fitword_to_drop = input("After reviewing the six scores, which root word will you drop? ")
word_to_drop = word_to_drop.strip().lower()

if word_to_drop not in root_words:
    raise ValueError("Enter one of the four original root words.")

remaining_roots = [word for word in root_words if word != word_to_drop]

report.add_heading("Root word selection", level=1)
report.add_paragraph(f"Dropped: {word_to_drop}")
report.add_paragraph(f"Retained: {', '.join(remaining_roots)}")
# look at all three comparisons involving each word, the word to drop should generally 
# have lower similarities with the other three. 
# A single low pair alone is unclear which member of that pair fits less well
print("Remaining root words:", remaining_roots)

# average the three retained vectors
remaining_vectors = [root_vectors[word] for word in remaining_roots]
average_vector = np.mean(remaining_vectors, axis=0)

report.add_heading("First five dimensions of the average vector", level=1)
for dimension, value in enumerate(average_vector[:5], start=1):
    report.add_paragraph(f"Dimension {dimension}: {value:.6f}")
# remaining_vectors as three rows by 100 columns. 
# axis=0 averages down the three rows, giving one 100-dimensional vector.
# [:5] selects the first five values for reporting. The searches still use all 100.

# Compare the average vector to each root word
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    length_a = np.linalg.norm(vector_a)
    length_b = np.linalg.norm(vector_b)
    return float(dot_product / (length_a * length_b))


report.add_heading("Cosine similarity to the average vector", level=1)

for word in root_words:
    score = cosine_similarity(average_vector, root_vectors[word])
    label = " (dropped)" if word == word_to_drop else ""
    report.add_paragraph(f"{word}{label}: {score:.4f}")

report.save(output_dir / "word_embeddings_results.docx")
# dividing the dot product by the two vector lengths, interpreting each root words
# as all four original words and lablled the dropped word, so we can see the comparsion.

# find the 50 closest words in all of GloVe
deductive_list = model.similar_by_vector(average_vector, topn=50)
# give a ranked list of (word, score) pairs from the full GloVe model's vocabulary

# Find the 50 closest words in the article
article_text = article_path.read_text(encoding="utf-8")
tokens = re.findall(r"[a-z]+", article_text.lower())
article_words = sorted(set(tokens))
article_scores = []
for word in article_words:
    if word in model:
        score = cosine_similarity(average_vector, model[word])
        article_scores.append((word, score))

article_scores.sort(key=lambda pair: pair[1], reverse=True)
inductive_list = article_scores[:50]

# Combine the list and write to csv 
combined_scores = dict(deductive_list)
combined_scores.update(inductive_list)

ranked_words = sorted(
    combined_scores.items(), key=lambda pair: pair[1], reverse=True
)

csv_path = output_dir / "word_list_for_evaluation.csv"

with csv_path.open("x", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["word", "score", "eval"])

    for word, score in ranked_words:
        writer.writerow([word, score, ""])
# The csv file has three columns: word, score, and eval.
# "" leave eval blank

# end it and evalute the words
print(f"Saved {len(ranked_words)} unique candidates to {csv_path}")


