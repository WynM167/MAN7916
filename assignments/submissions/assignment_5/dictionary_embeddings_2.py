from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE

script_folder = Path(__file__).resolve().parent
csv_path = script_folder / "word_list_for_evaluation.csv"
glove_path = Path("local_data") / "glove.6B.100d.txt"
plot_path = script_folder / "dictionary_embedding_plot.jpg"

# 1. Read the manual evaluations and keep only words marked 1.
evaluated_words = pd.read_csv(csv_path)
ratings = pd.to_numeric(evaluated_words["eval"], errors="coerce")

if not ratings.isin([0, 1]).all():
    raise ValueError("Enter 0 or 1 in every eval cell and save the CSV.")

words = evaluated_words.loc[ratings == 1, "word"].drop_duplicates().tolist()
if len(words) < 3:
    raise ValueError("This script needs at least three included words.")

# 2. Get one 100-dimensional GloVe vector for each included word.
print(f"Loading GloVe vectors for {len(words)} included words...")
glove = KeyedVectors.load_word2vec_format(
    str(glove_path), binary=False, no_header=True
)
word_vectors = glove[words]

# 3. Give each word two coordinates based on its vector.
perplexity = min(15, max(2, len(words) // 3))
coordinates = TSNE(
    n_components=2,
    perplexity=perplexity,
    metric="cosine",
    random_state=42,
).fit_transform(word_vectors)

# 4. Draw one dot and one label per word, then save the image.
fig, ax = plt.subplots(figsize=(12, 9), layout="constrained")
ax.scatter(coordinates[:, 0], coordinates[:, 1], s=22, color="steelblue")
ax.set_title("t-SNE Map of the Creativity Dictionary")
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
ax.margins(0.12)

labels = [
    ax.text(x, y, word, fontsize=9)
    for word, (x, y) in zip(words, coordinates)
]
adjust_text(labels, ax=ax, arrowprops={"arrowstyle": "-", "color": "gray"})

fig.savefig(plot_path, dpi=300)
plt.close(fig)
print(f"Saved {len(words)} labeled words to {plot_path}")
