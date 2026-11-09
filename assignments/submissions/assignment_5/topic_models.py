import csv

# make sure python has the training text
from pathlib import Path
import re

import pymupdf
import tomotopy as tp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

data_folder = Path("local_data") / "topic_model"
output_folder = Path(__file__).resolve().parent

def read_words(pdf_path):
    with pymupdf.open(pdf_path) as article:
        text = "\n".join(page.get_text() for page in article)

    # Stop at the references heading, including headings with spaced letters.
    text = re.split(r"(?im)^[ \t]*r *e *f *e *r *e *n *c *e *s[ \t]*$", text, maxsplit=1)[0]
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [word for word in words if word not in ENGLISH_STOP_WORDS]

pdf_files = sorted((data_folder / "train").glob("*.pdf"))
if len(pdf_files) < 10:
    raise ValueError(f"Expected at least 10 training PDFs. Found {len(pdf_files)}.")

training_docs = [read_words(pdf) for pdf in pdf_files]

# choose a topic count then tune the model, and save the word cloud
# First, vary the number of topics and keep alpha and eta fixed. Then fix the
# chosen topic count and vary alpha and eta. Think of alpha as controlling how 
# readily a paper mixes several topics; 
# and eta as controlling how broadly a topic spreads its weight across words.
def fit_lda(k, alpha, eta):
    model = tp.LDAModel(k=k, alpha=alpha, eta=eta, min_df=2, seed=24601)
    model.optim_interval = 0  # Keep the alpha and eta values I chose.

    for words in training_docs:
        model.add_doc(words)

    model.burn_in = 100
    model.train(500, workers=1)
    score = tp.coherence.Coherence(model, coherence="c_v", top_n=10).get_score()
    return model, score

scores = []

for topic_count in range(2, 11):
    model, score = fit_lda(topic_count, alpha=0.10, eta=0.01)
    scores.append((topic_count, score))
    print(f"{topic_count} topics: coherence = {score:.4f}")

csv_path = output_folder / "coherence_scores.csv"
with csv_path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.writer(output)
    writer.writerow(["topics", "coherence"])
    writer.writerows(scores)

print(f"Saved coherence scores to {csv_path}")

# tune alpha and eta in script, after see when topic number is 7, the coherence 
# score is highest
chosen_k = 8
best_score = float("-inf")

for alpha in (0.05, 0.10, 0.50):
    for eta in (0.005, 0.01, 0.05):
        model, score = fit_lda(chosen_k, alpha, eta)
        print(f"alpha={alpha:g}, eta={eta:g}, coherence={score:.4f}")

        if score > best_score:
            best_model = model
            best_score = score
            best_alpha = alpha
            best_eta = eta

print(f"\nBest model: {chosen_k} topics, alpha={best_alpha:g}, "
      f"eta={best_eta:g}, coherence={best_score:.4f}")

for topic_id in range(chosen_k):
    words = [word for word, weight in best_model.get_topic_words(topic_id, top_n=10)]
    print(f"Topic {topic_id + 1}: {', '.join(words)}")

# test the model on the eleventh paper
heldout_words = read_words(data_folder / "holdout.pdf")
vocabulary = set(best_model.used_vocabs)
known_words = [word for word in heldout_words if word in vocabulary]

if not known_words:
    raise ValueError("No words in the held-out article match the model vocabulary.")

heldout_doc = best_model.make_doc(known_words)
topic_shares, _ = best_model.infer(heldout_doc, iterations=200, workers=1)

ranked_topics = sorted(enumerate(topic_shares), key=lambda item: item[1], reverse=True)
for topic_id, share in ranked_topics:
    words = ", ".join(word for word, weight in best_model.get_topic_words(topic_id, top_n=6))
    print(f"Topic {topic_id + 1}: {share:.1%} | {words}")

# create the word cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")

for topic_id, ax in enumerate(axes.flat):
    word_weights = dict(best_model.get_topic_words(topic_id, top_n=20))
    cloud = WordCloud(
        width=700, height=400, background_color="white", random_state=42
    ).generate_from_frequencies(word_weights)

    ax.imshow(cloud, interpolation="bilinear")
    ax.set_title(f"Topic {topic_id + 1}")
    ax.axis("off")

cloud_path = Path(__file__).resolve().with_name("topic_model_wordcloud.jpg")
fig.savefig(cloud_path, dpi=300)
plt.close(fig)
print(f"Saved word clouds to {cloud_path}")