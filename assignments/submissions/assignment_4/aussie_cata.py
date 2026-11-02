# Assignment 4: Australian computer-assisted text analysis
from pathlib import Path
from collections import Counter
import re

import pandas as pd
import matplotlib.pyplot as plt
import spacy


# --------------------------------------------------
# 1. Locate folders and load the text files.
# --------------------------------------------------

output_folder = Path(__file__).resolve().parent
repository_folder = output_folder.parents[2]
data_folder = repository_folder / "local_data" / "aussie" / "About"

files = sorted(data_folder.glob("*.txt"))

if not files:
    raise FileNotFoundError(
        f"No .txt files found in: {data_folder}"
    )

records = []

for file in files:
    records.append({
        "name": file.stem,
        "filepath": file.relative_to(repository_folder).as_posix(),
        "text": file.read_text(encoding="utf-8-sig"),
    })

df = pd.DataFrame(records)


# --------------------------------------------------
# 2. Preprocess with spaCy.
# ws = with stopwords; wos = without stopwords.
# Lowercase and retain alphabetic tokens.
# Keep original word forms rather than lemmatizing.
# --------------------------------------------------

nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"],
)

with_stops = []
without_stops = []

for doc in nlp.pipe(df["text"]):
    with_stops.append([
        token.text.lower()
        for token in doc
        if token.is_alpha
    ])

    without_stops.append([
        token.text.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ])

df["preprocessed_ws"] = with_stops
df["preprocessed_wos"] = without_stops


# --------------------------------------------------
# 3. Count words across all companies.
# --------------------------------------------------

counter_ws = Counter(
    word
    for words in df["preprocessed_ws"]
    for word in words
)

counter_wos = Counter(
    word
    for words in df["preprocessed_wos"]
    for word in words
)


def save_line_chart(counter, top_n, title, filename):
    """Plot terms in descending order of frequency."""
    common = counter.most_common(top_n)

    fig, ax = plt.subplots(figsize=(18, 7))

    if common:
        words = [word for word, count in common]
        frequencies = [count for word, count in common]

        positions = range(len(words))

        ax.plot(positions, frequencies, marker="o")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(words, rotation=90)
    else:
        ax.text(
            0.5, 0.5,
            "No matching terms",
            transform=ax.transAxes,
            ha="center",
        )

    ax.set_title(title)
    ax.set_xlabel("Word")
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(
        output_folder / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


save_line_chart(
    counter_ws,
    100,
    "100 most common words: with stopwords",
    "common100_ws.jpg",
)

save_line_chart(
    counter_wos,
    100,
    "100 most common words: without stopwords",
    "common100_wos.jpg",
)


# --------------------------------------------------
# 4. Paste the dictionary's quoted word entries below.
# Do not paste its memo or metadata sections.
# --------------------------------------------------

dictionary_text = r"""
"ad lib"
"adroit"
"adroitness"
"bright idea"
"clever"
"cleverness"
"conceive"
"concoct"
"concoction"
"concoctive"
"conjure up"
"creative"
"creativity"
"develop"
"developed"
"dream"
"dream up"
"expert"
"formulation"
"freethinker"
"genesis"
"genius"
"gifted"
"hit upon"
"imagination"
"imaginative"
"improvise"
"ingenious"
"ingenuity"
"innovate"
"innovated"
"innovates"
"innovating"
"innovation"
"innovations"
"innovative"
"innovativeness"
"introduced"
"introducing"
"introduction"
"introductions"
"invent"
"invented"
"invention"
"inventive"
"inventiveness"
"inventor"
"launch"
"launched"
"launching"
"master stroke"
"mastermind"
"metamorphose"
"metamorphosis"
"neoteric"
"neoterism"
"neoterize"
"new capabilities"
"new capability"
"new compounds"
"new content"
"new core areas"
"new course"
"new directions"
"new family"
"new features"
"new generation"
"new generations"
"new idea"
"new ideas"
"new line of business"
"new medicine"
"new medicines"
"new molecular entities"
"new pharmaceuticals"
"new platform"
"new process"
"new processes"
"new product"
"new products"
"new solutions"
"new systems"
"new technique"
"new techniques"
"new technologies"
"new technology"
"new therapies"
"new thinking"
"new tools"
"new treatments"
"new ways"
"new wrinkle"
"new-generation"
"new-product"
"next generation"
"next-generation"
"novation"
"novel"
"novelty"
"patent"
"patented"
"patents"
"process development"
"product development"
"product launch"
"product launches"
"proprietary"
"prototype"
"prototyping"
"push the envelope"
"R&D"
"radical"
"re-engineering"
"reformulated"
"refreshed"
"reinvent"
"re-invent"
"reinvented"
"reinventing"
"reinvention"
"reinvents"
"released"
"renewal"
"renewing"
"research"
"reshape"
"reshaped"
"reshapes"
"reshaping"
"resourceful"
"resourcefulness"
"restyle"
"restyling"
"revolutionary"
"revolutionize"
"revolutionized"
"roll out"
"rolled out"
"see things"
"technologically advanced"
"think up"
"trademark"
"transform"
"transformation"
"transformed"
"transforming"
"visualize"
"""
innov_terms = re.findall(
    r'"([^"]+)"',
    dictionary_text,
)

innov_terms = {
    term.lower()
    for term in innov_terms
}

if not innov_terms:
    raise ValueError(
        "Paste the quoted Innovativeness dictionary entries "
        "inside dictionary_text, then rerun."
    )


# --------------------------------------------------
# 5. Filter the with-stopwords Counter.
# Exact matching of individual word tokens.
# --------------------------------------------------

innov_counter = Counter({
    word: frequency
    for word, frequency in counter_ws.items()
    if word in innov_terms
})

save_line_chart(
    innov_counter,
    20,
    "Most common Innovativeness dictionary words",
    "innov_freqs.jpg",
)


# --------------------------------------------------
# 6. Calculate company-level dictionary scores.
# --------------------------------------------------

def count_innov_words(words):
    return sum(
        word in innov_terms
        for word in words
    )


df["innov_ws"] = df["preprocessed_ws"].apply(
    count_innov_words
)

df["innov_perwd_ws"] = [
    count / len(words) if words else 0.0
    for count, words in zip(
        df["innov_ws"],
        df["preprocessed_ws"],
    )
]


# --------------------------------------------------
# 7. Save the dataset.
# Export preprocessed token lists as readable text.
# --------------------------------------------------

export_df = df.copy()

for column in ["preprocessed_ws", "preprocessed_wos"]:
    export_df[column] = export_df[column].apply(
        lambda words: " ".join(words)
    )

export_df.to_csv(
    output_folder / "innov_aussie_data.csv",
    index=False,
    encoding="utf-8",
)

print(f"Companies loaded: {len(df)}")
print(f"Dictionary entries loaded: {len(innov_terms)}")
print(f"Innovativeness word occurrences: {sum(innov_counter.values())}")
print(f"Results saved in: {output_folder}")

print(
    df[["name", "innov_ws", "innov_perwd_ws"]].head(10)
)