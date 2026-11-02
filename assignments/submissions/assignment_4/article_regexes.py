# Assignment 4: Article regular expressions
from pathlib import Path
from collections import Counter
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Find the input file and output folder.
output_folder = Path(__file__).resolve().parent
assignments_folder = output_folder.parent.parent

input_file = (
    assignments_folder
    / "materials"
    / "week_4"
    / "article_preprint.txt"
)

text = input_file.read_text(encoding="utf-8-sig")

# Exclude the references section if a standard heading exists.
body = re.split(
    r"(?im)^\s*(?:references|bibliography)\s*$",
    text,
    maxsplit=1,
)[0]

# Normalize whitespace using regular expressions.
body = re.sub(r"\s+", " ", body)

# Patterns for common author names and publication years.
surname = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'-]+"

# Allow surname prefixes such as "Von" and "von".
prefix = r"(?:(?:[Vv]an|[Vv]on|[Dd]e|[Dd]el|[Dd]a|[Dd]i)\s+)*"

# Allow suffixes such as "Jr", "Jr.", and "III".
suffix = r"(?:\s+(?:Jr\.?|Sr\.?|III|II|IV))?"

name = rf"{prefix}{surname}{suffix}"

authors = (
    rf"{name}"
    rf"(?:\s+et\s+al\.?"
    rf"|\s*(?:&|and)\s*{name})?"
)


# --------------------------------------------------
# 1. Extract citations with names and dates inside
#    parentheses.
# --------------------------------------------------
year = r"(?:18|19|20)\d{2}[a-z]?"

citation_pattern = re.compile(
    rf"(?P<authors>{authors})\s*,\s*"
    rf"(?P<years>{year}(?:\s*,\s*{year})*)"
)

parenthetical_citations = []

# First find complete parenthetical groups.
groups = re.findall(r"\(([^()]*)\)", body)

# Then extract individual citations from each group.
for group in groups:
    for match in citation_pattern.finditer(group):
        author_text = match.group("authors")

        author_text = re.sub(r"['’]s$", "", author_text)

        author_text = re.sub(
            r"\s+and\s+", " & ", author_text
        )
        author_text = re.sub(
            r"\bet\s+al\.?", "et al.", author_text
        )

        citation_years = re.findall(
            year, match.group("years")
        )

        for citation_year in citation_years:
            parenthetical_citations.append(
                f"{author_text}, {citation_year}"
            )


# --------------------------------------------------
# 2. Extract narrative citations.
#    Keep 50 preceding characters as permitted.
#    Identify common author forms for sorting and
#    avoiding duplicate references.
# --------------------------------------------------

date_pattern = re.compile(
    rf"\((?P<years>{year}(?:\s*,\s*{year})*)\)"
)

# Capture up to 50 characters ending with the date.
# The lookahead allows overlapping context windows.
context_pattern = re.compile(
    rf"(?=(?P<context>.{{0,50}}"
    rf"\((?:{year})(?:\s*,\s*{year})*\)))"
)

contexts_by_end = {}

for match in context_pattern.finditer(body):
    context = match.group("context")
    end_position = match.start() + len(context)

    # Keep the longest available context for each date.
    if end_position not in contexts_by_end:
        contexts_by_end[end_position] = context

author_at_end = re.compile(
    rf"(?P<authors>{authors})\s*$"
)

# Key: author and year. Value: context for the report.
narrative_references = {}
unresolved_contexts = set()

for match in date_pattern.finditer(body):
    context = contexts_by_end.get(match.end(), match.group())

    # Extract the text before the final date using regex.
    preceding = re.sub(
        rf"\({year}(?:\s*,\s*{year})*\)$",
        "",
        context,
    )

    author_match = author_at_end.search(preceding)

    if author_match:
        author_text = author_match.group("authors")
        author_text = re.sub(
            r"\s+and\s+", " & ", author_text
        )
        author_text = re.sub(
            r"\bet\s+al\.?", "et al.", author_text
        )

        for citation_year in re.findall(
            year, match.group("years")
        ):
            key = f"{author_text}, {citation_year}"
            narrative_references.setdefault(key, context)
    else:
        unresolved_contexts.add(context)


# --------------------------------------------------
# 3. Generate the top-20 citation chart.
#    Use parenthetical citations only.
# --------------------------------------------------

counts = Counter(parenthetical_citations)

if counts:
    most_common = counts.most_common(20)

    labels = [citation for citation, count in most_common]
    frequencies = [count for citation, count in most_common]

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(labels, frequencies)
    ax.set_xlabel("Citation")
    ax.set_ylabel("Number of occurrences")
    ax.set_title("Top 20 most frequently cited citations")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.setp(
        ax.get_xticklabels(),
        rotation=60,
        ha="right",
    )

    fig.tight_layout()

    fig.savefig(
        output_folder / "common_citations.jpg",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
else:
    print("No parenthetical citations found.")


# --------------------------------------------------
# 4. Save two alphabetical lists of unique citations.
# --------------------------------------------------

unique_parenthetical = sorted(
    set(parenthetical_citations),
    key=str.casefold,
)

unique_narrative = sorted(
    narrative_references,
    key=str.casefold,
)

report = [
    "CITATIONS WITH NAMES AND DATES INSIDE PARENTHESES",
    "",
    *unique_parenthetical,
    "",
    "CITATIONS WITH ONLY DATES INSIDE PARENTHESES",
    "",
]

for citation in unique_narrative:
    context = narrative_references[citation]
    report.append(f"{citation} | Context: {context}")

report_file = output_folder / "references_report.txt"

report_file.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print(f"Parenthetical occurrences: {len(parenthetical_citations)}")
print(f"Unique parenthetical citations: {len(unique_parenthetical)}")
print(f"Unique narrative citations: {len(unique_narrative)}")
print(f"Results saved in: {output_folder}")

# Display uncertain cases for manual checking.
if unresolved_contexts:
    print("\nCheck these date-only matches manually:")
    for context in sorted(unresolved_contexts):
        print(context)