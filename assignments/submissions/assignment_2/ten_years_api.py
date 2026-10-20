# Part 2: Ten years API

import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.stats import pearsonr, f_oneway


# STEP 1: Settings
API_URL = "https://api.crossref.org/works"
START_DATE = "2016-01-01"
END_DATE = date.today().isoformat()

OUTPUT_FOLDER = Path(__file__).resolve().parent
CSV_FILE = OUTPUT_FOLDER / "ten_years_api.csv"
STATS_FILE = OUTPUT_FOLDER / "publication_stats.txt"

# Both print and online ISSNs are included.
JOURNALS = {
    "Journal of Business Venturing": {"0883-9026", "1873-2003"},
    "Entrepreneurship Theory and Practice": {"1042-2587", "1540-6520"},
    "Strategic Entrepreneurship Journal": {"1932-4391", "1932-443X"},
}

COLUMNS = [
    "doi", "title", "n_refs", "n_authors", "n_cites",
    "url", "journal", "abstract", "pub_year",
]


def clean_text(text):
    """Remove HTML/XML tags and extra whitespace."""
    soup = BeautifulSoup(text or "", "html.parser")
    return " ".join(soup.get_text(" ", strip=True).split())


def publication_year(item):
    """Use Crossref's published date, with issued as a fallback."""
    for field in ("published", "issued"):
        parts = item.get(field, {}).get("date-parts", [])
        if parts and parts[0]:
            return parts[0][0]
    return None


def identify_journal(item):
    """Match a record to one of our three journals."""
    record_issns = set(item.get("ISSN", []))

    for name, issns in JOURNALS.items():
        if record_issns.intersection(issns):
            return name

    raise ValueError(f"Unexpected journal for DOI: {item.get('DOI')}")


def download_articles():
    # Repeated ISSN filters match any of the listed ISSNs.
    # The date and document-type restrictions apply to all of them.
    filters = [
        f"from-pub-date:{START_DATE}",
        f"until-pub-date:{END_DATE}",
        "type:journal-article",
    ]

    for issns in JOURNALS.values():
        for issn in sorted(issns):
            filters.append(f"issn:{issn}")

    params = {
        "filter": ",".join(filters),
        "rows": 1000,
        "cursor": "*",
    }

    # Retry temporary server errors and rate-limit responses.
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({
        "User-Agent": "MAN7916-publication-assignment/1.0"
    })

    records = []
    seen_dois = set()
    page = 1

    try:
        while True:
            response = session.get(
                API_URL, params=params, timeout=90
            )
            response.raise_for_status()
            message = response.json()["message"]
            items = message["items"]

            if page == 1:
                print(
                    "Crossref reports",
                    message["total-results"],
                    "matching records."
                )

            if not items:
                break

            for item in items:
                doi = item.get("DOI", "").lower()

                if not doi:
                    raise ValueError("A record has no DOI.")

                if doi in seen_dois:
                    continue

                seen_dois.add(doi)

                titles = item.get("title", [])
                authors = item.get("author")

                records.append({
                    "doi": doi,
                    "title": clean_text(
                        titles[0] if titles else ""
                    ),
                    # Missing metadata stays blank, rather than
                    # incorrectly being treated as zero.
                    "n_refs": item.get("reference-count"),
                    "n_authors": (
                        len(authors)
                        if authors is not None else None
                    ),
                    "n_cites": item.get(
                        "is-referenced-by-count"
                    ),
                    "url": item.get(
                        "URL", f"https://doi.org/{doi}"
                    ),
                    "journal": identify_journal(item),
                    "abstract": clean_text(
                        item.get("abstract", "")
                    ),
                    "pub_year": publication_year(item),
                })

            print(
                f"Page {page}: "
                f"{len(records)} unique articles collected."
            )

            # A short page marks the end of the results.
            if len(items) < params["rows"]:
                break

            next_cursor = message.get("next-cursor")

            if not next_cursor:
                raise RuntimeError(
                    "Pagination stopped unexpectedly."
                )

            if next_cursor == params["cursor"]:
                raise RuntimeError(
                    "Crossref returned a repeated cursor."
                )

            params["cursor"] = next_cursor
            page += 1
            time.sleep(1)

    finally:
        session.close()

    if not records:
        raise ValueError("No articles were retrieved.")

    df = pd.DataFrame(records, columns=COLUMNS)

    for column in ("n_refs", "n_authors", "n_cites", "pub_year"):
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        ).astype("Int64")

    return df.sort_values(
        ["journal", "pub_year", "doi"]
    ).reset_index(drop=True)


def run_statistics(df):
    # STEP 3: Pearson correlation
    correlation_data = df.dropna(
        subset=["pub_year", "n_cites"]
    )

    if (
        len(correlation_data) < 3
        or correlation_data["pub_year"].nunique() < 2
        or correlation_data["n_cites"].nunique() < 2
    ):
        raise ValueError(
            "Insufficient variation for the correlation."
        )

    r, correlation_p = pearsonr(
        correlation_data["pub_year"].astype(float),
        correlation_data["n_cites"].astype(float),
    )

    # STEP 4: One-way ANOVA
    anova_data = df.dropna(subset=["n_cites"])
    groups = []

    for journal in JOURNALS:
        values = anova_data.loc[
            anova_data["journal"] == journal, "n_cites"
        ].astype(float).to_numpy()

        if len(values) < 2:
            raise ValueError(
                f"Too few articles for ANOVA: {journal}"
            )

        groups.append(values)

    f_value, anova_p = f_oneway(*groups)
    df_between = len(groups) - 1
    df_within = sum(len(group) for group in groups) - len(groups)

    descriptives = (
        anova_data.groupby("journal")["n_cites"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "n", "std": "SD"})
    )

    def format_p(value):
        return "p < .001" if value < .001 else f"p = {value:.3f}"

    report = [
        "PUBLICATION STATISTICS",
        f"Retrieved: {END_DATE}",
        f"Publication window: {START_DATE} to {END_DATE}",
        f"Unique articles: {len(df)}",
        "",
        "CITATION COUNT BY JOURNAL",
        descriptives.to_string(float_format=lambda x: f"{x:.2f}"),
        "",
        "PEARSON CORRELATION: publication year and citations",
        (
            f"r({len(correlation_data) - 2}) = {r:.3f}, "
            f"{format_p(correlation_p)}, "
            f"N = {len(correlation_data)}."
        ),
        "",
        "ONE-WAY ANOVA: citation count by journal",
        (
            f"F({df_between}, {df_within}) = {f_value:.3f}, "
            f"{format_p(anova_p)}."
        ),
        "",
        "NOTES",
        "Citation counts are Crossref counts at retrieval.",
        "Missing values were excluded separately for each analysis.",
        "Missing abstracts and other unavailable fields are blank.",
        "Journal-article records may include editorials or corrections.",
        "Older articles have had more time to accumulate citations.",
        "ANOVA compares raw means without adjusting for publication year.",
        "Citation counts can be skewed and have unequal variances;",
        "the classical ANOVA should therefore be interpreted cautiously.",
    ]

    STATS_FILE.write_text(
        "\n".join(report), encoding="utf-8"
    )
    print("\n" + "\n".join(report))


def main():
    # STEP 2: Download and save the metadata
    df = download_articles()
    df.to_csv(
        CSV_FILE, index=False, encoding="utf-8-sig"
    )
    print(f"\nCSV saved: {CSV_FILE}")

    run_statistics(df)
    print(f"\nStatistics saved: {STATS_FILE}")


if __name__ == "__main__":
    main()