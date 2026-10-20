# Part 1: Dean scraping
import csv
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BLOG_URL = "https://pauljarley.wordpress.com/"

# Save the CSV beside this Python script.
OUTPUT_FILE = Path(__file__).resolve().parent / "dean_scraping.csv"


def get_soup(url):
    """Download a webpage and parse its HTML."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")


def main():
    # STEP 1: Collect titles, dates, and URLs from the first page.
    homepage = get_soup(BLOG_URL)
    posts = []

    for article in homepage.select("article.post"):
        title_link = article.select_one(".entry-title a")
        date = article.select_one("time.entry-date")

        if title_link is None or date is None:
            raise ValueError("A post is missing its title or date.")

        posts.append({
            "title": title_link.get_text(" ", strip=True),
            "url": urljoin(BLOG_URL, title_link["href"]),
            "entry_date": date["datetime"],
        })

    if not posts:
        raise ValueError("No posts found. Check the webpage structure.")

    # STEP 2: Visit each post and collect its text and comments.
    for post in posts:
        print(f"Scraping: {post['title']}")
        time.sleep(1)

        soup = get_soup(post["url"])
        content = soup.select_one("article.post .entry-content")

        if content is None:
            raise ValueError(f"Post text not found: {post['url']}")

        # Remove sharing buttons and other non-post content.
        for extra in content.select(
            ".sharedaddy, .jp-relatedposts, script, style"
        ):
            extra.decompose()

        post["post_text"] = content.get_text("\n", strip=True)

        comments = []

        # Selecting each comment body also includes nested replies
        # without combining a parent's text with its replies.
        for comment in soup.select("#comments .comment-body"):
            author = comment.select_one(".comment-author .fn")
            text = comment.select_one(".comment-content")

            if text is None:
                continue

            author_name = (
                author.get_text(" ", strip=True)
                if author is not None
                else "Unknown"
            )

            comment_text = text.get_text(" ", strip=True)
            comments.append(f"{author_name}: {comment_text}")

        post["comments"] = "\n".join(comments)

    # STEP 3: Write one row per post to the CSV.
    columns = [
        "title",
        "url",
        "entry_date",
        "post_text",
        "comments",
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(posts)

    print(f"\nSaved {len(posts)} posts to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()