"""Stage 2, step 2.1: explore MarkdownHeaderTextSplitter on ONE article.

Not part of the pipeline: a small experiment to SEE what splitting by section does
before we change chunking.py.

Our cleaned text uses Markdown headings (wikitext.py turns "==Eat==" into "## Eat"):

    # Dhaka/Old Dhaka
    ## Eat
    Restaurants are crammed throughout the narrow alleys...
    #### Dhakaiya Biryani
    Traditional Dhakaiya spicy dish...

MarkdownHeaderTextSplitter cuts the text AT those headings (not at a character
count) and records the headings each piece sat under in its metadata.

Run (from the project root):  python -m stage2.explore_sections
(Needs stage2/data/raw/ first:  python -m stage2.s1_extract_articles)
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter

from stage2.config import CHUNK_SIZE, RAW_DIR
from stage2.documents import WikivoyageLoader

# Which article to look at. Try others too: "Bangkok/Khao San Road", "Thailand", ...
ARTICLE_TITLE = "Thailand"

# How much of each section's text to print.
PREVIEW_CHARS = 120

# Which Markdown heading levels to split on, and the metadata key for each.
#   "#"    -> metadata["h1"]   (the article title, which documents.py puts on top)
#   "##"   -> metadata["h2"]   (main sections: Get in, See, Eat, Sleep, ...)
#   "###"  -> metadata["h3"]   (sub-sections: Budget, Mid-range, ...)
#   "####" -> metadata["h4"]   (sub-sub-sections: Dhakaiya Biryani, ...)
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def find_article(docs, title):
    """Return the Document whose metadata title matches, or None."""
    for doc in docs:
        if doc.metadata["title"] == title:
            return doc
    return None


def section_path(metadata: dict) -> str:
    """Join the heading levels into one readable line.

    {"h1": "Dhaka/Old Dhaka", "h2": "Eat", "h4": "Dhakaiya Biryani"}
    -> "Dhaka/Old Dhaka > Eat > Dhakaiya Biryani"
    """
    parts = []
    for key in ("h1", "h2", "h3", "h4"):
        if key in metadata:              # not every section has every level
            parts.append(metadata[key])
    return " > ".join(parts)


def main():
    # 1. Load the cleaned article (same Documents as Stage 1 used).
    loader = WikivoyageLoader(RAW_DIR)
    docs = loader.load()
    article = find_article(docs, ARTICLE_TITLE)

    if article is None:
        print(f"No article titled {ARTICLE_TITLE!r}. Did you run s1_extract_articles?")
        return

    text = article.page_content
    print(f"Article: {ARTICLE_TITLE}  ({len(text):,} characters)\n")

    # 2. Create the splitter.
    #    By default it REMOVES the heading lines from the text (strip_headers=True),
    #    because the headings now live in the metadata instead.
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)

    # 3. Split. Note: split_text takes ONE string and returns a list of Documents,
    #    one per section. (Unlike RecursiveCharacterTextSplitter, there's no
    #    split_documents here, so the article's own metadata isn't copied over.)
    sections = splitter.split_text(text)

    # 4. Print every section: its heading path, its size, and the start of its text.
    too_long = 0
    for number, section in enumerate(sections, start=1):
        length = len(section.page_content)
        path = section_path(section.metadata)

        marker = ""
        if length > CHUNK_SIZE:
            marker = f"   <-- longer than CHUNK_SIZE ({CHUNK_SIZE}), will need splitting"
            too_long += 1

        preview = section.page_content[:PREVIEW_CHARS].replace("\n", " ")

        print(f"[{number}] {path}  ({length:,} chars){marker}")
        print(f"     metadata: {section.metadata}")
        print(f"     {preview}...")
        print()

    # 5. Summary.
    print(f"{len(sections)} sections, {too_long} of them longer than {CHUNK_SIZE} characters.")


if __name__ == "__main__":
    main()
