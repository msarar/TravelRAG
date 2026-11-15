"""Stage 2 helper: split Documents into section-aware chunks with a context line.

Stage 1 cut articles purely by character count, which caused two problems:
  - heading-only chunks: a heading got separated from the text under it
  - orphan chunks: most chunks never said which place or section they were about

Stage 2 splits in two steps, then labels every chunk:

    article ──MarkdownHeaderTextSplitter──► sections (headings go into metadata)
            ──RecursiveCharacterTextSplitter──► pieces ≤ CHUNK_SIZE (long sections only)
            ──add context line──► final chunks

Every final chunk starts with a line saying exactly where it comes from, e.g.

    Ao Nang (Krabi Province) > Eat > Budget
    - Somtam Stall — Soi 8; Price: 60 baht. Great papaya salad...

That line is part of page_content, so the EMBEDDING sees it (metadata alone is
never embedded). It also counts towards CHUNK_SIZE, so we leave room for it.

Used by s3_split_documents.py (to inspect the chunks) and s4_build_index.py
(to embed and store them), so both always split in exactly the same way.
"""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from stage2.config import CHUNK_OVERLAP, CHUNK_SIZE

# Which Markdown heading levels mark a section, and the metadata key for each.
#   "#"    -> "h1"  the article title (documents.py puts "# <title>" on top)
#   "##"   -> "h2"  main sections: Get in, See, Eat, Sleep, ...
#   "###"  -> "h3"  sub-sections: Budget, Mid-range, ...
#   "####" -> "h4"  sub-sub-sections: Dhakaiya Biryani, ...
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

# Never let the room for text drop below this, even with a very long context line.
MIN_TEXT_ROOM = 200


def build_place_label(article_metadata: dict) -> str:
    """Where the article is, as a readable label.

    "Dhaka/Old Dhaka"                       -> "Dhaka > Old Dhaka"
    "Ao Nang"  (is_part_of "Krabi Province") -> "Ao Nang (Krabi Province)"
    "Bangkok/Silom" (is_part_of "Bangkok")  -> "Bangkok > Silom"   (Bangkok already in the title)
    """
    title = article_metadata["title"]
    label = title.replace("/", " > ")

    # Add the bigger area the place belongs to, so questions about "Krabi" can
    # find "Ao Nang" chunks. Skip it if the title already says it.
    is_part_of = article_metadata.get("is_part_of")
    if is_part_of and is_part_of not in title:
        label = f"{label} ({is_part_of})"

    return label


def build_section_fields(section_metadata: dict) -> dict:
    """Turn MarkdownHeaderTextSplitter's h2/h3/h4 keys into fixed metadata keys.

    The splitter only adds the keys a section actually has, e.g. a section with
    no sub-section has no "h3". Chroma filtering works best when every chunk has
    the SAME keys (see Stage 1 findings), so missing levels become "".

    {"h1": "Thailand", "h2": "Get in", "h3": "Entry requirements"}
    -> {"section": "Get in", "subsection": "Entry requirements",
        "subsubsection": "", "section_path": "Get in > Entry requirements"}
    """
    section = section_metadata.get("h2", "")
    subsection = section_metadata.get("h3", "")
    subsubsection = section_metadata.get("h4", "")

    # Join only the levels that exist: "Get in > Entry requirements"
    path_parts = []
    for heading in (section, subsection, subsubsection):
        if heading:
            path_parts.append(heading)
    section_path = " > ".join(path_parts)

    return {
        "section": section,
        "subsection": subsection,
        "subsubsection": subsubsection,
        "section_path": section_path,
    }


def build_context_line(place_label: str, section_path: str) -> str:
    """"Ao Nang (Krabi Province)" + "Eat > Budget" -> "Ao Nang (Krabi Province) > Eat > Budget"

    The article's intro (text before the first "##") has no section, so it's
    just the place label.
    """
    if section_path:
        return f"{place_label} > {section_path}"
    return place_label


def split_one_article(article: Document) -> list[Document]:
    """Split ONE article into final chunks."""
    place_label = build_place_label(article.metadata)

    # Step 1: cut at headings. split_text takes one string and returns one
    # Document per section, with the headings in its metadata.
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    sections = header_splitter.split_text(article.page_content)

    chunks = []
    for section in sections:
        section_fields = build_section_fields(section.metadata)
        context_line = build_context_line(place_label, section_fields["section_path"])

        # Step 2: cut long sections into pieces. Leave room for the context line
        # (+1 for the newline after it), so the FINAL chunk stays ≤ CHUNK_SIZE.
        room_for_text = CHUNK_SIZE - len(context_line) - 1
        room_for_text = max(room_for_text, MIN_TEXT_ROOM)

        piece_splitter = RecursiveCharacterTextSplitter(
            chunk_size=room_for_text,
            chunk_overlap=CHUNK_OVERLAP,
        )
        # Short sections come back as a single piece, unchanged.
        pieces = piece_splitter.split_text(section.page_content)

        for piece in pieces:
            # Step 3: label the piece. The context line goes INTO the text, so
            # the embedding knows where this chunk is from.
            page_content = context_line + "\n" + piece

            # Metadata = the article's (title, place, source URL...)
            #          + the section's (section, subsection, section_path).
            # MarkdownHeaderTextSplitter.split_text doesn't copy the article's
            # metadata for us, so we do it here.
            metadata = dict(article.metadata)       # a copy, not the same dict
            metadata.update(section_fields)

            chunk = Document(page_content=page_content, metadata=metadata)
            chunks.append(chunk)

    return chunks


def split_into_chunks(docs: list[Document]) -> list[Document]:
    """Split every article into section-aware chunks (same name as Stage 1's function)."""
    all_chunks = []
    for article in docs:
        article_chunks = split_one_article(article)
        all_chunks.extend(article_chunks)        # add this article's chunks to the list
    return all_chunks
