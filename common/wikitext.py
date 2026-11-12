"""Turn Wikivoyage wikitext into clean, readable text.

WHAT IS WIKITEXT?
Wikitext is the markup language Wikipedia and Wikivoyage are written in.
These are the pieces we need to deal with:

  {{name|param|key=value}}   "templates": listings, banners, maps, unit converters...
  [[Target|label]]           internal links        -> keep only the label
  [https://url label]        external links        -> keep only the label
  '''bold''' / ''italic''    emphasis              -> drop the quote marks
  ==Heading== / ===Sub===    section headings      -> "## Heading" / "### Sub"
  <ref>..</ref>, <!-- -->    footnotes, comments   -> drop
  {| ... |}                  tables                -> drop

WHY NOT JUST USE REGEX FOR EVERYTHING?
Templates can be nested, e.g.  {{eat | price={{THB|100}} }}.
A regex can't reliably find the matching "}}" for each "{{" when they nest,
so for templates and links we walk through the text and count brackets ourselves.
"""

import html
import re

# ---------------------------------------------------------------------------
# Configuration: which templates mean what
# ---------------------------------------------------------------------------

# Listing templates: each one describes ONE place (a restaurant, hotel, sight...).
# We turn these into a readable sentence.
LISTING_TYPES = {"see", "do", "buy", "eat", "drink", "sleep", "go", "listing"}

# Templates that carry no useful text for answering questions
# (maps, page banners, navigation boxes...). We delete these entirely.
DROP_TEMPLATES = {
    "pagebanner", "mapframe", "mapshape", "mapshapes", "mapmask", "geo", "ispartof",
    "routebox", "dead link", "seedistricts", "infobox", "quickbar", "climate chart",
    "regionlist", "printdistricts", "cnydates", "exchange rates", "hatnote",
    "relatedwikipedia", "relatedcommonscat", "related", "commons", "wikipedia",
    "footer", "disamb", "disambiguation", "dmoz", "toc", "-", "clear",
}

# Any template whose name STARTS with one of these is also dropped.
# e.g. "usablecity", "guidedistrict", "outlinecity", "starcity" are article-status
# banners, and names starting with "#" are wiki functions like {{#expr:...}}.
DROP_PREFIXES = ("usable", "guide", "outline", "star", "#")

# The listing fields we keep, in the order they should appear in the sentence.
# (field name, label to print before it). None means "print the value with no label".
LISTING_FIELDS = [
    ("address", None),
    ("directions", None),
    ("phone", "Phone"),
    ("hours", "Hours"),
    ("price", "Price"),
]


# ---------------------------------------------------------------------------
# Step 1 helpers: matching nested brackets and splitting parameters
# ---------------------------------------------------------------------------

def _find_closing(text: str, start: int, open_tok: str, close_tok: str) -> int:
    """Find where a bracket that opens at `start` closes.

    Example: text = "a {{b {{c}} d}} e", start = 2, open_tok = "{{", close_tok = "}}"
    We count: "{{" -> depth 1, "{{" -> depth 2, "}}" -> depth 1, "}}" -> depth 0 (done!)
    Returns the index just AFTER the matching "}}", or -1 if it never closes.
    """
    depth = 0
    i = start

    while i < len(text):
        if text.startswith(open_tok, i):
            # Found an opening bracket: we are one level deeper.
            depth += 1
            i += len(open_tok)

        elif text.startswith(close_tok, i):
            # Found a closing bracket: we are one level shallower.
            depth -= 1
            i += len(close_tok)

            if depth == 0:
                # Back to the level we started at: this is the matching close.
                return i

        else:
            # Any other character: just move on.
            i += 1

    # We ran out of text without closing: the wikitext is broken here.
    return -1


def _split_params(inner: str) -> list[str]:
    """Split the inside of a template on "|", but only at the TOP level.

    Example:  'eat|name=Somtam Shop|price={{THB|100}}'
    A naive inner.split("|") would also split inside {{THB|100}}.
    We only split when we are not inside nested {{ }} or [[ ]], giving:
              ['eat', 'name=Somtam Shop', 'price={{THB|100}}']
    """
    parts = []          # finished pieces
    current = []        # characters of the piece we're building now
    depth = 0           # how deep inside nested brackets we are
    i = 0

    while i < len(inner):
        two_chars = inner[i:i + 2]

        if two_chars == "{{" or two_chars == "[[":
            depth += 1
            current.append(two_chars)
            i += 2

        elif two_chars == "}}" or two_chars == "]]":
            depth -= 1
            current.append(two_chars)
            i += 2

        elif inner[i] == "|" and depth == 0:
            # A top-level separator: finish the current piece and start a new one.
            parts.append("".join(current))
            current = []
            i += 1

        else:
            current.append(inner[i])
            i += 1

    # Don't forget the last piece (there's no "|" after it).
    parts.append("".join(current))
    return parts


def _parse_template(inner: str) -> tuple[str, list[str], dict[str, str]]:
    """Break a template into its name, positional params and named params.

    'eat|name=X|price=Y'  -> ('eat', [],          {'name': 'X', 'price': 'Y'})
    'c|32'                -> ('c',   ['32'],      {})
    """
    parts = _split_params(inner)

    # The first piece is always the template's name.
    name = parts[0].strip().lower()

    positional = []     # values without a "key=" in front
    named = {}          # values written as key=value

    for part in parts[1:]:
        key, equals_sign, value = part.partition("=")

        # It's a named parameter only if there was an "=" AND the key looks like
        # a simple name (letters, digits, spaces, dashes). This stops text like
        # "1+1=2" from being treated as key "1+1".
        key_looks_valid = re.fullmatch(r"\s*[\w -]+\s*", key) is not None

        if equals_sign and key_looks_valid:
            named[key.strip().lower()] = value.strip()
        else:
            positional.append(part.strip())

    return name, positional, named


# ---------------------------------------------------------------------------
# Step 2: turning each kind of template into text
# ---------------------------------------------------------------------------

def _render_listing(named: dict[str, str]) -> str:
    """Turn a listing template into one readable line.

    {{eat | name=Hotel Star | address=Taher Bagh Lane | content=Fantastic biryani.}}
    ->  "Hotel Star — Taher Bagh Lane. Fantastic biryani."
    """
    # Listing values can contain their own markup (links, templates), so clean each one.
    values = {}
    for key, raw_value in named.items():
        values[key] = clean_inline(raw_value)

    name = values.get("name", "").strip()
    if not name:
        # A listing without a name isn't useful.
        return ""

    # Start with the name, plus the alternative name in brackets if there is one.
    text = name
    if values.get("alt"):
        text += f" ({values['alt']})"

    # Collect the details we care about, e.g. ["Taher Bagh Lane", "Price: Tk 50-150"].
    details = []
    for field, label in LISTING_FIELDS:
        value = values.get(field)
        if not value:
            continue
        if label is None:
            details.append(value)
        else:
            details.append(f"{label}: {value}")

    if details:
        text += " — " + "; ".join(details)
    text += "."

    # The free-text description goes at the end.
    if values.get("content"):
        text += " " + values["content"]

    # Squash all whitespace (including newlines) into single spaces,
    # because each listing must stay on ONE line of the output.
    words = text.split()
    return " ".join(words)


def _render_template(inner: str) -> str:
    """Decide what text (if any) a single template should become."""
    name, positional, named = _parse_template(inner)

    # Listings (restaurants, hotels, sights...) become sentences.
    if name in LISTING_TYPES:
        return _render_listing(named)

    # Maps, banners, navigation boxes... disappear.
    if name in DROP_TEMPLATES:
        return ""
    if name.startswith(DROP_PREFIXES):
        return ""

    # Many small templates have a single value; grab it once for convenience.
    first = ""
    if positional:
        first = positional[0]

    if name == "marker":
        # {{marker|name=Phuket Town|type=city}} -> "Phuket Town"
        return named.get("name", "")

    if name in ("cautionbox", "warningbox"):
        # Safety warnings are important: keep them, clearly labelled.
        return f"Caution: {first}"

    if name in ("sleeppricerange", "eatpricerange"):
        # {{sleeppricerange|under 500 baht|500-1500 baht|over 1500 baht}}
        labels = ["Budget", "Mid-range", "Splurge"]
        ranges = []
        for label, price in zip(labels, positional):
            ranges.append(f"{label}: {price}")
        return "Price ranges — " + "; ".join(ranges) + "."

    if name == "main":
        # {{main|Suvarnabhumi Airport}} -> "(See also: Suvarnabhumi Airport.)"
        return f"(See also: {first}.)"

    if name == "c":
        # {{c|32}} -> "32°C"
        return f"{first}°C"

    if name in ("km", "acre", "mi", "metre", "feet"):
        # {{km|5}} -> "5 km",  {{acre|205}} -> "205 acres"
        unit_names = {"acre": "acres", "metre": "m", "feet": "ft"}
        unit = unit_names.get(name, name)
        return f"{first} {unit}"

    if name == "convert" and len(positional) >= 2:
        # {{convert|109|km|0|abbr=on}} -> "109 km"
        return f"{positional[0]} {positional[1]}"

    if name in ("thb", "bkk"):
        # {{THB|500}} -> "500 baht"
        return f"{first} baht"

    if name == "rint":
        # {{rint|bangkok|Sukhumvit}} -> "Sukhumvit" (a transit line icon + name)
        if positional:
            return positional[-1]
        return ""

    if name == "flag":
        # {{flag|Albania}} just draws a flag icon.
        return ""

    # Anything else, e.g. {{phone|+66 2 123}}, {{IATA|BKK}}:
    # keep the first value, which is usually the meaningful part.
    return first


def _replace_templates(text: str) -> str:
    """Find every top-level {{...}} in the text and replace it with its rendered text."""
    output = []     # pieces of the result, joined at the end
    i = 0           # where we are in the text

    while True:
        start = text.find("{{", i)

        if start == -1:
            # No more templates: keep the rest of the text and stop.
            output.append(text[i:])
            break

        end = _find_closing(text, start, "{{", "}}")

        if end == -1:
            # A "{{" that never closes (broken wikitext). Keep the text before it,
            # skip the two stray braces, and carry on.
            output.append(text[i:start])
            i = start + 2
            continue

        # Keep the normal text before the template...
        output.append(text[i:start])

        # ...then replace the template with its rendered version.
        inside = text[start + 2:end - 2]        # strip the outer "{{" and "}}"
        output.append(_render_template(inside))

        i = end

    return "".join(output)


# ---------------------------------------------------------------------------
# Step 3: internal links [[...]]
# ---------------------------------------------------------------------------

def _replace_internal_links(text: str) -> str:
    """[[Bangkok|the capital]] -> "the capital",  [[Bangkok]] -> "Bangkok",
    [[File:photo.jpg|...]] -> removed (it's an image, not text)."""
    output = []
    i = 0

    while True:
        start = text.find("[[", i)

        if start == -1:
            output.append(text[i:])
            break

        end = _find_closing(text, start, "[[", "]]")

        if end == -1:
            output.append(text[i:start])
            i = start + 2
            continue

        output.append(text[i:start])

        inside = text[start + 2:end - 2]
        target = inside.split("|", 1)[0]          # the page being linked to
        target = target.strip().lower()

        is_image_or_category = target.startswith(("file:", "image:", "category:"))

        if not is_image_or_category:
            # Use the text after the last "|" if there is one, else the target itself.
            if "|" in inside:
                label = inside.split("|")[-1]
            else:
                label = inside
            # Labels can contain their own links, so clean them too (recursion).
            output.append(_replace_internal_links(label))

        i = end

    return "".join(output)


# ---------------------------------------------------------------------------
# Step 4: putting the cleaning steps together
# ---------------------------------------------------------------------------

def _clean(text: str, keep_newlines: bool) -> str:
    """Run every cleaning step, in order. Shared by clean_inline and wikitext_to_text."""
    # 1. Remove HTML comments:  <!-- note to editors -->
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)

    # 2. Remove footnotes:  <ref name="x"/>  and  <ref>source...</ref>
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)

    # 3. Templates {{...}} -> rendered text (listings, units, etc.)
    text = _replace_templates(text)

    # 4. Internal links [[...]] -> their label
    text = _replace_internal_links(text)

    # 5. External links:  [https://example.com Example] -> "Example"
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    #    ...and bare ones with no label:  [https://example.com] -> ""
    text = re.sub(r"\[https?://[^\]]+\]", "", text)

    # 6. Bold / italic markers:  '''bold''' -> bold,  ''italic'' -> italic
    text = re.sub(r"'{2,}", "", text)

    # 7. Any remaining HTML tags:  <br />, <span ...>, </small> ...
    text = re.sub(r"<[^>]+>", "", text)

    # 8. HTML entities:  &nbsp; -> space,  &amp; -> &
    text = html.unescape(text)

    # 9. Collapse runs of spaces/tabs (and non-breaking spaces) into one space.
    text = re.sub(r"[ \t ]+", " ", text)

    if keep_newlines:
        return text
    return text.strip()


def clean_inline(text: str) -> str:
    """Clean a short fragment of wikitext, e.g. one field of a listing."""
    return _clean(text, keep_newlines=False)


def wikitext_to_text(wikitext: str) -> str:
    """Convert a whole article into Markdown-style plain text with ## headings."""
    # Image galleries are just lists of photo file names: remove them.
    text = re.sub(r"<gallery.*?</gallery>", "", wikitext, flags=re.S | re.I)

    # Tables start with "{|" at the beginning of a line and end with "|}".
    text = re.sub(r"^\{\|.*?^\|\}", "", text, flags=re.S | re.M)

    # Magic words that only control how the page is displayed.
    text = text.replace("__NOTOC__", "")
    text = text.replace("__NOEDITSECTION__", "")

    # Run all the cleaning steps, but keep line breaks: we still need them
    # to recognise headings and list items below.
    text = _clean(text, keep_newlines=True)

    # Now go line by line to format headings and list items.
    lines = []
    for line in text.splitlines():
        line = line.strip()

        # Headings: "==Eat==" -> "## Eat",  "===Budget===" -> "### Budget"
        heading = re.fullmatch(r"(={2,6})\s*(.*?)\s*\1", line)
        if heading:
            equals_signs = heading.group(1)
            heading_text = heading.group(2)
            lines.append("")                                  # blank line before a heading
            lines.append("#" * len(equals_signs) + " " + heading_text)
            continue

        # Leftover table syntax that the table regex didn't catch.
        if line in ("|}", "{|") or line.startswith(("{|", "|-")):
            continue

        # List items: "* item" or "# item" -> "- item"
        if line.startswith(("*", "#")):
            item = line.lstrip("*#: ").strip()
            if item:
                lines.append("- " + item)
            continue

        lines.append(line)

    text = "\n".join(lines)

    # No more than one blank line in a row.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_is_part_of(wikitext: str) -> str | None:
    """Find where a place sits in Wikivoyage's hierarchy.

    The article for Ao Nang contains {{IsPartOf|Krabi Province}} -> "Krabi Province".
    Returns None if the article has no such template.
    """
    match = re.search(r"\{\{\s*ispartof\s*\|\s*([^}|]+)", wikitext, re.I)
    if match is None:
        return None
    return match.group(1).strip()
