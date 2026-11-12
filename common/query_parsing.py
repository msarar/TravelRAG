"""Shared helper: read what the user typed at the prompt.

Used by s5_search.py and s6_ask.py, so both understand the same input format:
    where can I eat street food?            -> no filter
    @bangkok: where can I eat street food?  -> only search place == "Bangkok"
"""


def parse_input(user_input: str) -> tuple[str, str | None]:
    """Split "@Chiang Mai: massage" into ("massage", "Chiang Mai").

    Without an "@...:" prefix, there's no place: ("massage", None).
    """
    text = user_input.strip()

    if text.startswith("@") and ":" in text:
        place_part, question_part = text.split(":", 1)   # split at the FIRST ":" only
        place = place_part[1:].strip()                    # drop the "@"
        question = question_part.strip()
        return question, place

    return text, None


def resolve_place(typed_place: str, known_places: list[str]) -> str | None:
    """Match what the user typed against the real place names, ignoring case.

    "dhaka" -> "Dhaka",   "CHIANG MAI" -> "Chiang Mai",   "Paris" -> None

    Chroma filters are EXACT matches, so we must pass the exact stored spelling.
    """
    typed_lower = typed_place.lower()

    for place in known_places:
        if place.lower() == typed_lower:
            return place

    return None


def list_places(vector_store) -> list[str]:
    """Collect every distinct "place" value in the index.

    vector_store.get() reads stored records straight from SQLite, WITHOUT a
    similarity search. We only ask for the metadata, which is quick.
    """
    stored = vector_store.get(include=["metadatas"])

    places = set()
    for metadata in stored["metadatas"]:
        places.add(metadata["place"])

    return sorted(places)
