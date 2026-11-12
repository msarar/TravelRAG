"""Shared helper: reading Chroma's distance scores."""


def distance_to_cosine(distance: float) -> float:
    """Chroma's default score is squared L2 distance (0 = perfect, ~2 = unrelated).

    For NORMALISED vectors (ours are: length ≈ 1.0):  cosine = 1 - distance / 2
      distance 0 -> cosine 1   (same meaning)
      distance 2 -> cosine 0   (unrelated)
      distance 4 -> cosine -1  (opposite)
    """
    return 1 - distance / 2
