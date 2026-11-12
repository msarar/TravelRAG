"""Shared helper: turning article titles into file names."""


def safe_filename(title: str) -> str:
    """Turn an article title into a safe file name.

    "Bangkok/Sukhumvit" -> "Bangkok__Sukhumvit"   (a "/" would mean a sub-folder)
    "Chiang Mai"        -> "Chiang_Mai"
    """
    name = title.replace("/", "__")
    name = name.replace(" ", "_")
    return name
