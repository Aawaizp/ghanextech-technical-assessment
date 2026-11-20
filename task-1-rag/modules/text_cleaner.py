import re


def clean_text(text: str) -> str:
    """Join wrapped lines while keeping paragraph boundaries for the splitter."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in paragraphs if paragraph.strip()
    )
