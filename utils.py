import re

from bs4 import BeautifulSoup
from markdownify import markdownify as md


def html_as_markdown(html: str):
    soup = BeautifulSoup(html, "html.parser")

    # Strip unnecessary data
    for data in soup(
        [
            "style",
            "script",
            "header",
            "footer",
            "img",
            "textarea",
        ]
    ):
        # Remove tags
        data.decompose()

    return md(soup.body.text) if soup.body else None
