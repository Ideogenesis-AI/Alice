"""MkDocs build hook for Alice documentation.

Post-processes rendered HTML to fix bullet lists that mkdocstrings emits
inside table cells as raw markdown text rather than <ul>/<li> elements.
"""

import re
import logging

log = logging.getLogger("mkdocs")


def on_post_page(output: str, **kwargs) -> str:
    """Convert inline bullet-point text inside <td> elements to proper lists.

    mkdocstrings sometimes renders a Parameters/Returns section inside a
    table cell as a plain string such as "- item1\\n- item2".  This hook
    detects those cells and wraps them in <ul><li>…</li></ul> so they
    display correctly in all browsers.
    """

    def _replace_cell(match: re.Match) -> str:
        inner = match.group(1)
        # Only transform cells that contain at least one "- " bullet line.
        if not re.search(r"^\s*-\s+", inner, re.MULTILINE):
            return match.group(0)
        lines = inner.split("\n")
        items = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- "):
                items.append(f"<li>{stripped[2:].strip()}</li>")
            elif stripped:
                items.append(stripped)
        if items:
            return f"<td><ul>{''.join(items)}</ul></td>"
        return match.group(0)

    output = re.sub(r"<td>(.*?)</td>", _replace_cell, output, flags=re.DOTALL)
    return output
