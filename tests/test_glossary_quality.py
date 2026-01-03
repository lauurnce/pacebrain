"""Glossary docs must stay structured."""

from pathlib import Path


def _entries():
    return sorted((Path(__file__).parents[1] / "docs" / "glossary").glob("*.md"))


def test_each_entry_has_heading_and_body():
    for p in _entries():
        text = p.read_text().strip()
        assert text.startswith("# "), f"{p.name} missing heading"
        assert len(text) > 120, f"{p.name} suspiciously short"


def test_see_also_links_resolve():
    for p in _entries():
        for line in p.read_text().splitlines():
            s = line.strip()
            if s.startswith("- [") and "](" in s and s.endswith(")") and ".md)" in s:
                target = s[s.rindex("(") + 1:-1]
                assert (p.parent / target).exists(), f"{p.name} broken link {target}"
