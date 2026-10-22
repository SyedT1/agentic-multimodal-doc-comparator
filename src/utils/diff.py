"""Safe, downloadable side-by-side text changes."""
from difflib import HtmlDiff, SequenceMatcher
from html import escape
import re


def document_text(document):
    tables = ["\n".join(" | ".join(str(cell or "") for cell in row) for row in table.get("data", []))
              for table in document.raw_tables]
    return document.raw_text + ("\n\nTables\n" + "\n\n".join(tables) if tables else "")


def changed_values(left, right):
    pattern = r"\$?\d[\d,.]*(?:\s*(?:million|billion|percent|days|months|years))?%?|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b"
    first = list(dict.fromkeys(re.findall(pattern, left, re.I)))
    second = list(dict.fromkeys(re.findall(pattern, right, re.I)))
    return [value for value in first if value not in second], [value for value in second if value not in first]


def diff_summary(left, right):
    added = removed = changed = 0
    for tag, i, j, k, l in SequenceMatcher(None, left.splitlines(), right.splitlines(), autojunk=False).get_opcodes():
        if tag == "insert":
            added += l - k
        elif tag == "delete":
            removed += j - i
        elif tag == "replace":
            changed += max(j - i, l - k)
    return {"added": added, "removed": removed, "changed": changed}


def diff_html(left, right, left_name, right_name):
    table = HtmlDiff(wrapcolumn=70).make_table(left.splitlines()[:5000], right.splitlines()[:5000],
        fromdesc=escape(left_name), todesc=escape(right_name), context=True, numlines=3)
    return '<!doctype html><html><head><meta charset="utf-8"><style>' + '''
body{margin:0;padding:16px;background:#f8fafc;color:#172033;font:13px system-ui}
table.diff{width:100%;border-collapse:collapse;background:white;border:1px solid #e2e8f0}
td,th{padding:8px;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}
.diff_header{background:#eef2f6;color:#475569}.diff_next{display:none}
.diff_add{background:#dcfce7;color:#14532d}.diff_sub{background:#fee2e2;color:#7f1d1d}
.diff_chg{background:#fef3c7;color:#78350f}
''' + '</style></head><body>' + table + '</body></html>'
