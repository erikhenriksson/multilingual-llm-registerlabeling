import json
import re
import sys
from html import escape
from pathlib import Path
from xml.etree import ElementTree as ET

ELEMENT_MAP = {
    "doc": "doc",
    "main": "main",
    "comments": "comments",
    "p": "p",
    "head": "head",
    "h1": "head",
    "h2": "head",
    "h3": "head",
    "h4": "head",
    "h5": "head",
    "h6": "head",
    "item": "li",
    "list": "ul",
    "quote": "blockquote",
    "blockquote": "blockquote",
    "table": "table",
    "row": "tr",
    "cell": "td",
    "td": "td",
    "th": "th",
    "tr": "tr",
    "li": "li",
    "code": "code",
}

KEEP_ATTRS = {
    "doc": ["fingerprint"],
}


# ---------- Simplification helpers ----------


def _clean(text: str) -> str:
    if not text:
        return ""
    text = escape(text)
    return re.sub(r"[\n\r\t ]+", " ", text)


def _is_bold_only(elem) -> bool:
    if elem.text and elem.text.strip():
        return False
    children = list(elem)
    if not children:
        return False
    for child in children:
        if child.tag != "hi":
            return False
        if "#b" not in child.attrib.get("rend", ""):
            return False
        if child.tail and child.tail.strip():
            return False
    return True


def _transform(elem) -> str:
    tag = elem.tag
    role = elem.attrib.get("role", "")

    new_tag = ELEMENT_MAP.get(tag)

    if tag == "cell" and role == "head":
        new_tag = "th"

    if tag == "p" and _is_bold_only(elem):
        new_tag = "head"

    if tag == "hi" and "#b" in elem.attrib.get("rend", ""):
        children = list(elem)
        if len(children) == 1 and children[0].tag == "p":
            if not (elem.text and elem.text.strip()):
                if not (children[0].tail and children[0].tail.strip()):
                    inner_text = _clean(children[0].text)
                    for grandchild in children[0]:
                        inner_text += _transform(grandchild)
                        inner_text += _clean(grandchild.tail)
                    inner_text = inner_text.strip()
                    if not inner_text:
                        return ""
                    return f"<head>{inner_text}</head>"

    if tag in ("hi", "strong", "em"):
        new_tag = None

    inner = _clean(elem.text)
    for child in elem:
        inner += _transform(child)
        inner += _clean(child.tail)

    if new_tag:
        inner = inner.strip()
        if not inner:
            return ""
        attrs = ""
        if tag in KEEP_ATTRS:
            for attr in KEEP_ATTRS[tag]:
                if attr in elem.attrib:
                    attrs += f' {attr}="{escape(elem.attrib[attr])}"'
        return f"<{new_tag}{attrs}>{inner}</{new_tag}>"

    return inner


def simplify_xml(xml_string: str) -> str:
    root = ET.fromstring(xml_string)
    simplified = _transform(root)
    return re.sub(r" +", " ", simplified)


# ---------- Markdown conversion from simplified XML ----------


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _line_safe(s: str) -> str:
    s = _collapse_ws(s)
    s = s.replace("\n", " ").replace("\r", " ")
    return _collapse_ws(s)


def _node_text_with_inline_code(elem) -> str:
    parts = []
    if elem.text:
        parts.append(elem.text)

    for child in list(elem):
        if child.tag == "code":
            code_txt = _line_safe("".join(child.itertext()))
            if code_txt:
                parts.append(f"CODE: {code_txt}")
        else:
            parts.append(_node_text_with_inline_code(child))

        if child.tail:
            parts.append(child.tail)

    return _line_safe("".join(parts))


def _render_ul_one_line(ul_elem) -> str:
    items = []
    for li in ul_elem.findall("./li"):
        txt = _node_text_with_inline_code(li)
        if txt:
            items.append(txt)
    if not items:
        return ""
    return _line_safe("- " + "; ".join(items))


def _render_table_one_line(table_elem) -> str:
    rows_out = []
    for tr in table_elem.findall("./tr"):
        cells = []
        has_th = False
        for cell in list(tr):
            if cell.tag not in ("td", "th"):
                continue
            if cell.tag == "th":
                has_th = True
            cell_txt = _node_text_with_inline_code(cell)
            cells.append(cell_txt)

        cells = [c for c in cells if c]
        if not cells:
            continue

        prefix = "H" if has_th else "R"
        rows_out.append(_line_safe(f"{prefix}: " + " | ".join(cells)))

    if not rows_out:
        return ""
    return _line_safe("TABLE: " + " ; ".join(rows_out))


def _render_code_one_line(code_elem) -> str:
    txt = _line_safe("".join(code_elem.itertext()))
    return _line_safe(f"CODE: {txt}") if txt else ""


def _section_to_numbered_markdown(section_elem) -> str:
    if section_elem is None:
        return ""

    lines: list[str] = []

    for child in list(section_elem):
        tag = child.tag

        if tag == "head":
            txt = _node_text_with_inline_code(child)
            if txt:
                lines.append(_line_safe(f"# {txt}"))

        elif tag == "p":
            txt = _node_text_with_inline_code(child)
            if txt:
                lines.append(_line_safe(txt))

        elif tag == "blockquote":
            txt = _node_text_with_inline_code(child)
            if txt:
                lines.append(_line_safe(f"> {txt}"))

        elif tag == "ul":
            txt = _render_ul_one_line(child)
            if txt:
                lines.append(_line_safe(txt))

        elif tag == "table":
            txt = _render_table_one_line(child)
            if txt:
                lines.append(_line_safe(txt))

        elif tag == "code":
            txt = _render_code_one_line(child)
            if txt:
                lines.append(_line_safe(txt))

        else:
            continue

    lines = [_line_safe(l) for l in lines if _line_safe(l)]

    return "\n".join(f"[{i}] {line}" for i, line in enumerate(lines, start=1))


def markdown_main_comments_from_simplified(xml_simplified: str) -> tuple[str, str]:
    if not xml_simplified:
        return "", ""

    try:
        root = ET.fromstring(xml_simplified)
    except Exception:
        return "", ""

    if root.tag != "doc":
        return "", ""

    main_elem = root.find("./main")
    comments_elem = root.find("./comments")

    md_main = _section_to_numbered_markdown(main_elem) if main_elem is not None else ""
    md_comments = (
        _section_to_numbered_markdown(comments_elem)
        if comments_elem is not None
        else ""
    )
    return md_main, md_comments


# ---------- JSONL processing ----------


def process_jsonl(input_path: str, output_path: str):
    written = 0
    skipped = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            row = json.loads(line)
            try:
                simplified = simplify_xml(row["xml"])
                md_main, md_comments = markdown_main_comments_from_simplified(
                    simplified
                )

                # Skip rows where any enriched field is empty
                if not simplified or not md_main or not md_comments:
                    skipped += 1
                    continue

                row["xml_simplified"] = simplified
                row["markdown_main"] = md_main
                row["markdown_comments"] = md_comments

            except Exception as e:
                print(f"Row {i}: {e}", file=sys.stderr)
                skipped += 1
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Written: {written}, Skipped: {skipped}")


def find_and_process_all(base_dir: str):
    base = Path(base_dir)
    input_files = sorted(base.rglob("shuffled.jsonl"))

    if not input_files:
        print(f"No 'shuffled.jsonl' files found under {base_dir}")
        return

    print(f"Found {len(input_files)} file(s) to process.\n")

    for input_path in input_files:
        output_path = input_path.parent / "shuffled_simplified.jsonl"
        print(f"Processing: {input_path}")
        print(f"  Output:   {output_path}")
        process_jsonl(str(input_path), str(output_path))
        print()


if __name__ == "__main__":
    find_and_process_all("/scratch/project_2011770/ehenriks/data/")
