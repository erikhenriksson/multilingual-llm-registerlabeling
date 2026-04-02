"""
Register classification of web document segments via LLM (Gemini).

Usage:
    export GEMINI_API_KEY="..."
    python classify_registers.py --input sample_10k_converted.jsonl --output sample_10k_annotated.jsonl

Optional:
    --model           Model string (default: gemini-3-flash-preview)
    --max-docs        Only process first N docs
    --start-from      Resume from doc index N (0-based)
    --verbose-prompts Print full system + user prompt before each LLM call
"""

import argparse
import json
import mmap
import os
import re
import sys
import time

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3-flash-preview"
CHUNK_SIZE = 20  # lines per API call
CONTEXT_LINES = 8  # surrounding lines shown as context
MAX_LINE_CHARS = 750  # truncate long lines
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry
SLEEP_BETWEEN_CALLS = 0.5  # FIX #7: meaningful rate-limit pause
DOC_PREAMBLE_LINES = 8  # opening lines sent as document-level context

# ---------------------------------------------------------------------------
# Valid values for annotation validation (FIX #6)
# ---------------------------------------------------------------------------
VALID_MODE_MEDIUM = {"written", "transcribed", "cannot_rate"}
VALID_MODE_TURN = {"monologic", "dialogic", None}
VALID_FIELD_ACTIVITY = {
    "recounting",
    "explaining",
    "directing",
    "evaluating",
    "promoting",
    "creating",
    None,
}
VALID_TENOR_FORMALITY = {"formal", "informal", None}

# ---------------------------------------------------------------------------
# Gemini client (reads GEMINI_API_KEY env var automatically)
# ---------------------------------------------------------------------------
gemini_client = genai.Client()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You classify text segments from web documents along four register dimensions \
based on Systemic Functional Linguistics.
 
# Line format conventions
 
Each line is numbered like [1], [2], etc. Lines use simplified markdown:
- `# text` = heading
- Plain text = paragraph
- `> text` = blockquote
- `- item1; item2; item3` = list items joined by semicolons
- `TABLE: ...` = table data
- `CODE: ...` = code snippet
 
Use headings, tables, code, and surrounding lines to understand the document's \
overall purpose, but classify ONLY the lines in the [CLASSIFY THESE LINES] \
section. Do NOT return JSON objects for lines marked [CONTEXT] or \
[DOCUMENT OPENING].
 
# Register dimensions
 
For each line, assign four fields:
 
## mode_medium (always required)
- "written" — composed as written text. Default for web content.
- "transcribed" — originally spoken, then written down. Use ONLY with clear \
evidence of oral origin: transcript labels, speaker turn markers \
(e.g. "Interviewer:", "Q:", "[Speaker]"), or text clearly from a speech, \
podcast, interview, or hearing. Informal writing is still "written."
- "cannot_rate" — use when the text is (1) machine-generated, boilerplate, \
navigational, or structural content (e.g. cookie notices, footers, breadcrumbs, \
copyright lines, share buttons, metadata), or (2) not a complete sentence or \
coherent phrase. A sentence split across multiple lines should be classified \
by its full meaning, not marked cannot_rate for being a fragment. Any complete \
sentence or coherent phrase with human communicative intent is classifiable.
 
## mode_turn (required if rateable; null if cannot_rate)
- "monologic" — one author/speaker addressing a general audience: articles, \
blog posts, encyclopedia entries, guides, stories. Editorially structured Q&A \
(FAQ pages, wiki-style Q&A, how-to sites) is monologic — it is reference \
content, not real conversation.
- "dialogic" — produced in a participatory context where responses from other \
participants are expected or possible: forum threads, comment sections, \
social media posts, chat messages, interviews with alternating speakers.
 
## field_activity (required if rateable; null if cannot_rate)
The communicative activity the language is performing. Classify by the DOMINANT \
activity. "explaining" is the residual category — use it only when no more \
specific activity applies.
 
- "recounting" — reporting specific events in temporal sequence. Requires \
events presented as having happened. \
Example: "The company was founded in 2005 and has grown to 500 employees" \
→ explaining. "In March 2005, the founders quit their jobs, pooled $10,000, \
and launched from a garage" → recounting.
- "explaining" — presenting factual information, concepts, states of affairs, \
or general knowledge. Neutral reference material, encyclopedic content, \
analytical writing.
- "directing" — telling the reader how to do something: tutorials, recipes, \
technical instructions, troubleshooting steps, Q&A answers that provide \
guidance. Recipe components (ingredient lists, quantities) are "directing."
- "evaluating" — expressing subjective views, arguments, complaints, praise, \
or commentary where stance is the primary purpose. Includes short expressive \
utterances ("Thanks!", "Great post!"). If text argues what should be done or \
advocates a position, it is "evaluating" even if the topic is factual.
- "promoting" — selling, advertising, or marketing a product, service, brand, \
or organization. Key test: WHO is writing and WHY. When an organization \
describes its own services or offerings, the activity is "promoting" even if \
the surface text reads as neutral. Includes fundraising appeals.
- "creating" — poetry, song lyrics, verse, or artistic literary expression \
presented as primary content.
 
## tenor_formality (required if rateable; null if cannot_rate)
- "formal" — institutional, professional, or academic register. Signals: \
complex syntax, technical/specialized vocabulary, impersonal constructions, \
no contractions or colloquialisms.
- "informal" — casual, conversational, or personal register. Signals: \
contractions, colloquialisms, first-person address, simple syntax, slang.
 
When signals are mixed, choose the dominant register of the line.
 
# Boundary examples
 
Promoting vs explaining:
  "We offer comprehensive dental care including cleanings, implants, and \
cosmetic dentistry." → promoting // author represents the business
  "Dental care includes preventive treatments such as cleanings, as well as \
restorative procedures like implants." → explaining // neutral explanation
 
Recounting vs explaining:
  "On June 12, protesters gathered outside city hall and clashed with police \
after an officer fired tear gas into the crowd." → recounting
  "The city has a history of political protests, particularly around police \
reform issues." → explaining
 
# Output format
 
Return ONLY a JSON array with one object per line in [CLASSIFY THESE LINES]. \
Each classified line gets exactly one JSON object. No markdown fences, \
no commentary, no extra text.
 
Example:
[{"line":1,"mode_medium":"written","mode_turn":"monologic","field_activity":"explaining","tenor_formality":"formal"},\
{"line":2,"mode_medium":"cannot_rate","mode_turn":null,"field_activity":null,"tenor_formality":null}]
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Fallback annotation for unrateable lines
def _cannot_rate(line_num):
    return {
        "line": line_num,
        "mode_medium": "cannot_rate",
        "mode_turn": None,
        "field_activity": None,
        "tenor_formality": None,
    }


def _structural(line_num):
    return {
        "line": line_num,
        "mode_medium": "cannot_rate",  # FIX #4: use valid schema value
        "mode_turn": None,
        "field_activity": None,
        "tenor_formality": None,
        "is_structural": True,  # preserve the distinction as a flag
    }


def validate_annotation(ann):
    """FIX #6: Validate and repair an annotation dict. Returns corrected copy."""
    ann = dict(ann)  # shallow copy

    # Validate mode_medium
    if ann.get("mode_medium") not in VALID_MODE_MEDIUM:
        print(
            f"  Invalid mode_medium '{ann.get('mode_medium')}' on line {ann.get('line')}, "
            f"defaulting to cannot_rate",
            file=sys.stderr,
        )
        ann["mode_medium"] = "cannot_rate"

    # If cannot_rate, force nulls on dependent fields
    if ann["mode_medium"] == "cannot_rate":
        ann["mode_turn"] = None
        ann["field_activity"] = None
        ann["tenor_formality"] = None
        return ann

    # Validate mode_turn
    if ann.get("mode_turn") not in VALID_MODE_TURN:
        print(
            f"  Invalid mode_turn '{ann.get('mode_turn')}' on line {ann.get('line')}, "
            f"defaulting to monologic",
            file=sys.stderr,
        )
        ann["mode_turn"] = "monologic"

    # Validate field_activity
    if ann.get("field_activity") not in VALID_FIELD_ACTIVITY:
        print(
            f"  Invalid field_activity '{ann.get('field_activity')}' on line {ann.get('line')}, "
            f"defaulting to explaining",
            file=sys.stderr,
        )
        ann["field_activity"] = "explaining"

    # Validate tenor_formality
    if ann.get("tenor_formality") not in VALID_TENOR_FORMALITY:
        print(
            f"  Invalid tenor_formality '{ann.get('tenor_formality')}' on line {ann.get('line')}, "
            f"defaulting to formal",
            file=sys.stderr,
        )
        ann["tenor_formality"] = "formal"

    # Rateable lines should not have null dependent fields
    if ann.get("mode_turn") is None:
        ann["mode_turn"] = "monologic"
    if ann.get("field_activity") is None:
        ann["field_activity"] = "explaining"
    if ann.get("tenor_formality") is None:
        ann["tenor_formality"] = "formal"

    return ann


def parse_lines(markdown_text):
    """Parse '[N] content' lines into list of {num, text}."""
    if not markdown_text or not markdown_text.strip():
        return []
    lines = []
    for raw in markdown_text.split("\n"):
        m = re.match(r"\[(\d+)\]\s*(.*)", raw)
        if m:
            lines.append({"num": int(m.group(1)), "text": m.group(2)})
    return lines


def truncate(text, max_chars=MAX_LINE_CHARS):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def is_structural(text):
    """Headings, tables, and code are structural — tagged locally, not by LLM."""
    t = text.strip()
    return t.startswith("# ") or t.startswith("TABLE: ") or t.startswith("CODE: ")


def get_doc_preamble(lines, max_lines=DOC_PREAMBLE_LINES):
    """First N non-structural lines as document-level context."""
    selected = []
    for line in lines:
        if not is_structural(line["text"]) and line["text"].strip():
            selected.append(f"[{line['num']}] {truncate(line['text'])}")
            if len(selected) >= max_lines:
                break
    return "\n".join(selected)


def build_user_prompt(url, section, lines, chunk_start, chunk_end, doc_preamble=""):
    """Build user message for one chunk.

    FIX #8 (new): Structural lines inside the classify range are sent as
    context-only (outside the CLASSIFY block) so the LLM doesn't waste
    tokens classifying them, but still sees them for context.
    """
    parts = []
    parts.append(f"URL: {url}")
    parts.append(f"Section: {section}")

    # Document-level context: opening lines of the document
    if doc_preamble:
        # Avoid duplicating preamble lines if the chunk already covers them
        chunk_nums = {lines[i]["num"] for i in range(chunk_start, chunk_end)}
        preamble_filtered = []
        for pline in doc_preamble.split("\n"):
            m = re.match(r"\[(\d+)\]", pline)
            if m and int(m.group(1)) in chunk_nums:
                continue  # skip — already in the classify section
            preamble_filtered.append(pline)
        if preamble_filtered:
            parts.append("")
            parts.append("[DOCUMENT OPENING — for context only, do not classify]")
            parts.extend(preamble_filtered)

    ctx_start = max(0, chunk_start - CONTEXT_LINES)
    if ctx_start < chunk_start:
        parts.append("")
        parts.append("[CONTEXT — do not classify]")
        for i in range(ctx_start, chunk_start):
            parts.append(f"[{lines[i]['num']}] {truncate(lines[i]['text'])}")

    # FIX #8: Split classify range into classifiable lines and inline
    # structural lines shown as context
    parts.append("")
    parts.append("[CLASSIFY THESE LINES]")
    for i in range(chunk_start, chunk_end):
        if is_structural(lines[i]["text"]):
            # Show structural lines for context but mark them
            parts.append(
                f"[{lines[i]['num']}] {truncate(lines[i]['text'])}  "
                f"{{structural — do not classify}}"
            )
        else:
            parts.append(f"[{lines[i]['num']}] {truncate(lines[i]['text'])}")

    ctx_end = min(len(lines), chunk_end + CONTEXT_LINES)
    if chunk_end < ctx_end:
        parts.append("")
        parts.append("[CONTEXT — do not classify]")
        for i in range(chunk_end, ctx_end):
            parts.append(f"[{lines[i]['num']}] {truncate(lines[i]['text'])}")

    return "\n".join(parts)


def _extract_text(response):
    """Extract only non-thought text parts from a Gemini response.

    response.text can include thinking content on Gemini 3 models.
    We iterate over parts and skip anything flagged as thought.
    """
    parts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if not part.text:
                continue
            if getattr(part, "thought", False):
                continue  # skip thinking content
            parts.append(part.text)
    if not parts:
        raise ValueError(
            f"No non-thought text in response "
            f"(finish_reason={getattr(response.candidates[0], 'finish_reason', 'unknown')})"
        )
    return "".join(parts)


def call_llm(model, user_prompt, verbose=False):
    """Call Gemini API with retries."""
    if verbose:
        print("\n" + "=" * 80, file=sys.stderr)
        print(">>> SYSTEM PROMPT:", file=sys.stderr)
        print(SYSTEM_PROMPT, file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        print(">>> USER PROMPT:", file=sys.stderr)
        print(user_prompt, file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)

    for attempt in range(MAX_RETRIES):
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    max_output_tokens=4096,
                    # Gemini 3 models cannot fully disable thinking, but
                    # MINIMAL constrains it to near-zero tokens and keeps
                    # thinking out of the main response text.
                    thinking_config=types.ThinkingConfig(
                        thinking_level="MINIMAL",
                    ),
                ),
            )
            return _extract_text(response)
        except Exception as e:
            wait = RETRY_BACKOFF * (2**attempt)
            print(
                f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES - 1:
                print(f"  Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise


def parse_json_response(text):
    """Extract JSON array from LLM response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Some models wrap output in <think>...</think> tags; strip those
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # FIX #10 (new): Find the outermost JSON array even if surrounded by text
    bracket_start = text.find("[")
    if bracket_start == -1:
        raise json.JSONDecodeError("No JSON array found in response", text, 0)

    # Try parsing from the first '[' to avoid preamble text
    candidate = text[bracket_start:]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Original truncation recovery
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    last_brace = candidate.rfind("}")
    if last_brace > 0:
        truncated = candidate[: last_brace + 1]
        if not truncated.rstrip().endswith("]"):
            truncated = truncated + "]"
        try:
            result = json.loads(truncated)
            print(
                f"  Salvaged {len(result)} items from truncated response",
                file=sys.stderr,
            )
            return result
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse response", text, 0)


def classify_section(model, url, section, markdown_text, verbose=False):
    """Classify all lines in a section."""
    lines = parse_lines(markdown_text)
    if not lines:
        return []

    ann_by_line = {}

    # Build document preamble once for all chunks in this section
    doc_preamble = get_doc_preamble(lines)

    for chunk_start in range(0, len(lines), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(lines))

        # FIX #11 (new): Skip chunks that are entirely structural
        classifiable = [
            i
            for i in range(chunk_start, chunk_end)
            if not is_structural(lines[i]["text"])
        ]
        if not classifiable:
            # All lines in this chunk are structural; tag them locally
            for i in range(chunk_start, chunk_end):
                ann_by_line[lines[i]["num"]] = _structural(lines[i]["num"])
            continue

        prompt = build_user_prompt(
            url, section, lines, chunk_start, chunk_end, doc_preamble
        )

        raw = call_llm(model, prompt, verbose=verbose)

        try:
            result = parse_json_response(raw)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}", file=sys.stderr)
            print(f"  Raw: {raw[:300]}", file=sys.stderr)
            result = [
                _cannot_rate(lines[i]["num"]) for i in range(chunk_start, chunk_end)
            ]

        # FIX #3: Store original results FIRST, so retries can overwrite
        for ann in result:
            ann_by_line[ann.get("line")] = ann

        # Only check non-structural lines for missing annotations
        expected = {
            lines[i]["num"]
            for i in range(chunk_start, chunk_end)
            if not is_structural(lines[i]["text"])
        }
        returned = {a.get("line") for a in result}
        missing = expected - returned
        if missing:
            print(
                f"  Retrying {len(missing)} missed lines: {sorted(missing)}",
                file=sys.stderr,
            )
            missing_indices = [
                i for i in range(chunk_start, chunk_end) if lines[i]["num"] in missing
            ]
            retry_start = missing_indices[0]
            retry_end = missing_indices[-1] + 1
            retry_prompt = build_user_prompt(
                url, section, lines, retry_start, retry_end, doc_preamble
            )
            try:
                retry_raw = call_llm(model, retry_prompt, verbose=verbose)
                retry_result = parse_json_response(retry_raw)
                # FIX #3: Retry results overwrite originals (applied AFTER)
                for ann in retry_result:
                    ann_by_line[ann.get("line")] = ann
                still_missing = missing - {a.get("line") for a in retry_result}
                if still_missing:
                    print(
                        f"  Still missing after retry: {sorted(still_missing)}",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"  Retry failed: {e}", file=sys.stderr)
            time.sleep(SLEEP_BETWEEN_CALLS)

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Build final list with post-processing
    final = []
    for line in lines:
        if is_structural(line["text"]):
            final.append(_structural(line["num"]))
        elif line["num"] in ann_by_line:
            ann = ann_by_line[line["num"]]
            # Comments are never transcribed — force to written
            if section == "comments" and ann.get("mode_medium") == "transcribed":
                ann["mode_medium"] = "written"
            # FIX #6: Validate annotation
            ann = validate_annotation(ann)
            final.append(ann)
        else:
            final.append(_cannot_rate(line["num"]))

    return final


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


def doc_has_errors(row):
    """Check if a document has annotations that need retrying."""
    ann = row.get("llm_register_annotation", {})
    if not ann:
        return True
    for section in ("main", "comments"):
        md = row.get(f"markdown_{section}", "")
        items = ann.get(section, [])
        if md and md.strip() and not items:
            return True
    return False


def _build_offset_index(path):
    """Build a list of byte offsets for each line in a file.

    Returns list where index[i] = byte offset of line i. This lets us
    seek to any line without loading the whole file.
    Memory: ~8 bytes per line (just the offset integer), vs the full line content.
    """
    offsets = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    return offsets


def _read_line_by_offset(path, offset):
    """Read a single line from a file given its byte offset."""
    with open(path, "rb") as f:
        f.seek(offset)
        return f.readline().decode("utf-8")


def retry_mode(args):
    """Re-process documents with errors in-place.

    Streams through the output file to find errors, builds a byte-offset
    index for the input file so we can retrieve specific lines without
    loading everything into memory.
    """
    print(f"Retry mode: scanning {args.output} for failed docs...")

    # First pass: find which output indices have errors
    error_indices = []
    total_output = 0
    with open(args.output) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if doc_has_errors(row):
                error_indices.append(i)
            total_output += 1

    if not error_indices:
        print("No errors found. Nothing to retry.")
        return

    # Build byte-offset index for the input file (lightweight: ~8 bytes/line)
    print(f"Building offset index for {args.input}...")
    input_offsets = _build_offset_index(args.input)
    total_input = len(input_offsets)

    # Guard against mismatched lengths
    valid_errors = [i for i in error_indices if i < total_input]
    if len(valid_errors) < len(error_indices):
        skipped = len(error_indices) - len(valid_errors)
        print(
            f"  WARNING: skipping {skipped} error indices beyond input "
            f"({total_input} lines). Output has {total_output} lines.",
            file=sys.stderr,
        )
        error_indices = valid_errors

    if not error_indices:
        print("No retryable errors found after filtering.")
        return

    print(f"Found {len(error_indices)} docs with errors. Retrying...")

    # Build byte-offset index for the output file too — we need to
    # rewrite it, but we do it in a streaming fashion:
    # 1. Copy output to a temp file
    # 2. Stream through temp, replacing error rows
    output_offsets = _build_offset_index(args.output)

    # Process error docs and collect replacements
    replacements = {}  # idx -> new JSON string
    t_start = time.time()

    for count, idx in enumerate(error_indices):
        doc_start = time.time()

        # Read the specific input line
        input_line = _read_line_by_offset(args.input, input_offsets[idx])
        input_row = json.loads(input_line)
        url = input_row.get("u", "")

        # Read existing output row to preserve any extra keys
        output_line = _read_line_by_offset(args.output, output_offsets[idx])
        row = json.loads(output_line)

        try:
            main_ann = classify_section(
                args.model,
                url,
                "main",
                input_row.get("markdown_main", ""),
                verbose=args.verbose_prompts,
            )
        except Exception as e:
            print(f"  FAILED main: {e}", file=sys.stderr)
            main_ann = row.get("llm_register_annotation", {}).get("main", [])

        try:
            comments_ann = classify_section(
                args.model,
                url,
                "comments",
                input_row.get("markdown_comments", ""),
                verbose=args.verbose_prompts,
            )
        except Exception as e:
            print(f"  FAILED comments: {e}", file=sys.stderr)
            comments_ann = row.get("llm_register_annotation", {}).get("comments", [])

        row["llm_register_annotation"] = {
            "main": main_ann,
            "comments": comments_ann,
        }
        replacements[idx] = json.dumps(row, ensure_ascii=False)

        doc_time = time.time() - doc_start
        done = count + 1
        elapsed = time.time() - t_start
        avg = elapsed / done
        remaining = (len(error_indices) - done) * avg
        print(
            f"[{idx}] {doc_time:.1f}s | {done}/{len(error_indices)} retries | "
            f"ETA {remaining / 60:.0f}min | {url[:60]}"
        )

    # Rewrite the output file, streaming line by line
    tmp_output = args.output + ".tmp"
    with open(args.output) as fin, open(tmp_output, "w") as fout:
        for i, line in enumerate(fin):
            if i in replacements:
                fout.write(replacements[i] + "\n")
            else:
                fout.write(line)

    os.replace(tmp_output, args.output)
    print(f"Done. Retried {len(error_indices)} docs.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _count_lines(path):
    """Count lines in a file without loading it into memory."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def _read_line_at_index(path, target_idx):
    """Read a single line from a JSONL file by index (0-based).

    Iterates without loading the whole file. For the forward pass this is
    called once per doc so the overhead is acceptable — each call skips at
    most `target_idx` newlines, which is fast on buffered I/O.
    """
    with open(path) as f:
        for i, line in enumerate(f):
            if i == target_idx:
                return line
    raise IndexError(f"Line {target_idx} not found in {path}")


def _iter_lines(path, start, end):
    """Yield (index, line_string) for lines [start, end) without loading all."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= end:
                break
            if i >= start:
                yield i, line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument(
        "--retry", action="store_true", help="Re-process failed docs in existing output"
    )
    parser.add_argument(
        "--verbose-prompts",
        action="store_true",
        help="Print the full system + user prompt before each LLM call",
    )
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY env variable.", file=sys.stderr)
        sys.exit(1)

    if args.retry:
        retry_mode(args)
        return

    # Count lines without loading file into memory
    print(f"Counting lines in {args.input}...", file=sys.stderr)
    total = _count_lines(args.input)
    print(f"Total docs: {total}")

    start = args.start_from
    if os.path.exists(args.output) and start == 0:
        existing = _count_lines(args.output)
        if existing > 0:
            start = existing
            print(f"Resuming: {existing} docs already in {args.output}")

    end = total if args.max_docs is None else min(start + args.max_docs, total)

    if start >= end:
        print(f"Nothing to do: start={start}, end={end}")
        return

    print(f"Docs {start}--{end - 1} of {total} | Model: {args.model}")

    t_start = time.time()

    with open(args.output, "a") as fout:
        for idx, raw_line in _iter_lines(args.input, start, end):
            doc_start = time.time()
            row = json.loads(raw_line)
            url = row.get("u", "")

            try:
                main_ann = classify_section(
                    args.model,
                    url,
                    "main",
                    row.get("markdown_main", ""),
                    verbose=args.verbose_prompts,
                )
            except Exception as e:
                print(f"  FAILED main: {e}", file=sys.stderr)
                main_ann = []

            try:
                comments_ann = classify_section(
                    args.model,
                    url,
                    "comments",
                    row.get("markdown_comments", ""),
                    verbose=args.verbose_prompts,
                )
            except Exception as e:
                print(f"  FAILED comments: {e}", file=sys.stderr)
                comments_ann = []

            row["llm_register_annotation"] = {
                "main": main_ann,
                "comments": comments_ann,
            }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            doc_time = time.time() - doc_start
            done = idx - start + 1
            elapsed = time.time() - t_start
            avg = elapsed / done
            remaining = (end - start - done) * avg
            eta_min = remaining / 60
            print(
                f"[{idx}] {doc_time:.1f}s | {done}/{end - start} | "
                f"avg {avg:.1f}s/doc | ETA {eta_min:.0f}min | {url[:60]}"
            )

    total_time = time.time() - t_start
    print(
        f"Done. {end - start} docs in {total_time / 60:.1f} min "
        f"({total_time / (end - start):.1f}s/doc avg)"
    )


if __name__ == "__main__":
    main()
