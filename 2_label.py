"""
Register classification of web document segments via LLM (OpenRouter).

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python classify_registers.py --input sample_10k_converted.jsonl --output sample_10k_annotated.jsonl

Optional:
    --model           Model string (default: openai/gpt-4.1-nano)
    --max-docs        Only process first N docs
    --start-from      Resume from doc index N (0-based)
    --verbose-prompts Print full system + user prompt before each LLM call
"""

import argparse
import json
import os
import re
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# DEFAULT_MODEL = "openai/gpt-5.2"
# DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
CHUNK_SIZE = 20  # lines per API call
CONTEXT_LINES = 8  # surrounding lines shown as context
MAX_LINE_CHARS = 750  # truncate long lines
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry
SLEEP_BETWEEN_CALLS = 0.01
DOC_PREAMBLE_LINES = 8  # opening lines sent as document-level context

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You classify text segments from web documents into register categories.

# Line format conventions

Each line is numbered like [1], [2], etc. Lines use simplified markdown:
- `# text` = heading
- Plain text = paragraph
- `> text` = blockquote
- `- item1; item2; item3` = list items joined by semicolons
- `TABLE: ...` = table data
- `CODE: ...` = code snippet

Use headings, tables, code, and surrounding lines to understand the document's \
overall purpose, but classify every line in the [CLASSIFY THESE LINES] section.

# Categories

For each line, assign three fields:

## source (always required)
- "written" — originally composed as written text. This is the default for \
most web content. Informal or ungrammatical writing is still "written" — poor \
grammar does not mean speech.
- "transcribed" — text that was originally spoken aloud and then written down. \
Use ONLY when there is clear evidence of oral origin: explicit transcript \
labels, speaker turn markers (e.g. "Interviewer:", "Q:", "[Speaker]"), or the \
text is clearly a transcript of a speech, podcast, interview, or hearing. Do \
NOT use just because the writing is informal, conversational, or ungrammatical. \
A letter or written document that is quoted or embedded in a page is "written" \
even if introduced with "here is the full text."
- "cannot_rate" — use for: (1) machine-generated, boilerplate, navigational, \
or structural content with no substantive human-authored message (e.g. cookie \
notices, auto-generated footers, breadcrumb navigation, "Related posts" lists, \
copyright lines, "Share on Facebook", short bylines/datelines, section labels \
like "7 Parts:", calls to edit/contribute like "Click EDIT to write this \
answer", and metadata). Note: an author bio with full sentences describing \
qualifications is substantive — classify it by its function, not as cannot_rate. \
(2) Incoherent, garbled, or spam-like text that lacks coherent human intent, \
including bad machine translations and keyword stuffing. \
(3) Fragments too short to classify — isolated numbers, single words. \
A complete sentence with clear communicative intent is always classifiable \
as written or transcribed, not cannot_rate.

## interactive (required if source is "written" or "transcribed"; null otherwise)
- true — the text is produced in a participatory context where responses from \
other participants are expected or possible. This includes: forum threads, \
comment sections, discussion boards, social media posts with replies, chat \
messages, and interviews with alternating speakers. A single comment or reply \
is interactive even without visible back-and-forth, because it is produced \
within a discussion context.
- false — monologic: one author/speaker addressing a general audience with no \
participatory context. Articles, blog posts (the post itself, not comments), \
encyclopedia entries, guides, stories.

Important: editorially structured Q&A content is NOT interactive. This \
includes FAQ pages, wiki-style Q&A, and how-to sites where questions and \
answers are organized as reference content rather than real conversation \
between participants.

## function (required if written/transcribed; null otherwise)
The key question: what communicative act is the author performing? \
Classify by the DOMINANT purpose of the line. If a line serves multiple \
functions (e.g. describes a feature while also promoting it), choose the \
function that best captures the author's primary intent. Comments and forum \
posts can serve any function — people in discussions also narrate events, \
describe facts, ask questions, and give instructions.

Decision priority: "describing" is the residual category — use it only when \
no more specific function applies. When a line could plausibly be "describing" \
or another function, ask: is the author's primary goal to inform neutrally, \
or to do something more specific — sell, argue, recount events, instruct? \
If the latter, prefer the more specific label.

Valid values (no other values are permitted): "narrating", "describing", \
"instructing", "opinion", "promoting", "lyrical".
- "narrating" — recounting specific events or stories in temporal sequence. \
Requires specific events that actually happened (or are presented as having \
happened). This includes news reports of events, historical accounts, and \
biographical passages that recount what someone did. Factual reference \
material about a topic is NOT narration even if it mentions dates or \
timelines. Hypothetical scenarios and descriptions of typical situations \
are also NOT narration. \
Decision test: does the text recount a sequence of specific events? \
"The company was founded in 2005 and has grown to 500 employees" is \
describing (general background). "In March 2005, the founders quit their \
jobs, pooled $10,000, and launched from a garage" is narrating (specific \
events in sequence).
- "describing" — presenting factual information, concepts, states of affairs, \
or general knowledge. This covers neutral reference material, encyclopedic \
content, and explanatory or analytical writing. Minor framing or incidental \
evaluative language does not disqualify a line from "describing" — the test \
is whether the primary purpose is to inform the reader about facts rather \
than to argue a position or express a judgment. However, if evaluative \
language is prominent or the line's purpose shifts toward advocacy or \
critique, prefer "opinion."
- "instructing" — teaching the reader how to do something, or providing \
direct answers to how-to or problem-solving questions. How-to guides, \
tutorials, recipes, technical instructions, troubleshooting steps, and \
Q&A content where the answer provides information or guidance in response \
to a question. Recipe components such as ingredient lists, quantities, and \
preparation notes are "instructing" — they are part of the instructional \
act of telling the reader how to make something.
- "opinion" — expressing subjective views, evaluations, arguments, \
complaints, praise, or commentary where the evaluative stance is the \
primary purpose of the line. Includes text that uses narrative framing \
primarily to support an evaluative point (when an anecdote serves to \
complain, critique, praise, or argue, the function is "opinion"). Also \
includes short expressive utterances like "Thanks!", "Great post!", \
"Agreed" — these express a stance or sentiment, not information. Signals \
include: sustained value judgments, subjective adjectives ("essential", \
"crucial", "best"), rhetorical questions, prescriptive statements \
("you should", "it's a must"), and recommendations. If text argues what \
should be done or advocates a position, it is "opinion" even if the \
topic is factual.
- "promoting" — selling, advertising, or marketing a product, service, \
brand, or organization. This includes: explicit ads and calls to action; \
business pages describing their own services or qualifications to attract \
clients; SEO-style content stuffed with keywords for commercial purposes; \
fundraising appeals and donation requests; text where the author represents \
an organization and frames its offerings positively to potential customers. \
"Promoting" does not require explicit "buy now" language — if the text's \
purpose is to attract clients or donors, it is promoting. \
Key test: consider WHO is writing and WHY. When an organization describes \
its own services, qualifications, or offerings, the function is "promoting" \
even if the surface text reads as neutral factual description. Example: a \
law firm's page stating "Our team has 20 years of experience in immigration \
law. We handle H-1B visas, green cards, and asylum cases" is promoting — \
the author represents the firm and frames its qualifications to attract \
clients.
- "lyrical" — poetry, song lyrics, verse, or artistic literary expression \
presented as primary content. A poem or song quoted within a review or \
critical essay should be classified by the outer framing purpose, not as \
lyrical.

# Boundary examples

These illustrate common difficult cases. The reasoning after // is for your \
reference — do not include it in output.

Promoting vs describing:
  "We offer comprehensive dental care including cleanings, implants, and \
cosmetic dentistry." → promoting // author represents the business, framing \
services to attract patients
  "Dental care includes preventive treatments such as cleanings, as well as \
restorative procedures like implants." → describing // neutral encyclopedia-style \
explanation, no organizational self-representation

Narrating vs describing:
  "On June 12, protesters gathered outside city hall and clashed with police \
after an officer fired tear gas into the crowd." → narrating // specific events \
in temporal sequence
  "The city has a history of political protests, particularly around police \
reform issues." → describing // general background, no specific event sequence

Opinion vs describing:
  "The framework provides dependency injection and supports middleware." \
→ describing // neutral factual statement about capabilities
  "The framework's dependency injection is excellent and makes it the best \
choice for enterprise applications." → opinion // evaluative stance is primary

# Output format

Return ONLY a JSON array with one object per line in [CLASSIFY THESE LINES]. \
Each classified line gets exactly one JSON object. No markdown fences, \
no commentary, no extra text.

Example:
[{"line":1,"source":"written","interactive":false,"function":"describing"},\
{"line":2,"source":"cannot_rate","interactive":null,"function":null}]
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Build user message for one chunk."""
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

    parts.append("")
    parts.append("[CLASSIFY THESE LINES]")
    for i in range(chunk_start, chunk_end):
        parts.append(f"[{lines[i]['num']}] {truncate(lines[i]['text'])}")

    ctx_end = min(len(lines), chunk_end + CONTEXT_LINES)
    if chunk_end < ctx_end:
        parts.append("")
        parts.append("[CONTEXT — do not classify]")
        for i in range(chunk_end, ctx_end):
            parts.append(f"[{lines[i]['num']}] {truncate(lines[i]['text'])}")

    return "\n".join(parts)


def call_llm(api_key, model, user_prompt, verbose=False):
    """Call OpenRouter with retries."""
    if verbose:
        print("\n" + "=" * 80, file=sys.stderr)
        print(">>> SYSTEM PROMPT:", file=sys.stderr)
        print(SYSTEM_PROMPT, file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        print(">>> USER PROMPT:", file=sys.stderr)
        print(user_prompt, file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
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

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    last_brace = text.rfind("}")
    if last_brace > 0:
        truncated = text[: last_brace + 1]
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


def classify_section(api_key, model, url, section, markdown_text, verbose=False):
    """Classify all lines in a section."""
    lines = parse_lines(markdown_text)
    if not lines:
        return []

    ann_by_line = {}

    # Build document preamble once for all chunks in this section
    doc_preamble = get_doc_preamble(lines)

    for chunk_start in range(0, len(lines), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(lines))
        prompt = build_user_prompt(
            url, section, lines, chunk_start, chunk_end, doc_preamble
        )

        raw = call_llm(api_key, model, prompt, verbose=verbose)

        try:
            result = parse_json_response(raw)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}", file=sys.stderr)
            print(f"  Raw: {raw[:300]}", file=sys.stderr)
            result = [
                {
                    "line": lines[i]["num"],
                    "source": "cannot_rate",
                    "interactive": None,
                    "function": None,
                }
                for i in range(chunk_start, chunk_end)
            ]

        expected = {lines[i]["num"] for i in range(chunk_start, chunk_end)}
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
                retry_raw = call_llm(api_key, model, retry_prompt, verbose=verbose)
                retry_result = parse_json_response(retry_raw)
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

        for ann in result:
            ann_by_line[ann.get("line")] = ann

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Build final list with post-processing
    final = []
    for line in lines:
        if is_structural(line["text"]):
            final.append(
                {
                    "line": line["num"],
                    "source": "structural",
                    "interactive": None,
                    "function": None,
                }
            )
        elif line["num"] in ann_by_line:
            ann = ann_by_line[line["num"]]
            # Comments are never transcribed — force to written
            if section == "comments" and ann.get("source") == "transcribed":
                ann["source"] = "written"
            final.append(ann)
        else:
            final.append(
                {
                    "line": line["num"],
                    "source": "cannot_rate",
                    "interactive": None,
                    "function": None,
                }
            )

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


def retry_mode(args, api_key):
    """Re-process documents with errors in-place."""
    print(f"Retry mode: scanning {args.output} for failed docs...")

    with open(args.output) as f:
        output_rows = [json.loads(line) for line in f]

    with open(args.input) as f:
        input_rows = [json.loads(line) for line in f]

    error_indices = [i for i, row in enumerate(output_rows) if doc_has_errors(row)]

    if not error_indices:
        print("No errors found. Nothing to retry.")
        return

    print(f"Found {len(error_indices)} docs with errors. Retrying...")

    t_start = time.time()
    for count, idx in enumerate(error_indices):
        doc_start = time.time()
        row = input_rows[idx]
        url = row.get("u", "")

        try:
            main_ann = classify_section(
                api_key,
                args.model,
                url,
                "main",
                row.get("markdown_main", ""),
                verbose=args.verbose_prompts,
            )
        except Exception as e:
            print(f"  FAILED main: {e}", file=sys.stderr)
            main_ann = (
                output_rows[idx].get("llm_register_annotation", {}).get("main", [])
            )

        try:
            comments_ann = classify_section(
                api_key,
                args.model,
                url,
                "comments",
                row.get("markdown_comments", ""),
                verbose=args.verbose_prompts,
            )
        except Exception as e:
            print(f"  FAILED comments: {e}", file=sys.stderr)
            comments_ann = (
                output_rows[idx].get("llm_register_annotation", {}).get("comments", [])
            )

        row["llm_register_annotation"] = {
            "main": main_ann,
            "comments": comments_ann,
        }
        output_rows[idx] = row

        doc_time = time.time() - doc_start
        done = count + 1
        elapsed = time.time() - t_start
        avg = elapsed / done
        remaining = (len(error_indices) - done) * avg
        print(
            f"[{idx}] {doc_time:.1f}s | {done}/{len(error_indices)} retries | "
            f"ETA {remaining / 60:.0f}min | {url[:60]}"
        )

    with open(args.output, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Retried {len(error_indices)} docs.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY env variable.", file=sys.stderr)
        sys.exit(1)

    if args.retry:
        retry_mode(args, api_key)
        return

    with open(args.input) as f:
        input_lines = f.readlines()

    total = len(input_lines)

    start = args.start_from
    if os.path.exists(args.output) and start == 0:
        with open(args.output) as f:
            existing = sum(1 for _ in f)
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
        for idx in range(start, end):
            doc_start = time.time()
            row = json.loads(input_lines[idx])
            url = row.get("u", "")

            try:
                main_ann = classify_section(
                    api_key,
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
                    api_key,
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
                f"[{idx}] {doc_time:.1f}s | {done}/{end - args.start_from} | "
                f"avg {avg:.1f}s/doc | ETA {eta_min:.0f}min | {url[:60]}"
            )

    total_time = time.time() - t_start
    print(
        f"Done. {end - start} docs in {total_time / 60:.1f} min "
        f"({total_time / (end - start):.1f}s/doc avg)"
    )


if __name__ == "__main__":
    main()
