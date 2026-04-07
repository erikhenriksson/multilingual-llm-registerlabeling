"""
Register classification of web document segments using a local LLM on LUMI.

Loads Gemma-4-31b-it once, then classifies documents from a JSONL file.
Writes results one doc at a time so you can resume by re-running the same command.

Usage:
    python classify_registers_local.py --input sample.jsonl --output annotated.jsonl

Optional:
    --model-id        HuggingFace model ID (default: google/gemma-4-31b-it)
    --max-docs        Only process first N docs (from the resume point)
    --start-from      Resume from doc index N (0-based)
    --verbose-prompts Print full prompts before each LLM call
"""

import argparse
import json
import os

os.environ["HF_HOME"] = "./hf_cache"


import re
import sys
import time

import torch
from accelerate import infer_auto_device_map
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = "./hf_cache"

DEFAULT_MODEL_ID = "google/gemma-4-31b-it"
CHUNK_SIZE = 15  # lines per LLM call
CONTEXT_LINES = 10  # surrounding lines shown as context
MAX_LINE_CHARS = 500  # truncate long lines
MAX_NEW_TOKENS = 2048  # enough for 15 lines of JSON annotations
DOC_PREAMBLE_LINES = 5  # opening lines sent as document-level context

# ---------------------------------------------------------------------------
# Valid values for annotation validation
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
# System prompt (same as original — defines the classification task)
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
copyright lines, share buttons, metadata labels like "Type: Oscillator"), \
(2) garbled or spam-like text with no coherent communicative intent, or \
(3) not a complete sentence or coherent phrase — including isolated words or \
fragments (e.g. "Now", "Next", "231") that only function as transitions or \
labels within a larger text. A sentence split across multiple lines should be \
classified by its full meaning, not marked cannot_rate for being a fragment. \
Any complete sentence or coherent phrase with human communicative intent is \
classifiable.
 
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
activity.
 
- "explaining" — presenting factual information, concepts, states of affairs, \
or general knowledge. Neutral reference material, encyclopedic content, \
analytical writing. This is the default for informational web content. Use a \
more specific category below only when there is clear evidence that the line's \
primary communicative purpose is something other than conveying information.
- "recounting" — reporting specific events in temporal sequence. Requires \
events presented as having happened, with narrative presentation — not just \
the presence of a date or an event-denoting verb. A reference-style mention \
of an event (e.g. a citation or catalog entry) is explaining, not recounting. \
Example: "The company was founded in 2005 and has grown to 500 employees" \
→ explaining. "In March 2005, the founders quit their jobs, pooled $10,000, \
and launched from a garage" → recounting.
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
the surface text reads as neutral. Includes fundraising appeals and statements \
about the availability or features of one's own product or service. \
Apply this test actively: if context suggests the author represents the entity \
being described, classify as promoting.
- "creating" — poetry, song lyrics, verse, or artistic literary expression \
presented as primary content. This includes religious scripture and liturgical \
text when presented as primary content (e.g. Bible verses, Quranic ayat, \
hymns).
 
## tenor_formality (required if rateable; null if cannot_rate)
- "formal" — the default for web content. The author maintains social \
distance, addressing a general or institutional audience.
- "informal" — the author reduces social distance, addressing the audience \
as peers or in-group members. Marked by colloquial vocabulary, in-group \
references, or a casual interpersonal stance.
 
When signals are mixed, choose the dominant register of the line.
 
# Register consistency
 
Lines within a coherent section of a document (e.g. a list of items, a \
sequence of captions, consecutive paragraphs by the same author) typically \
share the same register values. Do not assign a different field_activity or \
tenor_formality to an individual line unless the shift is clearly motivated \
by a change in communicative purpose — not merely by variation in surface \
phrasing. A caption that happens to contain an imperative-like phrase in a \
photo gallery is still the same activity as the surrounding captions. Reserve \
register shifts for genuine transitions: article body to comment section, \
instructions to editorial aside, author prose to embedded quotation from a \
different source.
 
# Using context
 
When a target line is ambiguous, lean on surrounding [CONTEXT] and \
[DOCUMENT OPENING] lines — especially for field_activity and tenor_formality.
 
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


# ===================================================================
# MODEL LOADING
# ===================================================================


def load_model(model_id):
    """Load model and tokenizer onto available GPUs. Returns (model, tokenizer)."""
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    print(f"Building device map...")
    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        meta_model = AutoModelForCausalLM.from_config(config)

    device_map = infer_auto_device_map(meta_model)
    # Gemma vision tower params need to be on GPU 0
    device_map["model.vision_tower.std_bias"] = 0
    device_map["model.vision_tower.std_scale"] = 0

    print(f"Loading model weights (float16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=torch.float16,
    )
    print(f"Model loaded.")
    return model, tokenizer


# ===================================================================
# LOCAL INFERENCE (replaces the HTTP API call)
# ===================================================================


def call_llm(model, tokenizer, user_prompt, verbose=False):
    """Run one classification prompt through the local model.

    Formats the system + user prompt as a chat, tokenizes, generates,
    and returns the decoded text.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if verbose:
        print("\n" + "=" * 80, file=sys.stderr)
        print(">>> SYSTEM PROMPT:", file=sys.stderr)
        print(SYSTEM_PROMPT, file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        print(">>> USER PROMPT:", file=sys.stderr)
        print(user_prompt, file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy, like temperature=0
        )

    # Decode only the newly generated tokens
    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response


# ===================================================================
# HELPERS (unchanged from original)
# ===================================================================


def _cannot_rate(line_num):
    """Fallback annotation for unrateable lines."""
    return {
        "line": line_num,
        "mode_medium": "cannot_rate",
        "mode_turn": None,
        "field_activity": None,
        "tenor_formality": None,
    }


def _structural(line_num):
    """Annotation for structural lines (headings, tables, code)."""
    return {
        "line": line_num,
        "mode_medium": "cannot_rate",
        "mode_turn": None,
        "field_activity": None,
        "tenor_formality": None,
        "is_structural": True,
    }


def validate_annotation(ann):
    """Validate and repair an annotation dict. Returns corrected copy."""
    ann = dict(ann)

    if ann.get("mode_medium") not in VALID_MODE_MEDIUM:
        print(
            f"  Invalid mode_medium '{ann.get('mode_medium')}' on line {ann.get('line')}, "
            f"defaulting to cannot_rate",
            file=sys.stderr,
        )
        ann["mode_medium"] = "cannot_rate"

    if ann["mode_medium"] == "cannot_rate":
        ann["mode_turn"] = None
        ann["field_activity"] = None
        ann["tenor_formality"] = None
        return ann

    if ann.get("mode_turn") not in VALID_MODE_TURN:
        print(
            f"  Invalid mode_turn '{ann.get('mode_turn')}' on line {ann.get('line')}, "
            f"defaulting to monologic",
            file=sys.stderr,
        )
        ann["mode_turn"] = "monologic"

    if ann.get("field_activity") not in VALID_FIELD_ACTIVITY:
        print(
            f"  Invalid field_activity '{ann.get('field_activity')}' on line {ann.get('line')}, "
            f"defaulting to explaining",
            file=sys.stderr,
        )
        ann["field_activity"] = "explaining"

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
    """Build user message for one chunk."""
    parts = []
    parts.append(f"URL: {url}")
    parts.append(f"Section: {section}")

    # Document-level context
    if doc_preamble:
        chunk_nums = {lines[i]["num"] for i in range(chunk_start, chunk_end)}
        preamble_filtered = []
        for pline in doc_preamble.split("\n"):
            m = re.match(r"\[(\d+)\]", pline)
            if m and int(m.group(1)) in chunk_nums:
                continue
            preamble_filtered.append(pline)
        if preamble_filtered:
            parts.append("")
            parts.append("[DOCUMENT OPENING — for context only, do not classify]")
            parts.extend(preamble_filtered)

    # Context before
    ctx_start = max(0, chunk_start - CONTEXT_LINES)
    if ctx_start < chunk_start:
        parts.append("")
        parts.append("[CONTEXT — do not classify]")
        for i in range(ctx_start, chunk_start):
            parts.append(f"[{lines[i]['num']}] {truncate(lines[i]['text'])}")

    # Lines to classify
    parts.append("")
    parts.append("[CLASSIFY THESE LINES]")
    for i in range(chunk_start, chunk_end):
        if is_structural(lines[i]["text"]):
            parts.append(
                f"[{lines[i]['num']}] {truncate(lines[i]['text'])}  "
                f"{{structural — do not classify}}"
            )
        else:
            parts.append(f"[{lines[i]['num']}] {truncate(lines[i]['text'])}")

    # Context after
    ctx_end = min(len(lines), chunk_end + CONTEXT_LINES)
    if chunk_end < ctx_end:
        parts.append("")
        parts.append("[CONTEXT — do not classify]")
        for i in range(chunk_end, ctx_end):
            parts.append(f"[{lines[i]['num']}] {truncate(lines[i]['text'])}")

    return "\n".join(parts)


def parse_json_response(text):
    """Extract JSON array from LLM response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Strip <think>...</think> tags some models produce
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Find the outermost JSON array
    bracket_start = text.find("[")
    if bracket_start == -1:
        raise json.JSONDecodeError("No JSON array found in response", text, 0)

    candidate = text[bracket_start:]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Try to salvage truncated JSON
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


# ===================================================================
# CLASSIFICATION LOGIC
# ===================================================================


def classify_section(model, tokenizer, url, section, markdown_text, verbose=False):
    """Classify all lines in one section (main or comments) of a document."""
    lines = parse_lines(markdown_text)
    if not lines:
        return []

    ann_by_line = {}
    doc_preamble = get_doc_preamble(lines)

    for chunk_start in range(0, len(lines), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(lines))

        # Skip chunks that are entirely structural
        classifiable = [
            i
            for i in range(chunk_start, chunk_end)
            if not is_structural(lines[i]["text"])
        ]
        if not classifiable:
            for i in range(chunk_start, chunk_end):
                ann_by_line[lines[i]["num"]] = _structural(lines[i]["num"])
            continue

        prompt = build_user_prompt(
            url, section, lines, chunk_start, chunk_end, doc_preamble
        )
        raw = call_llm(model, tokenizer, prompt, verbose=verbose)

        try:
            result = parse_json_response(raw)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}", file=sys.stderr)
            print(f"  Raw response: {raw[:300]}", file=sys.stderr)
            result = [
                _cannot_rate(lines[i]["num"]) for i in range(chunk_start, chunk_end)
            ]

        for ann in result:
            ann_by_line[ann.get("line")] = ann

        # Check for missing lines and retry once
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
                retry_raw = call_llm(model, tokenizer, retry_prompt, verbose=verbose)
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

    # Build final list
    final = []
    for line in lines:
        if is_structural(line["text"]):
            final.append(_structural(line["num"]))
        elif line["num"] in ann_by_line:
            ann = ann_by_line[line["num"]]
            if section == "comments" and ann.get("mode_medium") == "transcribed":
                ann["mode_medium"] = "written"
            ann = validate_annotation(ann)
            final.append(ann)
        else:
            final.append(_cannot_rate(line["num"]))

    return final


# ===================================================================
# FILE I/O HELPERS
# ===================================================================


def count_lines(path):
    """Count lines in a file without loading it into memory."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def iter_lines(path, start, end):
    """Yield (index, line_string) for lines [start, end) without loading all."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= end:
                break
            if i >= start:
                yield i, line


# ===================================================================
# MAIN
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Classify web document registers using a local LLM on LUMI."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID, help="HuggingFace model ID"
    )
    parser.add_argument(
        "--max-docs", type=int, default=None, help="Max docs to process"
    )
    parser.add_argument(
        "--start-from", type=int, default=0, help="Resume from this doc index"
    )
    parser.add_argument(
        "--verbose-prompts", action="store_true", help="Print prompts to stderr"
    )
    args = parser.parse_args()

    # --- Load model ---
    model, tokenizer = load_model(args.model_id)

    # --- Figure out where to start ---
    print(f"Counting lines in {args.input}...", file=sys.stderr)
    total = count_lines(args.input)
    print(f"Total docs: {total}")

    start = args.start_from
    if os.path.exists(args.output) and start == 0:
        existing = count_lines(args.output)
        if existing > 0:
            start = existing
            print(f"Resuming: {existing} docs already in {args.output}")

    end = total if args.max_docs is None else min(start + args.max_docs, total)

    if start >= end:
        print(f"Nothing to do: start={start}, end={end}")
        return

    print(f"Processing docs {start}–{end - 1} of {total}")

    # --- Process documents ---
    t_start = time.time()

    with open(args.output, "a") as fout:
        for idx, raw_line in iter_lines(args.input, start, end):
            doc_start = time.time()
            row = json.loads(raw_line)
            url = row.get("u", "")

            try:
                main_ann = classify_section(
                    model,
                    tokenizer,
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
                    model,
                    tokenizer,
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
            print(
                f"[{idx}] {doc_time:.1f}s | {done}/{end - start} | "
                f"avg {avg:.1f}s/doc | ETA {remaining / 60:.0f}min | {url[:60]}"
            )

    total_time = time.time() - t_start
    print(
        f"Done. {end - start} docs in {total_time / 60:.1f} min "
        f"({total_time / (end - start):.1f}s/doc avg)"
    )


if __name__ == "__main__":
    main()
