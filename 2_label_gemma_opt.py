"""
Register classification of web document segments using a local LLM on LUMI.

Loads Gemma-4-26B-A4B-it (MoE, 26B total / 4B active) once, then classifies
documents from a JSONL file. Much faster than the 31B dense model with
near-identical quality. Writes results one doc at a time so you can resume
by re-running the same command.

Usage:
    python classify_registers_local.py --input sample.jsonl --output annotated.jsonl

Optional:
    --model-id        HuggingFace model ID (default: google/gemma-4-26B-A4B-it)
    --max-docs        Only process first N docs (from the resume point)
    --start-from      Resume from doc index N (0-based)
    --verbose-prompts Print full prompts before each LLM call
    --batch-size      Number of prompts per batched generate call (default: 4)
    --no-compile      Skip torch.compile (useful for debugging)
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

DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"
CHUNK_SIZE = 15  # lines per LLM call
CONTEXT_LINES = 10  # surrounding lines shown as context
MAX_LINE_CHARS = 500  # truncate long lines
MAX_NEW_TOKENS = 1024  # reduced from 2048 — 15 lines of JSON fits in ~400 tokens
DOC_PREAMBLE_LINES = 5  # opening lines sent as document-level context
MAX_CHUNK_RETRIES = 3  # retries per chunk before giving up on the document
DEFAULT_BATCH_SIZE = 4  # prompts per batched generate() call

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
# System prompt (defines the classification task)
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


def load_model(model_id, use_compile=True):
    """Load model and tokenizer onto available GPUs. Returns (model, tokenizer)."""
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    # Ensure pad token is set (needed for batched generation with left-padding)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Building device map...")
    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        meta_model = AutoModelForCausalLM.from_config(config)

    device_map = infer_auto_device_map(meta_model)

    # Vision tower params must land on GPU 0 (they're small but
    # infer_auto_device_map sometimes puts them on cpu/disk).
    for key in list(device_map.keys()):
        if "vision_tower" in key:
            device_map[key] = 0

    print(f"Loading model weights (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    # Show which GPUs are in use
    gpus_used = sorted(set(v for v in device_map.values() if isinstance(v, int)))
    print(f"Model loaded across {len(gpus_used)} GPU(s): {gpus_used}")

    if use_compile:
        print("Compiling model with torch.compile (this may take a few minutes)...")
        try:
            model = torch.compile(model)
            print("torch.compile succeeded")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without compilation")

    return model, tokenizer


# ===================================================================
# LOCAL INFERENCE — BATCHED
# ===================================================================


def call_llm_batch(model, tokenizer, user_prompts, verbose=False):
    """Run a batch of classification prompts through the local model.

    Args:
        user_prompts: list of user prompt strings
    Returns:
        list of response strings, one per prompt
    """
    if not user_prompts:
        return []

    all_messages = []
    for user_prompt in user_prompts:
        all_messages.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

    if verbose:
        for i, msgs in enumerate(all_messages):
            print(f"\n{'=' * 80}", file=sys.stderr)
            print(f">>> BATCH ITEM {i} — USER PROMPT:", file=sys.stderr)
            print(msgs[1]["content"], file=sys.stderr)
            print("=" * 80 + "\n", file=sys.stderr)

    # Tokenize each conversation separately, then batch with left-padding
    tokenized = [
        tokenizer.apply_chat_template(
            msgs,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        for msgs in all_messages
    ]

    # Record individual input lengths (before padding) to slice outputs later
    input_lengths = [t["input_ids"].shape[1] for t in tokenized]

    # Left-pad to the longest sequence in the batch
    max_len = max(input_lengths)
    padded_input_ids = []
    padded_attention_mask = []

    for t in tokenized:
        ids = t["input_ids"].squeeze(0)  # (seq_len,)
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)
            ids = torch.cat([pad_ids, ids])
            mask = torch.cat(
                [
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.ones(t["input_ids"].shape[1], dtype=torch.long),
                ]
            )
        else:
            mask = torch.ones(ids.shape[0], dtype=torch.long)
        padded_input_ids.append(ids)
        padded_attention_mask.append(mask)

    batch_input_ids = torch.stack(padded_input_ids).to(model.device)
    batch_attention_mask = torch.stack(padded_attention_mask).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode each sequence, slicing off the (padded) input portion
    responses = []
    for i in range(len(user_prompts)):
        # The input was left-padded to max_len, so generated tokens start at max_len
        gen_ids = output_ids[i][max_len:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        responses.append(response)

    return responses


# Keep single-prompt call as a convenience wrapper
def call_llm(model, tokenizer, user_prompt, verbose=False):
    """Run one classification prompt through the local model."""
    return call_llm_batch(model, tokenizer, [user_prompt], verbose=verbose)[0]


# ===================================================================
# HELPERS
# ===================================================================


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
        "mode_medium": "cannot_rate",
        "mode_turn": None,
        "field_activity": None,
        "tenor_formality": None,
        "is_structural": True,
    }


def validate_annotation(ann):
    """Validate an annotation dict.

    Returns the annotation (possibly with nulls normalised for cannot_rate),
    or None if any field has an invalid value.
    """
    ann = dict(ann)

    if ann.get("mode_medium") not in VALID_MODE_MEDIUM:
        print(
            f"  Invalid mode_medium '{ann.get('mode_medium')}' on line {ann.get('line')}",
            file=sys.stderr,
        )
        return None

    if ann["mode_medium"] == "cannot_rate":
        ann["mode_turn"] = None
        ann["field_activity"] = None
        ann["tenor_formality"] = None
        return ann

    if ann.get("mode_turn") not in VALID_MODE_TURN or ann.get("mode_turn") is None:
        print(
            f"  Invalid mode_turn '{ann.get('mode_turn')}' on line {ann.get('line')}",
            file=sys.stderr,
        )
        return None

    if (
        ann.get("field_activity") not in VALID_FIELD_ACTIVITY
        or ann.get("field_activity") is None
    ):
        print(
            f"  Invalid field_activity '{ann.get('field_activity')}' on line {ann.get('line')}",
            file=sys.stderr,
        )
        return None

    if (
        ann.get("tenor_formality") not in VALID_TENOR_FORMALITY
        or ann.get("tenor_formality") is None
    ):
        print(
            f"  Invalid tenor_formality '{ann.get('tenor_formality')}' on line {ann.get('line')}",
            file=sys.stderr,
        )
        return None

    return ann


def parse_lines(markdown_text):
    if not markdown_text or not markdown_text.strip():
        return []
    lines = []
    for raw in markdown_text.split("\n"):
        m = re.match(r"\[(\d+)\]\s*(.*)", raw)
        if m:
            lines.append({"num": int(m.group(1)), "text": m.group(2)})
    return lines


def truncate(text, max_chars=MAX_LINE_CHARS):
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def is_structural(text):
    t = text.strip()
    return t.startswith("# ") or t.startswith("TABLE: ") or t.startswith("CODE: ")


def get_doc_preamble(lines, max_lines=DOC_PREAMBLE_LINES):
    selected = []
    for line in lines:
        if not is_structural(line["text"]) and line["text"].strip():
            selected.append(f"[{line['num']}] {truncate(line['text'])}")
            if len(selected) >= max_lines:
                break
    return "\n".join(selected)


def build_user_prompt(url, section, lines, chunk_start, chunk_end, doc_preamble=""):
    parts = [f"URL: {url}", f"Section: {section}"]

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

    ctx_start = max(0, chunk_start - CONTEXT_LINES)
    if ctx_start < chunk_start:
        parts.append("")
        parts.append("[CONTEXT — do not classify]")
        for i in range(ctx_start, chunk_start):
            parts.append(f"[{lines[i]['num']}] {truncate(lines[i]['text'])}")

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

    ctx_end = min(len(lines), chunk_end + CONTEXT_LINES)
    if chunk_end < ctx_end:
        parts.append("")
        parts.append("[CONTEXT — do not classify]")
        for i in range(chunk_end, ctx_end):
            parts.append(f"[{lines[i]['num']}] {truncate(lines[i]['text'])}")

    return "\n".join(parts)


def parse_json_response(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    bracket_start = text.find("[")
    if bracket_start == -1:
        raise json.JSONDecodeError("No JSON array found in response", text, 0)

    candidate = text[bracket_start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    last_brace = candidate.rfind("}")
    if last_brace > 0:
        truncated = candidate[: last_brace + 1]
        if not truncated.rstrip().endswith("]"):
            truncated += "]"
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
# CLASSIFICATION LOGIC — BATCHED
# ===================================================================


class ClassificationFailed(Exception):
    """Raised when a chunk cannot be classified after all retries."""

    pass


def _process_chunk_response(raw_response, expected_lines, section):
    """Parse one LLM response and return (valid_dict, still_needed_set).

    valid_dict: {line_num: validated_annotation}
    still_needed: set of line numbers that didn't get a valid annotation
    """
    valid = {}
    try:
        result = parse_json_response(raw_response)
    except Exception as e:
        print(f"  JSON parse failed: {e}", file=sys.stderr)
        return valid, set(expected_lines)

    for ann in result:
        line_num = ann.get("line")
        if line_num not in expected_lines:
            continue

        if section == "comments" and ann.get("mode_medium") == "transcribed":
            ann["mode_medium"] = "written"

        checked = validate_annotation(ann)
        if checked is not None:
            valid[line_num] = checked
        else:
            print(f"  Line {line_num}: invalid annotation", file=sys.stderr)

    still_needed = expected_lines - set(valid.keys())
    return valid, still_needed


def classify_section(
    model,
    tokenizer,
    url,
    section,
    markdown_text,
    batch_size=DEFAULT_BATCH_SIZE,
    verbose=False,
):
    """Classify all lines in one section (main or comments) using batched inference.

    Raises ClassificationFailed if any chunk fails after all retries.
    """
    lines = parse_lines(markdown_text)
    if not lines:
        return []

    doc_preamble = get_doc_preamble(lines)

    # ------------------------------------------------------------------
    # Phase 1: Build all chunk metadata
    # ------------------------------------------------------------------
    chunks = []  # list of dicts with chunk info
    for chunk_start in range(0, len(lines), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(lines))

        classifiable = [
            i
            for i in range(chunk_start, chunk_end)
            if not is_structural(lines[i]["text"])
        ]
        if not classifiable:
            # All-structural chunk — no LLM call needed
            chunks.append(
                {
                    "start": chunk_start,
                    "end": chunk_end,
                    "expected": set(),
                    "all_structural": True,
                }
            )
        else:
            expected = {
                lines[i]["num"]
                for i in range(chunk_start, chunk_end)
                if not is_structural(lines[i]["text"])
            }
            chunks.append(
                {
                    "start": chunk_start,
                    "end": chunk_end,
                    "expected": expected,
                    "all_structural": False,
                }
            )

    # ------------------------------------------------------------------
    # Phase 2: Batched first-pass classification
    # ------------------------------------------------------------------
    ann_by_line = {}  # line_num -> annotation (accumulates across attempts)

    # Tag structural chunks immediately
    for chunk in chunks:
        if chunk["all_structural"]:
            for i in range(chunk["start"], chunk["end"]):
                ann_by_line[lines[i]["num"]] = _structural(lines[i]["num"])

    # Collect prompts for non-structural chunks
    pending_chunks = [c for c in chunks if not c["all_structural"]]

    # Build prompts for all pending chunks
    for chunk in pending_chunks:
        chunk["prompt"] = build_user_prompt(
            url, section, lines, chunk["start"], chunk["end"], doc_preamble
        )
        chunk["still_needed"] = set(chunk["expected"])
        chunk["attempts"] = 0

    # Process in batches
    def _run_batch(work_items):
        """Send a batch of (chunk, prompt) pairs through the LLM."""
        prompts = [item["prompt"] for item in work_items]
        responses = call_llm_batch(model, tokenizer, prompts, verbose=verbose)

        for item, raw in zip(work_items, responses):
            item["attempts"] += 1
            valid, still_needed = _process_chunk_response(
                raw, item["still_needed"], section
            )
            ann_by_line.update(valid)
            item["still_needed"] = still_needed

    # First pass: all pending chunks
    for batch_start in range(0, len(pending_chunks), batch_size):
        batch = pending_chunks[batch_start : batch_start + batch_size]
        _run_batch(batch)

    # ------------------------------------------------------------------
    # Phase 3: Retries for chunks with missing lines
    # ------------------------------------------------------------------
    for attempt in range(2, MAX_CHUNK_RETRIES + 1):
        # Find chunks that still have missing lines
        retry_chunks = [c for c in pending_chunks if c["still_needed"]]
        if not retry_chunks:
            break

        print(
            f"  Retry attempt {attempt}/{MAX_CHUNK_RETRIES}: "
            f"{len(retry_chunks)} chunks need retries",
            file=sys.stderr,
        )

        # Rebuild prompts narrowed to just the missing lines
        for chunk in retry_chunks:
            needed_indices = [
                i
                for i in range(chunk["start"], chunk["end"])
                if lines[i]["num"] in chunk["still_needed"]
            ]
            p_start = needed_indices[0]
            p_end = needed_indices[-1] + 1
            chunk["prompt"] = build_user_prompt(
                url, section, lines, p_start, p_end, doc_preamble
            )

        for batch_start in range(0, len(retry_chunks), batch_size):
            batch = retry_chunks[batch_start : batch_start + batch_size]
            _run_batch(batch)

    # ------------------------------------------------------------------
    # Phase 4: Check for failures
    # ------------------------------------------------------------------
    failed_lines = set()
    for chunk in pending_chunks:
        if chunk["still_needed"]:
            failed_lines |= chunk["still_needed"]

    if failed_lines:
        raise ClassificationFailed(
            f"Lines {sorted(failed_lines)} failed after {MAX_CHUNK_RETRIES} attempts"
        )

    # ------------------------------------------------------------------
    # Phase 5: Assemble results in line order
    # ------------------------------------------------------------------
    final = []
    for line in lines:
        if is_structural(line["text"]):
            final.append(_structural(line["num"]))
        else:
            final.append(ann_by_line[line["num"]])

    return final


# ===================================================================
# FILE I/O
# ===================================================================


def count_lines(path):
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def iter_lines(path, start, end):
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--verbose-prompts", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of prompts per batched generate() call (default: 4)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (useful for debugging or if it causes issues)",
    )
    args = parser.parse_args()

    # --- Load model ---
    model, tokenizer = load_model(args.model_id, use_compile=not args.no_compile)

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

    print(
        f"Processing docs {start}–{end - 1} of {total} (batch_size={args.batch_size})"
    )

    # --- Process documents ---
    t_start = time.time()
    n_failed = 0

    with open(args.output, "a") as fout:
        for idx, raw_line in iter_lines(args.input, start, end):
            doc_start = time.time()
            row = json.loads(raw_line)
            url = row.get("u", "")
            failed = False

            try:
                main_ann = classify_section(
                    model,
                    tokenizer,
                    url,
                    "main",
                    row.get("markdown_main", ""),
                    batch_size=args.batch_size,
                    verbose=args.verbose_prompts,
                )
            except ClassificationFailed as e:
                print(f"  SKIPPING doc [{idx}]: main section: {e}", file=sys.stderr)
                failed = True
            except Exception as e:
                print(
                    f"  SKIPPING doc [{idx}]: unexpected error in main: {e}",
                    file=sys.stderr,
                )
                failed = True

            if not failed:
                try:
                    comments_ann = classify_section(
                        model,
                        tokenizer,
                        url,
                        "comments",
                        row.get("markdown_comments", ""),
                        batch_size=args.batch_size,
                        verbose=args.verbose_prompts,
                    )
                except ClassificationFailed as e:
                    print(
                        f"  SKIPPING doc [{idx}]: comments section: {e}",
                        file=sys.stderr,
                    )
                    failed = True
                except Exception as e:
                    print(
                        f"  SKIPPING doc [{idx}]: unexpected error in comments: {e}",
                        file=sys.stderr,
                    )
                    failed = True

            if failed:
                row["classification_failed"] = True
                row["llm_register_annotation"] = None
                n_failed += 1
            else:
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
            status = "FAILED" if failed else "ok"
            print(
                f"[{idx}] {doc_time:.1f}s {status} | {done}/{end - start} | "
                f"avg {avg:.1f}s/doc | ETA {remaining / 60:.0f}min | {url[:60]}"
            )

    total_time = time.time() - t_start
    print(
        f"\nDone. {end - start} docs in {total_time / 60:.1f} min "
        f"({total_time / (end - start):.1f}s/doc avg)"
    )
    if n_failed:
        print(
            f"{n_failed} docs failed classification. "
            f"Find them with: jq 'select(.classification_failed)' {args.output}"
        )


if __name__ == "__main__":
    main()
