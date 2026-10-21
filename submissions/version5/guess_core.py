"""Claude's guess of a subject's success rate on a benchmark (option C), standard library only.

The request shows the subject's recorded fields, exactly as the platform passes them, and one
task of the benchmark (the item being predicted when the guess is first needed); it names no
benchmark and contains no labels. One guess is kept per subject and benchmark.
"""

import hashlib
import json

# The subject fields the platform passes, in the order of paiec.data.SUBJECT_FIELDS.
FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
          "reasoning_effort", "harness_version", "subject_features_extra")
EXAMPLE_CHARS = 1500  # longer tasks show their first and last 750 characters

SYSTEM = """You estimate how well AI systems perform on benchmarks. An AI system is a language \
model, possibly run inside an agent harness, with its settings. You will see the details recorded \
for one AI system (some fields may be empty) and one example task from one benchmark.

Estimate the percentage of this benchmark's tasks that this AI system completes correctly, from \
0 (none) to 100 (all). Use what you know about the system: its developer, release date, size, \
harness and settings. Judge how hard the benchmark is from the example task. Do not try to solve \
the task.

Reply with only the integer percentage."""

SCHEMA = {"type": "object", "properties": {"success_percent": {"type": "integer"}},
          "required": ["success_percent"], "additionalProperties": False}


def subject_text(subject):
    lines = [f"{field}: {subject.get(field, '')}" for field in FIELDS if str(subject.get(field, "")).strip()]
    return "\n".join(lines) or "(no details recorded)"


def shown(content):
    if len(content) <= EXAMPLE_CHARS:
        return content
    half = EXAMPLE_CHARS // 2
    return content[:half] + "\n[...]\n" + content[-half:]


def guess_key(subject, item):
    """One key per subject (all its fields) and benchmark."""
    text = json.dumps({f: str(subject.get(f, "")) for f in FIELDS}, sort_keys=True)
    text += "\x00" + str(item.get("benchmark_id", ""))
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def prompt(subject, item):
    features = item.get("item_features", "") or "(none)"
    return (f"AI system:\n{subject_text(subject)}\n\nExample task from the benchmark:\n"
            f"Features: {features}\n{shown(str(item.get('item_content', '')))}")


def request_args(subject, item, settings):
    """The guess request, as keyword arguments of client.beta.messages.create."""
    return dict(
        model=settings["model"],
        max_tokens=4000,
        betas=["server-side-fallback-2026-06-01"],
        fallbacks=[{"model": "claude-opus-4-8"}],
        output_config={"effort": settings["effort"], "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt(subject, item)}],
    )


def parse(text, stop_reason):
    """The guess as a probability in [0, 1], or None for a refusal or a malformed reply."""
    if stop_reason == "refusal":
        return None
    try:
        value = int(json.loads(text)["success_percent"])
    except Exception:
        return None
    return value / 100 if 0 <= value <= 100 else None
