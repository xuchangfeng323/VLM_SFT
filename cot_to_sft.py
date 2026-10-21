#!/usr/bin/env python3
"""Convert teacher CoT JSONL to messages/images SFT JSONL (standard library only).

With --reference-sft, match by normalized text + image filename, preserving
the reference id, prompt and images. Labels come from CoT by default.
Without a reference, build a prompt and optionally remove an image path prefix.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import tempfile


DEFAULT_PROMPT = (
    "Perform grounded MNER by extracting entities from the text.\n"
    "For each entity, output `entity|type|[x1,y1,x2,y2]` if it is visible in the image; "
    "otherwise output `entity|type|None`.\n"
    "Separate multiple results with `;`, or output `None` if no entity is found.\n\n"
    "Use both textual evidence and visual evidence when available.\n\nText: {text}"
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
                yield line_no, row


def require_text(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field!r} must be a nonempty string")
    return value.strip()


def normalize_text(text):
    text = text.replace("<image>", "").strip()
    # Reference prompts place the tweet after a final line beginning with Text:.
    markers = list(re.finditer(r"(?:^|\n)Text:\s*", text))
    if markers:
        text = text[markers[-1].end():]
    return text.strip()


def reference_index(path):
    index = {}
    for line_no, row in read_jsonl(path):
        try:
            messages, images = row["messages"], row["images"]
            if (not isinstance(messages, list) or len(messages) != 2
                    or not all(isinstance(m, dict) for m in messages)
                    or [m.get("role") for m in messages] != ["user", "assistant"]):
                raise ValueError("reference must have one user and one assistant message")
            if (not isinstance(images, list) or len(images) != 1
                    or not isinstance(images[0], str) or not images[0].strip()):
                raise ValueError("reference must have exactly one image")
            require_text(messages[1], "content")
            key = (normalize_text(require_text(messages[0], "content")), Path(images[0]).name)
            if key in index:
                raise ValueError("ambiguous duplicate text/image pair in reference")
            index[key] = row
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not index:
        raise ValueError("reference is empty")
    return index


def convert_row(row, args, references, prompt):
    if row.get("error") not in (None, "", False):
        raise ValueError(f"teacher generation error: {row['error']}")
    text = normalize_text(require_text(row, args.text_field))
    image = require_text(row, args.image_field)
    think = require_text(row, args.think_field)
    # Accept already wrapped reasoning without nesting tags.
    wrapped = re.fullmatch(r"<think>\s*(.*?)\s*</think>", think, re.S)
    if wrapped:
        think = wrapped.group(1).strip()
    if not think or "<think>" in think or "</think>" in think:
        raise ValueError("empty reasoning or malformed/nested think tags")
    cot_answer = require_text(row, args.answer_field)
    different = False
    if references is not None:
        key = (text, Path(image).name)
        if key not in references:
            raise ValueError("no matching reference for text + image filename")
        output = copy.deepcopy(references[key])
        reference_answer = output["messages"][-1]["content"]
        different = cot_answer != reference_answer.strip()
        answer = reference_answer.strip() if args.answer_source == "reference" else cot_answer
    else:
        if args.strip_image_prefix:
            try:
                image = Path(image).relative_to(args.strip_image_prefix).as_posix()
            except ValueError as exc:
                raise ValueError("image is outside --strip-image-prefix") from exc
        output = {
            "messages": [{"role": "user", "content": prompt.replace("{text}", text)},
                         {"role": "assistant", "content": ""}],
            "images": [image],
        }
        if args.id_field in row:
            output["id"] = row[args.id_field]
        answer = cot_answer
    if args.check_images:
        for image_path in output["images"]:
            if not (args.image_root / image_path).is_file():
                raise ValueError(f"image does not exist: {args.image_root / image_path}")
    if args.answer_format == "tagged":
        answer = f"<answer>\n{answer}\n</answer>"
    output["messages"][-1]["content"] = f"<think>\n{think}\n</think>\n{answer}"
    return output, different


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="teacher CoT JSONL")
    parser.add_argument("--output", required=True, type=Path, help="new SFT JSONL")
    parser.add_argument("--reference-sft", type=Path, help="preserve matched reference metadata/prompt")
    parser.add_argument("--answer-source", choices=["cot", "reference"], default="cot")
    parser.add_argument("--answer-format", choices=["tagged", "plain"], default="tagged",
                        help="tagged: <think>...</think> + <answer>...</answer>; plain omits answer tags")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 prompt containing {text}; no reference mode")
    parser.add_argument("--strip-image-prefix", type=Path, help="remove old image root; no reference mode")
    parser.add_argument("--image-root", type=Path, default=Path("."), help="root for relative image checks")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true", help="explicitly skip bad rows; default: abort")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output after successful conversion")
    for name, default in [("text", "tweet_text"), ("image", "image_path"),
                          ("think", "think"), ("answer", "gold_triples"), ("id", "id")]:
        parser.add_argument(f"--{name}-field", default=default)
    args = parser.parse_args()
    if args.answer_source == "reference" and not args.reference_sft:
        parser.error("--answer-source reference requires --reference-sft")
    if args.reference_sft and (args.prompt_file or args.strip_image_prefix):
        parser.error("--prompt-file/--strip-image-prefix apply only without --reference-sft")
    for source in (args.input, args.reference_sft, args.prompt_file):
        if source and args.output.resolve() == source.resolve():
            parser.error("output must not overwrite an input/reference/prompt file")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; use --overwrite to replace")
    references = reference_index(args.reference_sft) if args.reference_sft else None
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    if "{text}" not in prompt:
        parser.error("prompt must contain {text}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = differences = 0
    seen = set()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=f".{args.output.name}.", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            for line_no, row in read_jsonl(args.input):
                try:
                    output, different = convert_row(row, args, references, prompt)
                    if "id" in output:
                        identity = json.dumps(output["id"], sort_keys=True, ensure_ascii=False)
                        if identity in seen:
                            raise ValueError(f"duplicate output id: {output['id']!r}")
                        seen.add(identity)
                except (KeyError, TypeError, ValueError) as exc:
                    if not args.skip_invalid:
                        raise ValueError(f"{args.input}:{line_no}: {exc}") from exc
                    skipped += 1
                    print(f"SKIP {args.input}:{line_no}: {exc}", file=sys.stderr)
                    continue
                stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                written += 1
                differences += int(different)
            if not written:
                raise ValueError("no valid records; output was not published")
        if args.overwrite:
            os.replace(temp_path, args.output)
        else:
            # Atomic no-clobber publication, even if another process created output.
            os.link(temp_path, args.output)
        print(json.dumps({"input": str(args.input), "output": str(args.output),
                          "written": written, "skipped": skipped,
                          "answers_different_from_reference": differences,
                          "answer_source": args.answer_source}, ensure_ascii=False, indent=2))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
