#!/usr/bin/env python3

"""Validate manifest entries by ensuring audio files are readable."""

import argparse
import concurrent.futures
import json
import logging
import os
from collections import deque
from pathlib import Path

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a cleaned copy of a NeMo-style manifest, keeping only entries whose "
            "audio files exist and can be decoded."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=str,
        help="Path to the source manifest JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional path to write the validated manifest. If omitted, a sibling file "
            "with the suffix '.validated.json' is written."
        ),
    )
    parser.add_argument(
        "--log-invalid",
        type=str,
        default=None,
        help="Optional path to write a JSONL file containing entries that failed validation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity (warnings only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers to use when probing audio files (defaults to CPU count).",
    )
    return parser.parse_args()


def determine_output_path(manifest_path: Path, explicit_output: str | None = None) -> Path:
    if explicit_output:
        return Path(explicit_output)
    name = manifest_path.name
    if name.endswith('.validated.json'):
        return manifest_path
    return manifest_path.with_suffix(".validated.json")


def _is_audio_readable(audio_path: Path) -> tuple[bool, str | None]:
    if not audio_path.exists():
        return False, "missing"

    try:
        AudioSegment.from_file(str(audio_path))
    except Exception as exc:  # noqa: BLE001 - external decoder errors are varied
        return False, str(exc)

    return True, None


def _validate_audio_async(audio_path: Path) -> tuple[bool, str | None]:
    return _is_audio_readable(audio_path)


def validate_manifest(
    manifest_path: Path,
    output_path: Path,
    invalid_path: Path | None,
    *,
    workers: int | None = None,
) -> tuple[int, int]:
    kept = 0
    dropped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_file = None

    if invalid_path:
        invalid_path = invalid_path.expanduser().resolve()
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_file = invalid_path.open("w", encoding="utf-8")

    max_workers = workers or max(1, (os.cpu_count() or 1))

    with manifest_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        pending: deque = deque()
        executor: concurrent.futures.Executor | None = None

        if max_workers > 1:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        total_lines = 0

        def _flush_one():
            nonlocal kept, dropped
            line_meta = pending.popleft()
            if line_meta is None:
                return
            line_num, entry, audio_path, future, raw_line = line_meta
            ok = True
            error_msg = None
            if future is None:
                ok, error_msg = _is_audio_readable(audio_path)
            else:
                ok, error_msg = future.result()

            if not ok:
                logging.warning(
                    "Skipping line %d: audio '%s' invalid (%s)",
                    line_num,
                    audio_path,
                    error_msg,
                )
                dropped += 1
                if invalid_file:
                    invalid_file.write(json.dumps({
                        "line": line_num,
                        "error": error_msg,
                        "entry": entry,
                    }, ensure_ascii=False) + "\n")
            else:
                dst.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept += 1

        for line_num, raw_line in enumerate(src, start=1):
            total_lines = line_num
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logging.warning(
                    "Skipping line %d: JSON decode error (%s)",
                    line_num,
                    exc,
                )
                dropped += 1
                if invalid_file:
                    invalid_file.write(json.dumps({
                        "line": line_num,
                        "error": f"json:{exc}",
                        "raw": raw_line,
                    }, ensure_ascii=False) + "\n")
                continue

            audio_path_value = entry.get("audio_filepath")
            if not audio_path_value:
                logging.warning("Skipping line %d: missing 'audio_filepath' field", line_num)
                dropped += 1
                if invalid_file:
                    invalid_file.write(json.dumps({
                        "line": line_num,
                        "error": "missing_audio",
                        "entry": entry,
                    }, ensure_ascii=False) + "\n")
                continue

            audio_path = Path(audio_path_value).expanduser().resolve()

            if executor is None:
                pending.append((line_num, entry, audio_path, None, raw_line))
                _flush_one()
            else:
                future = executor.submit(_validate_audio_async, audio_path)
                pending.append((line_num, entry, audio_path, future, raw_line))
                # keep queue bounded to avoid unbounded memory in huge manifests
                while len(pending) > max_workers * 4:
                    _flush_one()

            if not (line_num % 10000):
                logging.info(
                    "Validation progress: processed=%d kept=%d dropped=%d pending=%d",
                    line_num,
                    kept,
                    dropped,
                    len(pending),
                )

        # flush remaining pending items preserving order
        while pending:
            _flush_one()

        logging.info(
            "Validation complete for %s: total_lines=%d kept=%d dropped=%d",
            manifest_path,
            total_lines,
            kept,
            dropped,
        )

        if executor is not None:
            executor.shutdown(wait=True)

    if invalid_file:
        invalid_file.close()

    return kept, dropped


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_path = determine_output_path(manifest_path, args.output)
    invalid_path = Path(args.log_invalid) if args.log_invalid else None

    logging.info("Validating manifest: %s", manifest_path)
    kept, dropped = validate_manifest(manifest_path, output_path, invalid_path, workers=args.workers)
    logging.info("Validation complete. Kept=%d, Dropped=%d", kept, dropped)
    logging.info("Validated manifest written to: %s", output_path)
    if invalid_path:
        logging.info("Invalid entries logged to: %s", invalid_path.resolve())


if __name__ == "__main__":
    main()

