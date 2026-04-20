"""Manifest validation and path utilities."""

import concurrent.futures
import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Dict

from omegaconf import open_dict


def _resolve_manifest_path(manifest_value: str, data_dir: Path) -> Path:
    if not manifest_value:
        raise ValueError("Manifest value is empty")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = data_dir / manifest_value
    return manifest_path.expanduser().resolve()


def _relativize_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def determine_output_path(manifest_path: Path, explicit_output=None) -> Path:
    if explicit_output:
        return Path(explicit_output)
    name = manifest_path.name
    if name.endswith('.validated.json'):
        return manifest_path
    return manifest_path.with_suffix(".validated.json")


def _is_audio_readable(audio_path: Path):
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

    if not audio_path.exists():
        return False, "missing"
    try:
        AudioSegment.from_file(str(audio_path))
    except Exception as exc:
        return False, str(exc)
    return True, None


def validate_manifest_file(
    manifest_path: Path,
    output_path: Path,
    invalid_path=None,
    *,
    workers=None,
):
    """Validate a single manifest by checking audio files are readable.

    Returns (kept_count, dropped_count).
    """
    kept = 0
    dropped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_file = None

    if invalid_path:
        invalid_path = Path(invalid_path).expanduser().resolve()
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_file = invalid_path.open("w", encoding="utf-8")

    max_workers = workers or max(1, (os.cpu_count() or 1))

    with manifest_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        pending: deque = deque()
        executor = None

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
                    line_num, audio_path, error_msg,
                )
                dropped += 1
                if invalid_file:
                    invalid_file.write(json.dumps({
                        "line": line_num, "error": error_msg, "entry": entry,
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
                logging.warning("Skipping line %d: JSON decode error (%s)", line_num, exc)
                dropped += 1
                if invalid_file:
                    invalid_file.write(json.dumps({
                        "line": line_num, "error": f"json:{exc}", "raw": raw_line,
                    }, ensure_ascii=False) + "\n")
                continue

            audio_path_value = entry.get("audio_filepath")
            if not audio_path_value:
                logging.warning("Skipping line %d: missing 'audio_filepath' field", line_num)
                dropped += 1
                if invalid_file:
                    invalid_file.write(json.dumps({
                        "line": line_num, "error": "missing_audio", "entry": entry,
                    }, ensure_ascii=False) + "\n")
                continue

            audio_path = Path(audio_path_value).expanduser().resolve()

            if executor is None:
                pending.append((line_num, entry, audio_path, None, raw_line))
                _flush_one()
            else:
                future = executor.submit(_is_audio_readable, audio_path)
                pending.append((line_num, entry, audio_path, future, raw_line))
                while len(pending) > max_workers * 4:
                    _flush_one()

            if not (line_num % 10000):
                logging.info(
                    "Validation progress: processed=%d kept=%d dropped=%d pending=%d",
                    line_num, kept, dropped, len(pending),
                )

        while pending:
            _flush_one()

        logging.info(
            "Validation complete for %s: total_lines=%d kept=%d dropped=%d",
            manifest_path, total_lines, kept, dropped,
        )

        if executor is not None:
            executor.shutdown(wait=True)

    if invalid_file:
        invalid_file.close()

    return kept, dropped


def validate_manifests(cfg) -> Dict[str, Dict[str, int]]:
    """Validate configured manifests and update config to point at cleaned copies."""
    data_dir = Path(cfg.training.data_dir).expanduser().resolve()
    manifest_specs = []
    workers = cfg.training.get('validation_workers', None)

    train_manifest = cfg.training.get('train_manifest')
    if train_manifest:
        manifest_specs.append(('train_manifest', None, _resolve_manifest_path(train_manifest, data_dir)))

    test_manifest = cfg.training.get('test_manifest')
    if test_manifest:
        manifest_specs.append(('test_manifest', None, _resolve_manifest_path(test_manifest, data_dir)))

    extra_manifests = cfg.training.get('tokenizer_extra_manifests') or []
    for idx, manifest_name in enumerate(extra_manifests):
        try:
            manifest_path = _resolve_manifest_path(manifest_name, data_dir)
        except ValueError:
            continue
        manifest_specs.append(('tokenizer_extra_manifests', idx, manifest_path))

    validated_cache: Dict[Path, dict] = {}
    results: Dict[str, Dict[str, int]] = {}

    for field, index, manifest_path in manifest_specs:
        if manifest_path in validated_cache:
            stats = validated_cache[manifest_path]
        else:
            output_path = determine_output_path(manifest_path)
            invalid_log_path = manifest_path.with_suffix('.invalid.json')
            logging.info(
                "Validating manifest field=%s idx=%s input=%s output=%s",
                field, index, manifest_path, output_path,
            )
            kept, dropped = validate_manifest_file(
                manifest_path, output_path, invalid_log_path, workers=workers,
            )
            stats = {'kept': kept, 'dropped': dropped, 'output_path': output_path}
            validated_cache[manifest_path] = stats

        if index is None:
            with open_dict(cfg):
                cfg.training[field] = _relativize_path(stats['output_path'], data_dir)
        else:
            manifests = list(cfg.training.get(field, []))
            if index < len(manifests):
                manifests[index] = _relativize_path(stats['output_path'], data_dir)
                with open_dict(cfg):
                    cfg.training[field] = manifests

        results[str(manifest_path)] = {'kept': stats['kept'], 'dropped': stats['dropped']}

    return results
