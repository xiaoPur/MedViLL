#!/usr/bin/env python3
"""Prepare an OpenI-only server layout for MedViLL.

This script reads the repo's OpenI jsonl splits and the raw IU X-ray CSVs,
chooses a frontal image for each study id, and materializes the expected
repo layout:

    data/preprocessed/openi/{train,valid,test}/{id}.jpg

By default it creates symlinks. Use --copy to copy files instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FRONTAL_HINTS = {
    "pa",
    "p-a",
    "posteroanterior",
    "posterioranterior",
    "posterior-anterior",
    "ap",
    "a-p",
    "anteroposterior",
    "anteriorposterior",
    "anterior-posterior",
    "frontal",
    "front",
}
NON_LATERAL_HINTS = FRONTAL_HINTS | {"cxr", "chest"}


@dataclass(frozen=True)
class StudyImage:
    study_id: str
    projection: str
    source_path: Path
    row: dict[str, str]


def normalize_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text = text.strip("\"'[](){}")
    text = re.sub(r"\.0$", "", text)
    normalized = normalize_text(text)
    if normalized:
        return normalized
    return text.lower()


def safe_int_key(value: object) -> Optional[str]:
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        return str(int(text))
    return None


def score_projection(projection: str) -> tuple[int, int]:
    text = normalize_text(projection)
    if not text:
        return (0, 0)
    if text in FRONTAL_HINTS:
        return (4, 0)
    if any(hint in text for hint in ("pa", "ap", "frontal")):
        return (3, 0)
    if any(hint in text for hint in NON_LATERAL_HINTS):
        return (2, 0)
    if "lateral" in text:
        return (0, 1)
    return (1, 0)


def detect_columns(fieldnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {normalize_text(name): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def row_values(row: dict[str, str], *keys: Optional[str]) -> list[str]:
    values: list[str] = []
    for key in keys:
        if not key:
            continue
        raw = row.get(key)
        if raw is None:
            continue
        raw = str(raw).strip()
        if raw:
            values.append(raw)
    return values


def index_raw_images(images_root: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    if not images_root.exists():
        raise FileNotFoundError(f"Raw image directory not found: {images_root}")

    for path in images_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        candidates = {
            normalize_id(path.name),
            normalize_id(path.stem),
            normalize_id(path.relative_to(images_root).as_posix()),
        }
        for candidate in list(candidates):
            if candidate and candidate not in indexed:
                indexed[candidate] = path
    return indexed


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def build_projection_index(
    projections_csv: Path,
    reports_csv: Optional[Path],
    images_index: dict[str, Path],
) -> dict[str, list[StudyImage]]:
    rows = read_csv_rows(projections_csv)
    report_rows = read_csv_rows(reports_csv) if reports_csv and reports_csv.exists() else []

    all_rows = rows + report_rows
    all_fieldnames = set()
    for row in all_rows:
        all_fieldnames.update(row.keys())

    id_cols = [
        detect_columns(all_fieldnames, candidates)
        for candidates in (
            ("uid", "studyid", "study_id", "studyuid", "study", "id"),
            ("imageid", "image_id", "dicomid", "dicom_id"),
            ("patientid", "patient_id", "subjectid", "subject_id"),
        )
    ]
    id_cols = [col for col in id_cols if col]

    projection_cols = [
        detect_columns(all_fieldnames, candidates)
        for candidates in (
            ("projection", "viewposition", "view", "position"),
            ("imageview", "view_code", "seriesdescription"),
        )
    ]
    projection_cols = [col for col in projection_cols if col]

    path_cols = [
        detect_columns(all_fieldnames, candidates)
        for candidates in (
            ("path", "imagepath", "image_path", "filename", "file", "file_name", "img", "image"),
            ("dicompath", "dicom_path"),
        )
    ]
    path_cols = [col for col in path_cols if col]

    indexed: dict[str, list[StudyImage]] = defaultdict(list)
    for row in rows:
        projection = " ".join(row_values(row, *projection_cols))
        study_ids = extract_candidate_ids(row, id_cols, None)
        path_tokens = row_values(row, *path_cols)
        source_path = resolve_source_path(
            path_tokens,
            study_ids,
            images_index,
            projections_csv.parent,
        )
        if source_path is None:
            continue

        study_ids = extract_candidate_ids(row, id_cols, source_path)
        if not study_ids:
            continue

        for study_id in study_ids:
            indexed[study_id].append(
                StudyImage(
                    study_id=study_id,
                    projection=projection,
                    source_path=source_path,
                    row=row,
                )
            )

    # Some CSVs place the usable image path only in the reports file.
    for row in report_rows:
        study_ids = extract_candidate_ids(row, id_cols, None)
        path_tokens = row_values(row, *path_cols)
        source_path = resolve_source_path(
            path_tokens,
            study_ids,
            images_index,
            projections_csv.parent,
        )
        if source_path is None:
            continue

        study_ids = extract_candidate_ids(row, id_cols, source_path)
        if not study_ids:
            continue

        projection = " ".join(row_values(row, *projection_cols))
        for study_id in study_ids:
            indexed[study_id].append(
                StudyImage(
                    study_id=study_id,
                    projection=projection,
                    source_path=source_path,
                    row=row,
                )
            )

    return indexed


def extract_candidate_ids(
    row: dict[str, str], id_cols: list[Optional[str]], source_path: Optional[Path]
) -> list[str]:
    candidates: list[str] = []
    for key in id_cols:
        if not key:
            continue
        raw = row.get(key)
        if raw is None:
            continue
        normalized = normalize_id(raw)
        if normalized:
            candidates.append(normalized)
        int_key = safe_int_key(raw)
        if int_key:
            candidates.append(normalize_id(int_key))

    if source_path is not None:
        candidates.append(normalize_id(source_path.stem))
        candidates.append(normalize_id(source_path.name))

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def resolve_source_path(
    path_tokens: list[str],
    candidate_ids: list[str],
    images_index: dict[str, Path],
    raw_root: Path,
) -> Optional[Path]:
    for token in path_tokens:
        candidate = Path(token.replace("\\", "/")).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if candidate.exists():
            return candidate.resolve()

    for token in path_tokens:
        candidate = Path(token.replace("\\", "/")).name
        normalized = normalize_id(candidate)
        if normalized in images_index:
            return images_index[normalized]

        stem = normalize_id(Path(candidate).stem)
        if stem in images_index:
            return images_index[stem]

        suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
        for suffix in suffixes:
            lookup = normalize_id(f"{Path(candidate).stem}{suffix}")
            if lookup in images_index:
                return images_index[lookup]

    # Fall back to a basename lookup across the full tree.
    for token in path_tokens:
        normalized = normalize_id(token)
        if normalized in images_index:
            return images_index[normalized]

    for candidate_id in candidate_ids:
        if candidate_id in images_index:
            return images_index[candidate_id]

    # Some IU X-ray exports store just an identifier and keep images under
    # images_normalized. Search by stem against that directory.
    images_dir = raw_root / "images_normalized"
    if images_dir.exists():
        for candidate_id in candidate_ids:
            if not candidate_id:
                continue
            for ext in IMAGE_EXTS:
                candidate = images_dir / f"{candidate_id}{ext}"
                if candidate.exists():
                    return candidate

    return None


def choose_best_image(studies: list[StudyImage]) -> Optional[StudyImage]:
    if not studies:
        return None
    ranked = sorted(
        studies,
        key=lambda item: (
            -score_projection(item.projection)[0],
            score_projection(item.projection)[1],
            normalize_id(item.source_path.name),
        ),
    )
    return ranked[0]


def ensure_link_or_copy(source: Path, target: Path, copy_mode: bool, overwrite: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return "exists"
        if not overwrite:
            return "skipped"
        if target.is_dir():
            raise IsADirectoryError(f"Target exists as a directory: {target}")
        target.unlink()

    if copy_mode:
        shutil.copy2(source, target)
        return "copied"

    try:
        target.symlink_to(source)
        return "linked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def load_split_ids(jsonl_path: Path) -> list[str]:
    ids: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(normalize_id(row["id"]))
    return ids


def process_split(
    split_name: str,
    jsonl_path: Path,
    out_root: Path,
    study_index: dict[str, list[StudyImage]],
    copy_mode: bool,
    overwrite: bool,
) -> dict[str, object]:
    ids = load_split_ids(jsonl_path)
    split_dir = out_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "split": split_name,
        "total_ids": len(ids),
        "linked": 0,
        "copied": 0,
        "skipped": 0,
        "missing": [],
    }

    for study_id in ids:
        candidates = study_index.get(study_id, [])
        choice = choose_best_image(candidates)
        if choice is None:
            stats["missing"].append(study_id)
            continue

        target = split_dir / f"{study_id}.jpg"
        action = ensure_link_or_copy(choice.source_path, target, copy_mode, overwrite)
        if action == "linked":
            stats["linked"] += 1
        elif action == "copied":
            stats["copied"] += 1
        else:
            stats["skipped"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare OpenI images for MedViLL on a server.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/root/autodl-tmp/MedViLL"),
        help="MedViLL repository root.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/root/autodl-tmp/IU x-ray"),
        help="Raw IU X-ray dataset root containing images_normalized and CSVs.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace stale target files if they already exist.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path for a JSON summary manifest.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    raw_root = args.raw_root.resolve()
    images_root = raw_root / "images_normalized"
    projections_csv = raw_root / "indiana_projections.csv"
    reports_csv = raw_root / "indiana_reports.csv"
    out_root = repo_root / "data" / "preprocessed" / "openi"

    if not projections_csv.exists():
        raise FileNotFoundError(f"Missing raw projections CSV: {projections_csv}")
    if not reports_csv.exists():
        raise FileNotFoundError(f"Missing raw reports CSV: {reports_csv}")

    images_index = index_raw_images(images_root)
    study_index = build_projection_index(projections_csv, reports_csv, images_index)

    split_map = {
        "train": repo_root / "data" / "openi" / "Train.jsonl",
        "valid": repo_root / "data" / "openi" / "Valid.jsonl",
        "test": repo_root / "data" / "openi" / "Test.jsonl",
    }

    summary = {
        "repo_root": str(repo_root),
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "splits": [],
        "stats": {"indexed_studies": len(study_index), "image_index_size": len(images_index)},
    }

    for split_name, jsonl_path in split_map.items():
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing OpenI jsonl split: {jsonl_path}")
        stats = process_split(
            split_name=split_name,
            jsonl_path=jsonl_path,
            out_root=out_root,
            study_index=study_index,
            copy_mode=args.copy,
            overwrite=args.overwrite,
        )
        summary["splits"].append(stats)
        missing_count = len(stats["missing"])
        print(
            f"[{split_name}] total={stats['total_ids']} linked={stats['linked']} copied={stats['copied']} "
            f"skipped={stats['skipped']} missing={missing_count}"
        )
        if missing_count:
            preview = ", ".join(stats["missing"][:10])
            print(f"  missing ids: {preview}")

    manifest_path = args.manifest or (repo_root / "scripts" / "openi_prep_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
