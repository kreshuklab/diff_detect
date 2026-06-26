from __future__ import annotations

import argparse
import csv
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASET_URL = "https://huggingface.co/datasets/imageomics/Heliconius-Collection_Cambridge-Butterfly"
HELICONIUS_CSV_URL = f"{DATASET_URL}/resolve/main/Heliconius_img_master.csv"
DEFAULT_OUTPUT = Path("data/rounds.json")
DEFAULT_SEEDED_OUTPUT = Path("data/seeded_annotations.json")
IMAGE_FILE_TYPES = {"jpg", "jpeg", "png"}
STRICT_RULE = "strict_same_subspecies"
MIMIC_RULE = "relaxed_same_mimic_group"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate butterfly study rounds from the Heliconius Cambridge dataset.")
    parser.add_argument("--manifest-url", default=HELICONIUS_CSV_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeded-output", type=Path, default=DEFAULT_SEEDED_OUTPUT)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    rows = load_rows(args.manifest_url)
    eligible = eligible_rows(rows)
    rounds = build_rounds(eligible, args.rounds, args.seed)
    if not rounds:
        raise SystemExit(
            "Could not build any rounds. The generator requires non-hybrid same-view mimic groups with "
            "one species+subspecies group containing >=3 images and another species containing >=1 image."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rounds, indent=2) + "\n", encoding="utf-8")
    args.seeded_output.parent.mkdir(parents=True, exist_ok=True)
    args.seeded_output.write_text(json.dumps(build_seeded_annotations(rounds), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rounds)} rounds to {args.output}")
    print(f"Wrote seeded annotations to {args.seeded_output}")


def load_rows(manifest_url: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(manifest_url) as response:
        decoded = (line.decode("utf-8") for line in response)
        return list(csv.DictReader(decoded))


def eligible_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    clean_rows = []
    seen_md5: set[str] = set()
    for row in rows:
        if row.get("hybrid_stat") != "non-hybrid":
            continue
        if row.get("file_type", "").lower() not in IMAGE_FILE_TYPES:
            continue
        if row.get("CAM_dupe") == "True":
            continue
        species = row.get("species", "").strip()
        subspecies = row.get("subspecies", "").strip()
        mimic_group = row.get("mimic_group", "").strip()
        file_url = row.get("file_url", "").strip()
        md5 = row.get("md5", "").strip()
        if not species or not subspecies or not mimic_group or not file_url:
            continue
        if " x " in subspecies.lower() or " x " in species.lower():
            continue
        if md5 and md5 in seen_md5:
            continue
        if md5:
            seen_md5.add(md5)
        clean_rows.append(row)
    return clean_rows


def build_rounds(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rounds: list[dict[str, Any]] = []
    used_image_ids: set[str] = set()

    strict = build_strict_round(rows, len(rounds) + 1, seed, rng)
    if strict:
        rounds.append(strict)
        used_image_ids.update(image["image_id"] for image in strict["images"])

    rounds.extend(build_mimic_group_rounds(rows, count - len(rounds), len(rounds) + 1, seed, rng, used_image_ids))
    return rounds[:count]


def build_strict_round(
    rows: list[dict[str, str]],
    task_index: int,
    seed: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group_key = (row["subspecies"].strip(), row["View"].strip(), row["mimic_group"].strip())
        grouped[group_key][row["species"].strip()].append(row)

    candidates: list[tuple[tuple[str, str, str], str, str]] = []
    for group_key, species_rows in grouped.items():
        for ref_species, ref_rows in species_rows.items():
            if len(ref_rows) < 3:
                continue
            for odd_species, odd_rows in species_rows.items():
                if odd_species != ref_species and odd_rows:
                    candidates.append((group_key, ref_species, odd_species))

    rng.shuffle(candidates)
    for group_key, ref_species, odd_species in candidates:
        subspecies, view, mimic_group = group_key
        ref_pool = grouped[group_key][ref_species]
        odd_pool = grouped[group_key][odd_species]
        if len(ref_pool) < 3 or not odd_pool:
            continue
        refs = rng.sample(ref_pool, 3)
        odd = rng.choice(odd_pool)
        return round_entry(
            task_index=task_index,
            refs=refs,
            odd=odd,
            view=view,
            mimic_group=mimic_group,
            seed=seed,
            rng=rng,
            rule=STRICT_RULE,
        )
    return None


def build_mimic_group_rounds(
    rows: list[dict[str, str]],
    count: int,
    start_index: int,
    seed: int,
    rng: random.Random,
    used_image_ids: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[tuple[str, str], list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group_key = (row["View"].strip(), row["mimic_group"].strip())
        taxon_key = (row["species"].strip(), row["subspecies"].strip())
        grouped[group_key][taxon_key].append(row)

    candidates: list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]] = []
    for group_key, taxa in grouped.items():
        for ref_taxon, ref_rows in taxa.items():
            if len(ref_rows) < 3:
                continue
            for odd_taxon, odd_rows in taxa.items():
                if odd_taxon[0] != ref_taxon[0] and odd_rows:
                    candidates.append((group_key, ref_taxon, odd_taxon))

    rng.shuffle(candidates)
    rounds: list[dict[str, Any]] = []
    for group_key, ref_taxon, odd_taxon in candidates:
        if len(rounds) >= count:
            break
        view, mimic_group = group_key
        ref_pool = [row for row in grouped[group_key][ref_taxon] if image_id(row) not in used_image_ids]
        odd_pool = [row for row in grouped[group_key][odd_taxon] if image_id(row) not in used_image_ids]
        if len(ref_pool) < 3:
            continue
        if not odd_pool:
            continue
        refs = rng.sample(ref_pool, 3)
        odd = rng.choice(odd_pool)
        round_data = round_entry(
            task_index=start_index + len(rounds),
            refs=refs,
            odd=odd,
            view=view,
            mimic_group=mimic_group,
            seed=seed,
            rng=rng,
            rule=MIMIC_RULE,
        )
        rounds.append(round_data)
        used_image_ids.update(image["image_id"] for image in round_data["images"])
    return rounds


def round_entry(
    task_index: int,
    refs: list[dict[str, str]],
    odd: dict[str, str],
    view: str,
    mimic_group: str,
    seed: int,
    rng: random.Random,
    rule: str,
) -> dict[str, Any]:
    images = [image_entry(row, "reference") for row in refs] + [image_entry(odd, "odd")]
    rng.shuffle(images)
    odd_image_id = image_id(odd)
    ref_species = refs[0]["species"]
    ref_subspecies = refs[0]["subspecies"]
    odd_species = odd["species"]
    odd_subspecies = odd["subspecies"]
    all_four_share_subspecies = ref_subspecies == odd_subspecies
    task_prefix = "hf_strict" if rule == STRICT_RULE else "hf_mimic"
    return {
        "task_id": f"{task_prefix}_{task_index:03d}",
        "odd_image_id": odd_image_id,
        "images": images,
        "metadata": {
            "dataset": "imageomics/Heliconius-Collection_Cambridge-Butterfly",
            "dataset_url": DATASET_URL,
            "manifest_url": HELICONIUS_CSV_URL,
            "view": view,
            "generation_seed": seed,
            "generation_rule": (
                "strict same-subspecies non-hybrid round"
                if rule == STRICT_RULE
                else "relaxed non-hybrid round: same view and mimic_group; references share species and subspecies; odd species differs"
            ),
            "round_rule": rule,
            "reference_species": ref_species,
            "reference_subspecies": ref_subspecies,
            "odd_species": odd_species,
            "odd_subspecies": odd_subspecies,
            "mimic_group": mimic_group,
            "hybrid_stat": "non-hybrid",
            "references_share_subspecies": True,
            "all_four_share_subspecies": all_four_share_subspecies,
            "all_four_share_mimic_group": True,
        },
    }


def image_entry(row: dict[str, str], role: str) -> dict[str, Any]:
    row_image_id = image_id(row)
    return {
        "image_id": row_image_id,
        "path": f"data/images/huggingface/{safe_filename(row['filename'])}",
        "source_url": row["file_url"],
        "species_role": role,
        "species": row["species"],
        "subspecies": row["subspecies"],
        "view": row["View"],
        "mimic_group": row["mimic_group"],
        "hybrid_stat": row["hybrid_stat"],
        "source": {
            "camid": row.get("CAMID", ""),
            "filename": row.get("filename", ""),
            "filepath": row.get("filepath", ""),
            "md5": row.get("md5", ""),
            "record_number": row.get("record_number", ""),
            "zenodo_link": row.get("zenodo_link", ""),
        },
    }


def build_seeded_annotations(rounds: list[dict[str, Any]]) -> list[dict[str, str]]:
    labels = {
        "ai": ("color", "#e83e8c"),
        "peer": ("shape", "#111111"),
    }
    annotations = []
    for task in rounds:
        odd_image = next(image for image in task["images"] if image["image_id"] == task["odd_image_id"])
        metadata = task["metadata"]
        for source, (label, color) in labels.items():
            annotations.append(
                {
                    "task_id": task["task_id"],
                    "source": source,
                    "selected_image_id": task["odd_image_id"],
                    "label": label,
                    "explanation": (
                        f"The selected {odd_image['species']} ({odd_image['subspecies']}) differs from "
                        f"the three {metadata['reference_species']} ({metadata['reference_subspecies']}) references."
                    ),
                    "annotation_color": color,
                }
            )
    return annotations


def image_id(row: dict[str, str]) -> str:
    identifier = row.get("filename") or row.get("Image_name") or row.get("X") or row.get("CAMID")
    return safe_id(identifier or "")


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


if __name__ == "__main__":
    main()
