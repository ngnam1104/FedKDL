"""Utilities for measuring encoded image payloads from dataset files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def list_unique_image_files(image_dir: str | Path) -> list[Path]:
    """Return one deterministic entry per image file below ``image_dir``.

    Filtering by ``Path.suffix`` avoids the duplicate matches produced on
    case-insensitive filesystems when separate ``*.jpg`` and ``*.JPG`` glob
    patterns are combined.
    """
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    image_paths = sorted(
        {
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        },
        key=str,
    )
    if not image_paths:
        raise FileNotFoundError(f"No JPEG/PNG images found below: {root}")
    return image_paths


def image_bytes_by_owner(
    image_paths: Sequence[Path],
    owner_indices: Mapping[int, Iterable[int]],
) -> dict[int, int]:
    """Sum encoded on-disk image bytes for every owner.

    Repeated indices are counted repeatedly because they represent repeated
    local ownership/transmission in the supplied partition snapshot.
    """
    file_sizes = [path.stat().st_size for path in image_paths]
    n_images = len(file_sizes)
    totals: dict[int, int] = {}

    for owner_id, indices in owner_indices.items():
        total = 0
        for index in indices:
            index = int(index)
            if index < 0 or index >= n_images:
                raise IndexError(
                    f"Image index {index} for owner {owner_id} is outside "
                    f"the valid range [0, {n_images - 1}]."
                )
            total += file_sizes[index]
        totals[int(owner_id)] = total

    return totals
