"""Validate externally supplied training audio. No microphone or downloads."""
import csv
import hashlib
from pathlib import Path
import wave

FIELDS = {"path", "label", "source_id", "split", "license", "subtype"}


def load_manifest(manifest):
    manifest = Path(manifest).resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not FIELDS.issubset(reader.fieldnames or []):
            raise ValueError("Required columns: " + ", ".join(sorted(FIELDS)))
        rows = list(reader)
    sources, paths, hashes = {}, set(), {}
    for row in rows:
        if any(not (row.get(field) or "").strip() for field in FIELDS):
            raise ValueError("Every required field must be nonempty")
        if row["label"] not in {"doorbell", "other"}:
            raise ValueError("label must be doorbell or other")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        source, split = row["source_id"], row["split"]
        if sources.setdefault(source, split) != split:
            raise ValueError("Source leakage across splits: " + source)
        path = (manifest.parent / row["path"]).resolve()
        if path in paths:
            raise ValueError("Duplicate audio path: " + str(path))
        paths.add(path)
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("Expected mono PCM16 16000 Hz WAV: " + str(path))
            if audio.getnframes() < 15600:
                raise ValueError("Audio must be at least 0.975 seconds: " + str(path))
            pcm = audio.readframes(audio.getnframes())
            if len(pcm) != audio.getnframes() * 2:
                raise ValueError("Truncated WAV: " + str(path))
        digest = hashlib.sha256(pcm).hexdigest()
        if digest in hashes:
            raise ValueError("Duplicate audio content: " + str(path))
        hashes[digest] = split
        row["resolved_path"] = path
        row["sha256"] = digest
    for split in ("train", "val", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {"doorbell", "other"}:
            raise ValueError("Each split needs doorbell and other clips: " + split)
    return rows
