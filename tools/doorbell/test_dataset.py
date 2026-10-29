import csv
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from dataset import FIELDS, load_manifest
from train import choose_threshold, metrics


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "clips.csv"
        self.rows = []
        for split in ("train", "val", "test"):
            for label in ("doorbell", "other"):
                name = f"{split}-{label}"
                with wave.open(str(self.root / (name + ".wav")), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    audio.writeframes(struct.pack("<h", len(self.rows) + 1) * 16000)
                self.rows.append(dict(path=name + ".wav", label=label, source_id=name,
                                      split=split, license="test fixture", subtype="electronic"))

    def save(self):
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(FIELDS))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_valid(self):
        self.save()
        self.assertEqual(len(load_manifest(self.manifest)), 6)

    def test_source_leakage(self):
        self.rows[2]["source_id"] = self.rows[0]["source_id"]
        self.save()
        with self.assertRaisesRegex(ValueError, "Source leakage"):
            load_manifest(self.manifest)

    def test_duplicate_path(self):
        self.rows[1]["path"] = self.rows[0]["path"]
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate audio path"):
            load_manifest(self.manifest)

    def test_duplicate_content(self):
        (self.root / self.rows[1]["path"]).write_bytes((self.root / self.rows[0]["path"]).read_bytes())
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate audio content"):
            load_manifest(self.manifest)

    def test_missing_license(self):
        self.rows[0]["license"] = ""
        self.save()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            load_manifest(self.manifest)

    def test_missing_negative_test_data(self):
        self.rows.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "Each split"):
            load_manifest(self.manifest)

    def test_wrong_sample_rate(self):
        with wave.open(str(self.root / self.rows[0]["path"]), "wb") as audio:
            audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x00\x00" * 16000)
        self.save()
        with self.assertRaisesRegex(ValueError, "16000 Hz"):
            load_manifest(self.manifest)

    def test_metrics_include_threshold_equality(self):
        result = metrics([1, 1, 0, 0], [0.5, 0.1, 0.6, 0.2], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0.5)

    def test_threshold_respects_false_positive_limit(self):
        labels, scores = [1, 1, 0, 0], [0.9, 0.6, 0.7, 0.1]
        threshold = choose_threshold(labels, scores, 0.0)
        self.assertEqual(threshold, 0.9)
        self.assertEqual(metrics(labels, scores, threshold)["false_positive_rate"], 0.0)

    def test_threshold_can_reject_all_when_negative_has_highest_score(self):
        threshold = choose_threshold([1, 0], [0.3, 1.0], 0.0)
        self.assertGreater(threshold, 1.0)


if __name__ == "__main__":
    unittest.main()
