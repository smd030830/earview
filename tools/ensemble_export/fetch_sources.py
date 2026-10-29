"""Fetch pinned official reference code and pretrained weights for export.

Run with build-only `requests` installed. Checkpoint downloads are resumable.
"""
from pathlib import Path
import hashlib
import io
import requests
import time
import zipfile

HERE = Path(__file__).resolve().parent
EFFICIENTAT_REV = "a425fdce92572e602a1d5634799bd9f1f2efa806"
PANNS_REV = "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"


def download(url, name, algorithm=None, expected=None):
    path = HERE / name
    for attempt in range(6):
        if path.exists() and expected and hashlib.new(algorithm, path.read_bytes()).hexdigest() == expected:
            return path
        offset = path.stat().st_size if expected and path.exists() else 0
        try:
            response = requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                                    stream=True, timeout=180)
            response.raise_for_status()
            with path.open("ab" if offset and response.status_code == 206 else "wb") as f:
                for chunk in response.iter_content(256 * 1024):
                    f.write(chunk)
            if expected:
                if hashlib.new(algorithm, path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Checkpoint checksum mismatch: {name}")
            return path
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2)
    raise RuntimeError(name)


def main():
    archive = download(f"https://github.com/fschmid56/EfficientAT/archive/{EFFICIENTAT_REV}.zip", "efficientat.zip")
    source_root = (HERE / "EfficientAT-main").resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            relative = Path(member.filename).parts[1:]
            if not relative or member.is_dir():
                continue
            target = (source_root / Path(*relative)).resolve()
            if not target.is_relative_to(source_root):
                raise ValueError(f"Archive member escapes source directory: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))
    base = f"https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/{PANNS_REV}/pytorch"
    download(f"{base}/models.py", "panns_model_reference.py")
    download(f"{base}/pytorch_utils.py", "pytorch_utils.py")
    download("https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt",
             "mn04_as.pt", "sha256", "899a8c6217063f6941793102a2780b3dd788ab11fce468d9cf5f0577c453321c")
    download("https://zenodo.org/records/3987831/files/MobileNetV2_mAP%3D0.383.pth?download=1",
             "panns_mobilenetv2.pth", "sha256", "22437af9eb5ed1854b5ad285b4a17d5fc566f1a3b23df40db5541f5705ca340f")


if __name__ == "__main__":
    main()
