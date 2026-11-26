"""Download and verify the pinned OPUS-MT snapshot for offline inference."""

import argparse
from hashlib import sha1, sha256
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
REPO = "Helsinki-NLP/opus-mt-en-bcl"
REVISION = "5012a13b56eea980f9c393f1e354aa5c3ef00d12"
# The LFS weight hash is SHA-256. The other entries are Git blob object IDs.
FILES = {
    "README.md": (845, "80e517337fd93df9945f3527874851fc907216db"),
    "config.json": (1382, "d30e1a120bd15c54366ade0bde31dbe55b3a1662"),
    "generation_config.json": (293, "d548afd7d2c0584ea2b2197b26d3219ba162e54d"),
    "pytorch_model.bin": (298654617, "3ed444f32c2bf21d934377a78aadad9d8b7e545e31ffbfbb64bb249815b57c81"),
    "source.spm": (821252, "15f17852a0a59efc0ea5e95ec487f80f1af1d38c"),
    "target.spm": (829916, "7e477eb07624c5352ecbf1ebf8610a2ec961a255"),
    "tokenizer_config.json": (43, "3aa07c6234a67cddd69fb2eee52c9acaf379497b"),
    "vocab.json": (1328511, "9738959f9a55f64a2e2c643c5eb290f697b1a362"),
}


def file_hash(size, expected):
    return sha256() if len(expected) == 64 else sha1(f"blob {size}\0".encode("ascii"))


def verified(path, size, expected):
    if not path.is_file() or path.stat().st_size != size:
        return False
    checksum = file_hash(size, expected)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest() == expected


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (size, expected) in FILES.items():
        destination = directory / name
        if verified(destination, size, expected):
            print(f"Verified {name}", flush=True)
            continue
        print(f"Downloading {name} ({size / 1e6:.2f} MB)", flush=True)
        with NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as target:
            temporary = Path(target.name)
            try:
                checksum = file_hash(size, expected)
                count = 0
                url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true"
                with urlopen(url, timeout=90) as stream:
                    while chunk := stream.read(1024 * 1024):
                        count += len(chunk)
                        if count > size:
                            raise ValueError(f"Unexpected size for {name}")
                        checksum.update(chunk)
                        target.write(chunk)
                if count != size or checksum.hexdigest() != expected:
                    raise ValueError(f"Checksum or size mismatch for {name}")
                target.close()
                temporary.replace(destination)
            finally:
                target.close()
                temporary.unlink(missing_ok=True)
    print(f"Verified {REPO}@{REVISION} in {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/marian/opus-mt-en-bcl")
    args = parser.parse_args()
    try:
        download(args.output_dir)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
