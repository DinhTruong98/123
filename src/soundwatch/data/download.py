"""WP1: download raw datasets into artifacts/data/raw (idempotent, resumable).

- ESC-50: ``git clone --depth 1`` (zip downloads from codeload are blocked in the dev env).
- UrbanSound8K: Zenodo record 1203745, HTTP Range resume, md5 verified, extracted,
  tarball deleted afterwards (unless ``--keep-archives``).
- FSD50K: only with ``--with-fsd50k`` (not implemented in the PoC -> prints a notice).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from soundwatch import paths

ESC50_GIT = "https://github.com/karolpiczak/ESC-50"
US8K_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"
US8K_SIZE = 6023741708
US8K_MD5 = "9aa69802bbf37fb986f71ec1483a196e"
ESC50_N_CLIPS = 2000
US8K_N_CLIPS = 8732


def esc50_dir(raw: Path | None = None) -> Path:
    return (raw or paths.RAW) / "esc50"


def us8k_dir(raw: Path | None = None) -> Path:
    """Directory containing ``metadata/UrbanSound8K.csv`` and ``audio/fold*/``."""
    return (raw or paths.RAW) / "urbansound8k" / "UrbanSound8K"


def esc50_ready(raw: Path | None = None) -> bool:
    d = esc50_dir(raw)
    return (d / "meta" / "esc50.csv").exists() and len(list((d / "audio").glob("*.wav"))) >= ESC50_N_CLIPS


def us8k_ready(raw: Path | None = None) -> bool:
    d = us8k_dir(raw)
    return (d / "metadata" / "UrbanSound8K.csv").exists() and len(
        list((d / "audio").glob("fold*/*.wav"))
    ) >= US8K_N_CLIPS


def download_esc50(raw: Path | None = None) -> Path:
    d = esc50_dir(raw)
    if esc50_ready(raw):
        print(f"[esc50] already present: {d}")
        return d
    if d.exists():
        shutil.rmtree(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    print(f"[esc50] git clone {ESC50_GIT} -> {d}")
    subprocess.run(["git", "clone", "--depth", "1", ESC50_GIT, str(d)], check=True)
    shutil.rmtree(d / ".git", ignore_errors=True)
    if not esc50_ready(raw):
        raise RuntimeError("ESC-50 clone incomplete")
    return d


def md5_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fetch_resumable(url: str, dest: Path, total: int | None = None, chunk: int = 1 << 20) -> None:
    """Download ``url`` to ``dest`` resuming a partial ``dest.part`` via HTTP Range."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(20):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have >= total:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored Range -> restart
                    have = 0
                mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    done, last = have, have
                    while b := r.read(chunk):
                        f.write(b)
                        done += len(b)
                        if done - last >= 256 << 20:
                            pct = f" ({100 * done / total:.0f}%)" if total else ""
                            print(f"  {done / 2**30:.2f} GiB{pct}", flush=True)
                            last = done
            if total is None or part.stat().st_size >= total:
                break
        except OSError as e:  # includes URLError, timeouts, resets
            print(f"  download interrupted ({e!r}); retry {attempt + 1}", flush=True)
    else:
        raise RuntimeError(f"failed to download {url}")
    part.rename(dest)


def download_us8k(raw: Path | None = None, keep_archive: bool = False) -> Path:
    d = us8k_dir(raw)
    if us8k_ready(raw):
        print(f"[us8k] already present: {d}")
        return d
    base = d.parent
    base.mkdir(parents=True, exist_ok=True)
    tgz = base / "UrbanSound8K.tar.gz"
    if not (tgz.exists() and tgz.stat().st_size == US8K_SIZE):
        print(f"[us8k] downloading {US8K_URL} ({US8K_SIZE / 2**30:.1f} GiB, resumable)")
        fetch_resumable(US8K_URL, tgz, total=US8K_SIZE)
    print("[us8k] verifying md5 ...")
    got = md5_file(tgz)
    if got != US8K_MD5:
        tgz.unlink()
        raise RuntimeError(f"UrbanSound8K md5 mismatch: {got} != {US8K_MD5} (archive deleted, rerun)")
    print("[us8k] extracting ...")
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(base, filter="data")
    if not us8k_ready(raw):
        raise RuntimeError("UrbanSound8K extraction incomplete")
    if not keep_archive:
        tgz.unlink()
    return d


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action="store_true", help="ESC-50 only")
    parser.add_argument("--with-fsd50k", action="store_true", help="also FSD50K (not implemented in PoC)")
    parser.add_argument("--keep-archives", action="store_true", help="keep the US8K tarball after extraction")


def main(args: argparse.Namespace) -> int:
    download_esc50()
    if not args.quick:
        download_us8k(keep_archive=args.keep_archives)
    if args.with_fsd50k:
        print("[fsd50k] not implemented in the PoC (dataset too large); skipped")
    return 0
