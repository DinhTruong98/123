"""Rebuild the VN sound dataset (vn-sound-data) from its manifest: download every source again and cut
the exact clips listed, no searching or relabelling.

The audio itself is never committed (YouTube clips are for internal coursework training only, see
data/vn/SOURCES.md). Only data/vn/manifest.csv is in the repo: one row per clip with source URL,
start/end (seconds in the original video), label, split (grouped by channel) and metadata.

    uv run --with yt-dlp python scripts/rebuild_vn_data.py --out ../vn-sound-data

Needs ffmpeg on PATH. Resumable: clips already on disk are skipped. YouTube sources that ask to sign in
or confirm "not a bot" are skipped (no cookies, no login); the script stops after a few such answers in
a row (IP rate limit: wait and rerun). Output: <out>/<class_key>/<file>.wav + .json, <out>/manifest.csv
(only clips that were rebuilt), <out>/rebuild_report.json (missing sources, counts per class/split).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

SR = 16000
ROOT = Path(__file__).resolve().parents[1]
JSON_FIELDS = [
    "class_key",
    "source_url",
    "platform",
    "source_id",
    "uploader",
    "channel",
    "license",
    "start_s",
    "end_s",
    "duration_s",
    "vehicle_type",
    "passing",
    "scene",
    "distance",
    "label_confidence",
    "split",
    "group_key",
    "title",
    "notes",
]
_bot_hits: list[str] = []


def _decode(src: Path, dst: Path, seconds: float | None) -> None:
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src)]
    if seconds:
        cmd += ["-t", str(seconds)]
    subprocess.run(cmd + ["-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(dst)], check=True)


def fetch_youtube(vid: str, tmp: Path, max_end: float) -> Path | None:
    if len(_bot_hits) >= 4:
        return None
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--no-playlist",
        "-f",
        "249/250/139/bestaudio",
        "--retries",
        "3",
        "--sleep-requests",
        "1",
        "--sleep-interval",
        "2",
        "--max-sleep-interval",
        "5",
        "-o",
        str(tmp / f"{vid}.%(ext)s"),
        f"https://www.youtube.com/watch?v={vid}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    got = [p for p in tmp.glob(f"{vid}.*") if p.suffix in (".webm", ".m4a", ".opus", ".mp4")]
    if not got:
        if "not a bot" in r.stderr:
            _bot_hits.append(vid)
        print(f"  ! yt {vid}: {r.stderr.strip()[-160:]}", flush=True)
        return None
    _bot_hits.clear()
    wav = tmp / f"{vid}.wav"
    _decode(got[0], wav, max_end + 1)
    got[0].unlink()
    return wav


def fetch_freesound(page_url: str, sid: str, tmp: Path) -> Path | None:
    """Public HQ preview (no account): the page carries data-mp3=".../<id>_<user>-lq.mp3"."""
    req = lambda u: urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})  # noqa: E731
    try:
        page = urllib.request.urlopen(req(page_url), timeout=60).read().decode("utf-8", "replace")
        m = re.search(r'data-mp3="([^"]+)"', page)
        if not m:
            raise RuntimeError("no preview url on page")
        mp3 = tmp / f"fs-{sid}.mp3"
        mp3.write_bytes(
            urllib.request.urlopen(req(m.group(1).replace("-lq.mp3", "-hq.mp3")), timeout=180).read()
        )
    except Exception as e:  # noqa: BLE001
        print(f"  ! fs {sid}: {e!r}", flush=True)
        return None
    wav = tmp / f"fs-{sid}.wav"
    _decode(mp3, wav, 600)  # the original build kept the first 10 minutes of each preview
    mp3.unlink()
    return wav


def rebuild_source(
    source_id: str, rows: list[dict], out: Path, tmp_root: Path
) -> tuple[str, int, str | None]:
    todo = [r for r in rows if not (out / r["file"]).exists()]
    if not todo:
        return source_id, len(rows), None
    tmp = Path(tempfile.mkdtemp(dir=tmp_root))
    try:
        platform, ident = source_id.split("-", 1)
        if platform == "yt":
            wav = fetch_youtube(ident, tmp, max(float(r["end_s"]) for r in todo))
        else:
            wav = fetch_freesound(rows[0]["source_url"], ident, tmp)
        if wav is None:
            return source_id, len(rows) - len(todo), "download failed"
        x, sr = sf.read(wav, dtype="int16")
        assert sr == SR
        for r in todo:
            a, b = int(float(r["start_s"]) * SR), int(float(r["end_s"]) * SR)
            if b > len(x):
                print(f"  ! {r['file']}: source shorter than manifest ({len(x) / SR:.1f}s)", flush=True)
                continue
            dst = out / r["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, x[a:b], SR, subtype="PCM_16")
            meta = {k: r.get(k) for k in JSON_FIELDS}
            meta["passing"] = r.get("passing") == "True"
            for k in ("start_s", "end_s", "duration_s"):
                meta[k] = float(r[k])
            dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        done = sum((out / r["file"]).exists() for r in rows)
        return source_id, done, None if done == len(rows) else "some clips missing"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=str(ROOT / "data" / "vn" / "manifest.csv"))
    ap.add_argument("--out", default=str(ROOT.parent / "vn-sound-data"))
    ap.add_argument("--workers", type=int, default=3, help="parallel sources (YouTube rate-limits above ~3)")
    ap.add_argument("--only", default=None, help="comma list of class_key to rebuild")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["class_key"] in keep]
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source_id"]].append(r)
    print(f"{len(rows)} clips from {len(by_src)} sources -> {out}", flush=True)
    tmp_root = out / "_tmp"
    tmp_root.mkdir(exist_ok=True)
    failed = {}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, (sid, done, err) in enumerate(
            ex.map(lambda kv: rebuild_source(kv[0], kv[1], out, tmp_root), by_src.items()), 1
        ):
            if err:
                failed[sid] = {"error": err, "clips_ok": done, "clips": len(by_src[sid])}
            print(
                f"[{i}/{len(by_src)}] {sid}: {done}/{len(by_src[sid])} {err or ''} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    shutil.rmtree(tmp_root, ignore_errors=True)

    ok = [r for r in rows if (out / r["file"]).exists()]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(ok)
    want = Counter((r["class_key"], r["split"]) for r in rows)
    got = Counter((r["class_key"], r["split"]) for r in ok)
    report = {
        "clips_expected": len(rows),
        "clips_rebuilt": len(ok),
        "per_class_split": {f"{c}/{s}": [got[(c, s)], n] for (c, s), n in sorted(want.items())},
        "passing_siren": [
            sum(r["passing"] == "True" for r in ok if r["class_key"] == "siren"),
            sum(r["passing"] == "True" for r in rows if r["class_key"] == "siren"),
        ],
        "failed_sources": failed,
        "stopped_on_bot_check": len(_bot_hits) >= 4,
    }
    (out / "rebuild_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "failed_sources"}, indent=1))
    print(f"{len(failed)} sources failed (see rebuild_report.json)")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
