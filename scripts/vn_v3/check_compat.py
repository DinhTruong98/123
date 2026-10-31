# ruff: noqa: E501
"""Backward-compatibility check (vn-v3): the default config/model must export exactly the iOS resources in
ios/SoundWatch/Resources (config JSON + demo expected events / per-hop probs). Nothing is written.

usage: uv run python scripts/vn_v3/check_compat.py
"""
import importlib.util
import json
import sys

from soundwatch import paths
from soundwatch.audio import features as F
from soundwatch.config import load_config

spec = importlib.util.spec_from_file_location("ios_export", paths.ROOT / "ios/scripts/export_ios_assets.py")
ios = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ios)
res = paths.ROOT / "ios/SoundWatch/Resources"
stats = F.load_stats(ios.STATS)
cfg = load_config()
ok = True
cur = json.loads(json.dumps(ios.export_config(cfg, stats), ensure_ascii=False))
ref = json.loads((res / "soundwatch_config.json").read_text(encoding="utf-8"))
print("soundwatch_config.json identical:", cur == ref)
ok &= cur == ref
demo = json.loads(json.dumps(ios.export_demo(cfg, stats)))
ref = json.loads((res / "demo_expected_events.json").read_text(encoding="utf-8"))
same_ev = demo["events"] == ref["events"]
same_hops = demo["hops"] == ref["hops"]
print(f"demo events identical: {same_ev} ({len(demo['events'])} events), per-hop probs identical: {same_hops}")
ok &= same_ev and same_hops
sys.exit(0 if ok else 1)
