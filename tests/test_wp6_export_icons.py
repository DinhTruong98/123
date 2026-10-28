"""WP6 icon export for firmware (XBM 48x48 1-bit + RGB565 96x96)."""

from __future__ import annotations

import argparse
import re

import numpy as np

from soundwatch.config import load_config
from soundwatch.tools import export_icons as ei


def _arrays(text: str) -> dict[str, bytes]:
    out = {}
    for m in re.finditer(r"static const uint8_t (\w+)\[(\d+)\][^=]*=\s*\{([^}]*)\};", text):
        data = bytes(int(v, 16) for v in re.findall(r"0x([0-9a-fA-F]{2})", m.group(3)))
        assert len(data) == int(m.group(2))
        out[m.group(1)] = data
    return out


def test_export_icons_header_parses(tmp_path):
    cfg = load_config()
    rc = ei.main(argparse.Namespace(config=None, out=str(tmp_path)))
    assert rc == 0
    text = (tmp_path / "icons.h").read_text()
    assert (tmp_path / "icons_preview.png").stat().st_size > 1000
    arrays = _arrays(text)
    icons = sorted({c.icon for c in cfg.classes})
    assert len(arrays) == 2 * len(icons)
    for name, data in arrays.items():
        if name.endswith("_xbm"):
            assert len(data) == 288
            bits = ei.from_xbm(data, 48, 48)
            assert 0.05 < bits.mean() < 0.8  # an actual icon, not blank / solid
        else:
            assert name.endswith("_rgb565") and len(data) == 18432
    # lookup table in config order
    rows = re.findall(r'\{"(\w+)", (\w+)_xbm, (\w+)_rgb565\}', text)
    assert [r[0] for r in rows] == cfg.class_keys
    assert f"#define SW_NUM_CLASSES {cfg.num_classes}" in text


def test_xbm_bit_order_and_rgb565_byte_order():
    a = np.zeros((2, 10), dtype=np.float32)
    a[0, 0] = 1.0  # leftmost pixel of row 0 -> LSB of byte 0
    a[1, 9] = 1.0  # pixel 9 of row 1 -> bit 1 of the 2nd byte of row 1
    data = ei.to_xbm(a)
    assert data == bytes([0x01, 0x00, 0x00, 0x02])
    np.testing.assert_array_equal(ei.from_xbm(data, 10, 2), a >= 0.5)
    px = ei.to_rgb565_be(np.ones((1, 1), np.float32), fg=(255, 0, 0))
    assert px == bytes([0xF8, 0x00])  # red = 0xF800, high byte first
    assert ei.to_rgb565_be(np.ones((1, 1), np.float32), fg=(0, 0, 255)) == bytes([0x00, 0x1F])
