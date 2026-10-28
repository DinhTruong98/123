"""Offline tests for soundwatch.data.remote_zip (split-zip selective extraction)."""

import io
import struct
import zipfile

import numpy as np
import pytest

from soundwatch.data import remote_zip as rz


def _members(rng):
    return {
        "dir/a.wav": rng.integers(0, 256, 5000, dtype=np.uint8).tobytes(),  # incompressible
        "dir/b.txt": b"hello world " * 3000,  # very compressible
        "dir/c.bin": rng.integers(0, 4, 20000, dtype=np.uint8).tobytes(),
        "dir/empty": b"",
    }


def _single_zip(members, stored=("dir/a.wav",)):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in members.items():
            z.writestr(n, d, compress_type=zipfile.ZIP_STORED if n in stored else zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def _to_split(data: bytes, seg: int) -> list[bytes]:
    """Turn a single-disk zip into a PKWARE split archive with segments of ``seg`` bytes:
    spanning marker at the start of disk 0, disk-relative offsets in CD + EOCD."""
    data = struct.pack("<I", 0x08074B50) + data
    eocd = data.rfind(struct.pack("<I", rz.SIG_EOCD))
    _, _, _, _, n, cd_size, cd_off, _ = struct.unpack_from("<IHHHHIIH", data, eocd)
    cd_off += 4
    cd = bytearray(data[cd_off : cd_off + cd_size])
    p = 0
    while p < len(cd):
        nlen, xlen, clen = struct.unpack_from("<HHH", cd, p + 28)
        (off,) = struct.unpack_from("<I", cd, p + 42)
        g = off + 4
        struct.pack_into("<H", cd, p + 34, g // seg)
        struct.pack_into("<I", cd, p + 42, g % seg)
        p += 46 + nlen + xlen + clen
    body = data[:cd_off]
    end = len(body) + len(cd)
    if end // seg != (end + 21) // seg:  # writers never split the EOCD record across disks
        body += b"\0" * (seg - end % seg)
        cd_off = len(body)
    out = body + bytes(cd)
    last = (len(out) + 21) // seg
    eocd_rec = struct.pack("<IHHHHIIH", rz.SIG_EOCD, last, cd_off // seg, n, n, cd_size, cd_off % seg, 0)
    out += eocd_rec
    segs = [out[i : i + seg] for i in range(0, len(out), seg)]
    assert len(segs) == last + 1
    return segs


def test_segment_spans_arithmetic():
    src = rz.SegmentedSource([10, 10, 5], lambda d, o, n: b"")
    assert src.spans(0, 2, 5) == [(0, 2, 5)]
    assert src.spans(0, 8, 5) == [(0, 8, 2), (1, 0, 3)]
    assert src.spans(0, 8, 15) == [(0, 8, 2), (1, 0, 10), (2, 0, 3)]
    assert src.spans(0, 12, 3) == [(1, 2, 3)]  # offset past end of its disk
    with pytest.raises(ValueError):
        src.spans(1, 5, 20)
    segs = [bytes(range(10)), bytes(range(10, 20)), bytes(range(20, 25))]
    s = rz.bytes_source(segs)
    assert s.read(0, 7, 10) == bytes(range(7, 17))
    assert rz.split_segment_names("X", 3) == ["X.z01", "X.z02", "X.zip"]


def test_single_disk_zip():
    members = _members(np.random.default_rng(0))
    z = rz.SplitZip(rz.bytes_source([_single_zip(members)]))
    assert {e.name for e in z.entries} == set(members)
    for n, d in members.items():
        assert z.read(n) == d
    assert z.by_name["dir/a.wav"].method == 0 and z.by_name["dir/b.txt"].method == 8


@pytest.mark.parametrize("seg", [1000, 4096, 7777])
def test_multi_disk_split_zip(seg):
    members = _members(np.random.default_rng(1))
    segs = _to_split(_single_zip(members), seg)
    assert len(segs) >= 2
    z = rz.SplitZip(rz.bytes_source(segs), tail_size=64)
    assert z.info.num_disks == len(segs)
    assert any(e.disk > 0 for e in z.entries)
    for n, d in members.items():
        assert z.read(n, slack=8) == d  # small slack also exercises the exact re-fetch path


def test_crc_mismatch_detected():
    members = {"x.txt": b"abc" * 100}
    data = bytearray(_single_zip(members, stored=("x.txt",)))
    i = data.find(b"abcabc")
    data[i] = ord("z")
    z = rz.SplitZip(rz.bytes_source([bytes(data)]))
    with pytest.raises(rz.ZipFormatError, match="CRC"):
        z.read("x.txt")


def test_zip64_extra_field():
    extra = struct.pack("<HHQQQI", 1, 28, 5_000_000_000, 4_000_000_000, 7, 3)
    assert rz._zip64_extra(extra, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF) == (
        5_000_000_000,
        4_000_000_000,
        7,
        3,
    )
    assert rz._zip64_extra(b"", 10, 9, 8, 0) == (10, 9, 8, 0)
