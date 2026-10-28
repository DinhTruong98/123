"""WP1: read selected members of a (split / multi-disk) ZIP archive over HTTP Range requests.

Used for FSD50K, whose audio is a 6-segment split zip (``.z01`` .. ``.z05`` + ``.zip``, ~17 GB):
only the central directory (at the end of the last segment) and the local header + data of the
members we need are fetched, so nothing close to the full archive ever touches the disk.

Layout of a split archive (PKWARE APPNOTE 8.5): segment ``i`` (``.z01`` = disk 0, ..., ``.zip`` =
last disk) is a plain byte slice; offsets in the central directory are *relative to the start of
the disk* given by "disk number start". Disk 0 may start with the 4-byte spanning marker
``PK\\x07\\x08`` (it is counted in the offsets). A member's data may continue at the start of the
next disk; :class:`SegmentedSource` hides those boundaries.

Supported: compression methods 0 (stored) and 8 (deflate), ZIP64 sizes/offsets/disk numbers,
CRC32 verification. Not supported: encryption.
"""

from __future__ import annotations

import struct
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

SIG_EOCD = 0x06054B50
SIG_ZIP64_EOCD = 0x06064B50
SIG_ZIP64_LOCATOR = 0x07064B50
SIG_CENTRAL = 0x02014B50
SIG_LOCAL = 0x04034B50
EOCD_MIN = 22
EOCD_SEARCH = EOCD_MIN + 0xFFFF  # max comment length
LOCAL_HEADER = 30

Fetch = Callable[[int, int, int], bytes]  # (disk, offset, length) -> bytes (exactly length)


class ZipFormatError(ValueError):
    pass


@dataclass(frozen=True)
class ZipEntry:
    name: str
    disk: int  # disk number start
    offset: int  # local header offset, relative to the start of ``disk``
    compressed_size: int
    size: int
    method: int
    crc: int
    flags: int


@dataclass(frozen=True)
class CentralDirInfo:
    disk: int  # disk where the central directory starts
    offset: int  # relative to that disk
    size: int
    entries: int
    num_disks: int  # total number of disks (= index of last disk + 1)


class SegmentedSource:
    """Random access over consecutive segments (disks) of known sizes."""

    def __init__(self, sizes: Sequence[int], fetch: Fetch):
        if not sizes or any(s <= 0 for s in sizes):
            raise ValueError("segment sizes must be positive")
        self.sizes = list(sizes)
        self.fetch = fetch

    @property
    def num_disks(self) -> int:
        return len(self.sizes)

    def spans(self, disk: int, offset: int, length: int) -> list[tuple[int, int, int]]:
        """Split a read of ``length`` bytes starting at (``disk``, ``offset``) into per-disk ranges."""
        if not 0 <= disk < len(self.sizes) or offset < 0 or length < 0:
            raise ValueError(f"bad read: disk={disk} offset={offset} length={length}")
        out = []
        # an offset may already point past the end of its disk (data starting on the next one)
        while length > 0:
            if disk >= len(self.sizes):
                raise ValueError("read past the end of the last segment")
            avail = self.sizes[disk] - offset
            if avail <= 0:
                offset -= self.sizes[disk]
                disk += 1
                continue
            n = min(avail, length)
            out.append((disk, offset, n))
            length -= n
            disk, offset = disk + 1, 0
        return out

    def read(self, disk: int, offset: int, length: int) -> bytes:
        parts = [self.fetch(d, o, n) for d, o, n in self.spans(disk, offset, length)]
        data = b"".join(parts)
        if len(data) != length:
            raise OSError(f"short read: got {len(data)} of {length} bytes")
        return data


# ---------------------------------------------------------------- parsing


def parse_eocd(tail: bytes, tail_start: int = 0) -> tuple[CentralDirInfo, int | None]:
    """Parse the End Of Central Directory record found in ``tail`` (the last bytes of the last
    disk, starting at ``tail_start`` within it). Returns the info and the position of the record
    inside ``tail`` (a ZIP64 locator, if any, sits in the 20 bytes just before it)."""
    i = tail.rfind(struct.pack("<I", SIG_EOCD))
    while i >= 0 and len(tail) - i < EOCD_MIN:
        i = tail.rfind(struct.pack("<I", SIG_EOCD), 0, i)
    if i < 0:
        raise ZipFormatError("end of central directory record not found")
    _, this_disk, cd_disk, _n_this, n_total, cd_size, cd_off, _clen = struct.unpack_from("<IHHHHIIH", tail, i)
    info = CentralDirInfo(disk=cd_disk, offset=cd_off, size=cd_size, entries=n_total, num_disks=this_disk + 1)
    return info, i


def needs_zip64(info: CentralDirInfo) -> bool:
    return (
        info.disk == 0xFFFF
        or info.offset == 0xFFFFFFFF
        or info.size == 0xFFFFFFFF
        or info.entries == 0xFFFF
        or info.num_disks == 0x10000
    )


def parse_zip64_locator(buf: bytes, pos: int) -> tuple[int, int, int]:
    """ZIP64 EOCD locator at ``buf[pos:pos+20]`` -> (disk of zip64 EOCD, offset in it, total disks)."""
    sig, disk, off, total = struct.unpack_from("<IIQI", buf, pos)
    if sig != SIG_ZIP64_LOCATOR:
        raise ZipFormatError("zip64 end of central directory locator not found")
    return disk, off, total


def parse_zip64_eocd(buf: bytes) -> CentralDirInfo:
    sig, _size, _vm, _vn, this_disk, cd_disk, _n_this, n_total, cd_size, cd_off = struct.unpack_from(
        "<IQHHIIQQQQ", buf, 0
    )
    if sig != SIG_ZIP64_EOCD:
        raise ZipFormatError("bad zip64 end of central directory record")
    return CentralDirInfo(disk=cd_disk, offset=cd_off, size=cd_size, entries=n_total, num_disks=this_disk + 1)


def _zip64_extra(extra: bytes, usize: int, csize: int, offset: int, disk: int) -> tuple[int, int, int, int]:
    p = 0
    while p + 4 <= len(extra):
        hid, ln = struct.unpack_from("<HH", extra, p)
        body = extra[p + 4 : p + 4 + ln]
        if hid == 0x0001:
            q = 0
            if usize == 0xFFFFFFFF:
                (usize,) = struct.unpack_from("<Q", body, q)
                q += 8
            if csize == 0xFFFFFFFF:
                (csize,) = struct.unpack_from("<Q", body, q)
                q += 8
            if offset == 0xFFFFFFFF:
                (offset,) = struct.unpack_from("<Q", body, q)
                q += 8
            if disk == 0xFFFF:
                (disk,) = struct.unpack_from("<I", body, q)
            break
        p += 4 + ln
    return usize, csize, offset, disk


def parse_central_directory(data: bytes) -> list[ZipEntry]:
    out = []
    p = 0
    while p + 46 <= len(data):
        (sig, _vm, _vn, flags, method, _t, _d, crc, csize, usize, nlen, xlen, clen, disk, _ia, _ea, off) = (
            struct.unpack_from("<IHHHHHHIIIHHHHHII", data, p)
        )
        if sig != SIG_CENTRAL:
            break
        raw_name = data[p + 46 : p + 46 + nlen]
        extra = data[p + 46 + nlen : p + 46 + nlen + xlen]
        name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
        usize, csize, off, disk = _zip64_extra(extra, usize, csize, off, disk)
        out.append(ZipEntry(name, disk, off, csize, usize, method, crc, flags))
        p += 46 + nlen + xlen + clen
    return out


def decompress(entry: ZipEntry, data: bytes) -> bytes:
    if entry.flags & 0x1:
        raise ZipFormatError(f"{entry.name}: encrypted members are not supported")
    if entry.method == 0:
        out = data
    elif entry.method == 8:
        d = zlib.decompressobj(-15)
        out = d.decompress(data) + d.flush()
    else:
        raise ZipFormatError(f"{entry.name}: unsupported compression method {entry.method}")
    if len(out) != entry.size:
        raise ZipFormatError(f"{entry.name}: size {len(out)} != {entry.size}")
    if zlib.crc32(out) & 0xFFFFFFFF != entry.crc:
        raise ZipFormatError(f"{entry.name}: CRC32 mismatch")
    return out


# ---------------------------------------------------------------- archive


class SplitZip:
    """Read the central directory and selected members of a (possibly multi-disk) archive."""

    def __init__(self, source: SegmentedSource, tail_size: int = 1 << 16):
        self.source = source
        last = source.num_disks - 1
        n = min(source.sizes[last], max(tail_size, EOCD_SEARCH))
        start = source.sizes[last] - n
        tail = source.read(last, start, n)
        info, pos = parse_eocd(tail, start)
        if needs_zip64(info):
            if pos < 20:
                raise ZipFormatError("zip64 locator outside the fetched tail")
            z_disk, z_off, _total = parse_zip64_locator(tail, pos - 20)
            info = parse_zip64_eocd(source.read(z_disk, z_off, 56))
        if info.num_disks != source.num_disks:
            raise ZipFormatError(f"archive has {info.num_disks} disks, got {source.num_disks} segments")
        self.info = info
        cd = source.read(info.disk, info.offset, info.size)
        self.entries = parse_central_directory(cd)
        if len(self.entries) != info.entries:
            raise ZipFormatError(f"central directory: {len(self.entries)} entries, expected {info.entries}")
        self.by_name = {e.name: e for e in self.entries}

    def data_range(self, entry: ZipEntry) -> tuple[int, int]:
        """(disk, offset) where ``entry``'s compressed data starts (reads the local header)."""
        hdr = self.source.read(entry.disk, entry.offset, LOCAL_HEADER)
        sig, *_rest, nlen, xlen = struct.unpack("<IHHHHHIIIHH", hdr)
        if sig != SIG_LOCAL:
            raise ZipFormatError(f"{entry.name}: bad local header signature")
        return entry.disk, entry.offset + LOCAL_HEADER + nlen + xlen

    def read(self, name_or_entry: str | ZipEntry, slack: int = 1024) -> bytes:
        """Fetch + decompress + CRC-check one member. The local header and data are fetched in
        a single request (``slack`` bytes reserved for the local name/extra fields)."""
        e = self.by_name[name_or_entry] if isinstance(name_or_entry, str) else name_or_entry
        want = LOCAL_HEADER + slack + e.compressed_size
        total_left = sum(self.source.sizes[e.disk :]) - e.offset
        buf = self.source.read(e.disk, e.offset, min(want, total_left))
        sig, *_rest, nlen, xlen = struct.unpack_from("<IHHHHHIIIHH", buf, 0)
        if sig != SIG_LOCAL:
            raise ZipFormatError(f"{e.name}: bad local header signature")
        start = LOCAL_HEADER + nlen + xlen
        data = buf[start : start + e.compressed_size]
        if len(data) < e.compressed_size:  # very long local extra field: fetch exactly
            disk, off = self.data_range(e)
            data = self.source.read(disk, off, e.compressed_size)
        return decompress(e, data)


# ---------------------------------------------------------------- HTTP


def http_range(url: str, start: int, length: int, retries: int = 6, timeout: float = 60.0) -> bytes:
    """GET bytes [start, start+length) of ``url`` (requires a server honouring Range: 206)."""
    if length == 0:
        return b""
    hdr = {"Range": f"bytes={start}-{start + length - 1}"}
    err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout) as r:
                if r.status != 206:
                    raise OSError(f"server ignored Range request (HTTP {r.status}) for {url}")
                data = r.read()
            if len(data) != length:
                raise OSError(f"short range read {len(data)}/{length}")
            return data
        except (OSError, urllib.error.URLError) as e:
            err = e
            time.sleep(min(30.0, 2.0**attempt))
    raise OSError(f"range request failed for {url} [{start}, +{length}): {err!r}")


class HttpSegments(SegmentedSource):
    """Segments served over HTTP; counts the bytes downloaded."""

    def __init__(self, urls: Sequence[str], sizes: Sequence[int]):
        self.urls = list(urls)
        self.bytes_fetched = 0
        super().__init__(sizes, self._fetch)

    def _fetch(self, disk: int, offset: int, length: int) -> bytes:
        data = http_range(self.urls[disk], offset, length)
        self.bytes_fetched += len(data)
        return data


def http_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def split_segment_names(stem: str, num_disks: int) -> list[str]:
    """``stem.z01`` .. ``stem.z{n-1}`` followed by ``stem.zip`` (the last disk)."""
    return [f"{stem}.z{i:02d}" for i in range(1, num_disks)] + [f"{stem}.zip"]


def bytes_source(segments: Sequence[bytes]) -> SegmentedSource:
    """In-memory segments (tests / local files)."""
    segs = list(segments)
    return SegmentedSource([len(s) for s in segs], lambda d, o, n: segs[d][o : o + n])
