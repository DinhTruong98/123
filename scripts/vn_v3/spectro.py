# ruff: noqa: E501
"""Spectrogram contact sheets + a simple 'wail' score for relabel review (vn-v3).

usage: spectro.py OUT_PREFIX PER_PAGE file[::caption] ...
Each panel: 0-5 kHz linear spectrogram (dB) of the clip, with its dominant-frequency track.
Also prints, per clip, ``sweep`` = median |d log2(f_peak)/dt| (octaves/s) over tonal frames whose
peak moves smoothly (wailing sirens glide: ~0.5-4 oct/s; beeps/bells are flat or jump).
"""

from __future__ import annotations

import sys

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SR = 16000


def wail_stats(x: np.ndarray) -> dict:
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    band = (f >= 400) & (f <= 4000)
    Sb = S[band]
    fb = f[band]
    pk = Sb.argmax(axis=0)
    tonal = Sb.max(axis=0) / (np.median(Sb, axis=0) + 1e-12) > 30
    loud = 10 * np.log10(Sb.sum(axis=0) + 1e-12)
    tonal &= loud > loud.max() - 30
    lf = np.log2(fb[pk])
    dt = 256 / SR
    d = np.abs(np.diff(lf)) / dt
    both = tonal[1:] & tonal[:-1]
    smooth = both & (d < 8.0)  # < 8 oct/s: continuous glide, not a jump between beep tones
    moving = smooth & (d > 0.3)
    return {
        "tonal": float(tonal.mean()),
        "glide_frac": float(moving.sum() / max(both.sum(), 1)),
        "sweep": float(np.median(d[smooth])) if smooth.any() else 0.0,
        "range_oct": float(np.percentile(lf[tonal], 90) - np.percentile(lf[tonal], 10)) if tonal.sum() > 5 else 0.0,
        "track": (np.arange(len(pk)) * dt, np.where(tonal, fb[pk], np.nan)),
    }


def main() -> None:
    out, per = sys.argv[1], int(sys.argv[2])
    items = [a.split("::", 1) if "::" in a else (a, a.rsplit("/", 1)[-1]) for a in sys.argv[3:]]
    cols = 3
    for page in range(0, len(items), per):
        chunk = items[page:page + per]
        rows = (len(chunk) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 2.4), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (path, cap) in zip(axes.flat, chunk, strict=False):
            x, _ = librosa.load(path, sr=SR, mono=True, duration=12.0)
            st = wail_stats(x)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=1024, hop_length=256)), ref=np.max)
            ax.axis("on")
            ax.imshow(D[:321], origin="lower", aspect="auto", vmin=-60, vmax=0, cmap="magma",
                      extent=[0, len(x) / SR, 0, 5000])
            ax.set_title(f"{cap}\nsweep {st['sweep']:.2f} glide {st['glide_frac']:.2f} rng {st['range_oct']:.2f}",
                         fontsize=7)
            ax.tick_params(labelsize=6)
            print(f"{path}\tsweep={st['sweep']:.2f}\tglide={st['glide_frac']:.2f}\trange={st['range_oct']:.2f}\t"
                  f"tonal={st['tonal']:.2f}")
        fig.tight_layout()
        fig.savefig(f"{out}_{page // per:02d}.png", dpi=72)
        plt.close(fig)


if __name__ == "__main__":
    main()
