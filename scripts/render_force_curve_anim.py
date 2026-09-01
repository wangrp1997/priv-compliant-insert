#!/usr/bin/env python3
"""Render animated wrist residual force curve synced to ego mp4 length."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


def _probe_frames(path: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        text=True,
    ).strip()
    if out.isdigit():
        return int(out)
    # fallback duration*fps
    out2 = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    info = json.loads(out2)["streams"][0]
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return int(round(float(info["duration"]) * fps))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ego", type=Path, default=None, help="pad force anim to ego frame count")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    sm = summary.get("surface_meta") or summary.get("approach_noise") or {}
    trace = list(sm.get("force_trace") or summary.get("force_trace") or [])
    if not trace:
        raise SystemExit(f"no force_trace in {args.summary}")

    fps = max(1, int(args.fps))
    stride = max(1, int(args.stride))
    trace = trace[::stride]

    # Pad front so force curve timeline matches full ego video.
    pad_n = 0
    if args.ego is not None and args.ego.is_file():
        ego_n = _probe_frames(args.ego)
        force_n = len(trace)
        pad_n = max(0, ego_n - force_n)
        if pad_n > 0:
            t0 = float(trace[0]["t"])
            fdes0 = float(trace[0].get("f_des", 0.15))
            pad = []
            for i in range(pad_n):
                pad.append(
                    {
                        "t": (i - pad_n) / float(fps) + t0,
                        "step": float(i - pad_n),
                        "phase": "approach",
                        "resid_r": float("nan"),
                        "resid_l": float("nan"),
                        "f_des": fdes0,
                    }
                )
            # re-index time to start at 0 for full episode
            trace = pad + trace
            for i, x in enumerate(trace):
                x["t"] = i / float(fps)

    t = np.asarray([float(x["t"]) for x in trace], dtype=np.float64)
    rr = np.asarray([float(x["resid_r"]) for x in trace], dtype=np.float64)
    rl = np.asarray([float(x["resid_l"]) for x in trace], dtype=np.float64)
    fdes = np.asarray([float(x.get("f_des", 0.15)) for x in trace], dtype=np.float64)
    phases = [str(x.get("phase", "")) for x in trace]

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
    ax.set_xlim(0.0, float(t[-1]) + 1e-3)
    finite = rr[np.isfinite(rr)]
    finite_l = rl[np.isfinite(rl)]
    ymax = 0.6
    if finite.size:
        ymax = max(ymax, float(np.nanmax(finite)) * 1.2)
    if finite_l.size:
        ymax = max(ymax, float(np.nanmax(finite_l)) * 1.2)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("time (s)  [synced to ego]")
    ax.set_ylabel("|wrist residual| (N)")
    ax.set_title("Dual-wrist residual vs ego timeline")
    ax.grid(True, alpha=0.3)
    (line_r,) = ax.plot([], [], color="#c0392b", lw=2.0, label="right |r|")
    (line_l,) = ax.plot([], [], color="#2980b9", lw=2.0, label="left |r|")
    (line_des,) = ax.plot([], [], color="#27ae60", lw=1.5, ls="--", label="f_des")
    marker = ax.axvline(0.0, color="#7f8c8d", lw=1.0, alpha=0.8)
    phase_txt = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="upper right")

    def _init():
        line_r.set_data([], [])
        line_l.set_data([], [])
        line_des.set_data([], [])
        return line_r, line_l, line_des, marker, phase_txt

    def _update(i: int):
        # skip nan in approach pad for line drawing
        tt = t[: i + 1]
        rr_i = rr[: i + 1].copy()
        rl_i = rl[: i + 1].copy()
        line_r.set_data(tt, rr_i)
        line_l.set_data(tt, rl_i)
        line_des.set_data(tt, fdes[: i + 1])
        marker.set_xdata([t[i], t[i]])
        rv = rr[i] if np.isfinite(rr[i]) else float("nan")
        lv = rl[i] if np.isfinite(rl[i]) else float("nan")
        phase_txt.set_text(
            f"phase={phases[i]}  t={t[i]:.2f}s\n"
            f"R={rv:.2f}N  L={lv:.2f}N  des={fdes[i]:.2f}N"
        )
        return line_r, line_l, line_des, marker, phase_txt

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(t),
        init_func=_init,
        blit=True,
        interval=1000.0 / fps,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(str(args.out), writer=writer)
    plt.close(fig)
    print(
        f"wrote {args.out} frames={len(t)} fps={fps} pad_approach={pad_n}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
