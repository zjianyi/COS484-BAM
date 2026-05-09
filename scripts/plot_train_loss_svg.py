#!/usr/bin/env python3
"""Plot train loss curves from ablation metrics JSONs or Neuronic logs.

The ablation runner now writes ``train_history`` to metrics JSONs. For older
runs, this script can also parse lines like:
``epoch 7 done: 150 examples, avg_loss=1.2345 W_out_norm=...``
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(r"epoch\s+(\d+)\s+done:.*avg_loss=([0-9.]+)")


def load_curve(path: Path) -> tuple[str, list[tuple[int, float]]]:
    text = path.read_text()
    if path.suffix == ".json":
        payload = json.loads(text)
        history = payload.get("train_history", [])
        points = [
            (int(row["epoch"]), float(row["avg_loss"]))
            for row in history
            if "epoch" in row and "avg_loss" in row
        ]
        if points:
            label = payload.get("config", {}).get("run_id") or path.stem
            return str(label), points

    points = [(int(m.group(1)), float(m.group(2))) for m in EPOCH_RE.finditer(text)]
    if not points:
        raise ValueError(f"No train loss history found in {path}")
    return path.stem, points


def build_svg(curves: list[tuple[str, list[tuple[int, float]]]]) -> str:
    width, height = 960, 540
    left, right, top, bottom = 72, 220, 36, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]

    xs = [x for _, points in curves for x, _ in points]
    ys = [y for _, points in curves for _, y in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1.0
    y_pad = 0.05 * (y_max - y_min)
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad

    def sx(x: int) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    elems: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="14">Epoch</text>',
        f'<text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 18 {top + plot_h / 2:.1f})">Average loss</text>',
        f'<text x="{left + plot_w / 2:.1f}" y="22" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700">StateCache Training Loss</text>',
    ]

    for i in range(6):
        frac = i / 5
        y_val = y_min + frac * (y_max - y_min)
        y = sy(y_val)
        elems.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        elems.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y_val:.3g}</text>')

    for i in range(5):
        frac = i / 4
        x_val = round(x_min + frac * (x_max - x_min))
        x = sx(x_val)
        elems.append(f'<text x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="12">{x_val}</text>')

    for idx, (label, points) in enumerate(curves):
        color = colors[idx % len(colors)]
        poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        elems.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            elems.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>')
        legend_y = top + 22 * idx
        elems.append(f'<line x1="{left + plot_w + 24}" y1="{legend_y}" x2="{left + plot_w + 48}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elems.append(f'<text x="{left + plot_w + 56}" y="{legend_y + 4}" font-family="Arial" font-size="12">{html.escape(label)}</text>')

    elems.append("</svg>")
    return "\n".join(elems)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="metrics JSONs or Neuronic log files")
    parser.add_argument("--output", type=Path, default=Path("metrics/l6-sweeps/train_loss.svg"))
    args = parser.parse_args()

    curves = [load_curve(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(curves))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
