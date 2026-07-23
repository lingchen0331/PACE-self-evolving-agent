#!/usr/bin/env python3
"""Generate a minimal IFEval figure emphasizing structural optimization."""

from __future__ import annotations

import argparse
from pathlib import Path


WIDTH = 980
HEIGHT = 680
MARGIN_LEFT = 120
MARGIN_RIGHT = 120
MARGIN_TOP = 60
MARGIN_BOTTOM = 150

BG = "#ffffff"
PANEL = "#ffffff"
GRID = "#d9dde3"
AXIS = "#4a5565"
TEXT = "#111827"
MUTED = "#5b6472"
PROMPT = "#9a3412"
STRUCTURE = "#0f766e"
ARROW = "#1f2937"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="figures/ifeval_structural_motivation.svg",
        help="Output SVG path.",
    )
    parser.add_argument("--base", type=float, default=0.690, help="Base IFEval strict accuracy.")
    parser.add_argument("--prompt-only", type=float, default=0.712, help="Prompt-only optimized accuracy.")
    parser.add_argument("--structure", type=float, default=0.764, help="Structural optimization accuracy.")
    return parser.parse_args()


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 24,
    weight: int = 400,
    anchor: str = "start",
    fill: str = TEXT,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{value}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, *, opacity: float = 1.0, rx: int = 14) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" opacity="{opacity:.2f}" rx="{rx}" />'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str, *, width: int = 2) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" />'
    )


def build_svg(base: float, prompt_only: float, structure: float) -> str:
    values = [base, prompt_only, structure]
    labels = ["Base", "Prompt\noptimization", "Structural\noptimization"]
    colors = [MUTED, PROMPT, STRUCTURE]

    y_min = min(values) - 0.02
    y_max = max(values) + 0.025
    plot_w = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    axis_right = WIDTH - MARGIN_RIGHT

    def y_pos(value: float) -> float:
        return MARGIN_TOP + (y_max - value) * plot_h / (y_max - y_min)

    bar_w = 110
    centers = [
        MARGIN_LEFT + plot_w * 0.18,
        MARGIN_LEFT + plot_w * 0.50,
        MARGIN_LEFT + plot_w * 0.82,
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, BG, rx=18),
        rect(40, 24, WIDTH - 80, HEIGHT - 48, PANEL, rx=18),
    ]

    for tick in [0.68, 0.70, 0.72, 0.74, 0.76]:
        y = y_pos(tick)
        svg.append(line(MARGIN_LEFT, y, axis_right, y, GRID, width=1))
        svg.append(text(MARGIN_LEFT - 16, y + 6, f"{tick:.2f}", size=17, anchor="end", fill=MUTED))

    svg.extend(
        [
            line(MARGIN_LEFT, MARGIN_TOP, MARGIN_LEFT, HEIGHT - MARGIN_BOTTOM, AXIS, width=2),
            line(MARGIN_LEFT, HEIGHT - MARGIN_BOTTOM, axis_right, HEIGHT - MARGIN_BOTTOM, AXIS, width=2),
        ]
    )

    for idx, value in enumerate(values):
        x = centers[idx] - bar_w / 2
        y = y_pos(value)
        svg.append(rect(x, y, bar_w, HEIGHT - MARGIN_BOTTOM - y, colors[idx], rx=10))
        svg.append(text(centers[idx], y - 14, f"{value:.3f}", size=18, weight=700, anchor="middle"))
        base_y = HEIGHT - MARGIN_BOTTOM + 36
        for offset, label in enumerate(labels[idx].split("\n")):
            svg.append(text(centers[idx], base_y + 24 * offset, label, size=17, anchor="middle"))

    prompt_gain_y = y_pos(prompt_only) - 22
    svg.append(line(centers[0] + bar_w / 2, prompt_gain_y, centers[1] - bar_w / 2, prompt_gain_y, PROMPT, width=2))
    svg.append(text((centers[0] + centers[1]) / 2, prompt_gain_y - 8, "+ prompt gain", size=16, weight=700, anchor="middle", fill=PROMPT))

    structure_gain_y = y_pos(structure) - 26
    svg.append(line(centers[1] + bar_w / 2, structure_gain_y, centers[2] - bar_w / 2, structure_gain_y, STRUCTURE, width=2))
    svg.append(text((centers[1] + centers[2]) / 2, structure_gain_y - 8, "+ structural gain", size=16, weight=700, anchor="middle", fill=STRUCTURE))
    svg.append(text(centers[2], y_pos(structure) - 42, "Addresses control-logic bottlenecks", size=17, weight=700, anchor="middle", fill=STRUCTURE))

    legend_y = HEIGHT - 46
    svg.extend(
        [
            line(MARGIN_LEFT + 60, legend_y, MARGIN_LEFT + 96, legend_y, PROMPT, width=5),
            text(MARGIN_LEFT + 110, legend_y + 5, "Prompt optimization", size=17),
            line(MARGIN_LEFT + 330, legend_y, MARGIN_LEFT + 366, legend_y, STRUCTURE, width=5),
            text(MARGIN_LEFT + 380, legend_y + 5, "Structural optimization", size=17),
            "</svg>",
        ]
    )
    return "\n".join(svg)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_svg(base=args.base, prompt_only=args.prompt_only, structure=args.structure),
        encoding="utf-8",
    )
    print(f"Wrote IFEval structural motivation figure to {output_path}")


if __name__ == "__main__":
    main()
