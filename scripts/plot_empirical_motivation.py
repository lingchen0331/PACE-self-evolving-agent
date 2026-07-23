#!/usr/bin/env python3
"""Generate an empirical motivation figure for the paper.

The figure is intentionally lightweight: it writes a self-contained SVG and
does not depend on matplotlib. The default values follow the requested
Qwen3-4B narrative:

1. Prompt evolution improves MMLU from 0.771 to 0.788.
2. Additional prompt refinements oscillate.
3. A structural parser fix yields an immediate +0.02 jump.
4. Prompt tuning on the repaired structure yields another +0.02.
"""

from __future__ import annotations

import argparse
from pathlib import Path


WIDTH = 1200
HEIGHT = 680
MARGIN_LEFT = 120
MARGIN_RIGHT = 220
MARGIN_TOP = 48
MARGIN_BOTTOM = 150

BG = "#ffffff"
PANEL = "#ffffff"
GRID = "#d9dde3"
AXIS = "#4a5565"
TEXT = "#111827"
MUTED = "#5b6472"
PROMPT = "#9a3412"
STRUCTURE = "#0f766e"
HILITE = "#e6f4f1"
ARROW = "#1f2937"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="figures/empirical_motivation_qwen3_4b.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--title",
        default="Neither Prompt Optimization Nor Logic Updates Alone Are Sufficient",
        help="Figure title.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.771,
        help="Initial MMLU score before prompt evolution.",
    )
    parser.add_argument(
        "--prompt-peak",
        type=float,
        default=0.788,
        help="Best MMLU score reached from prompt-only evolution.",
    )
    parser.add_argument(
        "--prompt-oscillation",
        type=float,
        default=0.782,
        help="Representative score after additional prompt refinements oscillate.",
    )
    parser.add_argument(
        "--structure-gain",
        type=float,
        default=0.020,
        help="Immediate improvement after the structural fix.",
    )
    parser.add_argument(
        "--post-structure-gain",
        type=float,
        default=0.020,
        help="Additional prompt gain after the structure is fixed.",
    )
    return parser.parse_args()


def x_position(index: int, count: int) -> float:
    usable_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    return MARGIN_LEFT + usable_width * index / (count - 1)


def y_position(value: float, y_min: float, y_max: float) -> float:
    usable_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    return MARGIN_TOP + usable_height * (y_max - value) / (y_max - y_min)


def polyline(points: list[tuple[float, float]], color: str, dash: str = "") -> str:
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="5" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash_attr} '
        f'points="{path}" />'
    )


def circle(x: float, y: float, color: str, radius: int = 8) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
        f'fill="{color}" stroke="{PANEL}" stroke-width="3" />'
    )


def text(
    x: float,
    y: float,
    value: str,
    size: int = 24,
    weight: int = 400,
    anchor: str = "start",
    fill: str = TEXT,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{value}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" opacity="{opacity:.2f}" rx="18" />'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: int = 2) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" />'
    )


def build_svg(title: str, scores: list[float]) -> str:
    labels = [
        "Initial\nprompt",
        "Prompt\nevolution",
        "More prompt\nrefinement",
        "Agent logic\nupdate",
        "Prompt tuning\nafter logic update",
    ]
    x_vals = [x_position(i, len(scores)) for i in range(len(scores))]

    y_min = min(scores) - 0.008
    y_max = max(scores) + 0.010
    prompt_points = [(x_vals[i], y_position(scores[i], y_min, y_max)) for i in range(3)]
    structure_points = [(x_vals[i], y_position(scores[i], y_min, y_max)) for i in range(2, 5)]
    all_points = [(x_vals[i], y_position(scores[i], y_min, y_max)) for i in range(len(scores))]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, BG),
        rect(40, 24, WIDTH - 80, HEIGHT - 48, PANEL),
    ]

    y_ticks = [0.77, 0.78, 0.79, 0.80, 0.81, 0.82]
    for tick in y_ticks:
        y = y_position(tick, y_min, y_max)
        svg.append(line(MARGIN_LEFT, y, WIDTH - MARGIN_RIGHT, y, GRID, width=1))
        svg.append(text(MARGIN_LEFT - 18, y + 6, f"{tick:.2f}", size=17, anchor="end", fill=MUTED))

    svg.append(line(MARGIN_LEFT, MARGIN_TOP, MARGIN_LEFT, HEIGHT - MARGIN_BOTTOM, AXIS, width=2))
    svg.append(line(MARGIN_LEFT, HEIGHT - MARGIN_BOTTOM, WIDTH - MARGIN_RIGHT, HEIGHT - MARGIN_BOTTOM, AXIS, width=2))

    band_x = x_vals[2] - 72
    band_w = x_vals[4] - x_vals[2] + 120
    svg.append(rect(band_x, MARGIN_TOP + 16, band_w, HEIGHT - MARGIN_TOP - MARGIN_BOTTOM - 28, HILITE, opacity=0.65))
    svg.append(text(x_vals[3], MARGIN_TOP + 38, "Prompt landscape after logic update", size=17, weight=700, anchor="middle", fill=STRUCTURE))

    svg.append(polyline(prompt_points, PROMPT))
    svg.append(polyline(structure_points, STRUCTURE))
    svg.append(polyline(all_points, "#7c8798", dash="8 8"))

    for idx, (x, y) in enumerate(all_points):
        color = PROMPT if idx <= 2 else STRUCTURE
        svg.append(circle(x, y, color))
        svg.append(text(x, y - 18, f"{scores[idx]:.3f}", size=17, weight=700, anchor="middle"))
        label_lines = labels[idx].split("\n")
        base_y = HEIGHT - MARGIN_BOTTOM + 36
        for offset, label in enumerate(label_lines):
            svg.append(text(x, base_y + offset * 24, label, size=17, anchor="middle"))

    jump_y = (all_points[2][1] + all_points[3][1]) / 2
    svg.append(line(all_points[2][0] + 12, jump_y, all_points[3][0] - 18, jump_y - 42, ARROW, width=2))
    svg.append(text((all_points[2][0] + all_points[3][0]) / 2 + 18, jump_y - 50, "Agent logic update: +0.020", size=17, weight=700, fill=ARROW))

    gain_y = (all_points[3][1] + all_points[4][1]) / 2
    svg.append(line(all_points[3][0] + 12, gain_y, all_points[4][0] - 18, gain_y - 34, STRUCTURE, width=2))
    svg.append(text(all_points[4][0] - 12, gain_y - 42, "Prompt optimization on new logic: +0.020", size=17, weight=700, anchor="end", fill=STRUCTURE))

    svg.append(text(x_vals[1] - 20, all_points[1][1] - 44, "Prompt optimization yields early gains", size=17, weight=700, fill=PROMPT))
    svg.append(text(x_vals[2] - 6, all_points[2][1] + 40, "Further prompt updates plateau", size=17, weight=700, fill=PROMPT))

    legend_y = HEIGHT - 58
    svg.append(line(MARGIN_LEFT, legend_y, MARGIN_LEFT + 42, legend_y, PROMPT, width=4))
    svg.append(text(MARGIN_LEFT + 54, legend_y + 5, "Prompt-only optimization", size=17))
    svg.append(line(MARGIN_LEFT + 318, legend_y, MARGIN_LEFT + 360, legend_y, STRUCTURE, width=4))
    svg.append(text(MARGIN_LEFT + 372, legend_y + 5, "Logic update followed by prompt optimization", size=17))

    svg.append("</svg>")
    return "\n".join(svg)


def main() -> None:
    args = parse_args()
    scores = [
        args.start,
        args.prompt_peak,
        args.prompt_oscillation,
        args.prompt_oscillation + args.structure_gain,
        args.prompt_oscillation + args.structure_gain + args.post_structure_gain,
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_svg(args.title, scores), encoding="utf-8")
    print(f"Wrote empirical motivation figure to {output_path}")


if __name__ == "__main__":
    main()
