#!/usr/bin/env python3
"""Generate a single-panel intro companion figure for IFEval."""

from __future__ import annotations

import argparse
import html
import textwrap
from pathlib import Path


WIDTH = 1200
HEIGHT = 680

BG = "#ffffff"
PANEL = "#ffffff"
SUBPANEL = "#f8fafc"
GRID = "#d9dde3"
AXIS = "#4a5565"
TEXT = "#111827"
MUTED = "#5b6472"
INITIAL = "#6b7280"
PROMPT = "#9a3412"
LOGIC = "#14b8a6"
BAE = "#0f766e"
PROMPT_BG = "#fff3eb"
BAE_BG = "#ecfdf5"
FAIL = "#b91c1c"
FAIL_BG = "#fee2e2"
PASS = "#166534"
PASS_BG = "#dcfce7"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="figures/ifeval_intro_companion.svg",
        help="Output SVG path.",
    )
    parser.add_argument("--initial", type=float, default=0.697, help="Initial agent IFEval score.")
    parser.add_argument(
        "--prompt-only",
        type=float,
        default=0.718,
        help="Prompt-only evolution IFEval score.",
    )
    parser.add_argument(
        "--logic-only",
        type=float,
        default=0.744,
        help="Illustrative logic-only evolution IFEval score.",
    )
    parser.add_argument("--bae", type=float, default=0.761, help="BAE IFEval score.")
    return parser.parse_args()


def escape(value: str) -> str:
    return html.escape(value, quote=False)


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    *,
    stroke: str | None = None,
    stroke_width: int = 1,
    opacity: float = 1.0,
    rx: int = 18,
) -> str:
    stroke_attr = ""
    if stroke:
        stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width}"'
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'fill="{fill}" opacity="{opacity:.2f}" rx="{rx}"{stroke_attr} />'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str, *, width: int = 2) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}" />'
    )


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 24,
    weight: int = 400,
    anchor: str = "start",
    fill: str = TEXT,
    font_family: str = "Helvetica, Arial, sans-serif",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font_family}" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{escape(value)}</text>'
    )


def multiline_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 18,
    weight: int = 400,
    anchor: str = "start",
    fill: str = TEXT,
    line_height: float = 1.35,
    font_family: str = "Helvetica, Arial, sans-serif",
) -> str:
    lines = value.split("\n")
    chunks = [
        (
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font_family}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        )
    ]
    for idx, line_value in enumerate(lines):
        dy = "0" if idx == 0 else f"{line_height:.2f}em"
        chunks.append(f'<tspan x="{x:.1f}" dy="{dy}">{escape(line_value)}</tspan>')
    chunks.append("</text>")
    return "".join(chunks)


def wrapped_text(
    x: float,
    y: float,
    value: str,
    *,
    width_chars: int,
    size: int = 18,
    weight: int = 400,
    anchor: str = "start",
    fill: str = TEXT,
    line_height: float = 1.35,
    font_family: str = "Helvetica, Arial, sans-serif",
) -> str:
    wrapped_lines: list[str] = []
    for paragraph in value.split("\n"):
        if not paragraph:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(paragraph, width=width_chars) or [""])
    return multiline_text(
        x,
        y,
        "\n".join(wrapped_lines),
        size=size,
        weight=weight,
        anchor=anchor,
        fill=fill,
        line_height=line_height,
        font_family=font_family,
    )


def pill(x: float, y: float, w: float, h: float, label: str, fill: str, text_fill: str) -> str:
    cy = y + h / 2 + 5
    return (
        rect(x, y, w, h, fill, rx=999)
        + text(x + w / 2, cy, label, size=15, weight=700, anchor="middle", fill=text_fill)
    )


def build_svg(initial: float, prompt_only: float, logic_only: float, bae: float) -> str:
    del initial, prompt_only, logic_only, bae

    panel_x, panel_y, panel_w, panel_h = 40, 24, WIDTH - 80, HEIGHT - 48

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        rect(0, 0, WIDTH, HEIGHT, BG, rx=18),
        rect(panel_x, panel_y, panel_w, panel_h, PANEL, stroke="#e5e7eb", rx=20),
    ]

    svg.append(text(panel_x + 28, panel_y + 38, "One Real IFEval Example", size=24, weight=700))
    svg.append(
        text(
            panel_x + 28,
            panel_y + 64,
            "Prompt-only gets close; the brittle outer format constraint still breaks.",
            size=17,
            fill=MUTED,
        )
    )

    instr_x = panel_x + 24
    instr_y = panel_y + 82
    instr_w = panel_w - 48
    instr_h = 58
    svg.append(rect(instr_x, instr_y, instr_w, instr_h, SUBPANEL, stroke="#dbe3ea", rx=16))
    instruction = (
        "Write a song about innovation with a positive tone that is appealing to teenagers. "
        "Put your entire response in double quotation marks."
    )
    svg.append(text(instr_x + 18, instr_y + 36, f"Instruction: {instruction}", size=15))

    content_y = instr_y + instr_h + 14
    raw_x = panel_x + 24
    raw_w = 180
    prompt_x = raw_x + raw_w + 14
    prompt_w = 210
    logic_x = prompt_x + prompt_w + 14
    logic_w = 250
    checklist_x = logic_x + logic_w + 14
    checklist_w = panel_x + panel_w - 24 - checklist_x
    card_h = 390

    svg.append(rect(raw_x, content_y, raw_w, card_h, "#f5f5f4", stroke="#d6d3d1", rx=18))
    svg.append(text(raw_x + 18, content_y + 28, "Raw output", size=16, weight=700, fill="#57534e"))
    svg.append(pill(raw_x + raw_w - 84, content_y + 14, 62, 26, "FAIL", FAIL_BG, FAIL))
    raw_output = (
        "Verse 1:\n"
        "New ideas light the hallway.\n"
        "We build tomorrow with code.\n"
        "Innovation makes us bold.\n"
        "The future starts right now."
    )
    svg.append(
        multiline_text(
            raw_x + 18,
            content_y + 62,
            raw_output,
            size=16,
            font_family="Courier New, monospace",
            line_height=1.28,
        )
    )

    svg.append(rect(prompt_x, content_y, prompt_w, card_h, PROMPT_BG, stroke="#f0c7ae", rx=18))
    svg.append(text(prompt_x + 18, content_y + 28, "Prompt-only output", size=16, weight=700, fill=PROMPT))
    svg.append(pill(prompt_x + prompt_w - 84, content_y + 14, 62, 26, "FAIL", FAIL_BG, FAIL))
    prompt_output = (
        '"Future Made Now"\n'
        "Verse 1:\n"
        "New ideas light the hallway.\n"
        "We build tomorrow with code.\n"
        "Innovation makes us bold.\n"
        "The future starts right now."
    )
    svg.append(
        multiline_text(
            prompt_x + 18,
            content_y + 62,
            prompt_output,
            size=16,
            font_family="Courier New, monospace",
            line_height=1.28,
        )
    )

    svg.append(rect(logic_x, content_y, logic_w, card_h, BAE_BG, stroke="#b8e4d3", rx=18))
    svg.append(
        multiline_text(
            logic_x + 18,
            content_y + 26,
            "Logic-constrained output\non top of prompt change",
            size=15,
            weight=700,
            fill=BAE,
            line_height=1.10,
        )
    )
    svg.append(pill(logic_x + logic_w - 84, content_y + 14, 62, 26, "PASS", PASS_BG, PASS))
    logic_output = (
        '"Verse 1:\n'
        "New ideas light the hallway.\n"
        "We build tomorrow with code.\n"
        "Innovation makes us bold.\n"
        'The future starts right now."'
    )
    svg.append(
        multiline_text(
            logic_x + 18,
            content_y + 80,
            logic_output,
            size=16,
            font_family="Courier New, monospace",
            line_height=1.28,
        )
    )

    svg.append(rect(checklist_x, content_y, checklist_w, card_h, SUBPANEL, stroke="#dbe3ea", rx=18))
    svg.append(text(checklist_x + 18, content_y + 28, "Constraint checklist", size=16, weight=700, fill=MUTED))
    svg.append(text(checklist_x + checklist_w - 168, content_y + 48, "Raw", size=12, weight=700, fill=MUTED, anchor="middle"))
    svg.append(text(checklist_x + checklist_w - 104, content_y + 48, "Prompt", size=12, weight=700, fill=MUTED, anchor="middle"))
    svg.append(text(checklist_x + checklist_w - 42, content_y + 48, "Logic", size=12, weight=700, fill=MUTED, anchor="middle"))

    checklist_items = [
        ("Song about innovation", True, True, True),
        ("Positive tone", True, True, True),
        ("Appeals to teenagers", True, True, True),
        ("Entire response in double quotes", False, False, True),
    ]

    row_y = content_y + 64
    row_h = 74
    for label, raw_pass, prompt_pass, logic_pass in checklist_items:
        svg.append(rect(checklist_x + 14, row_y, checklist_w - 28, 54, "#ffffff", stroke="#e5e7eb", rx=12))
        svg.append(wrapped_text(checklist_x + 24, row_y + 24, label, width_chars=22, size=13, line_height=1.14))
        svg.append(
            pill(
                checklist_x + checklist_w - 190,
                row_y + 18,
                44,
                22,
                "PASS" if raw_pass else "FAIL",
                PASS_BG if raw_pass else FAIL_BG,
                PASS if raw_pass else FAIL,
            )
        )
        svg.append(
            pill(
                checklist_x + checklist_w - 126,
                row_y + 18,
                44,
                22,
                "PASS" if prompt_pass else "FAIL",
                PASS_BG if prompt_pass else FAIL_BG,
                PASS if prompt_pass else FAIL,
            )
        )
        svg.append(
            pill(
                checklist_x + checklist_w - 62,
                row_y + 18,
                44,
                22,
                "PASS" if logic_pass else "FAIL",
                PASS_BG if logic_pass else FAIL_BG,
                PASS if logic_pass else FAIL,
            )
        )
        row_y += row_h

    svg.append("</svg>")
    return "\n".join(svg)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_svg(
            initial=args.initial,
            prompt_only=args.prompt_only,
            logic_only=args.logic_only,
            bae=args.bae,
        ),
        encoding="utf-8",
    )
    print(f"Wrote IFEval intro companion figure to {output_path}")


if __name__ == "__main__":
    main()
