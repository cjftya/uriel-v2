from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence


def write_budget_bar_chart_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    budgets: Sequence[int],
    values: Sequence[float],
    y_label: str,
    value_format: str,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    if len(budgets) != len(values) or not budgets:
        raise ValueError("budgets와 values는 같은 길이의 비어 있지 않은 목록이어야 합니다")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("budget은 양수여야 합니다")

    width, height = 960, 540
    left, right, top, bottom = 105, 55, 112, 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = y_max if y_max is not None else max(values)
    if maximum <= y_min:
        maximum = y_min + 1.0
    if y_max is None:
        maximum = maximum * 1.12

    def y_position(value: float) -> float:
        return top + (maximum - value) / (maximum - y_min) * plot_height

    slot_width = plot_width / len(budgets)
    bar_width = min(150.0, slot_width * 0.55)
    lines: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">',
        '<rect width="960" height="540" fill="#ffffff"/>',
        f'<text x="{left}" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#172033">{html.escape(title)}</text>',
        f'<text x="{left}" y="70" font-family="Arial, sans-serif" font-size="14" fill="#5c667a">{html.escape(subtitle)}</text>',
    ]

    for index in range(5):
        value = y_min + (maximum - y_min) * index / 4
        y = y_position(value)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e4e8ef" stroke-width="1"/>')
        lines.append(
            f'<text x="{left-14}" y="{y+5:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#667085">'
            f'{html.escape(format(value, value_format))}</text>'
        )

    lines.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#344054" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#344054" stroke-width="1.2"/>',
        f'<text x="24" y="{top + plot_height/2:.2f}" transform="rotate(-90 24 {top + plot_height/2:.2f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#475467">{html.escape(y_label)}</text>',
    ])

    for index, (budget, value) in enumerate(zip(budgets, values, strict=True)):
        x = left + slot_width * (index + 0.5)
        y = y_position(value)
        bar_height = height - bottom - y
        label = f"{budget:,}"
        lines.extend([
            f'<rect x="{x-bar_width/2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#2f6fed" stroke="#1f4fae" stroke-width="1.2"/>',
            f'<text x="{x:.2f}" y="{y-14:.2f}" text-anchor="middle" font-family="monospace" font-size="13" font-weight="700" fill="#172033">{html.escape(format(value, value_format))}</text>',
            f'<text x="{x:.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="monospace" font-size="12" fill="#475467">{label}</text>',
        ])

    lines.extend([
        f'<text x="{left + plot_width/2:.2f}" y="{height-24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#475467">Seed budget per round</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
