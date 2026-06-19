from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "when-systems-break.md"
PDF_PATH = ROOT / "when-systems-break.pdf"

PAGE_W = 8.5
PAGE_H = 11.0
LEFT = 0.85
RIGHT = 0.85
TOP = 0.8
BOTTOM = 0.7
CONTENT_W = PAGE_W - LEFT - RIGHT


def strip_inline(text):
    text = text.strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("`", "")
    return text


def wrap_text(text, width):
    return textwrap.wrap(strip_inline(text), width=width) or [""]


def text_at(ax, x, y, text, size=10.5, weight="normal", family="DejaVu Sans", color="#111111"):
    ax.text(
        x / PAGE_W,
        y / PAGE_H,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=size,
        fontweight=weight,
        fontfamily=family,
        color=color,
    )


def centered_text(ax, y, text, size=14, weight="normal", color="#111111"):
    ax.text(
        0.5,
        y / PAGE_H,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        fontfamily="DejaVu Sans",
        color=color,
    )


def new_page(pdf, page_no):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax, PAGE_H - TOP, page_no


def finish_page(pdf, fig, ax, page_no):
    ax.plot(
        [LEFT / PAGE_W, (PAGE_W - RIGHT) / PAGE_W],
        [0.055, 0.055],
        transform=ax.transAxes,
        color="#DDDDDD",
        linewidth=0.8,
    )
    ax.text(
        (PAGE_W - RIGHT) / PAGE_W,
        0.035,
        str(page_no),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    pdf.savefig(fig)
    plt.close(fig)


def parse_markdown(lines):
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            blocks.append(
                {
                    "type": "image",
                    "title": image_match.group(1),
                    "path": ROOT / image_match.group(2),
                }
            )
            i += 1
            continue

        if stripped.startswith("## "):
            blocks.append({"type": "h2", "text": stripped[3:]})
            i += 1
            continue

        if stripped.startswith("### "):
            blocks.append({"type": "h3", "text": stripped[4:]})
            i += 1
            continue

        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip())
                i += 1
            blocks.append({"type": "code", "lines": code_lines})
            i += 1
            continue

        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            blocks.append({"type": "list", "text": stripped})
            i += 1
            continue

        paragraph = [stripped]
        i += 1
        while i < len(lines):
            candidate = lines[i].strip()
            if not candidate:
                break
            if (
                candidate.startswith("## ")
                or candidate.startswith("### ")
                or candidate.startswith("```")
                or candidate.startswith("![")
                or re.match(r"^[-*]\s+", candidate)
                or re.match(r"^\d+\.\s+", candidate)
            ):
                break
            paragraph.append(candidate)
            i += 1
        blocks.append({"type": "p", "text": " ".join(paragraph)})

    return blocks


def draw_title_page(pdf, title, subtitle):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    centered_text(ax, 7.2, title.replace(":", ""), size=25, weight="bold")
    y = 6.55
    for line in textwrap.wrap(subtitle, width=58):
        centered_text(ax, y, line, size=14.5, weight="normal", color="#333333")
        y -= 0.34

    ax.plot([0.22, 0.78], [5.55 / PAGE_H, 5.55 / PAGE_H], transform=ax.transAxes, color="#111111", linewidth=1.2)
    centered_text(ax, 5.15, "Research Report", size=12.5, weight="bold")
    centered_text(ax, 4.82, "Robustness, Confidence, and Refusal in Machine Learning Systems", size=10.5, color="#555555")
    centered_text(ax, 1.25, "Generated from the when-systems-break project", size=9.5, color="#777777")

    pdf.savefig(fig)
    plt.close(fig)


def ensure_space(pdf, fig, ax, y, needed, page_no):
    if y - needed >= BOTTOM:
        return fig, ax, y, page_no
    finish_page(pdf, fig, ax, page_no)
    return new_page(pdf, page_no + 1)


def draw_blocks(pdf, blocks):
    fig, ax, y, page_no = new_page(pdf, 2)

    for block in blocks:
        kind = block["type"]

        if kind == "h2":
            fig, ax, y, page_no = ensure_space(pdf, fig, ax, y, 0.65, page_no)
            y -= 0.12
            for line in wrap_text(block["text"], 72):
                text_at(ax, LEFT, y, line, size=14.5, weight="bold")
                y -= 0.31
            y -= 0.14

        elif kind == "h3":
            fig, ax, y, page_no = ensure_space(pdf, fig, ax, y, 0.45, page_no)
            for line in wrap_text(block["text"], 80):
                text_at(ax, LEFT, y, line, size=12, weight="bold")
                y -= 0.25
            y -= 0.06

        elif kind == "p":
            lines = wrap_text(block["text"], 98)
            fig, ax, y, page_no = ensure_space(pdf, fig, ax, y, len(lines) * 0.21 + 0.14, page_no)
            for line in lines:
                text_at(ax, LEFT, y, line, size=10.2)
                y -= 0.21
            y -= 0.12

        elif kind == "list":
            text = strip_inline(block["text"])
            text = re.sub(r"^[-*]\s+", "- ", text)
            lines = wrap_text(text, 92)
            fig, ax, y, page_no = ensure_space(pdf, fig, ax, y, len(lines) * 0.21 + 0.08, page_no)
            for index, line in enumerate(lines):
                prefix = "" if index else ""
                text_at(ax, LEFT + 0.22, y, prefix + line, size=10.1)
                y -= 0.21
            y -= 0.04

        elif kind == "code":
            code_height = max(0.48, len(block["lines"]) * 0.22 + 0.22)
            fig, ax, y, page_no = ensure_space(pdf, fig, ax, y, code_height + 0.1, page_no)
            rect = plt.Rectangle(
                (LEFT / PAGE_W, (y - code_height + 0.08) / PAGE_H),
                CONTENT_W / PAGE_W,
                code_height / PAGE_H,
                transform=ax.transAxes,
                facecolor="#F5F5F5",
                edgecolor="#DDDDDD",
                linewidth=0.8,
            )
            ax.add_patch(rect)
            code_y = y - 0.12
            for code_line in block["lines"]:
                text_at(ax, LEFT + 0.18, code_y, code_line, size=9.4, family="DejaVu Sans Mono", color="#222222")
                code_y -= 0.22
            y -= code_height + 0.12

        elif kind == "image":
            image = Image.open(block["path"])
            aspect = image.size[0] / image.size[1]
            max_w = CONTENT_W
            max_h = 3.95
            img_w = max_w
            img_h = img_w / aspect
            if img_h > max_h:
                img_h = max_h
                img_w = img_h * aspect

            needed = img_h + 0.58
            fig, ax, y, page_no = ensure_space(pdf, fig, ax, y, needed, page_no)
            text_at(ax, LEFT, y, block["title"], size=10.5, weight="bold", color="#333333")
            y -= 0.32

            x = LEFT + (CONTENT_W - img_w) / 2
            img_ax = fig.add_axes([x / PAGE_W, (y - img_h) / PAGE_H, img_w / PAGE_W, img_h / PAGE_H])
            img_ax.imshow(image)
            img_ax.axis("off")
            y -= img_h + 0.22

    finish_page(pdf, fig, ax, page_no)


def main():
    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    title_lines = []
    body_start = 0
    for index, line in enumerate(lines):
        if line.startswith("# "):
            title_lines.append(strip_inline(line[2:]))
            body_start = index + 1
        elif title_lines:
            body_start = index
            break

    title = title_lines[0] if title_lines else "When Systems Break"
    subtitle = " ".join(title_lines[1:]) if len(title_lines) > 1 else ""
    blocks = parse_markdown(lines[body_start:])

    with PdfPages(PDF_PATH) as pdf:
        draw_title_page(pdf, title, subtitle)
        draw_blocks(pdf, blocks)


if __name__ == "__main__":
    main()
