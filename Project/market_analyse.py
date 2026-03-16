import ollama
import re
from datetime import datetime, timezone
from pathlib import Path

# -----------------------------
# Mock Dataset (Demo)
# -----------------------------

columns = ["Year", "Revenue (₹)", "Expenses (₹)", "Margin (₹)", "Profit (%)"]

Row = tuple[int, float, float, float, float]

DEFAULT_ROWS = [
    (2020, 1000.0, 700.0, 300.0, 30.0),
    (2021, 1200.0, 850.0, 350.0, 29.0),
    (2022, 1500.0, 1100.0, 400.0, 26.0),
    (2023, 1800.0, 1300.0, 500.0, 28.0),
]

rows: list[Row] = DEFAULT_ROWS

# -----------------------------
# Prepare Dataset Text
# -----------------------------

def format_dataset_table(cols: list[str], data_rows: list[Row]) -> str:
    header = " | ".join(cols)
    sep = " | ".join(["---"] * len(cols))
    lines = [header, sep]
    for year, revenue, expenses, margin, profit_pct in data_rows:
        lines.append(
            f"{int(year)} | {revenue:.2f} | {expenses:.2f} | {margin:.2f} | {profit_pct:.2f}"
        )
    return "\n".join(lines)


dataset_text = format_dataset_table(columns, rows)

# -----------------------------
# Prompt with Strict Output Format
# -----------------------------

prompt = f"""
You are an AI Financial & Business Analysis Assistant.

Your task is to analyze historical company financial data and produce a structured business analysis report.

STRICT RULES:
- Use ONLY the dataset provided below.
- Do NOT use external knowledge.
- Do NOT predict stock prices.
- Do NOT give investment advice (buy/sell/hold).
- Base all insights strictly on the dataset.

Dataset Table (Yearly Financial Data):
{dataset_text}

Columns:
Year, Revenue, Expenses, Margin, Profit Percentage

ANALYSIS TASK

Analyze how the company's financial metrics change over time.

Focus on:
- Revenue trends
- Expense behavior
- Margin and profitability patterns
- Overall financial stability

Support every insight with **data evidence from the dataset**.

IMPORTANT:
Follow the output structure EXACTLY.
Use the section headings exactly as written.

OUTPUT FORMAT

* Executive Summary
Provide a concise summary of the company’s historical financial performance.

* Key Performance Trends
Explain important trends observed in the dataset.

Examples of trends:
- Revenue trajectory
- Profitability changes
- Cost behavior
- Growth stability

Each trend must include supporting numbers from the dataset.

* Strengths
Strength:
Explanation:
Supporting Data:

* Weaknesses
Weakness:
Explanation:
Supporting Data:

* Risks and Uncertainties
Identify potential risks suggested by the dataset.

Examples:
- Declining revenue growth
- Increasing expenses
- Volatile profitability

Important:
These must be DATA-DRIVEN risks only, not predictions.

* Balanced Outlook
Provide a neutral interpretation of the company’s position based strictly on the dataset.

Avoid speculation.

* Assumptions and Limitations
List:
- Assumptions made during analysis
- Missing or incomplete information
- Limitations caused by the dataset scope

STYLE REQUIREMENTS

- Use bullet points where appropriate
- Reference numbers from the dataset
- Maintain a neutral professional tone
- Do not include any sections outside the required format
"""

# -----------------------------
# Call Ollama
# -----------------------------

response = ollama.chat(
    model="qwen3.5",
    messages=[
        {"role": "system", "content": "You are a financial analysis assistant."},
        {"role": "user", "content": prompt}
    ],
    options={
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40
    }
)

# -----------------------------
# Extract Sections -> Variables
# -----------------------------

report_text = response["message"]["content"]


def _normalize_heading(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def extract_report_sections(text: str) -> dict:
    # Primary strategy: split by lines that look like "* Executive Summary"
    # Fallback: also accept plain headings like "Executive Summary" or "Executive Summary:"
    wanted = [
        "Executive Summary",
        "Key Performance Trends",
        "Strengths",
        "Weaknesses",
        "Risks and Uncertainties",
        "Balanced Outlook",
        "Assumptions and Limitations",
    ]
    wanted_norm = {_normalize_heading(w): w for w in wanted}

    sections: dict[str, list[str]] = {w: [] for w in wanted}
    current: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m = re.match(r"^\s*(?:\*+|\d+\.)?\s*(.+?)\s*$", line)
        candidate = m.group(1) if m else line.strip().rstrip(":")
        cand_norm = _normalize_heading(candidate)

        if cand_norm in wanted_norm:
            current = wanted_norm[cand_norm]
            continue

        if current is not None:
            sections[current].append(line)

    # Join and trim
    return {k: "\n".join(v).strip() for k, v in sections.items()}


parts = extract_report_sections(report_text)
executive_summary = parts["Executive Summary"]
key_performance_trends = parts["Key Performance Trends"]
strengths = parts["Strengths"]
weaknesses = parts["Weaknesses"]
risks_and_uncertainties = parts["Risks and Uncertainties"]
balanced_outlook = parts["Balanced Outlook"]
assumptions_and_limitations = parts["Assumptions and Limitations"]

# -----------------------------
# Save To PDF
# -----------------------------

def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _sanitize_pdf_text(s: str) -> str:
    # Standard PDF fonts don't reliably support the rupee sign; keep PDF text mostly ASCII.
    s = s.replace("₹", "INR")
    return s.encode("latin-1", "replace").decode("latin-1")


def _wrap_lines(text: str, width: int) -> list[str]:
    out: list[str] = []
    for line in text.splitlines() or [""]:
        if not line:
            out.append("")
            continue
        cur = line
        while len(cur) > width:
            cut = cur.rfind(" ", 0, width + 1)
            if cut <= 0:
                cut = width
            out.append(cur[:cut].rstrip())
            cur = cur[cut:].lstrip()
        out.append(cur)
    return out


def write_simple_pdf(path: Path, text: str) -> None:
    # Minimal multi-page PDF using built-in Type1 Helvetica.
    a4_w, a4_h = 595, 842
    font_size = 11
    leading = 14
    margin_x = 50
    start_y = 800
    lines_per_page = max(1, (start_y - 60) // leading)

    clean = _sanitize_pdf_text(text)
    lines = _wrap_lines(clean, width=85)

    pages: list[list[str]] = []
    for i in range(0, len(lines), lines_per_page):
        pages.append(lines[i : i + lines_per_page])
    if not pages:
        pages = [[""]]

    objects: list[bytes] = []

    def add_obj(b: bytes) -> int:
        objects.append(b)
        return len(objects)

    add_obj(b"")  # 1: catalog placeholder
    add_obj(b"")  # 2: pages placeholder

    page_obj_nums: list[int] = []
    for page_lines in pages:
        stream_lines = []
        stream_lines.append("BT")
        stream_lines.append(f"/F1 {font_size} Tf")
        stream_lines.append(f"{leading} TL")
        stream_lines.append(f"{margin_x} {start_y} Td")
        for ln in page_lines:
            stream_lines.append(f"({_pdf_escape(ln)}) Tj")
            stream_lines.append("T*")
        stream_lines.append("ET")
        stream = ("\n".join(stream_lines) + "\n").encode("latin-1")

        content_num = add_obj(
            b"<< /Length %d >>\nstream\n%bendstream\n" % (len(stream), stream)
        )

        page_num = add_obj(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {a4_w} {a4_h}] "
                f"/Resources << /Font << /F1 {{FONT_OBJ}} 0 R >> >> "
                f"/Contents {content_num} 0 R >>\n"
            ).encode("ascii")
        )
        page_obj_nums.append(page_num)

    font_num = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n")

    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    objects[1] = f"<< /Type /Pages /Kids [ {kids} ] /Count {len(page_obj_nums)} >>\n".encode(
        "ascii"
    )
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>\n"

    for obj_num in page_obj_nums:
        obj_idx = obj_num - 1
        objects[obj_idx] = objects[obj_idx].replace(b"{FONT_OBJ}", str(font_num).encode("ascii"))

    out = bytearray()
    out.extend(b"%PDF-1.4\n")

    offsets: list[int] = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"endobj\n")

    xref_start = len(out)
    out.extend(b"xref\n")
    out.extend(f"0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    out.extend(b"trailer\n")
    out.extend(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"))
    out.extend(b"startxref\n")
    out.extend(f"{xref_start}\n".encode("ascii"))
    out.extend(b"%%EOF\n")

    path.write_bytes(bytes(out))


run_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out_path = Path(__file__).with_name(f"market_analysis_report_{run_utc}.pdf")

pdf_text = (
    "BUSINESS ANALYSIS REPORT\n\n"
    "DATASET USED\n"
    f"{dataset_text}\n\n"
    "EXECUTIVE SUMMARY\n"
    f"{executive_summary}\n\n"
    "KEY PERFORMANCE TRENDS\n"
    f"{key_performance_trends}\n\n"
    "STRENGTHS\n"
    f"{strengths}\n\n"
    "WEAKNESSES\n"
    f"{weaknesses}\n\n"
    "RISKS AND UNCERTAINTIES\n"
    f"{risks_and_uncertainties}\n\n"
    "BALANCED OUTLOOK\n"
    f"{balanced_outlook}\n\n"
    "ASSUMPTIONS AND LIMITATIONS\n"
    f"{assumptions_and_limitations}\n\n"
    "RAW REPORT\n"
    f"{report_text}\n"
)

write_simple_pdf(out_path, pdf_text)

# -----------------------------
# Print Result
# -----------------------------

print("\n===== BUSINESS ANALYSIS REPORT =====\n")
print(report_text)
print(f"\nSaved PDF to: {out_path}")
