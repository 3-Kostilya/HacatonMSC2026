import json
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
import pdfplumber
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "analysis/results/summary.json").read_text(encoding="utf-8"))
checks = []
for s in summary["years"] + [summary["sample"]]:
    for k in ["alarm", "values", "daily", "channels", "types", "systems"]:
        assert sum(s[k].values()) == s["rows"], (s["file"], k)
    assert sum(s["failure_channels"].values()) == s["values"].get("Неисправен", 0)
    assert sum(s["failure_types"].values()) == s["values"].get("Неисправен", 0)
    assert sum(s["daily_alarm"].values()) == sum(s["alarm"].get(v, 0) for v in ["t", "true"])
    assert sum(s["daily_failure"].values()) == s["values"].get("Неисправен", 0)
    assert sum(v["count"] for v in s["numeric_by_type"].values()) == s["numeric_rows"]
    if "archive_exit_code" in s:
        assert s["archive_exit_code"] == 0
    checks.append({"file": s["file"], "rows": s["rows"], "reconciled": True})
assert summary["totals"]["rows"] == sum(s["rows"] for s in summary["years"])
pdf = ROOT / "output/pdf/dataset_analysis_2019_2026.pdf"
qa = ROOT / "tmp/pdfs"
qa.mkdir(parents=True, exist_ok=True)
report = {"data_checks": checks, "pages": []}
with pdfplumber.open(pdf) as doc:
    for i, page in enumerate(doc.pages):
        words = page.extract_words()
        outside = [
            w["text"]
            for w in words
            if w["x0"] < 45 or w["x1"] > 550 or w["top"] < 35 or w["bottom"] > 813
        ]
        text = page.extract_text() or ""
        report["pages"].append(
            {
                "page": i + 1,
                "words": len(words),
                "outside_bounds": outside,
                "first_lines": text.splitlines()[:3],
            }
        )
        assert not outside, (i + 1, outside)
        assert len(words) > 35, (i + 1, "Nearly blank page")
    assert "Неисправен" in "".join(p.extract_text() for p in doc.pages)
doc = pdfium.PdfDocument(pdf)
images = []
for i in range(len(doc)):
    img = doc[i].render(scale=1.35).to_pil().convert("RGB")
    img.save(qa / f"page-{i + 1:02d}.png")
    images.append(img)
for start in range(0, len(images), 4):
    tw, th = 595, 842
    sheet = Image.new("RGB", (tw * 2 + 30, th * 2 + 50), "#dce2e5")
    draw = ImageDraw.Draw(sheet)
    for j, img in enumerate(images[start : start + 4]):
        thumb = ImageOps.contain(img, (tw, th))
        x = 10 + (j % 2) * (tw + 10)
        y = 20 + (j // 2) * (th + 20)
        sheet.paste(thumb, (x, y))
        draw.text((x, y - 15), str(start + j + 1), fill="black")
    sheet.save(qa / f"contact-{start // 4 + 1:02d}.png")
(qa / "verification.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {"pages": len(images), "data_checks": len(checks), "contacts": (len(images) + 3) // 4},
        ensure_ascii=False,
    )
)
