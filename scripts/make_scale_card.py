"""Generate the printable scale card (ArUco 4x4 id 7, 100 mm black square + 100 mm verification bar)."""
import os

import cv2
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make(path: str, page, size_mm: float = 100.0, ident: int = 7):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    bm = cv2.aruco.generateImageMarker(d, ident, 6 * 20)  # 6x6 cells (1-cell black border + 4x4 payload)
    cells = bm[10::20, 10::20] > 127  # sample cell centres
    W, H = page
    c = canvas.Canvas(path, pagesize=page)
    cx, top = W / 2, H - 28 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(cx, top, "SCALE CARD - print at 100 % (Actual size), do not 'fit to page'")
    c.setFont("Helvetica", 10)
    c.drawCentredString(cx, top - 7 * mm, "Tape flat on a wall at chest height (1.2 - 1.6 m), in clear view, no glare, not behind furniture.")
    x0, y0 = cx - size_mm * mm / 2, top - 25 * mm - size_mm * mm
    c.setFillColorRGB(1, 1, 1)
    c.rect(x0 - 8 * mm, y0 - 8 * mm, size_mm * mm + 16 * mm, size_mm * mm + 16 * mm, stroke=0, fill=1)
    cell = size_mm * mm / 6
    c.setFillColorRGB(0, 0, 0)
    for i in range(6):
        for j in range(6):
            if not cells[i, j]:
                c.rect(x0 + j * cell, y0 + (5 - i) * cell, cell, cell, stroke=0, fill=1)
    # verification bar: exactly 100 mm
    by = y0 - 28 * mm
    c.setLineWidth(1)
    c.line(cx - 50 * mm, by, cx + 50 * mm, by)
    for k in range(11):
        h = 4 * mm if k in (0, 10) else 2 * mm
        c.line(cx - 50 * mm + k * 10 * mm, by - h / 2, cx - 50 * mm + k * 10 * mm, by + h / 2)
    c.setFont("Helvetica", 11)
    c.drawCentredString(cx, by - 9 * mm, "CHECK: this bar must measure exactly 10.0 cm with a ruler. If not, re-print at 100 %.")
    c.setFont("Helvetica", 9)
    c.drawCentredString(cx, 18 * mm, f"floorscan scale card v1 - ArUco DICT_4X4_50 id {ident} - black square side = {size_mm:.0f} mm")
    c.showPage()
    c.save()


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "capture"), exist_ok=True)
    make(os.path.join(ROOT, "capture", "scale_card_A4.pdf"), A4)
    make(os.path.join(ROOT, "capture", "scale_card_Letter.pdf"), letter)
    print("wrote capture/scale_card_A4.pdf, capture/scale_card_Letter.pdf")
