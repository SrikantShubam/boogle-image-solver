import sys
from pathlib import Path
import cv2
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from autoplay_v2 import template_ocr as t
t._build_templates()
OUT = REPO / "tools" / "dbg_templates"
OUT.mkdir(parents=True, exist_ok=True)
for L, tpl in t._UPPER_TEMPLATES.items():
    cv2.imwrite(str(OUT / f"U_{L}.png"), tpl)
print("Templates written to", OUT)
print("C pixel count:", int((t._UPPER_TEMPLATES['C']>0).sum()))
print("E pixel count:", int((t._UPPER_TEMPLATES['E']>0).sum()))
print("G pixel count:", int((t._UPPER_TEMPLATES['G']>0).sum()))
print("B pixel count:", int((t._UPPER_TEMPLATES['B']>0).sum()))
