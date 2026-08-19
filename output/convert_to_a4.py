#!/usr/bin/env python3
"""Convert an A5-native thesis PDF to A4 pages.

Every page is uniformly scaled by the SAME factor (the one that makes the
common case, a plain A5 portrait page, fill an A4 sheet) and centered. The
mediabox for every page is identical A5 (14.8 x 21 cm) before conversion,
even for pages that carry a /Rotate flag for landscape figures, since
/Rotate is only a display-time flag applied on top of the same physical
page box. That means one global scale+translate transform, left untouched
by /Rotate, reproduces the exact current look at A4 size for every page,
including the rotated one, with no per-page special casing and no risk of
cropping.
"""
# python3 output/convert_to_a4.py thesis.pdf output/pdf/thesis-a4.pdf

import sys
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject

A4_W = 595.2756  # 210 mm in pt
A4_H = 841.8898  # 297 mm in pt


def convert(src, dst):
    reader = PdfReader(src)
    writer = PdfWriter()

    # Determine one global scale from the first page's mediabox. All pages
    # in this document share the same A5 mediabox regardless of /Rotate.
    first_mb = reader.pages[0].mediabox
    orig_w = float(first_mb.width)
    orig_h = float(first_mb.height)
    scale = min(A4_W / orig_w, A4_H / orig_h)
    new_w = orig_w * scale
    new_h = orig_h * scale
    tx = (A4_W - new_w) / 2
    ty = (A4_H - new_h) / 2
    print(f"Global scale factor: {scale:.5f}  (margin x={tx:.2f}pt y={ty:.2f}pt)")

    mismatches = 0
    for i, page in enumerate(reader.pages):
        mb = page.mediabox
        if (round(float(mb.width), 2), round(float(mb.height), 2)) != (
            round(orig_w, 2),
            round(orig_h, 2),
        ):
            mismatches += 1
            print(f"  WARNING page {i+1}: unexpected mediabox {mb}")

        page.add_transformation(Transformation().scale(scale).translate(tx, ty))
        page.mediabox = RectangleObject((0, 0, A4_W, A4_H))
        page.cropbox = RectangleObject((0, 0, A4_W, A4_H))
        for box in ("/TrimBox", "/ArtBox", "/BleedBox"):
            if box in page:
                del page[box]
        # /Rotate is left untouched on purpose.

        writer.add_page(page)

    with open(dst, "wb") as f:
        writer.write(f)

    print(f"Wrote {dst}: {len(writer.pages)} pages, {mismatches} mediabox mismatches")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
