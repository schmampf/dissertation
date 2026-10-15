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

The complete document catalog is cloned before the pages are transformed.
This preserves metadata, outlines, page labels, and named destinations. Link
annotation rectangles and all explicit destination coordinates are transformed
with the page content so that navigation remains aligned with the A4 pages.
"""
# python3 output/convert_to_a4.py thesis-a5.pdf thesis-a4.pdf

import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ArrayObject, FloatObject, NameObject, RectangleObject

A4_W = 595.2756  # 210 mm in pt
A4_H = 841.8898  # 297 mm in pt


def _number(value, scale, offset=0):
    """Transform a PDF coordinate while preserving null destination values."""
    if value is None or str(value) == "NullObject":
        return value
    return FloatObject(float(value) * scale + offset)


def _transform_destination(destination, scale, tx, ty):
    """Transform an explicit PDF destination array or a dictionary containing one."""
    if destination is None:
        return
    destination = destination.get_object()
    if hasattr(destination, "get") and "/D" in destination:
        _transform_destination(destination["/D"], scale, tx, ty)
        return
    if not isinstance(destination, ArrayObject) or len(destination) < 2:
        return

    mode = str(destination[1])
    if mode == "/XYZ" and len(destination) >= 5:
        destination[2] = _number(destination[2], scale, tx)
        destination[3] = _number(destination[3], scale, ty)
    elif mode in ("/FitH", "/FitBH") and len(destination) >= 3:
        destination[2] = _number(destination[2], scale, ty)
    elif mode in ("/FitV", "/FitBV") and len(destination) >= 3:
        destination[2] = _number(destination[2], scale, tx)
    elif mode == "/FitR" and len(destination) >= 6:
        destination[2] = _number(destination[2], scale, tx)
        destination[3] = _number(destination[3], scale, ty)
        destination[4] = _number(destination[4], scale, tx)
        destination[5] = _number(destination[5], scale, ty)


def _transform_name_tree(node, scale, tx, ty):
    """Transform every destination stored in a PDF name tree."""
    if node is None:
        return
    node = node.get_object()
    for kid in node.get("/Kids", []):
        _transform_name_tree(kid, scale, tx, ty)
    names = node.get("/Names", [])
    for index in range(1, len(names), 2):
        _transform_destination(names[index], scale, tx, ty)


def _transform_outlines(node, scale, tx, ty, visited=None):
    """Transform explicit destinations in the linked outline tree."""
    if node is None:
        return
    if visited is None:
        visited = set()
    while node is not None:
        key = (getattr(node, "idnum", None), getattr(node, "generation", None))
        if key == (None, None):
            key = id(node)
        if key in visited:
            return
        visited.add(key)
        item = node.get_object()
        _transform_destination(item.get("/Dest"), scale, tx, ty)
        action = item.get("/A")
        if action is not None:
            _transform_destination(action, scale, tx, ty)
        _transform_outlines(item.get("/First"), scale, tx, ty, visited)
        node = item.get("/Next")


def _transform_navigation(writer, scale, tx, ty):
    """Transform catalog-level destinations without changing their targets."""
    root = writer.root_object
    names = root.get("/Names")
    if names is not None:
        _transform_name_tree(names.get_object().get("/Dests"), scale, tx, ty)

    legacy_dests = root.get("/Dests")
    if legacy_dests is not None:
        for destination in legacy_dests.get_object().values():
            _transform_destination(destination, scale, tx, ty)

    outlines = root.get("/Outlines")
    if outlines is not None:
        _transform_outlines(outlines.get_object().get("/First"), scale, tx, ty)
    _transform_destination(root.get("/OpenAction"), scale, tx, ty)


def _transform_annotations(page, scale, tx, ty):
    """Keep link hitboxes and direct annotation destinations aligned with content."""
    annotations = page.get("/Annots")
    if annotations is None:
        return 0

    count = 0
    for reference in annotations.get_object():
        annotation = reference.get_object()
        rect = annotation.get("/Rect")
        if rect is not None and len(rect) == 4:
            annotation[NameObject("/Rect")] = RectangleObject(
                (
                    float(rect[0]) * scale + tx,
                    float(rect[1]) * scale + ty,
                    float(rect[2]) * scale + tx,
                    float(rect[3]) * scale + ty,
                )
            )
            count += 1

        quad_points = annotation.get("/QuadPoints")
        if quad_points is not None:
            annotation[NameObject("/QuadPoints")] = ArrayObject(
                [
                    FloatObject(float(value) * scale + (tx if index % 2 == 0 else ty))
                    for index, value in enumerate(quad_points)
                ]
            )

        _transform_destination(annotation.get("/Dest"), scale, tx, ty)
        action = annotation.get("/A")
        if action is not None:
            _transform_destination(action, scale, tx, ty)
    return count


def convert(src, dst):
    reader = PdfReader(src)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    writer.pdf_header = reader.pdf_header

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
    annotation_count = 0
    for i, page in enumerate(writer.pages):
        mb = page.mediabox
        if (round(float(mb.width), 2), round(float(mb.height), 2)) != (
            round(orig_w, 2),
            round(orig_h, 2),
        ):
            mismatches += 1
            print(f"  WARNING page {i+1}: unexpected mediabox {mb}")

        page.add_transformation(Transformation().scale(scale).translate(tx, ty))
        annotation_count += _transform_annotations(page, scale, tx, ty)
        page.mediabox = RectangleObject((0, 0, A4_W, A4_H))
        page.cropbox = RectangleObject((0, 0, A4_W, A4_H))
        for box in ("/TrimBox", "/ArtBox", "/BleedBox"):
            if box in page:
                del page[box]
        # /Rotate is left untouched on purpose.

    _transform_navigation(writer, scale, tx, ty)

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        writer.write(f)

    print(
        f"Wrote {dst}: {len(writer.pages)} pages, "
        f"{annotation_count} annotation rectangles transformed, "
        f"{mismatches} mediabox mismatches"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} SOURCE.pdf DESTINATION.pdf")
    convert(sys.argv[1], sys.argv[2])
