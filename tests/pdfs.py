"""Small PDFs made in the tests: pages of text, and pages that are only a picture, as a scan."""

from io import BytesIO

from PIL import Image


def text_pdf(pages: list[list[str]]) -> bytes:
    """A PDF with one page per list of lines, in Helvetica, from the top of an A4 page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", ""]  # the page list is filled in below
    font = 3
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    kids = []
    for lines in pages:
        text = " ".join(
            f"1 0 0 1 72 {800 - 16 * n} Tm ({line.replace('(', r'\(').replace(')', r'\)')}) Tj"
            for n, line in enumerate(lines)
        )
        stream = f"BT /F1 11 Tf {text} ET".encode("latin-1")
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n{stream.decode('latin-1')}\nendstream"
        )
        content = len(objects)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out = BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1"))
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


def scanned_pdf(pages: int = 1) -> bytes:
    """A PDF whose pages are pictures with no text, as a scanner makes."""
    images = [Image.new("L", (595, 842), color=255) for _ in range(pages)]
    out = BytesIO()
    images[0].save(out, format="PDF", save_all=True, append_images=images[1:])
    return out.getvalue()
