"""Export the GLACE pipeline flowchart from draw.io to the paper's PDF.

draw.io pads its PDF export unevenly (under 1 pt at the top, 4 to 6 pt at the right
and bottom), which draw.io itself cannot change. The exported page is therefore
reset to the drawn content plus an equal margin on every side, by changing only its
MediaBox and CropBox: the drawing and the embedded fonts stay as draw.io wrote them.

Run with ``pixi run -e notebooks python notebooks/pipeline-flowchart/export-flowchart.py``.
Needs draw.io (``drawio``, e.g. the snap) and ``xvfb-run`` on ``PATH``.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageOps
import pymupdf

from glace_production.config import settings

FIGURE_NAME = "pipeline-flowchart"
SOURCE = Path(__file__).resolve().parent / f"{FIGURE_NAME}.drawio"
FIGURE_DIR = settings.figures_dir / FIGURE_NAME  # one folder per notebook

MARGIN_PT = 3.0
"""Distance between the drawn content and the page edge, on every side."""

DETECT_DPI = 1200
"""Resolution at which the extent of the drawn content is measured."""

INK_THRESHOLD = 8
"""Darkness (0-255) above which a rendered pixel counts as drawn content."""


def export_drawio(source: Path, target: Path) -> None:
    """Export the first page of a draw.io diagram to a PDF cropped to its content.

    Parameters
    ----------
    source
        The ``.drawio`` file.
    target
        The PDF to write. The draw.io snap cannot reach hidden directories, ``/mnt``
        or ``/var/tmp``, so it has to lie in a plain directory under the home folder.
        xvfb-run keeps its X authority file in the same directory.
    """
    xvfb_run = shutil.which("xvfb-run")
    drawio = shutil.which("drawio")
    if xvfb_run is None or drawio is None:
        msg = "draw.io export needs both `xvfb-run` and `drawio` on PATH"
        raise FileNotFoundError(msg)
    command = [
        *(xvfb_run, "-a", drawio, "--no-sandbox", "--disable-gpu"),
        *("--export", "--format", "pdf", "--crop", "--output", str(target), str(source)),
    ]
    # xvfb-run writes the X authority file under TMPDIR, which .env points to /var/tmp.
    env = os.environ | {"TMPDIR": str(target.parent)}
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)  # noqa: S603 - fixed arguments
    if result.returncode != 0 or not target.is_file():
        msg = f"draw.io did not write {target}:\n{result.stderr}"
        raise RuntimeError(msg)


def pad_pdf(source: Path, margin: float = MARGIN_PT) -> bytes:
    """Set the page of a one-page PDF to its drawn content plus an equal margin.

    Parameters
    ----------
    source
        The PDF exported by draw.io.
    margin
        Margin around the drawn content, in points.

    Returns
    -------
    bytes
        The PDF with the new MediaBox and CropBox and otherwise unchanged content.
    """
    with pymupdf.open(source) as doc:
        page = doc[0]
        pixmap = page.get_pixmap(dpi=DETECT_DPI, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        ink = ImageOps.invert(image.convert("L")).point(lambda v: 255 if v > INK_THRESHOLD else 0)
        bbox = ink.getbbox()
        if bbox is None:
            msg = f"{source} renders as an empty page"
            raise ValueError(msg)
        x0, y0, x1, y1 = bbox
        scale = 72 / DETECT_DPI
        # The pixmap runs top-down, the MediaBox bottom-up in PDF space.
        height = page.mediabox.height
        page.set_mediabox(  # also resets the CropBox
            pymupdf.Rect(
                x0 * scale - margin,
                height - y1 * scale - margin,
                x1 * scale + margin,
                height - y0 * scale + margin,
            )
        )
        return doc.tobytes(garbage=0, deflate=False)


def main() -> None:
    """Export the flowchart and write the padded PDF to the figures directory."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target = FIGURE_DIR / f"{FIGURE_NAME}.pdf"
    # Next to the source, the one place the draw.io snap is sure to be able to write.
    with tempfile.TemporaryDirectory(dir=SOURCE.parent, prefix="export-") as tmp:
        exported = Path(tmp) / f"{FIGURE_NAME}.pdf"
        export_drawio(SOURCE, exported)
        target.write_bytes(pad_pdf(exported))
    print(target)


if __name__ == "__main__":
    main()
