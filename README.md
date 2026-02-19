# RollMarkEngravingProject

## SVG to Grayscale Heightmap Generator for Precision Laser Engraving

This tool converts an SVG roll mark into calibrated grayscale depth maps suitable for grayscale or multi-pass laser engraving workflows.

It exports individual SVG objects using the Inkscape CLI, computes distance-transform wall ramps, and generates controlled cross-section profiles for clean, consistent engraving geometry.

Designed for precision roll marks and controlled wall tapering.

---

## Features

- Per-shape SVG export using Inkscape CLI
- Distance-transform based sloped wall ramps
- Adjustable ramp width (manual or automatic estimation)
- Square plus trapezoid cross-section blending
- Optional left or right weakness compensation gradient
- Depth-to-pass calculation using calibration data
- High DPI raster export for smooth grayscale transitions

---

## What This Solves

Traditional grayscale engraving often produces harsh vertical walls or inconsistent tapering.

This tool generates:

- Controlled tapered wall geometry
- Adjustable straight and sloped depth blending
- Consistent wall width in millimeters
- Predictable depth based on calibrated millimeters per pass

Instead of guessing at grayscale power curves, you generate geometry-aware heightmaps.

---

## Requirements

- Python 3.13 or recent version
- Inkscape installed with CLI support
- Python packages:
  - numpy
  - opencv-python
  - lxml
  - Pillow

Install dependencies:

pip install -r requirements.txt


---

## Inkscape Configuration

Inside `main.py`, verify the executable path:

INKSCAPE_EXE = r"C:\Program Files\Inkscape\bin\inkscape.exe"


If you are on macOS or Linux, update this path to your local Inkscape binary.

---

## Project Structure

RollMarkEngravingProject/
input/
markings.svg
output/
main.py
requirements.txt


Place your SVG in:

input/markings.svg


Run the script:

python main.py


Generated files will appear in:

output/


---

## Output Files

### Sloped Only
Distance-transform ramp from edges inward.

### Full Depth
Binary ink mask where all ink equals full depth.

### Square plus Trapezoid
Blended straight-wall and sloped-wall profile.

The console also reports required laser passes based on your `MM_PER_PASS` calibration.

---

## License

MIT License  
Copyright 2026 Matthew Bickham
