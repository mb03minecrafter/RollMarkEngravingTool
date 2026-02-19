from __future__ import annotations

import subprocess
import webbrowser
from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np
from lxml import etree
from PIL import Image

# ================== CONFIG ==================
# Where your Inkscape executable lives. This script shells out to Inkscape a lot.
# If this path is wrong, everything dies immediately when we try to export.
INKSCAPE_EXE = r"C:\Program Files\Inkscape\bin\inkscape.exe"  # adjust if needed

# DPI we export the SVG at. Higher DPI = more pixels = smoother ramps/edges,
# but also bigger files + slower processing. 1270 is *spicy* on purpose.
EXPORT_DPI = 1270

# When we export a single object from SVG, it comes out as RGBA pixels.
# We convert to grayscale and then say:
#   - if pixel is visible (alpha > 0) AND grayscale <= THRESHOLD => "ink" (black)
#   - else => background (white)
# Lower threshold = stricter (thinner lines). Higher threshold = thicker/bleedier.
THRESHOLD = 15

# Padding we add around each exported object before distance transforms.
# Without padding, the distance transform can get clipped at the edge of the image.
CANVAS_PADDING_PX = 20

# How we pick the wall width for the distance-based ramp.
#   "manual" = use MANUAL_WALL_WIDTH_MM for everything
#   "auto"   = estimate from shape thickness (distance transform percentile)
WALL_WIDTH_MODE = "manual"  # "auto" or "manual"

# Auto mode heuristics
# The distance transform gives us a "radius to edge" for each ink pixel.
# If we take a high percentile of those distances (like 90%), we get a
# pretty decent "typical half-width" of the shape.
WALL_WIDTH_FRACTION = 1.0       # multiplier on the half-width -> final wall width
DIST_PERCENTILE = 90.0          # percentile used from dist values inside ink
WALL_WIDTH_MIN_PX = 2.0         # clamp to avoid useless tiny ramps
WALL_WIDTH_MAX_PX = 120.0       # clamp to avoid silly huge ramps

# SVG elements we consider "shapes" worth exporting individually.
# If your SVG is one single combined path, this is gonna be 1 shape.
SHAPE_TAGS = {"path", "rect", "circle", "ellipse", "polygon", "polyline", "line", "text"}

# Optional: only process shapes inside a certain group id in the SVG.
# Leave as None to process everything.
ONLY_WITHIN_GROUP_ID: str | None = None

# Optional: skip these ids if you know certain shapes should not be processed.
SKIP_IDS: set[str] = set()

# ---------------- Roll-mark weakness gradient ----------------
# This is a hacky-but-useful option: apply a left->right or right->left "washout"
# across the ink span. Basically: it makes one side of the mark weaker (lighter).
# Good for compensation if your process "over-cuts" on one side, or for style.
ENABLE_WEAKNESS_GRADIENT = False
WEAKNESS_DIRECTION = "right_to_left"  # "left_to_right" or "right_to_left"
WEAKNESS_MAX_WASHOUT = 1.0            # 0 = no effect, 1 = max effect
INK_THRESHOLD = 254                   # anything <= this is considered "ink"
# -------------------------------------------------------------

# If True, we open the final output image in your default viewer.
OPEN_FINAL = True
# ===========================================


@dataclass
class ShapeRef:
    # Simple container for "what shape are we exporting"
    id: str
    tag: str


def run_inkscape(cmd: list[str]) -> None:
    """
    Run Inkscape CLI command and hard-fail if it errors.
    I want loud failure here because if Inkscape silently fails,
    we end up generating garbage images and wasting time.
    """
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "Inkscape failed.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{res.stdout}\n\n"
            f"STDERR:\n{res.stderr}\n"
        )


def localname(tag: str) -> str:
    """
    SVG tags can come through like '{namespace}path'.
    This strips the namespace so we can compare to 'path', 'rect', etc.
    """
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def recolor_svg_black(tree: etree._ElementTree) -> None:
    """
    Force the SVG to be pure black ink (for anything that's not 'none').

    Why:
    - exported PNGs can have colors/styles
    - we want thresholding to behave consistently
    - black ink = obvious "on/off" after grayscale threshold

    We handle:
    - direct fill/stroke attributes
    - inline style="fill:...;stroke:...;"
    """
    for elem in tree.iter():
        # lxml sometimes includes comments or weird nodes; ignore anything not a real tag
        if not isinstance(elem.tag, str):
            continue

        # Direct attributes first
        if "fill" in elem.attrib and elem.attrib["fill"] not in ("none", ""):
            elem.attrib["fill"] = "#000000"
        if "stroke" in elem.attrib and elem.attrib["stroke"] not in ("none", ""):
            elem.attrib["stroke"] = "#000000"

        # Inline style string (classic SVG mess)
        style = elem.attrib.get("style")
        if style:
            parts = []
            for part in style.split(";"):
                part = part.strip()
                if not part:
                    continue
                k, _, v = part.partition(":")
                k = k.strip().lower()
                v = v.strip()
                if k == "fill" and v not in ("none", ""):
                    parts.append("fill:#000000")
                elif k == "stroke" and v not in ("none", ""):
                    parts.append("stroke:#000000")
                else:
                    # keep everything else (stroke-width, opacity, etc)
                    parts.append(part)
            elem.attrib["style"] = ";".join(parts)


def ensure_ids_for_shapes(tree: etree._ElementTree) -> list[ShapeRef]:
    """
    Walk the SVG and make sure every exportable shape has a unique id.

    Why:
    - Inkscape exports per-object using --export-id
    - If something has no id, we can't target it cleanly
    - So we auto-assign ids like auto_shape_0, auto_shape_1, ...

    Also supports:
    - ONLY_WITHIN_GROUP_ID: restrict scan to a specific group
    - SKIP_IDS: ignore shapes we don't want
    """
    root = tree.getroot()

    # If we only want one group, find it and use that as the search root
    if ONLY_WITHIN_GROUP_ID:
        group = root.xpath(f".//*[@id='{ONLY_WITHIN_GROUP_ID}']")
        if not group:
            raise ValueError(f"Group id '{ONLY_WITHIN_GROUP_ID}' not found in SVG.")
        search_root = group[0]
    else:
        search_root = root

    # Collect all ids already used in the whole doc (so we don't collide)
    used = {e.attrib["id"] for e in root.iter() if isinstance(e.tag, str) and "id" in e.attrib}

    shapes: list[ShapeRef] = []
    auto_i = 0

    for e in search_root.iter():
        if not isinstance(e.tag, str):
            continue

        tag = localname(e.tag)

        # Only process shape-ish tags
        if tag not in SHAPE_TAGS:
            continue

        # If no id, generate one
        eid = e.attrib.get("id")
        if not eid:
            while True:
                candidate = f"auto_shape_{auto_i}"
                auto_i += 1
                if candidate not in used:
                    eid = candidate
                    e.attrib["id"] = eid
                    used.add(eid)
                    break

        # Skip list support
        if eid in SKIP_IDS:
            continue

        shapes.append(ShapeRef(id=eid, tag=tag))

    return shapes


def export_object_png(svg_path: Path, out_png: Path, object_id: str, dpi: int) -> None:
    """
    Export exactly one SVG object (by id) to a PNG using Inkscape CLI.

    Notes:
    - --export-id-only means "only render that object", not the whole page.
    - --export-area-page keeps the output canvas consistent (same size for all objects),
      which is important because later we composite everything together.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        INKSCAPE_EXE,
        str(svg_path),
        "--export-type=png",
        f"--export-filename={str(out_png)}",
        f"--export-dpi={dpi}",
        f"--export-id={object_id}",
        "--export-id-only",
        "--export-area-page",
    ]
    run_inkscape(cmd)


def threshold_object_png_to_bw(png_path: Path, threshold: int) -> Image.Image:
    """
    Take the exported object PNG (RGBA), and turn it into a clean black/white mask.

    Rules:
    - only pixels with alpha > 0 are considered "real"
    - convert RGB -> grayscale
    - if grayscale <= threshold => black (ink)
    - else => white (background)

    Output: PIL Image, mode "L" (8-bit grayscale), but basically binary (0 or 255).
    """
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    rgb = arr[:, :, :3]
    a = arr[:, :, 3]

    # Standard luminance conversion (perceptual-ish)
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)

    # Start as all white
    out = np.full(gray.shape, 255, dtype=np.uint8)

    # Only consider visible pixels (alpha > 0)
    visible = a > 0

    # Anything dark enough becomes ink
    out[visible & (gray <= threshold)] = 0
    return Image.fromarray(out, mode="L")


def pad_image_white(img_l: Image.Image, pad_px: int) -> Image.Image:
    """
    Add a white border around the image.

    This is just safety so the distance transform doesn't get weird
    right at the edges of the exported canvas.
    """
    img_l = img_l.convert("L")
    if pad_px <= 0:
        return img_l
    w, h = img_l.size
    canvas = Image.new("L", (w + 2 * pad_px, h + 2 * pad_px), 255)
    canvas.paste(img_l, (pad_px, pad_px))
    return canvas


def compute_inside_and_distance(img_bw: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute:
    - inside mask: 1 where ink is (black pixels), 0 otherwise
    - distance transform: for each inside pixel, distance to nearest edge

    We use OpenCV distanceTransform which expects a binary-ish image where:
    - non-zero pixels are treated as "foreground"
    """
    arr = np.array(img_bw.convert("L"), dtype=np.uint8)

    # ink (black) becomes 1, background becomes 0
    inside = (arr < 128).astype(np.uint8)

    # L2 distance to nearest background pixel
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    return inside, dist


def px_to_mm(px: float, dpi: int) -> float:
    """
    Convert pixels to millimeters based on DPI.
    DPI = dots per inch, and 1 inch = 25.4 mm.
    """
    return (px / float(dpi)) * 25.4


def wall_width_mm_to_px(dpi: int, wall_width_mm: float) -> float:
    """
    Convert your manual wall width (mm) into pixels at the export DPI.
    Clamp to at least 1px because 0px ramp makes no sense.
    """
    px = (float(wall_width_mm) / 25.4) * float(dpi)
    return max(px, 1.0)


def estimate_wall_width_px_auto(img_bw: Image.Image) -> tuple[float, float]:
    """
    Auto-estimate the ramp width (in pixels) from the shape thickness.

    Process:
    - compute inside mask + distance transform
    - collect distance values for inside pixels
    - take DIST_PERCENTILE of those distances -> "typical half-width"
    - multiply by WALL_WIDTH_FRACTION and clamp

    Returns:
    - wall_width_px (final ramp width used)
    - half_width_px (raw percentile value before scaling)
    """
    inside, dist = compute_inside_and_distance(img_bw)

    # If there is no ink at all, just return the minimum so we don't crash
    if inside.sum() == 0:
        return WALL_WIDTH_MIN_PX, 0.0

    vals = dist[inside == 1]
    if vals.size == 0:
        return WALL_WIDTH_MIN_PX, 0.0

    half_width_px = float(np.percentile(vals, DIST_PERCENTILE))
    wall_width_px = float(np.clip(half_width_px * WALL_WIDTH_FRACTION, WALL_WIDTH_MIN_PX, WALL_WIDTH_MAX_PX))
    return wall_width_px, half_width_px


def linear_wall_falloff(img_bw: Image.Image, wall_width_px: float) -> Image.Image:
    """
    Build the actual "sloped wall" grayscale ramp for one shape.

    Idea:
    - edge of ink should be "light" (near white)
    - as you go inward from the edge, it should ramp down toward black
    - the ramp happens over wall_width_px distance

    Implementation:
    - distanceTransform gives dist-to-edge for inside pixels
    - normalize: t = dist / wall_width_px (0..1)
    - intensity = 255 * (1 - t)
      so:
        dist=0 => 255 (white-ish edge)
        dist>=wall_width => 0 (black interior)
    Background stays white (255).
    """
    inside, dist = compute_inside_and_distance(img_bw)

    # start output as all white
    out = np.full(inside.shape, 255, dtype=np.uint8)

    if inside.sum() == 0:
        return Image.fromarray(out, mode="L")

    max_dist = float(dist.max())
    if max_dist <= 0.0:
        # if dist is degenerate, just make the ink fully black
        out[inside == 1] = 0
        return Image.fromarray(out, mode="L")

    # safety clamp: ramp width can't exceed max_dist or be ~0
    wall_width_px = float(max(1e-6, min(wall_width_px, max_dist)))

    t = np.clip(dist / wall_width_px, 0.0, 1.0)
    ramp = (255.0 * (1.0 - t)).astype(np.uint8)
    out[inside == 1] = ramp[inside == 1]
    return Image.fromarray(out, mode="L")


def find_ink_span_x(img_arr: np.ndarray, ink_threshold: int = 254) -> tuple[int, int] | None:
    """
    Find the horizontal span (x0..x1) where there is ANY ink present.

    Why:
    - weakness gradient is applied across the ink region only
      (not the full canvas where there's just white padding)

    ink_threshold:
    - treat anything <= this as ink (so 254 basically means "anything not pure white")
    """
    if img_arr.ndim != 2:
        raise ValueError("Expected a 2D grayscale array.")

    ink_mask = img_arr <= ink_threshold
    if not ink_mask.any():
        return None

    # find columns that have at least one ink pixel
    col_has_ink = ink_mask.any(axis=0)
    xs = np.where(col_has_ink)[0]
    return int(xs[0]), int(xs[-1])


def apply_weakness_gradient_over_ink_span(
    img_arr: np.ndarray,
    enabled: bool,
    direction: str,
    max_washout: float,
    ink_threshold: int = 254,
) -> np.ndarray:
    """
    Apply a left/right gradient that "washes out" ink.

    Think:
    - we multiply the ink depth by some factor that changes with X.
    - more washout => closer to white => weaker engraving.

    We only touch ink pixels (<= ink_threshold).
    Background stays white.

    direction:
    - left_to_right: starts strong on left, weaker on right
    - right_to_left: starts strong on right, weaker on left
    """
    if not enabled or max_washout <= 0.0:
        return img_arr

    max_washout = float(max(0.0, max_washout))

    span = find_ink_span_x(img_arr, ink_threshold=ink_threshold)
    if span is None:
        return img_arr

    x0, x1 = span
    if x1 <= x0:
        return img_arr

    h, w = img_arr.shape
    width = (x1 - x0)

    # normalized x position across the ink span
    x = np.arange(w, dtype=np.float32)
    t = (x - x0) / float(width)
    t = np.clip(t, 0.0, 1.0)

    # gradient direction selection
    if direction.lower() == "left_to_right":
        g = t
    elif direction.lower() == "right_to_left":
        g = 1.0 - t
    else:
        raise ValueError('WEAKNESS_DIRECTION must be "left_to_right" or "right_to_left".')

    # expand to 2D factor field
    g2 = np.tile(g.reshape(1, -1), (h, 1))

    # factor goes from 1.0 down to (1.0 - max_washout)
    factor = 1.0 - (max_washout * g2)
    factor = np.clip(factor, 0.0, 1.0)

    arr = img_arr.astype(np.float32)
    ink = arr <= ink_threshold

    # Washout math:
    # - 255 is "no cut" (white)
    # - 0 is "full cut" (black)
    # We want to reduce the depth (darkness) by factor:
    #   depth = (255 - arr)
    #   new_depth = depth * factor
    #   new_arr = 255 - new_depth
    out = arr.copy()
    out[ink] = 255.0 - (255.0 - arr[ink]) * factor[ink]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def make_full_depth_ink_image(img_arr: np.ndarray, ink_threshold: int = 254) -> np.ndarray:
    """
    Build a pure "full depth" mask.

    Full-depth means:
    - if it's ink at all => black (0)
    - otherwise => white (255)

    This represents the "straight wall" contribution: a vertical cut down.
    """
    out = np.full(img_arr.shape, 255, dtype=np.uint8)
    out[img_arr <= ink_threshold] = 0
    return out


def blend_straight_and_sloped(
    sloped_img: np.ndarray,
    full_depth_img: np.ndarray,
    straight_wall_depth_mm: float,
    sloped_wall_depth_mm: float,
    ink_threshold: int = 254,
) -> np.ndarray:
    """
    Blend two different "profiles" into one image:
      - full_depth_img: the square/vertical (straight wall) part
      - sloped_img:     the ramp/trapezoid part

    The blend weight is based on the ratio of depth contributions:
      w_full = straight / (straight + sloped)
      w_slope = sloped / (straight + sloped)

    We blend in intensity space because this grayscale is basically a linear
    depth fraction (white=0 depth, black=max depth for that stage).

    Important:
    - Background stays white no matter what.
    - Only ink pixels get blended.
    """
    straight = float(max(0.0, straight_wall_depth_mm))
    sloped = float(max(0.0, sloped_wall_depth_mm))

    # If both are zero, just return the sloped image (doesn't matter much)
    if straight == 0.0 and sloped == 0.0:
        return sloped_img.copy()

    # If one side is zero, no need to blend, just return the other
    if sloped == 0.0:
        return full_depth_img.copy()
    if straight == 0.0:
        return sloped_img.copy()

    total = straight + sloped
    w_full = straight / total
    w_slope = sloped / total  # = 1 - w_full

    # Weighted average
    blended = (w_full * full_depth_img.astype(np.float32) + w_slope * sloped_img.astype(np.float32))

    # Background stays white; only apply where we consider it ink.
    ink = sloped_img <= ink_threshold
    out = np.full(sloped_img.shape, 255, dtype=np.uint8)
    out[ink] = np.clip(blended[ink], 0.0, 255.0).astype(np.uint8)
    return out


def open_file(path: Path) -> None:
    """
    Open a file in the default system viewer.
    This uses a file:// URI so it works reliably on Windows.
    """
    webbrowser.open(path.resolve().as_uri())


# Your calibration value: how much depth you get per laser pass.
# This is super process-specific. If your settings change, update this.
MM_PER_PASS = 0.004  # from your calibration data


def passes_for_depth(depth_mm: float, mm_per_pass: float = MM_PER_PASS) -> int:
    """
    Convert target depth (mm) into "how many passes do I need?"
    We ceil because you can't do 12.3 passes lol.
    """
    if depth_mm <= 0:
        return 0
    if mm_per_pass <= 0:
        raise ValueError("mm_per_pass must be > 0")
    return math.ceil(depth_mm / mm_per_pass)


# Manual sloped wall width (mm)
# This defines how wide the ramp is from edge -> interior.
MANUAL_WALL_WIDTH_MM = 0.20

# New depth controls (mm)
# Straight wall depth = the vertical "step down" part
# Sloped wall depth   = the additional depth contributed by the ramp profile
# Total depth = straight + sloped
STRAIGHT_WALL_DEPTH_MM = 0.0   # square/vertical portion depth
SLOPED_WALL_DEPTH_MM = 0.20     # trapezoid/ramp portion depth (additional)


if __name__ == "__main__":
    # Compute total depth + max passes based on your calibration
    total_depth_mm = float(STRAIGHT_WALL_DEPTH_MM) + float(SLOPED_WALL_DEPTH_MM)
    max_passes = passes_for_depth(total_depth_mm, MM_PER_PASS)

    # Print out our "cross section recipe" so you can sanity check it
    print("=== Cross-section settings ===")
    print(f"Straight wall depth: {STRAIGHT_WALL_DEPTH_MM:.4f} mm")
    print(f"Sloped wall depth:   {SLOPED_WALL_DEPTH_MM:.4f} mm")
    print(f"Total depth:         {total_depth_mm:.4f} mm")
    print(f"MM per pass:         {MM_PER_PASS:.6f} mm/pass")
    print(f"MAX passes:          {max_passes} passes")

    # If we're blending, it's nice to show the weight so you know what you're getting.
    if SLOPED_WALL_DEPTH_MM > 0:
        w_full = (
            STRAIGHT_WALL_DEPTH_MM / (STRAIGHT_WALL_DEPTH_MM + SLOPED_WALL_DEPTH_MM)
            if STRAIGHT_WALL_DEPTH_MM > 0
            else 0.0
        )
        print(f"Blend weight (full-depth / straight portion): {w_full:.3f}")
    print("==============================\n")

    # File locations (expects this exact structure)
    in_svg = Path("input/markings.svg")
    out_dir = Path("output")
    tmp_dir = out_dir / "_tmp_objects"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Hard fail if input missing (no point continuing)
    if not in_svg.exists():
        raise FileNotFoundError(f"SVG not found: {in_svg.resolve()}")

    # Parse SVG with comments removed (keeps the tree cleaner)
    parser = etree.XMLParser(remove_comments=True)
    tree = etree.parse(str(in_svg), parser)

    # Step 1: force everything to black so thresholding behaves
    recolor_svg_black(tree)

    # Step 2: make sure every shape has an ID so we can export it
    shapes = ensure_ids_for_shapes(tree)

    # Save the working version so you can debug it in Inkscape if needed
    working_svg = out_dir / f"{in_svg.stem}_working_black_ids.svg"
    tree.write(str(working_svg), encoding="utf-8", xml_declaration=True)

    # If we got zero shapes, your SVG is probably not made of exportable objects
    # (or it's one combined thing and you expected more)
    if not shapes:
        raise RuntimeError(
            "No shapes found to export individually. "
            "If your SVG is a single combined path, per-shape ramp via export-id won't work."
        )

    # Validate mode
    mode = WALL_WIDTH_MODE.strip().lower()
    if mode not in ("auto", "manual"):
        raise ValueError('WALL_WIDTH_MODE must be "auto" or "manual".')

    print(f"Found {len(shapes)} shape(s). Export DPI={EXPORT_DPI}, pad={CANVAS_PADDING_PX}px, wall width mode={mode}")

    # Export first shape so we can learn base canvas size.
    # Because we export-area-page, every object export should be same canvas size.
    first_png = tmp_dir / f"{shapes[0].id}.png"
    export_object_png(working_svg, first_png, shapes[0].id, EXPORT_DPI)
    base_w, base_h = Image.open(first_png).size

    # We pad all objects, so the final composite canvas is padded too
    padded_w = base_w + 2 * CANVAS_PADDING_PX
    padded_h = base_h + 2 * CANVAS_PADDING_PX
    print(f"Base canvas: {base_w}x{base_h} -> padded: {padded_w}x{padded_h}")

    # Composite starts all white. We'll "min()" in each shape ramp.
    # min works because darker = deeper cut. White stays white unless ink exists.
    composite = np.full((padded_h, padded_w), 255, dtype=np.uint8)

    # Manual wall width calculation once (same for every shape)
    manual_wall_width_px = None
    if mode == "manual":
        manual_wall_width_px = wall_width_mm_to_px(EXPORT_DPI, MANUAL_WALL_WIDTH_MM)
        mm = px_to_mm(manual_wall_width_px, EXPORT_DPI)
        print(f"[MANUAL] wall_width={manual_wall_width_px:.2f}px ({mm:.4f} mm)")

    # Main loop: export shape -> threshold -> pad -> pick wall width -> build ramp -> composite it
    for i, s in enumerate(shapes, start=1):
        obj_png = tmp_dir / f"{s.id}.png"
        export_object_png(working_svg, obj_png, s.id, EXPORT_DPI)

        # Make clean 0/255 mask from the export
        bw = threshold_object_png_to_bw(obj_png, THRESHOLD)

        # Pad so distance transform doesn't get edge-clipped
        bw = pad_image_white(bw, CANVAS_PADDING_PX)

        # Decide wall width per shape (auto) or global (manual)
        if mode == "auto":
            wall_width_px, half_w_px = estimate_wall_width_px_auto(bw)
            half_mm = px_to_mm(half_w_px, EXPORT_DPI)
            ww_mm = px_to_mm(wall_width_px, EXPORT_DPI)

            print(
                f"[{i:03d}/{len(shapes):03d}] id={s.id} | "
                f"half≈{half_w_px:.2f}px ({half_mm:.4f}mm) | "
                f"wall_width={wall_width_px:.2f}px ({ww_mm:.4f}mm)"
            )
        else:
            wall_width_px = float(manual_wall_width_px)
            print(f"[{i:03d}/{len(shapes):03d}] id={s.id} | wall_width={wall_width_px:.2f}px (manual)")

        # Build the actual ramp (sloped walls)
        falloff = linear_wall_falloff(bw, wall_width_px)

        # Combine into composite (min = darkest wins)
        composite = np.minimum(composite, np.array(falloff, dtype=np.uint8))

    # Optional: apply weakness gradient across the ink span
    # (only matters if ENABLE_WEAKNESS_GRADIENT = True)
    composite_final = apply_weakness_gradient_over_ink_span(
        composite,
        enabled=ENABLE_WEAKNESS_GRADIENT,
        direction=WEAKNESS_DIRECTION,
        max_washout=WEAKNESS_MAX_WASHOUT,
        ink_threshold=INK_THRESHOLD,
    )

    # Build a pure "full depth" mask (black anywhere there's ink)
    full_depth_img = make_full_depth_ink_image(composite_final, ink_threshold=INK_THRESHOLD)

    # Blend the sloped ramp + full-depth mask to get the cross-section you want:
    # square (straight wall) + trapezoid (sloped wall)
    composite_profile = blend_straight_and_sloped(
        sloped_img=composite_final,
        full_depth_img=full_depth_img,
        straight_wall_depth_mm=STRAIGHT_WALL_DEPTH_MM,
        sloped_wall_depth_mm=SLOPED_WALL_DEPTH_MM,
        ink_threshold=INK_THRESHOLD,
    )

    # Output filenames
    out_sloped = out_dir / f"{in_svg.stem}_sloped_only_{mode}.png"
    out_full = out_dir / f"{in_svg.stem}_full_depth_{mode}.png"
    out_profile = out_dir / f"{in_svg.stem}_square_on_trapezoid_{mode}.png"

    # Save images with DPI metadata so scaling stays sane downstream
    Image.fromarray(composite_final, mode="L").save(out_sloped, dpi=(EXPORT_DPI, EXPORT_DPI))
    Image.fromarray(full_depth_img, mode="L").save(out_full, dpi=(EXPORT_DPI, EXPORT_DPI))
    Image.fromarray(composite_profile, mode="L").save(out_profile, dpi=(EXPORT_DPI, EXPORT_DPI))

    # Print final paths so you can copy/paste them
    print("\nOutputs:")
    print(f"  Sloped only:         {out_sloped.resolve()}")
    print(f"  Full depth (ink):    {out_full.resolve()}")
    print(f"  Square+trapezoid:    {out_profile.resolve()}")
    print(f"\nUse MAX passes = {max_passes} for total depth {total_depth_mm:.4f} mm.")

    # Pop open the final image if you want instant feedback
    if OPEN_FINAL:
        open_file(out_profile)
