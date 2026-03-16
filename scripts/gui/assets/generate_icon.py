"""Generate FreeClimber icon - macOS Big Sur style rounded rectangle."""

import os
import shutil
import subprocess

from PIL import Image, ImageDraw

SIZE = 1024
NAVY_TOP = (15, 15, 26)       # #0f0f1a
NAVY_BOT = (26, 26, 62)       # #1a1a3e
TEAL = (83, 168, 182)          # #53a8b6
CORAL = (233, 69, 96)          # #e94560

CORNER_RADIUS = int(SIZE * 0.22)  # Big Sur style ~22%


def lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_rounded_rect_mask(size, radius):
    """Create anti-aliased rounded rect mask at 4x then downscale."""
    scale = 4
    big = size * scale
    r = radius * scale
    mask = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, big - 1, big - 1], radius=r, fill=255)
    return mask.resize((size, size), Image.LANCZOS)


def make_gradient(size, top_color, bot_color):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for y in range(size):
        t = y / (size - 1)
        c = lerp_color(top_color, bot_color, t)
        for x in range(size):
            img.putpixel((x, y), c + (255,))
    return img


def bezier_point(pts, t):
    """De Casteljau cubic bezier evaluation."""
    while len(pts) > 1:
        pts = [
            (p0[0] * (1 - t) + p1[0] * t, p0[1] * (1 - t) + p1[1] * t)
            for p0, p1 in zip(pts[:-1], pts[1:])
        ]
    return pts[0]


def draw_curve(draw, control_points, color, width, num_segments=200):
    """Draw a smooth cubic bezier curve with anti-aliasing via line segments."""
    points = []
    for i in range(num_segments + 1):
        t = i / num_segments
        points.append(bezier_point(control_points, t))
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=color, width=width)
    return points


def generate_master_icon():
    # Work at 4x for anti-aliasing, then downscale
    S = SIZE * 2
    scale = S / SIZE

    # Gradient background
    bg = make_gradient(S, NAVY_TOP, NAVY_BOT)

    # Rounded rect mask
    mask = draw_rounded_rect_mask(S, int(CORNER_RADIUS * scale))

    # Apply mask
    icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    icon.paste(bg, mask=mask)

    draw = ImageDraw.Draw(icon)

    # Margins for the curve content area
    margin = int(S * 0.18)
    content_w = S - 2 * margin
    content_h = S - 2 * margin

    # Main climbing velocity curve - smooth upward sweep from bottom-left to top-right
    # Starts shallow, accelerates upward (like a climbing velocity trajectory)
    p0 = (margin + content_w * 0.05, margin + content_h * 0.88)
    p1 = (margin + content_w * 0.35, margin + content_h * 0.82)
    p2 = (margin + content_w * 0.60, margin + content_h * 0.45)
    p3 = (margin + content_w * 0.95, margin + content_h * 0.10)
    main_pts = [p0, p1, p2, p3]

    # Draw a subtle glow/shadow under the curve
    glow_color = (TEAL[0] // 3, TEAL[1] // 3, TEAL[2] // 3, 80)
    for offset in range(12, 0, -2):
        shifted = [(p[0], p[1] + offset * scale) for p in main_pts]
        pts = []
        for i in range(201):
            t = i / 200
            pts.append(bezier_point(shifted, t))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=glow_color, width=int(6 * scale))

    # Main curve - thick teal line
    draw_curve(draw, main_pts, TEAL + (255,), int(8 * scale), 300)

    # Secondary subtle curve (fainter, representing variance/spread)
    secondary_offset = content_h * 0.08
    sec_pts = [(p[0], p[1] + secondary_offset) for p in main_pts]
    sec_color = (TEAL[0], TEAL[1], TEAL[2], 90)
    draw_curve(draw, sec_pts, sec_color, int(4 * scale), 200)

    # Fill area between curves with subtle gradient
    fill_overlay = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_overlay)

    main_sampled = [bezier_point(main_pts, t / 100) for t in range(101)]
    sec_sampled = [bezier_point(sec_pts, t / 100) for t in range(101)]

    polygon_pts = main_sampled + list(reversed(sec_sampled))
    fill_draw.polygon(
        [(int(p[0]), int(p[1])) for p in polygon_pts],
        fill=(TEAL[0], TEAL[1], TEAL[2], 35),
    )
    icon = Image.alpha_composite(icon, fill_overlay)
    draw = ImageDraw.Draw(icon)

    # Coral data points along the main curve at specific intervals
    data_point_ts = [0.12, 0.28, 0.42, 0.58, 0.72, 0.85, 0.95]
    dot_radius = int(10 * scale)
    small_dot_radius = int(7 * scale)

    for i, t in enumerate(data_point_ts):
        pt = bezier_point(main_pts, t)
        x, y = int(pt[0]), int(pt[1])
        r = dot_radius if i % 2 == 0 else small_dot_radius

        # Coral dot with slight glow
        draw.ellipse(
            [x - r - 3, y - r - 3, x + r + 3, y + r + 3],
            fill=(CORAL[0], CORAL[1], CORAL[2], 60),
        )
        draw.ellipse([x - r, y - r, x + r, y + r], fill=CORAL + (255,))
        # Highlight
        hr = max(2, r // 3)
        draw.ellipse(
            [x - hr, y - r + 2, x + hr, y - r + 2 + hr * 2],
            fill=(255, 120, 140, 100),
        )

    # Subtle axis lines (very faint)
    axis_color = (255, 255, 255, 30)
    # Y axis
    draw.line(
        [(margin, margin + content_h * 0.05), (margin, margin + content_h * 0.95)],
        fill=axis_color,
        width=int(2 * scale),
    )
    # X axis
    draw.line(
        [
            (margin, margin + content_h * 0.95),
            (margin + content_w * 0.98, margin + content_h * 0.95),
        ],
        fill=axis_color,
        width=int(2 * scale),
    )

    # Small tick marks on axes
    tick_color = (255, 255, 255, 20)
    for i in range(1, 5):
        ty = margin + content_h * 0.95 - (content_h * 0.9 * i / 4)
        draw.line(
            [(margin - 4 * scale, ty), (margin + 4 * scale, ty)],
            fill=tick_color,
            width=int(1.5 * scale),
        )
    for i in range(1, 5):
        tx = margin + content_w * i / 5
        draw.line(
            [
                (tx, margin + content_h * 0.95 - 4 * scale),
                (tx, margin + content_h * 0.95 + 4 * scale),
            ],
            fill=tick_color,
            width=int(1.5 * scale),
        )

    # Downscale to 1024 with high-quality resampling
    final = icon.resize((SIZE, SIZE), Image.LANCZOS)

    # Re-apply clean mask at final size
    final_mask = draw_rounded_rect_mask(SIZE, CORNER_RADIUS)
    result = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    result.paste(final, mask=final_mask)

    return result


def main():
    print("Generating 1024x1024 master icon...")
    master = generate_master_icon()

    assets_dir = os.path.dirname(os.path.abspath(__file__))
    iconset_dir = os.path.join(assets_dir, "FreeClimber.iconset")

    # Create iconset directory
    if os.path.exists(iconset_dir):
        shutil.rmtree(iconset_dir)
    os.makedirs(iconset_dir)

    # macOS icon sizes: name -> pixel size
    icon_sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_64x64.png": 64,       # sometimes needed
        "icon_64x64@2x.png": 128,   # sometimes needed
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }

    for name, px in icon_sizes.items():
        resized = master.resize((px, px), Image.LANCZOS)
        resized.save(os.path.join(iconset_dir, name), "PNG")
    print("  Created iconset with all sizes")

    # Build .icns
    icns_path = os.path.join(assets_dir, "..", "..", "..", "FreeClimber.icns")
    icns_path = os.path.abspath(icns_path)
    try:
        subprocess.run(
            ["iconutil", "-c", "icns", iconset_dir, "-o", icns_path],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  Created {icns_path}")
    except subprocess.CalledProcessError as e:
        print(f"  iconutil warning: {e.stderr.strip()}")
        # Remove non-standard sizes and retry
        for name in ["icon_64x64.png", "icon_64x64@2x.png"]:
            path = os.path.join(iconset_dir, name)
            if os.path.exists(path):
                os.remove(path)
        try:
            subprocess.run(
                ["iconutil", "-c", "icns", iconset_dir, "-o", icns_path],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"  Created {icns_path} (retry without 64x64)")
        except subprocess.CalledProcessError as e2:
            print(f"  iconutil failed: {e2.stderr.strip()}")

    # Save logo files for GUI use
    logo_64 = master.resize((64, 64), Image.LANCZOS)
    logo_64.save(os.path.join(assets_dir, "logo_64.png"), "PNG")
    print("  Saved logo_64.png (64x64)")

    logo_256 = master.resize((256, 256), Image.LANCZOS)
    logo_256.save(os.path.join(assets_dir, "logo_256.png"), "PNG")
    print("  Saved logo_256.png (256x256)")

    # Clean up iconset directory
    shutil.rmtree(iconset_dir)
    print("  Cleaned up iconset directory")

    print("Done!")


if __name__ == "__main__":
    main()
