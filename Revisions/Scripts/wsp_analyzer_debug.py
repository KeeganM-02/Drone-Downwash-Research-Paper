import os
import cv2
import numpy as np
import pandas as pd

# -----------------------------
# WSP Image Analysis with Debug
# -----------------------------
def analyze_wsp_with_debug(image_path, rel_subdir, debug_root, min_area=1):
    """
    Analyze a WSP image to detect droplets, coverage %, deposition, etc.
    Also saves a debug overlay image inside debug_root/rel_subdir.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect WSP card (ignore black background)
    _, wsp_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    wsp_mask = cv2.morphologyEx(wsp_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    wsp_mask = cv2.morphologyEx(wsp_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Step 2: Detect droplets inside WSP mask
    masked_gray = cv2.bitwise_and(gray, gray, mask=wsp_mask)
    _, thresh = cv2.threshold(masked_gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, wsp_mask)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    droplet_areas = []
    total_droplet_area = 0
    kept_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            perim = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perim*perim) if perim > 0 else 0
            droplet_areas.append(area)
            total_droplet_area += area
            kept_contours.append((cnt, area, circularity))

    card_area = np.count_nonzero(wsp_mask)

    # Step 3: Save debug overlay
    debug_filename = None
    if debug_root:
        debug_subdir = os.path.join(debug_root, rel_subdir)
        os.makedirs(debug_subdir, exist_ok=True)

        overlay = image.copy()

        mask_colored = np.zeros_like(image)
        mask_colored[wsp_mask == 255] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

        for (cnt, area, circ) in kept_contours:
            cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 1)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(overlay, f"{int(area)}",
                            (cx+3, cy+3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255,255,255), 1, cv2.LINE_AA)

        fname = os.path.basename(image_path)
        debug_filename = fname.replace(".png", "_overlay.png")
        out_path = os.path.join(debug_subdir, debug_filename)
        cv2.imwrite(out_path, overlay)

    return {
        "num_droplets": len(droplet_areas),
        "droplet_areas": droplet_areas,
        "total_droplet_area": total_droplet_area,
        "card_area": card_area,
        "coverage_percent": (total_droplet_area / card_area * 100) if card_area > 0 else 0,
        "debug_image": os.path.join(rel_subdir, debug_filename) if debug_filename else None
    }


def parse_filename(filename):
    """
    Parse filename like 'P1_Bot_Scan.png' -> Plant = P1, Canopy = Bot
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]  # Plant, Canopy Position
    return "Unknown", "Unknown"


def process_all_images(base_dir):
    results = []
    debug_root = os.path.join(base_dir, "debug_overlays")

    valid_dirs = ["5ft", "10ft", "15ft", "Backpack Sprayer"]

    # Conversion factors for px → mm
    px_to_mm = 25.4 / 600
    px_to_mm2 = px_to_mm ** 2

    for root, dirs, files in os.walk(base_dir):
        rel_root = os.path.relpath(root, base_dir)
        if not any(rel_root.startswith(vd) for vd in valid_dirs):
            continue

        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                rel_subdir = os.path.relpath(root, base_dir)

                # Top-level folder = height
                height = rel_root.split(os.sep)[0]

                plant, canopy = parse_filename(file)
                info = analyze_wsp_with_debug(image_path, rel_subdir, debug_root)

                # Convert px² → mm²
                deposition_mm2 = info["total_droplet_area"] * px_to_mm2
                card_area_mm2 = info["card_area"] * px_to_mm2

                # Coverage density (droplets/mm²)
                coverage_density = (info["num_droplets"] / card_area_mm2) if card_area_mm2 > 0 else 0

                results.append({
                    "Height": height,
                    "Plant": plant,
                    "Canopy_Position": canopy,
                    "DropletCount": info["num_droplets"],
                    "CoveragePercent": round(info["coverage_percent"], 3),
                    "CoverageDensity(droplets/mm2)": round(coverage_density, 5),
                    "Deposition_mm2": round(deposition_mm2, 3),
                    "CardArea_mm2": round(card_area_mm2, 3),
                    "DebugOverlay": info["debug_image"]
                })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Order Heights
    height_order = ["5ft", "10ft", "15ft", "Backpack Sprayer"]
    df["Height"] = pd.Categorical(df["Height"], categories=height_order, ordered=True)

    # Extract numeric plant index for sorting
    df["PlantNum"] = df["Plant"].str.extract(r"P(\d+)").astype(float)

    # Order by Height → PlantNum → Canopy_Position
    canopy_order = ["Bot", "Mid", "Top"]
    df["Canopy_Position"] = pd.Categorical(df["Canopy_Position"], categories=canopy_order, ordered=True)

    df = df.sort_values(["Height", "PlantNum", "Canopy_Position"]).reset_index(drop=True)
    df = df.drop(columns=["PlantNum"])  # don’t keep helper column in final CSV

    # Save to CSV
    output_csv = os.path.join(base_dir, "wsp_analysis_results.csv")
    df.to_csv(output_csv, index=False)

    print(f"\nAnalysis complete. Results saved to {output_csv}")
    print(f"Debug overlays saved to {debug_root}")


if __name__ == "__main__":
    base_dir = r"C:\Users\keega\Desktop\WSP_Images"
    process_all_images(base_dir)
