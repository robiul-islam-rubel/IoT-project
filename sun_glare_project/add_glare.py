import cv2
import numpy as np
import pandas as pd
import os
import cv2
import numpy as np

def add_glare_relative_to_sign(base_img, glare_img, bbox, position="right"):

    x1, y1, x2, y2 = bbox
    H, W = base_img.shape[:2]

    # Extract center of glare image (brightest part)
    gh, gw = glare_img.shape[:2]
    cx1 = int(gw * 0.375)
    cx2 = int(gw * 0.625)
    cy1 = int(gh * 0.375)
    cy2 = int(gh * 0.625)

    center_glare = glare_img[cy1:cy2, cx1:cx2]

    # Resize glare relative to sign size
    sign_w = x2 - x1
    sign_h = y2 - y1

    glare_w = int(sign_w * 0.7)
    glare_h = int(sign_h * 0.7)

    glare_small = cv2.resize(center_glare, (glare_w, glare_h), interpolation=cv2.INTER_AREA)

    # Extract alpha
    if glare_small.shape[2] == 4:
        b, g, r, a = cv2.split(glare_small)
        glare_rgb = cv2.merge([b, g, r])
        alpha = a.astype(float) / 255.0
    else:
        glare_rgb = glare_small
        alpha = np.ones((glare_h, glare_w), float) * 0.65

    # Soft radial fade
    yy, xx = np.mgrid[0:glare_h, 0:glare_w]
    cx = glare_w // 2
    cy = glare_h // 2
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    dist /= dist.max()
    falloff = np.clip(1 - dist, 0, 1)

    alpha = alpha * falloff

    # Compute region target
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    if position == "top_left":
        tx, ty = x1, y1
    elif position == "top_right":
        tx, ty = x2, y1
    elif position == "bottom_left":
        tx, ty = x1, y2
    elif position == "bottom_right":
        tx, ty = x2, y2
    elif position == "left":
        tx, ty = x1, mid_y
    elif position == "right":
        tx, ty = x2, mid_y
    elif position == "top":
        tx, ty = mid_x, y1
    elif position == "bottom":
        tx, ty = mid_x, y2
    elif position == "center":
        tx, ty = mid_x, mid_y
    else:
        raise ValueError("Invalid glare position")

    # Convert center target to placement region
    px = tx - glare_w // 2
    py = ty - glare_h // 2

    px2 = px + glare_w
    py2 = py + glare_h

    # Clamp to image
    ix1 = max(0, px)
    iy1 = max(0, py)
    ix2 = min(W, px2)
    iy2 = min(H, py2)

    if ix1 >= ix2 or iy1 >= iy2:
        return base_img

    gx1 = ix1 - px
    gy1 = iy1 - py
    gx2 = gx1 + (ix2 - ix1)
    gy2 = gy1 + (iy2 - iy1)

    glare_sub = glare_rgb[gy1:gy2, gx1:gx2]
    alpha_sub = alpha[gy1:gy2, gx1:gx2]

    roi = base_img[iy1:iy2, ix1:ix2]

    # Blend glare
    for c in range(3):
        roi[:, :, c] = (
            roi[:, :, c] * (1 - alpha_sub)
            + glare_sub[:, :, c] * alpha_sub
        ).astype(np.uint8)

    base_img[iy1:iy2, ix1:ix2] = roi
    return base_img





def add_glare_to_images(dataset_root, bbox_csv, glare_png, output_folder, position="center"):

    os.makedirs(output_folder, exist_ok=True)

    glare_img = cv2.imread(glare_png, cv2.IMREAD_UNCHANGED)
    df = pd.read_csv(bbox_csv)

    for _, row in df.iterrows():

        rel_path = row["Path"]
        full_path = os.path.join(dataset_root, rel_path)

        if not os.path.exists(full_path):
            print(f"Image not found: {full_path}")
            continue

        img = cv2.imread(full_path)
        if img is None:
            print(f"Cannot load: {full_path}")
            continue

        bbox = (
            int(row["Roi.X1"]),
            int(row["Roi.Y1"]),
            int(row["Roi.X2"]),
            int(row["Roi.Y2"])
        )

        output = add_glare_relative_to_sign(
            img,
            glare_img,
            bbox,
            position=position,
        )

        save_path = os.path.join(output_folder, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, output)

    print("Glare applied to all images.")


DATA_DIR = "./1_Datasets/GTSRB"

add_glare_to_images(
    dataset_root=DATA_DIR,
    bbox_csv=f"{DATA_DIR}/Test.csv",
    glare_png="./1_Datasets/sun_glare.png",
    output_folder="./1_Datasets/GTSRB_glare_center",
    position="center"       
)
