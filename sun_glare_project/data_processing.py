import json
import os
import csv

DATA_DIR = "./1_Datasets/dataset3"

def map_sign_type(sign_name: str) -> str:
    name = sign_name
    print(name)

    if name.startswith("St"):       # stop
        return "stop"
    if name.startswith("T"):        # traffic light
        return "trafficlight"
    if name.startswith("Sp"):        # speed limit
        return "speedlimit"
    if name.startswith("P"):        # crosswalk
        return "crosswalk"

    return "unknown"


def load_ground_truth(gt_csv_path: str):
    gt_dict = {}
    with open(gt_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            gt_label = row["sign_name"].strip()
            gt_dict[fname] = gt_label
    return gt_dict


def process_json_to_csv(json_folder: str, gt_csv_path: str, output_csv: str):
    
    # Load ground truth
    gt_dict = load_ground_truth(gt_csv_path)

    # Prepare rows for output
    rows = []

    for fname in os.listdir(json_folder):
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(json_folder, fname)

        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract prediction from JSON
        raw_name = data.get("traffic_sign_name", "")
        prediction = map_sign_type(raw_name)

        # Match ground-truth by the JSON filename (remove .json if needed)
        base_image_name = fname.replace(".json", ".png")  # adjust if jpg/jpeg
        ground_truth = gt_dict.get(base_image_name, "UNKNOWN")

        rows.append({
            "filename": base_image_name,
            "ground_truth": ground_truth,
            "prediction": prediction
        })

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "ground_truth", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


# Example usage:
process_json_to_csv("./1_Datasets/llama4scout_dataset3_glare_half", f"{DATA_DIR}/traffic_sign_labels.csv", "./Results/csv/predictions_glare_half.csv")
