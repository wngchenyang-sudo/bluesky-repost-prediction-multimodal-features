# Colab script for Bluesky post-level image feature extraction.
# Recommended use: open in Google Colab, enable T4 GPU, then run section by section.

# =========================
# Cell 1. Mount Drive And Unzip Images
# =========================

import os
import zipfile
from google.colab import drive

drive.mount("/content/drive")

ZIP_PATH = "/content/drive/MyDrive/images.zip"
ALIGNMENT_CSV = "/content/drive/MyDrive/post_image_alignment.csv"
EXTRACT_DIR = "/content/images"

# First run uses 500 posts for validation. Change to None for full run.
SAMPLE_N = 500

os.makedirs(EXTRACT_DIR, exist_ok=True)

print("Unzipping images to Colab local SSD...")
with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)
print("Unzip complete.")


# =========================
# Cell 2. Install And Import Dependencies
# =========================

# Run this line in Colab if dependencies are missing:
# !pip install -q transformers pillow pandas torch torchvision tqdm opencv-python

import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
CPU_WORKERS = min(4, cpu_count())

print("Device:", DEVICE)
print("Batch size:", BATCH_SIZE)
print("CPU workers:", CPU_WORKERS)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


# =========================
# Cell 3. Build Image Path Index Once
# =========================

def build_image_index(root_dir):
    """Build path indexes once to avoid repeated os.walk inside loops."""
    by_basename = {}
    by_relative_suffix = {}
    image_exts = (".jpg", ".jpeg", ".png", ".webp")

    for root, _, files in os.walk(root_dir):
        for filename in files:
            if not filename.lower().endswith(image_exts):
                continue

            full_path = os.path.join(root, filename)
            rel_from_extract = os.path.relpath(full_path, root_dir).replace("\\", "/")

            by_basename.setdefault(filename, []).append(full_path)
            by_relative_suffix[rel_from_extract] = full_path

            parts = rel_from_extract.split("/")
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                by_relative_suffix.setdefault(suffix, full_path)

    return by_basename, by_relative_suffix


basename_index, suffix_index = build_image_index(EXTRACT_DIR)

print("Unique basenames:", len(basename_index))
print("Relative suffix entries:", len(suffix_index))
print("Sample extracted image paths:")
for key, value in list(suffix_index.items())[:10]:
    print(key, "=>", value)


# =========================
# Cell 4. Read Alignment Table And Flatten To Image-Level
# =========================

df_posts = pd.read_csv(ALIGNMENT_CSV)

if SAMPLE_N is not None:
    df_posts = df_posts.head(SAMPLE_N).copy()
    print("Sample mode enabled. Posts:", len(df_posts))
else:
    print("Full mode enabled. Posts:", len(df_posts))

if "has_image" in df_posts.columns:
    df_posts["has_image"] = df_posts["has_image"].astype(str).str.lower().eq("true")
else:
    df_posts["has_image"] = False


def split_image_paths(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [p.strip() for p in text.split(";") if p.strip()]


def resolve_image_path(original_path):
    """
    Resolve original local path from Windows/project machine to Colab extracted path.
    Priority:
    1. suffix after images/
    2. any relative suffix match
    3. basename fallback only if unique
    """
    if not original_path or pd.isna(original_path):
        return None, "missing_path"

    normalized = str(original_path).replace("\\", "/").strip()
    lower_norm = normalized.lower()
    candidates = []

    if "/images/" in lower_norm:
        idx = lower_norm.rfind("/images/")
        suffix = normalized[idx + len("/images/"):]
        candidates.append(suffix)

    candidates.append(normalized)

    parts = normalized.split("/")
    for i in range(len(parts)):
        candidates.append("/".join(parts[i:]))

    for candidate in candidates:
        if candidate in suffix_index:
            return suffix_index[candidate], "relative_suffix"

    basename = os.path.basename(normalized)
    matches = basename_index.get(basename, [])

    if len(matches) == 1:
        return matches[0], "unique_basename"
    if len(matches) > 1:
        return None, "ambiguous_basename"
    return None, "not_found"


image_rows = []

for _, row in tqdm(df_posts.iterrows(), total=len(df_posts), desc="Flattening posts"):
    if not row.get("has_image", False):
        continue

    paths = split_image_paths(row.get("image_paths", ""))
    for image_order, original_path in enumerate(paths):
        resolved_path, match_method = resolve_image_path(original_path)
        image_rows.append({
            "post_uri": row["post_uri"],
            "image_order": image_order,
            "original_image_path": original_path,
            "local_image_path": resolved_path if resolved_path else "",
            "file_exists": bool(resolved_path and os.path.exists(resolved_path)),
            "path_match_method": match_method,
        })

df_images = pd.DataFrame(image_rows)

print("Image-level rows:", len(df_images))
if len(df_images) > 0:
    print("Existing image files:", int(df_images["file_exists"].sum()))
    print(df_images["path_match_method"].value_counts(dropna=False))


# =========================
# Cell 5. Extract Low/Mid-Level Features
# =========================

def extract_low_mid_features(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            aspect_ratio = width / height if height else 0.0

        file_size_kb = os.path.getsize(image_path) / 1024.0

        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return {"low_mid_valid": False}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        saturation = float(np.mean(hsv[:, :, 1]))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        hue = hsv[:, :, 0]
        warm_mask = ((hue >= 0) & (hue <= 30)) | ((hue >= 150) & (hue <= 180))
        cool_mask = (hue >= 80) & (hue <= 130)

        total_pixels = img_cv.shape[0] * img_cv.shape[1]
        warm_cool_score = float((np.sum(warm_mask) - np.sum(cool_mask)) / total_pixels)

        visual_intensity_score = float(
            (brightness / 255.0 + contrast / 128.0 + saturation / 255.0) / 3.0
        )

        return {
            "low_mid_valid": True,
            "width": float(width),
            "height": float(height),
            "aspect_ratio": float(aspect_ratio),
            "file_size_kb": float(file_size_kb),
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "sharpness": sharpness,
            "warm_cool_score": warm_cool_score,
            "visual_intensity_score": visual_intensity_score,
        }

    except Exception:
        return {"low_mid_valid": False}


valid_images = df_images[df_images["file_exists"]].copy() if len(df_images) else pd.DataFrame()
valid_paths = valid_images["local_image_path"].tolist() if len(valid_images) else []

print("Extracting low/mid-level features...")
if valid_paths:
    with Pool(processes=CPU_WORKERS) as pool:
        low_mid_results = list(
            tqdm(pool.imap(extract_low_mid_features, valid_paths), total=len(valid_paths))
        )
    low_mid_df = pd.DataFrame(low_mid_results)
    valid_images = pd.concat(
        [valid_images.reset_index(drop=True), low_mid_df.reset_index(drop=True)],
        axis=1,
    )
    print(valid_images["low_mid_valid"].value_counts(dropna=False))
else:
    valid_images = pd.DataFrame(columns=list(df_images.columns) if len(df_images) else [])
    print("No valid images found.")


# =========================
# Cell 6. Extract CLIP Embeddings With Real Batching
# =========================

class CLIPImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return image, path, True
        except Exception:
            return Image.new("RGB", (224, 224), color=(0, 0, 0)), path, False


def collate_clip_batch(batch):
    images, paths, valid_flags = zip(*batch)
    inputs = clip_processor(images=list(images), return_tensors="pt", padding=True)
    return inputs["pixel_values"], list(paths), torch.tensor(valid_flags, dtype=torch.bool)


clip_cols = [f"clip_{i}" for i in range(512)]

if valid_paths:
    clip_dataset = CLIPImageDataset(valid_paths)
    clip_loader = DataLoader(
        clip_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_clip_batch,
        pin_memory=True,
    )

    all_clip_vectors = []
    all_clip_valid = []

    print("Extracting CLIP embeddings with GPU batching...")
    with torch.no_grad():
        for pixel_values, paths, valid_flags in tqdm(clip_loader):
            pixel_values = pixel_values.to(DEVICE)
            features = clip_model.get_image_features(pixel_values=pixel_values)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_clip_vectors.append(features.cpu().numpy())
            all_clip_valid.append(valid_flags.numpy())

    clip_matrix = np.vstack(all_clip_vectors)
    clip_valid = np.concatenate(all_clip_valid)
else:
    clip_matrix = np.zeros((0, 512), dtype=np.float32)
    clip_valid = np.array([], dtype=bool)

clip_df = pd.DataFrame(clip_matrix, columns=clip_cols)

if len(valid_images):
    valid_images = pd.concat(
        [valid_images.reset_index(drop=True), clip_df.reset_index(drop=True)],
        axis=1,
    )
    valid_images["clip_valid"] = clip_valid
    valid_images["image_feature_valid"] = (
        valid_images["low_mid_valid"].fillna(False).astype(bool)
        & valid_images["clip_valid"].fillna(False).astype(bool)
    )
    print(valid_images["image_feature_valid"].value_counts(dropna=False))


# =========================
# Cell 7. Aggregate Back To Post-Level
# =========================

successful_images = valid_images[valid_images["image_feature_valid"]].copy() if len(valid_images) else pd.DataFrame()

low_mid_cols = [
    "width",
    "height",
    "aspect_ratio",
    "file_size_kb",
    "brightness",
    "contrast",
    "saturation",
    "sharpness",
    "warm_cool_score",
    "visual_intensity_score",
]

feature_cols = low_mid_cols + clip_cols

if len(successful_images) > 0:
    grouped_features = successful_images.groupby("post_uri")[feature_cols].mean().reset_index()
else:
    grouped_features = pd.DataFrame(columns=["post_uri"] + feature_cols)

rename_map = {
    "width": "avg_width",
    "height": "avg_height",
    "aspect_ratio": "avg_aspect_ratio",
    "file_size_kb": "avg_file_size_kb",
    "brightness": "avg_brightness",
    "contrast": "avg_contrast",
    "saturation": "avg_saturation",
    "sharpness": "avg_sharpness",
}

grouped_features = grouped_features.rename(columns=rename_map)

if len(df_images) > 0:
    image_stats = df_images.groupby("post_uri").agg(
        image_count_from_paths=("original_image_path", "count"),
        downloaded_image_count=("file_exists", "sum"),
    ).reset_index()
else:
    image_stats = pd.DataFrame(columns=["post_uri", "image_count_from_paths", "downloaded_image_count"])

if len(successful_images) > 0:
    extracted_counts = successful_images.groupby("post_uri").size().reset_index(name="images_used")
else:
    extracted_counts = pd.DataFrame(columns=["post_uri", "images_used"])

final_df = df_posts.copy()
final_df = final_df.merge(image_stats, on="post_uri", how="left")
final_df = final_df.merge(extracted_counts, on="post_uri", how="left")
final_df = final_df.merge(grouped_features, on="post_uri", how="left")

final_df["image_count_from_paths"] = final_df["image_count_from_paths"].fillna(0).astype(int)
final_df["downloaded_image_count"] = final_df["downloaded_image_count"].fillna(0).astype(int)
final_df["images_used"] = final_df["images_used"].fillna(0).astype(int)


def determine_image_feature_status(row):
    if not row.get("has_image", False):
        return "no_image"
    if row["image_count_from_paths"] == 0:
        return "no_image_path"
    if row["downloaded_image_count"] == 0:
        return "image_not_downloaded"
    if row["images_used"] == 0:
        return "image_open_failed"
    return "extracted"


final_df["image_feature_status"] = final_df.apply(determine_image_feature_status, axis=1)

output_feature_cols = [
    "avg_width",
    "avg_height",
    "avg_aspect_ratio",
    "avg_file_size_kb",
    "avg_brightness",
    "avg_contrast",
    "avg_saturation",
    "avg_sharpness",
    "warm_cool_score",
    "visual_intensity_score",
] + clip_cols

for col in output_feature_cols:
    if col not in final_df.columns:
        final_df[col] = 0.0

final_df[output_feature_cols] = final_df[output_feature_cols].fillna(0.0)

print("Final shape:", final_df.shape)
print(final_df["image_feature_status"].value_counts(dropna=False))


# =========================
# Cell 8. Save Outputs
# =========================

sample_suffix = "_sample500" if SAMPLE_N is not None else "_full"
OUTPUT_CSV = f"/content/drive/MyDrive/image_features_post_level{sample_suffix}.csv"
IMAGE_LEVEL_DEBUG_CSV = f"/content/drive/MyDrive/image_level_path_debug{sample_suffix}.csv"

final_df.to_csv(OUTPUT_CSV, index=False)
df_images.to_csv(IMAGE_LEVEL_DEBUG_CSV, index=False)

print("Saved post-level image features to:", OUTPUT_CSV)
print("Saved image-level path debug table to:", IMAGE_LEVEL_DEBUG_CSV)

