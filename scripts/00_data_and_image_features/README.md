# Data and image-feature construction

This folder contains the scripts used before model selection. 

1. `download_images_from_jonas_posts.py` reads Jonas post records, identifies image URLs and downloads the image files.
2. `run_full_osf37.py` with `full_osf37_worker.m` calculates 37 non-semantic visual features in MATLAB.
3. `run_full_extra21.py` with `full_extra21_worker.m` calculates 21 supplemental visual-feature candidates in MATLAB. `export_extra2_features.py` then retains the two candidates used in the formal models, pixel count and aspect ratio, to create `extra2_features_by_post.csv`.
4. `preflight_clip_images.py` checks downloaded image files before CLIP inference.
5. `extract_post_level_clip512_colab.py` calculates CLIP image embeddings in Colab. Set its sample setting to full-data mode when running a full extraction.
6. `aggregate_clip512_by_post.py` averages image-level CLIP512 vectors to the post level.
7. `extract_ocr_m_features_by_image.py` extracts recognised text from images and calculates the image-level OCR input table. The later OCR experiment calculates its final 13 OCR--post comparison features from this input table at runtime.

The MATLAB scripts use the Matz-style reference functions named in `run_full_osf37.py`. The complete calculated outputs are already included under `../../data/study_generated/`.

Python dependencies for this optional reconstruction stage are listed in `../../requirements-image-features.txt`. MATLAB is required only for the non-semantic feature stage.
