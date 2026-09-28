# OCR Complementarity Analysis

`run_ocr_new_content_groups.py` reproduces the exploratory OCR analysis in
Table 8.1. It compares the MI model with MI plus 14 OCR-related features, using only posts with non-empty recognised OCR text.

The script does not use the 66 OCR-derived M features. The 14 features are
calculated at runtime from recognised image text (`ocr_text`) and the matching
post text (`post_text`).
