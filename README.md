---
title: Pothole and Defect Detector
emoji: 🚧
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
---

# Pothole & Defect Detector

An AI-powered road image analysis application using a fine-tuned YOLOv8 model to detect potholes, display confidence scores, and estimate severity from detected bounding-box coverage.

## Model metrics

- Validation mAP50: 77.1%
- Validation mAP50-95: 48.8%

These are the recorded validation results; test-set results should be added after test evaluation.

## Limitations

Severity is estimated from bounding-box coverage and does not measure pothole depth or guarantee road safety.
