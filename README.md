# DiagnoScan-An-AI-based-Medical-Imaging-Analysis-and-Report-Generation-System
# 🩺 AI-based Medical Segmentation & Report Generator

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/streamlit-%E2%89%A54.0-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Project Summary

**Medical Segmentation & Report Generator** is an end-to-end Streamlit application that:
- accepts medical images (X-ray or CT),
- performs segmentation using pretrained deep learning models,
- visualizes probability maps and binary masks,
- generates downloadable PNG visualizations and a PDF report,
- includes a domain-specific document-based chatbot (LangChain + Ollama / embeddings) for additional information.

The app is designed for demo/educational purposes and **not** for clinical diagnosis. See the disclaimer at the bottom.

---

## Key Features

- Upload X-ray / CT images and receive segmentation output.
- Visual outputs: original image, probability map, binary mask, overlay with coverage percentage.
- Downloadable artifacts: visualization PNG, binary mask PNG, and a professionally formatted PDF report.
- Sidebar patient details and configurable segmentation threshold.
- (Optional) Document-based chatbot that answers domain questions using provided PDFs.
- Modular code structure: `app.py` (Streamlit UI), `chatbot.py` (retrieval QA setup), models in the repo.

---



