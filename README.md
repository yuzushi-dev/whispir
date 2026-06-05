# Whispir - Local Transcription, Translation & Document Intelligence

Whispir is a secure, private, and offline web application for:
- 🎙️ Transcribing and translating audio and video files.
- 📄 Processing documents: conversion to Markdown, local PII redaction, and visual OCR.

Built using **Streamlit**, **faster-whisper**, **Helsinki-NLP/MarianMT**, **Microsoft MarkItDown**, **OpenAI Privacy Filter**, and **GLM-OCR**. 

The application runs entirely offline within a Docker container, keeping all processed data private, and exposes its interface locally on port **9999**.

---

## Key Features

### Media Processing
- 🐳 **Dockerized Setup**: Quick and easy initialization. No need to install python, ffmpeg, or native C++ dependencies on the host system.
- 🏎️ **Optimized CPU Execution**: Powered by `faster-whisper` using CTranslate2 and `int8` quantization to run efficiently on CPU architectures.
- 💻 **Dynamic Hardware Awareness**: Automatically scans host system hardware (CPU cores, RAM size) to recommend the optimal Whisper model.
- 🔀 **Flexible Tasks Pipeline**:
  - **Transcription Only**: High-fidelity transcript in the source language.
  - **Transcription + English Translation**: Single-pass native Whisper translation to English.
  - **Transcription + Italian Translation**: Dual-pass cascading translation (Whisper source transcription -> MarianMT English-to-Italian translation).
- 🗂️ **Side-by-Side Live Preview**: Features a multi-tab view (`Original Transcript` and `Translation`) that updates in real-time as processing occurs.
- 📂 **Decoupled Downloads**: Separated columns allow users to download transcripts and translations independently as Plain Text (`.txt`), Standard Subtitles (`.srt`), or WebVTT (`.vtt`).
- 🎙️ **Voice Activity Detection (VAD)**: Built-in Silero VAD filtering to strip out silent intervals and prevent hallucinations during long files.

### Document Intelligence (NEW)
- 📝 **Markdown Conversion**: Instantly convert PDF, Word, Excel, PowerPoint, and HTML documents to Markdown using Microsoft's `markitdown`.
- 🔒 **Local PII Redaction**: Automatically redact sensitive information (names, emails, phones, addresses) using a fully offline instance of the `openai/privacy-filter` token classification model.
- 👁️ **Visual OCR**: Perform local OCR on images using the lightweight and state-of-the-art `zai-org/GLM-OCR` model, with optional Italian translation coupling.
- 💾 **Persistent Cache**: Model weights (Whisper, MarianMT, Privacy Filter, and GLM-OCR) are stored on the host under `./models_cache` to avoid re-downloading upon container restarts.

---

## Prerequisites

- **Docker**
- **Docker Compose**

---

## Getting Started

### 1. Build and Launch the Application

Navigate to the project root directory and spin up the Docker container:

```bash
docker compose up --build -d
```

### 2. Access the Web Interface

Open your browser and navigate to:
[http://localhost:9999](http://localhost:9999)

### 3. Stop the Application

To shut down and remove the active container:

```bash
docker compose down
```

---

## Project Structure

```
├── Dockerfile              # Python 3.12-slim based container environment (with torchvision CPU)
├── README.md               # English project documentation
├── app.py                  # Streamlit application code and pipeline execution logic
├── document_intelligence.py# Core helpers for markitdown, PII redaction, and GLM-OCR
├── docker-compose.yml      # Service container mapping (port 9999, model volume persistence)
├── requirements.txt        # Python library dependencies (PyTorch CPU, faster-whisper, transformers, markitdown)
└── verify_translation.py   # Automated end-to-end integration test for translation
```

---

## Model Comparison Guide

### Transcription Models
| Model | File Size | Approximate RAM | CPU Speed | Quality |
| :--- | :--- | :--- | :--- | :--- |
| **tiny** | ~75 MB | ~150 MB | Ultra-Fast | Low (Best for quick tests or drafts) |
| **base** | ~140 MB | ~250 MB | Very Fast | Medium |
| **small** | ~460 MB | ~600 MB | Fast | Good (Solid default selection) |
| **medium** | ~1.5 GB | ~2.0 GB | Moderate | High |
| **turbo** | ~1.6 GB | ~2.2 GB | Fast | Excellent (Great speed-to-accuracy balance) |
| **large-v3** | ~3.0 GB | ~3.8 GB | Slow | Maximum (Best for complex audio or rare dialects) |

### Document Intelligence Models
| Task / Model | Size | Hardware Recomm. | Description |
| :--- | :--- | :--- | :--- |
| **openai/privacy-filter** | ~350 MB | Any CPU | Local token classification for redaction |
| **zai-org/GLM-OCR** | ~1.8 GB | >= 8GB RAM | SOTA multimodal OCR (0.9B parameters) on CPU |

*Note: On the first run of any selected model, downloading the weights from HuggingFace might take a few minutes. Subsequent executions load instantly from the cache volume.*

---

## Troubleshooting

### HuggingFace Download Errors
If the container cannot reach HuggingFace to download models, verify your host network connection. The container needs initial outbound access to download model weights. Once downloaded, the application operates entirely offline.

### Model Storage Persistence
If models are redownloaded every time the container starts, ensure the `models_cache` directory exists in the project root on the host machine and has read/write permissions for the Docker user.

