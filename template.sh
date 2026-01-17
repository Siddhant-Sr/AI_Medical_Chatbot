#!/bin/bash

# ================================
# Medical Multimodal AI Assistant
# Project Scaffolding Script
# ================================

echo "Creating project structure..."

# Root folders
mkdir -p app
mkdir -p data/documents
mkdir -p data/embeddings
mkdir -p prompts
mkdir -p evaluation
mkdir -p observability/logs
mkdir -p scripts
mkdir -p tests

# App submodules
mkdir -p app/api
mkdir -p app/core
mkdir -p app/audio
mkdir -p app/vision
mkdir -p app/rag
mkdir -p app/safety
mkdir -p app/utils

# Core files
touch app/main.py
touch app/config.py

# API
touch app/api/routes.py

# Audio (STT + TTS)
touch app/audio/stt.py
touch app/audio/tts.py

# Vision
touch app/vision/image_analysis.py

# RAG
touch app/rag/retriever.py
touch app/rag/ingest.py

# Core orchestration
touch app/core/orchestrator.py
touch app/core/llm.py

# Safety
touch app/safety/guardrails.py

# Utils
touch app/utils/logger.py

# Prompts
touch prompts/medical_rag_v1.txt
touch prompts/vision_summary_v1.txt

# Evaluation
touch evaluation/rag_eval_set.json
touch evaluation/evaluate_rag.py

# Scripts
touch scripts/run_ingest.py

# Root files
touch requirements.txt
touch README.md
touch .env.example

echo "✅ Project structure created successfully"
