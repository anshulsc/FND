# Agentic Framework for Fake News Detection

**Author**: [Anshul Singh](https://anshulsc.live)
**Date**: 2025-09-16



## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
  - [High-Level Diagram](#high-level-diagram)
  - [Core Components](#core-components)
  - [Directory Structure](#directory-structure)
- [Technology Stack](#technology-stack)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [How to Use the System](#how-to-use-the-system)
  - [Running the Application](#running-the-application)
  - [Adding New Queries](#adding-new-queries)
  - [Monitoring and Administration](#monitoring-and-administration)
- [In-Depth Workflow](#in-depth-workflow)
  - [1. Automated Query Detection](#1-automated-query-detection)
  - [2. Offline Evidence Extraction](#2-offline-evidence-extraction)
  - [3. Staging for Inference](#3-staging-for-inference)
  - [4. Multimodal AI Inference](#4-multimodal-ai-inference)
  - [5. PDF Report Generation](#5-pdf-report-generation)
- [Key Modules and Logic](#key-modules-and-logic)
  - [`status_manager.py`](#status_managerpy)
  - [`evidence_searcher.py`](#evidence_searcherpy)
  - [`inference_pipeline.py`](#inference_pipelinepy)
  - [`pdf_generator.py`](#pdf_generatorpy)
- [Future Enhancements](#future-enhancements)

## Overview

This project is a sophisticated, multi-modal agentic framework designed for the automated detection of fake news. It leverages a powerful combination of Large Language Models (vLLM with Gemma 3), custom deep learning models (FraudNet), and a robust backend system to provide a comprehensive analysis of news samples.

The system is architected to be fully automated. It autonomously detects new queries, gathers relevant evidence from a local knowledge base, performs a multi-faceted AI analysis, and generates a detailed, professional PDF report of its findings. The entire process is managed and monitored through a clean, interactive web-based dashboard built with Streamlit.

## Key Features

-   **Fully Automated Pipeline**: A "drop-and-go" system. Simply add a new query, and the framework handles everything from evidence gathering to final report generation.
-   **Offline Evidence Extraction**: Utilizes a local vector database (ChromaDB) to find the most relevant textual and visual evidence without requiring an internet connection.
-   **Multi-Modal Analysis**: Fuses insights from multiple AI agents:
    -   **Image-Text Consistency**: Analyzes the semantic alignment between a news image and its caption.
    -   **Image-Image Similarity**: Compares the query image against visual evidence.
    -   **Text-Text Factual Alignment**: Verifies textual claims against a database of news articles.
    -   **FraudNet Model**: A specialized deep learning model for a final fake/true classification.
-   **LangGraph Workflow**: Orchestrates the AI agents in a logical graph, ensuring a structured and debuggable reasoning process.
-   **Professional PDF Reporting**: Automatically generates elegant, multi-page PDF reports that summarize the findings, showcase evidence, and detail the AI's reasoning.
-   **Interactive Web Dashboard**: A Streamlit application for monitoring query statuses in real-time, viewing results, and managing the system.
-   **Robust and Modular Architecture**: Built with a clear separation of concerns, making the system easy to maintain, debug, and extend.

## System Architecture

The framework is built on a modular, service-oriented architecture designed for robustness and scalability. It avoids complex setups like Docker or Redis, opting for a transparent, file-based task queue system.

### High-Level Diagram

```
+-------------------+      +----------------------+      +--------------------+
| Frontend (UI)     |      |    Backend (API)     |      |  File System Watcher |
| (Streamlit)       |<---->|      (FastAPI)       |<---->|      (Watchdog)      |
+-------------------+      +----------+-----------+      +----------+---------+
      ^                          ^            |                     |
      | (REST API Calls)         |            | (File I/O)          | (Monitors Dirs)
      v                          v            v                     v
+-------------------+      +----------------------+      +--------------------+
|  User's Browser   |      |   Status Manager     |      | Agentic Workspace  |
|                   |      |    (SQLite DB)       |      | (File System)      |
+-------------------+      +----------------------+      +--------------------+
                                      |
         +----------------------------+
         |
         v
+------------------+
| Main Worker      |
| (Processes Jobs) |
+--------+---------+
         |
         v
+------------------+ -> +-------------------+ -> +--------------------+
| Evidence         |    | Model Inference   |    | PDF Generation     |
| Extraction Module|    | Module (LangGraph)|    | Module             |
+------------------+    +-------------------+    +--------------------+
         ^                           ^
         |                           |
         v                           v
+------------------+        +-------------------+
| Vector Database  |        | AI Models (vLLM,  |
| (ChromaDB)       |        | PyTorch)          |
+------------------+        +-------------------+
```

### Core Components

-   **File System Watcher (`watcher.py`)**: A lightweight service that monitors the `1_queries` directory. Upon detecting a new query, it adds an entry to the status database and creates a job file in the file-based queue.
-   **Main Worker (`main_worker.py`)**: The engine of the system. It continuously polls the job queue for new tasks. When a job is found, it executes the full, multi-stage analysis pipeline.
-   **Backend API (`api/main.py`)**: A FastAPI server that acts as the communication layer. It provides REST endpoints for the frontend to fetch query statuses, retrieve PDF reports, and trigger administrative actions like re-indexing.
-   **Streamlit Frontend (`streamlit_app.py`)**: A multi-page web application that serves as the user's command center for interacting with and monitoring the system.
-   **Status Manager (`database/status_manager.py`)**: A simple, robust SQLite database that acts as the system's "brain," tracking the state and progress of every query.

### Directory Structure

The project is organized into two main parts: the source code (`src`) and the data workspace (`agentic_workspace`).

```
agentic_framework_v2/
├── agentic_workspace/          # All data, models, and results
│   ├── .system/                # System files (DB, logs, indexes)
│   ├── 1_queries/              # INPUT: Drop new query folders here
│   ├── 2_evidence_database/    # INPUT: The knowledge base of news articles
│   ├── 3_processed_for_model/  # STAGING: Intermediate files for the model
│   ├── 4_results/              # OUTPUT: Final PDF reports
│   └── 5_trash/                # For deleted items
├── src/                        # All Python source code
│   ├── api/                    # FastAPI backend code
│   ├── agents/                 # Core AI agent logic and prompts
│   ├── database/               # Status manager (SQLite)
│   ├── frontend/               # (Future use)
│   ├── modules/                # Reusable logic (PDF gen, search, inference)
│   ├── workers/                # Watcher and main worker scripts
│   └── tools/                  # Utility scripts (e.g., build_index.py)
├── start_system.sh             # Master script to run the application
└── README.md                   # This file
```

## Technology Stack

-   **Backend**: Python 3.10+
-   **API Framework**: FastAPI
-   **Web UI**: Streamlit
-   **Vector Database**: ChromaDB
-   **AI Orchestration**: LangGraph
-   **LLM Inference**: vLLM (with Hugging Face Transformers)
-   **Deep Learning**: PyTorch
-   **File Monitoring**: Watchdog
-   **PDF Generation**: FPDF2

## Setup and Installation

### Prerequisites

-   Python 3.10 or higher.
-   An NVIDIA GPU with CUDA installed (required for vLLM and FraudNet).
-   `wget` for downloading font files.

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd agentic_framework_v2
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt 
    # NOTE: You will need to create a requirements.txt file by running:
    # pip freeze > requirements.txt
    ```

4.  **Download Assets:** The PDF generator requires specific fonts. Run the setup script:
    ```bash
    # Create the necessary directories
    mkdir -p src/assets/fonts

    # Download the fonts into the new directory
    wget -P src/assets/fonts/ https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf
    wget -P src/assets/fonts/ https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf
    ```
5.  **Populate the Evidence Database:**
    Place your collection of news articles (folders containing an image and a `caption.txt`) into the `agentic_workspace/2_evidence_database/` directory.

6.  **Build the Initial Search Index:**
    Run the indexing tool to process your evidence database. This only needs to be done once initially, and then can be re-run from the UI.
    ```bash
    python -m tools.build_index
    ```

## How to Use the System

### Running the Application

A master bash script manages the entire application stack.

1.  **Make the script executable (only once):**
    ```bash
    chmod +x start_system.sh
    ```

2.  **Start the system:**
    ```bash
    ./start_system.sh
    ```
    This will start the watcher, worker, and API in the background and launch the Streamlit frontend. Your browser should open to the dashboard automatically.

3.  **Stopping the system:**
    Press `Ctrl+C` in the terminal where you ran the script. This will trigger a clean shutdown of all background services.

### Adding New Queries

There are three ways to add a new query for analysis:

1.  **File System (Automated)**: Simply create a new folder inside `agentic_workspace/1_queries/`. The folder must contain a single image file and a `query_cap.txt` file. The watcher will detect it automatically and start the process.
2.  **Web UI (Image & Caption)**: Navigate to the "Add New Query" page on the dashboard and upload an image and type the caption directly.
3.  **Web UI (Zipped Folder)**: On the same page, you can upload a `.zip` file that contains the image and caption file.

### Monitoring and Administration

-   **Dashboard**: The main page provides a real-time overview of all queries, their status, and progress through the pipeline stages.
-   **Query Details Page**: Click "View Details" on any query to see a beautifully rendered breakdown of the AI's analysis, including the evidence used.
-   **Settings Page**:
    -   **Re-Index Database**: Trigger the indexing process again if you have added new articles to your evidence database.
    -   **View Logs**: Access the latest logs from the API, worker, and watcher to debug any issues.

## In-Depth Workflow

The automated process for a new query follows these five steps:

#### 1. Automated Query Detection
The `watcher.py` service detects a new directory in `1_queries`. It immediately creates a job file (e.g., `sample_001.job`) in the `.system/job_queue/` directory and registers the query in the `app_state.db` with a "pending" status.

#### 2. Offline Evidence Extraction
The `main_worker.py` picks up the job. It reads the query image and caption, generates CLIP embeddings for them, and queries the ChromaDB vector store. It finds the top K most similar images and texts, ranking them by similarity score.

#### 3. Staging for Inference
The worker creates a new directory in `3_processed_for_model/`. It copies the original query files and the single best visual evidence into this folder. Crucially, it generates an `evidence_metadata.json` file, which contains paths to all the necessary files and the ranked list of all found evidence.

#### 4. Multimodal AI Inference
The `inference_pipeline.py` module is called with the path to the `evidence_metadata.json`. It loads the vLLM and FraudNet models (if not already in memory) and prepares the inputs. The LangGraph workflow is invoked, running the full chain of agentic reasoning. The final, structured output is saved as `inference_results.json`.

#### 5. PDF Report Generation
Finally, the `pdf_generator.py` module takes the `inference_results.json` and `evidence_metadata.json` files. It uses this data to generate a polished, multi-page `analysis_report.pdf` and saves it in the `4_results/` directory. The worker then updates the query's status in `app_state.db` to "completed" and records the path to the final PDF.

## Key Modules and Logic

-   **`status_manager.py`**: The single source of truth for the state of any query. Using SQLite makes it lightweight and serverless.
-   **`evidence_searcher.py`**: Encapsulates all interaction with ChromaDB. It performs a hybrid search by querying with both the image and text embeddings and intelligently merging the results to find the most relevant multimodal evidence.
-   **`inference_pipeline.py`**: A critical module that acts as a bridge between the file-based system and the complex AI models. It handles model loading (using a singleton pattern to prevent reloading) and data preparation.
-   **`pdf_generator.py`**: More than a simple file writer, this module is a sophisticated document renderer. It uses custom logic to parse the Markdown output from the LLM, apply conditional styling (e.g., coloring "Mismatch" in red), and handle media layout gracefully.

