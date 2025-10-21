# Euphemism Attack System

A comprehensive system for generating and evaluating euphemistic adversarial attacks on sentiment analysis models.

## Overview

This repository contains the implementation of a euphemism attack system that generates adversarial examples by transforming text using euphemistic language while maintaining semantic meaning. The system is designed to test the robustness of sentiment analysis models against subtle linguistic manipulations.

## Features

- **Euphemistic Attack Generation**: Uses large language models to generate euphemistic transformations
- **Multi-criteria Filtering**: Combines BERTScore and perplexity metrics for sample quality assessment
- **Attack Verification**: Validates attack success through sentiment prediction
- **Multi-dimensional Evaluation**: Comprehensive quality assessment across coherence, fluency, grammar, and naturalness
- **Multi-language Support**: Includes datasets and evaluation templates for both English and Chinese

## Project Structure

```
├── main.py                 # Main execution script
├── config.py              # Configuration management
├── euphem_attack.py       # Core attack generation module
├── dataloader.py          # Data loading utilities
├── filter.py              # Sample filtering and scoring
├── predictor.py           # Attack success verification
├── label_mapping.py       # Label mapping utilities
├── multi_eval/            # Multi-dimensional evaluation
│   ├── multi_eval.py      # Evaluation script
│   └── prompt/            # Evaluation prompt templates
├── data/                  # Datasets
│   ├── cn/               # Chinese datasets
│   └── en/               # English datasets
├── requirements.txt       # Dependencies
└── sample_data.csv        # Sample data
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EuphemAttack-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
   - GPT-2 model for perplexity calculation
   - BERT model for BERTScore calculation

## Usage

### Basic Usage

Run the main attack generation pipeline:

```bash
python main.py --api-key YOUR_API_KEY --base-url YOUR_BASE_URL --model-name YOUR_MODEL_NAME --input-file input.csv --output-file output.csv
```

### Configuration

The system supports various configuration options:

- **Model Settings**: API key, base URL, model name, temperature, top-p
- **Data Settings**: Input/output files, batch size, number of generations
- **Filter Settings**: GPT model path, BERT model type, scoring weights
- **Evaluation Settings**: Few-shot examples, device selection

### Multi-dimensional Evaluation

Evaluate generated texts across multiple quality dimensions:

```bash
python multi_eval/multi_eval.py --api_key YOUR_API_KEY --base_url YOUR_BASE_URL --model_name YOUR_MODEL_NAME --input_csv input.csv --output_dir results/
```

## Key Components

### 1. Attack Generation (`euphem_attack.py`)
- Generates euphemistic transformations using LLM
- Supports few-shot learning for better performance
- Maintains entity preservation and style consistency

### 2. Sample Filtering (`filter.py`)
- Combines BERTScore (semantic similarity) and perplexity (fluency) metrics
- Weighted scoring system for optimal sample selection
- GPU-accelerated processing with automatic device detection

### 3. Attack Verification (`predictor.py`)
- Validates attack success through sentiment prediction
- Supports both few-shot and zero-shot evaluation
- Comprehensive success rate statistics

### 4. Multi-dimensional Evaluation (`multi_eval/`)
- Evaluates coherence, fluency, grammar, and naturalness
- Uses Likert scale (1-5) scoring
- Supports both English and Chinese evaluation

## Datasets

The repository includes sample datasets for both English and Chinese:

- **English**: Amazon, IMDB, Yelp reviews
- **Chinese**: CTrip, DianPing, JD reviews

## Configuration Options

### Environment Variables
- `OPENAI_API_KEY`: Your API key
- `OPENAI_BASE_URL`: API base URL
- `MODEL_NAME`: Model name to use
- `INPUT_FILE`: Input CSV file path
- `OUTPUT_FILE`: Output CSV file path

### Command Line Arguments
- `--api-key`: OpenAI API key
- `--base-url`: OpenAI API base URL
- `--model-name`: Model name to use
- `--input-file`: Input CSV file path
- `--output-file`: Output CSV file path
- `--num-generations`: Number of adversarial generations per sample
- `--device`: Device to use (auto/cuda/cpu)

## Output Format

The system generates CSV files with the following columns:
- `original_label`: Original sentiment label
- `original_text`: Original input text
- `best_adversarial_text`: Best generated adversarial text
- `predicted_label`: Predicted label for adversarial text
- `attack_successful`: Boolean indicating attack success
- `best_score`: Quality score of the best sample

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- Transformers 4.20.0+
- OpenAI API access
- CUDA support (recommended for GPU acceleration)
