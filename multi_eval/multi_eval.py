import pandas as pd
from openai import OpenAI
import time
import re
import os
import argparse
import logging
from tqdm import tqdm

# Prompt template file paths
PROMPT_TEMPLATE_PATHS = {
    "coh": "prompt/likert/coh_likert.txt",
    "flu": "prompt/likert/flu_likert.txt",
    "gram": "prompt/likert/gram_likert.txt",
    "nat": "prompt/likert/nat_likert.txt",
}


def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger("evaluation")
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(output_dir, "evaluation.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_prompt_templates():
    """Load prompt templates from files"""
    templates = {}
    for dim, file_path in PROMPT_TEMPLATE_PATHS.items():
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                templates[dim] = file.read()
        except Exception as e:
            raise Exception(f"Error loading prompt template {file_path}: {e}")
    return templates


def load_csv_data(csv_path):
    """Load CSV data without header (first column: label, second column: text)"""
    try:
        # Read CSV without header
        data = pd.read_csv(csv_path, header=None, names=["label", "text"])
        return data
    except Exception as e:
        raise Exception(f"Error loading CSV file: {e}")


def get_model_score(prompt, client, model_name, logger):
    """Get model evaluation score for a given prompt"""
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt + " no think"}],
                stream=False,
            )
            break
        except Exception as e:
            logger.warning(f"Request failed, error message: {e}, retrying...")
            time.sleep(2)

    content = response.choices[0].message.content.strip()
    match = re.search(r"\d+(\.\d+)?", content)
    numeric_content = match.group(0) if match else None

    try:
        score = float(numeric_content)
        if 1 <= score <= 5:
            return score
        else:
            logger.warning(f"Score {score} is outside valid range [1,5], returning 0")
            return 0  # Return 0 if score is outside valid range
    except (ValueError, TypeError):
        logger.warning(f"Could not parse score from: {content}, returning 0")
        return 0


def evaluate_text(text, templates, client, model_name, logger):
    """Evaluate a single text across all dimensions"""
    scores = {}
    for dim, template in templates.items():
        formatted_prompt = template.replace("{{sentence}}", str(text))
        score = get_model_score(formatted_prompt, client, model_name, logger)
        scores[dim] = score
        logger.info(f"  {dim}: {score}")
    return scores


def generate_outputs(evaluation_results, output_dir, logger):
    """Generate TXT output file with averages only"""
    # Calculate overall averages for each dimension
    overall_averages = {}
    for dim in PROMPT_TEMPLATE_PATHS.keys():
        all_scores = []
        for result in evaluation_results:
            all_scores.append(result["scores"][dim])
        overall_averages[dim] = sum(all_scores) / len(all_scores) if all_scores else 0

    # Write TXT file with overall averages only
    txt_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(txt_path, "w", encoding="utf-8") as txtfile:
        txtfile.write("Overall Evaluation Results\n")
        txtfile.write("=" * 30 + "\n")
        for dim in PROMPT_TEMPLATE_PATHS.keys():
            txtfile.write(f"{dim}: {overall_averages[dim]:.4f}\n")

    logger.info(f"Results written to: {txt_path}")
    logger.info(f"Overall averages: {overall_averages}")

    return overall_averages


def main():
    """Main evaluation function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate texts using Qwen model")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--base_url", required=True, help="API base URL")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--input_csv", required=True, help="Input CSV file path")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting evaluation process")
    logger.info(f"API Key: {args.api_key[:10]}...")
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Output Directory: {args.output_dir}")

    # Initialize OpenAI client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Load prompt templates
    logger.info("Loading prompt templates...")
    templates = load_prompt_templates()
    logger.info(f"Loaded {len(templates)} prompt templates")

    # Load data
    logger.info(f"Loading data from: {args.input_csv}")
    data = load_csv_data(args.input_csv)
    logger.info(f"Loaded {len(data)} rows")

    # Store all evaluation results
    evaluation_results = []

    # Process each row
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        text = row["text"]
        label = row["label"]

        if pd.isna(text) or text == "":
            logger.warning(f"Skipping empty text at row {idx}")
            continue

        logger.info(f"Evaluating row {idx}: {text[:50]}...")
        scores = evaluate_text(text, templates, client, args.model_name, logger)

        # Store result
        evaluation_results.append({"label": label, "text": text, "scores": scores})

    logger.info(f"Completed evaluation of {len(evaluation_results)} texts")

    # Generate output files
    overall_averages = generate_outputs(evaluation_results, args.output_dir, logger)

    logger.info("Evaluation completed successfully!")
    print(f"Evaluation completed! Results saved to {args.output_dir}")
    print(f"Overall averages: {overall_averages}")


if __name__ == "__main__":
    main()
