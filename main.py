#!/usr/bin/env python3
"""
Main script for Euphemism Attack System
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, create_argument_parser
from dataloader import DataLoader
from euphem_attack import EuphemismAttack
from filter import SampleFilter
from predictor import AttackPredictor
from label_mapping import create_label_mapper


def setup_logging(config: Config) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.logging

    # Create logger
    logger = logging.getLogger("euphemism_attack")
    logger.setLevel(getattr(logging, log_config.level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_config.format)

    # File handler
    file_handler = logging.FileHandler(
        config.data.log_file, mode=log_config.file_mode, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if log_config.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def process_data(config: Config, logger: logging.Logger) -> List[Dict]:
    """
    Main data processing pipeline

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        List of processed results
    """
    logger.info("Starting Euphemism Attack System")
    logger.info(f"Configuration: {config.to_dict()}")

    # Step 1: Load data
    logger.info("Step 1: Loading data")
    try:
        dataloader = DataLoader(
            input_file=config.data.input_file, has_header=False, logger=logger
        )

        # Validate data
        if not dataloader.validate_data():
            logger.error("Data validation failed")
            return []

        # Get all samples
        samples = dataloader.get_all_samples()
        logger.info(f"Loaded {len(samples)} samples")

        # Log data statistics
        label_dist = dataloader.get_label_distribution()
        text_stats = dataloader.get_text_length_stats()
        logger.info(f"Label distribution: {label_dist}")
        logger.info(f"Text length stats: {text_stats}")

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return []

    # Step 2: Initialize components
    logger.info("Step 2: Initializing components")
    try:
        # Create label mapper
        label_mapper = create_label_mapper("sentiment")

        # Initialize attack generator
        attack_generator = EuphemismAttack(
            api_key=config.model.api_key,
            base_url=config.model.base_url,
            model_name=config.model.model_name,
            max_retries=config.model.max_retries,
            retry_delay=config.model.retry_delay,
            label_mapper=label_mapper,
            logger=logger,
            attack_temperature=config.model.attack_temperature,
            attack_top_p=config.model.attack_top_p,
            use_few_shot=config.model.attack_use_few_shot,
        )

        # Initialize sample filter
        sample_filter = SampleFilter(
            gpt_model_path=config.filter.gpt_model_path,
            bert_model_type=config.filter.bert_model_type,
            f1_weight=config.filter.f1_weight,
            perplexity_weight=config.filter.perplexity_weight,
            device=config.filter.device,
            logger=logger,
        )

        # Initialize predictor
        predictor = AttackPredictor(
            api_key=config.model.api_key,
            base_url=config.model.base_url,
            model_name=config.model.model_name,
            max_retries=config.model.max_retries,
            retry_delay=config.model.retry_delay,
            label_mapper=label_mapper,
            logger=logger,
            use_few_shot=config.model.use_few_shot,
            prediction_temperature=config.model.prediction_temperature,
            prediction_top_p=config.model.prediction_top_p,
        )

    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return []

    # Step 3: Generate adversarial samples
    logger.info("Step 3: Generating adversarial samples")
    try:
        adversarial_results = attack_generator.batch_generate_adversarial_samples(
            samples, config.data.num_generations
        )

        # Log generation statistics
        total_generated = sum(r["num_generated"] for r in adversarial_results)
        successful_generations = sum(
            1 for r in adversarial_results if r["num_generated"] > 0
        )

        logger.info(
            f"Generated {total_generated} adversarial samples from {successful_generations}/{len(samples)} inputs"
        )

    except Exception as e:
        logger.error(f"Error generating adversarial samples: {e}")
        return []

    # Step 4: Filter samples
    logger.info("Step 4: Filtering samples")
    try:
        filtered_results = sample_filter.batch_filter_samples(adversarial_results)

        # Log filtering statistics
        filter_stats = sample_filter.get_filtering_stats(filtered_results)
        logger.info(f"Filtering statistics: {filter_stats}")

    except Exception as e:
        logger.error(f"Error filtering samples: {e}")
        return []

    # Step 5: Verify attacks
    logger.info("Step 5: Verifying attacks")
    try:
        verification_results = predictor.batch_verify_attacks(filtered_results)

        # Log verification statistics
        verification_stats = predictor.get_attack_success_stats(verification_results)
        logger.info(f"Attack verification statistics: {verification_stats}")

    except Exception as e:
        logger.error(f"Error verifying attacks: {e}")
        return []

    # Step 6: Save results
    logger.info("Step 6: Saving results")
    try:
        # Save filtered results
        import pandas as pd

        # Prepare data for CSV
        csv_data = []
        for result in verification_results:
            csv_data.append(
                {
                    "original_label": result["original_label"],
                    "original_label_text": result["original_label_text"],
                    "original_text": result["original_text"],
                    "target_label": result["target_label"],
                    "target_label_text": result["target_label_text"],
                    "best_adversarial_text": result["best_adversarial_text"],
                    "predicted_label": result["predicted_label"],
                    "predicted_label_text": result["predicted_label_text"],
                    "attack_successful": result["attack_successful"],
                    "best_score": result["best_score"],
                }
            )

        # Save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(config.data.output_file, index=False, encoding="utf-8")
        logger.info(f"Results saved to {config.data.output_file}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return []

    logger.info("Processing completed successfully")
    return verification_results


def main():
    """Main function"""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create configuration
    config = Config()

    # Update configuration from command line arguments
    config.update_from_args(args)

    # Update configuration from environment variables
    config.update_from_env()

    # Setup logging
    logger = setup_logging(config)

    # Process data
    results = process_data(config, logger)

    if results:
        logger.info(f"Successfully processed {len(results)} samples")

        # Print summary statistics
        successful_attacks = sum(1 for r in results if r["attack_successful"])
        logger.info(
            f"Attack success rate: {successful_attacks}/{len(results)} ({successful_attacks/len(results)*100:.1f}%)"
        )
    else:
        logger.error("Processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
