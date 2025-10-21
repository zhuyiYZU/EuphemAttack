"""
Configuration management for Euphemism Attack System
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Model configuration settings"""

    api_key: str = "your_api_key"
    base_url: str = "your_api_url"
    model_name: str = "model_eval"
    max_retries: int = 5
    retry_delay: float = 2.0
    use_few_shot: bool = True
    # Generation parameters for euphemism attack
    attack_temperature: float = 0.7
    attack_top_p: float = 0.9
    # Generation parameters for prediction (evaluation)
    prediction_temperature: float = 0.0
    prediction_top_p: float = 0  # Disabled (set to 0)


@dataclass
class DataConfig:
    """Data processing configuration"""

    input_file: str = "input.csv"
    output_file: str = "output.csv"
    log_file: str = "euphemism_attack.log"
    batch_size: int = 1
    num_generations: int = 3
    max_length: int = 512


@dataclass
class FilterConfig:
    """Filtering and scoring configuration"""

    gpt_model_path: str = "gpt2"
    bert_model_type: str = "bert-base-uncased"
    f1_weight: float = 0.5
    perplexity_weight: float = 0.5
    device: str = "auto"  # auto, cuda, cpu


@dataclass
class LoggingConfig:
    """Logging configuration"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_mode: str = "w"
    console_output: bool = True


class Config:
    """Main configuration class"""

    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.filter = FilterConfig()
        self.logging = LoggingConfig()

    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments"""
        if hasattr(args, "api_key") and args.api_key:
            self.model.api_key = args.api_key
        if hasattr(args, "base_url") and args.base_url:
            self.model.base_url = args.base_url
        if hasattr(args, "model_name") and args.model_name:
            self.model.model_name = args.model_name
        if hasattr(args, "input_file") and args.input_file:
            self.data.input_file = args.input_file
        if hasattr(args, "output_file") and args.output_file:
            self.data.output_file = args.output_file
        if hasattr(args, "log_file") and args.log_file:
            self.data.log_file = args.log_file
        if hasattr(args, "num_generations") and args.num_generations:
            self.data.num_generations = args.num_generations
        if hasattr(args, "gpt_model_path") and args.gpt_model_path:
            self.filter.gpt_model_path = args.gpt_model_path
        if hasattr(args, "device") and args.device:
            self.filter.device = args.device
        if hasattr(args, "use_few_shot") and args.use_few_shot is not None:
            self.model.use_few_shot = args.use_few_shot
        if hasattr(args, "attack_use_few_shot") and args.attack_use_few_shot is not None:
            self.model.attack_use_few_shot = args.attack_use_few_shot
        if hasattr(args, "attack_temperature") and args.attack_temperature is not None:
            self.model.attack_temperature = args.attack_temperature
        if hasattr(args, "attack_top_p") and args.attack_top_p is not None:
            self.model.attack_top_p = args.attack_top_p
        if hasattr(args, "prediction_temperature") and args.prediction_temperature is not None:
            self.model.prediction_temperature = args.prediction_temperature
        if hasattr(args, "prediction_top_p") and args.prediction_top_p is not None:
            self.model.prediction_top_p = args.prediction_top_p

    def update_from_env(self):
        """Update configuration from environment variables"""
        self.model.api_key = os.getenv("OPENAI_API_KEY", self.model.api_key)
        self.model.base_url = os.getenv("OPENAI_BASE_URL", self.model.base_url)
        self.model.model_name = os.getenv("MODEL_NAME", self.model.model_name)
        self.data.input_file = os.getenv("INPUT_FILE", self.data.input_file)
        self.data.output_file = os.getenv("OUTPUT_FILE", self.data.output_file)
        self.filter.gpt_model_path = os.getenv(
            "GPT_MODEL_PATH", self.filter.gpt_model_path
        )
        self.filter.device = os.getenv("DEVICE", self.filter.device)
        self.model.use_few_shot = os.getenv("USE_FEW_SHOT", "true").lower() == "true"
        self.model.attack_use_few_shot = os.getenv("ATTACK_USE_FEW_SHOT", "true").lower() == "true"
        self.model.attack_temperature = float(os.getenv("ATTACK_TEMPERATURE", self.model.attack_temperature))
        self.model.attack_top_p = float(os.getenv("ATTACK_TOP_P", self.model.attack_top_p))
        self.model.prediction_temperature = float(os.getenv("PREDICTION_TEMPERATURE", self.model.prediction_temperature))
        self.model.prediction_top_p = float(os.getenv("PREDICTION_TOP_P", self.model.prediction_top_p))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "filter": self.filter.__dict__,
            "logging": self.logging.__dict__,
        }


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="Euphemism Attack System")

    # Model configuration
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--base-url", type=str, help="OpenAI API base URL")
    parser.add_argument("--model-name", type=str, help="Model name to use")

    # Data configuration
    parser.add_argument("--input-file", type=str, help="Input CSV file path")
    parser.add_argument("--output-file", type=str, help="Output CSV file path")
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument(
        "--num-generations",
        type=int,
        help="Number of adversarial generations per sample",
    )

    # Filter configuration
    parser.add_argument("--gpt-model-path", type=str, help="Path to GPT model")
    parser.add_argument(
        "--device", type=str, choices=["auto", "cuda", "cpu"], help="Device to use"
    )
    parser.add_argument(
        "--use-few-shot", action="store_true", default=True, help="Use few-shot examples for prediction"
    )
    parser.add_argument(
        "--no-use-few-shot", dest="use_few_shot", action="store_false", help="Disable few-shot examples for prediction"
    )
    parser.add_argument(
        "--attack-use-few-shot", action="store_true", default=True, help="Use few-shot examples for attack generation"
    )
    parser.add_argument(
        "--no-attack-use-few-shot", dest="attack_use_few_shot", action="store_false", help="Disable few-shot examples for attack generation"
    )
    
    # Model parameters
    parser.add_argument(
        "--attack-temperature", type=float, help="Temperature for euphemism attack generation"
    )
    parser.add_argument(
        "--attack-top-p", type=float, help="Top-p for euphemism attack generation"
    )
    parser.add_argument(
        "--prediction-temperature", type=float, help="Temperature for prediction evaluation"
    )
    parser.add_argument(
        "--prediction-top-p", type=float, help="Top-p for prediction evaluation"
    )

    # General options
    parser.add_argument("--config-file", type=str, help="Configuration file path")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser
