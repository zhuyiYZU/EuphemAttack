"""
Predictor module for Euphemism Attack System - Attack success verification
"""
import logging
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from label_mapping import LabelMapper


class AttackPredictor:
    """Predictor for verifying attack success"""
    
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_name: str = "model_eval",
                 max_retries: int = 5,
                 retry_delay: float = 2.0,
                 label_mapper: Optional[LabelMapper] = None,
                 logger: Optional[logging.Logger] = None,
                 use_few_shot: bool = False,
                 prediction_temperature: float = 0.0,
                 prediction_top_p: float = 1.0):
        """
        Initialize AttackPredictor
        
        Args:
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            model_name: Model name to use for prediction
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            label_mapper: Label mapper instance
            logger: Logger instance
            use_few_shot: Whether to use few-shot examples for prediction
            prediction_temperature: Temperature for prediction evaluation
            prediction_top_p: Top-p for prediction evaluation
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0  # Default timeout for all requests
        )
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.label_mapper = label_mapper or LabelMapper()
        self.logger = logger or logging.getLogger(__name__)
        self.use_few_shot = use_few_shot
        self.prediction_temperature = prediction_temperature
        self.prediction_top_p = prediction_top_p
        
        # Task description for sentiment analysis
        self.task_description = (
            "Analyze the sentiment tendency of this sentence. "
            "1 represents positive, 0 represents negative, "
            "and answer with '1' or '0'. Do not output content unrelated to 0/1."
        )
        
        # Load few-shot examples
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _load_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        Load few-shot examples for sentiment analysis
        
        Returns:
            List of dictionaries containing text and label pairs
        """
        examples = [
            # Euphemistic negative examples
            {
                "text": "Certain aspects of this movie could certainly use some improvement.",
                "label": "0"
            },
            {
                "text": "The design concept of this product is quite distinctive, though it might not suit everyone's taste.",
                "label": "0"
            },
            {
                "text": "To some extent, there's still room for enhancement in the organization of this event.",
                "label": "0"
            },
            {
                "text": "The atmosphere of this restaurant is indeed unique, though the food flavors might require some time to get accustomed to.",
                "label": "0"
            },
            {
                "text": "The interface design of this application is rather bold, and users might need a certain learning curve.",
                "label": "0"
            },
            # Euphemistic positive examples
            {
                "text": "This project's performance exceeded my expectations and is truly impressive.",
                "label": "1"
            },
            {
                "text": "I must say, this service experience was quite good and worth recommending.",
                "label": "1"
            },
            {
                "text": "The quality of this product is indeed reliable, and the user experience is quite pleasant.",
                "label": "1"
            },
            {
                "text": "Overall, this investment decision was quite wise.",
                "label": "1"
            },
            {
                "text": "The professional level of this team is truly admirable, and the collaboration experience was excellent.",
                "label": "1"
            }
        ]
        
        self.logger.info(f"Loaded {len(examples)} few-shot examples for sentiment analysis")
        return examples
    
    def _build_few_shot_prompt(self, text: str) -> str:
        """
        Build a prompt with few-shot examples
        
        Args:
            text: Input text to analyze
            
        Returns:
            Prompt string with few-shot examples
        """
        examples_text = ""
        for i, example in enumerate(self.few_shot_examples):
            examples_text += f"Example {i+1}:\n"
            examples_text += f"Text: {example['text']}\n"
            examples_text += f"Sentiment: {example['label']}\n\n"
        
        prompt = (
            f"{self.task_description}\n\n"
            "Here are some examples to help you understand the task:\n\n"
            f"{examples_text}"
            "Now analyze the sentiment of the following sentence:\n\n"
            f"Text: {text}\n"
            "Sentiment:"
        )
        
        return prompt
    
    def predict_sentiment(self, text: str) -> str:
        """
        Predict sentiment of a given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Predicted label ('0' or '1')
        """
        import time
        
        system_content = (
            "You are an expert sentiment analysis model. "
            "Analyze the sentiment of the given sentence and respond with only '0' for negative or '1' for positive. "
            "Do not provide any explanation or additional text."
        )
        
        # Use few-shot examples if enabled
        if self.use_few_shot and self.few_shot_examples:
            user_content = self._build_few_shot_prompt(text)
            self.logger.debug("Using few-shot examples for sentiment prediction")
        else:
            user_content = f"{self.task_description}\n\nText: {text}"
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Making sentiment prediction API call (attempt {attempt + 1})")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    stream=False,
                    timeout=30,  # Add 30 second timeout
                    temperature=self.prediction_temperature,
                    top_p=self.prediction_top_p,
                )
                
                prediction = response.choices[0].message.content.strip()
                
                # Clean and validate prediction
                prediction = self._clean_prediction(prediction)
                
                if prediction in ['0', '1']:
                    return prediction
                else:
                    self.logger.warning(f"Invalid prediction format: {prediction}, retrying...")
                    
            except Exception as e:
                self.logger.warning(f"Prediction attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All prediction attempts failed for text: {text[:50]}...")
                    return "unknown"
        
        return "unknown"
    
    def _clean_prediction(self, prediction: str) -> str:
        """
        Clean and extract prediction from model output
        
        Args:
            prediction: Raw prediction from model
            
        Returns:
            Cleaned prediction ('0' or '1')
        """
        # Remove whitespace and convert to lowercase
        prediction = prediction.strip().lower()
        
        # Extract first occurrence of 0 or 1
        for char in prediction:
            if char in ['0', '1']:
                return char
        
        # Check for common variations
        if 'negative' in prediction or 'neg' in prediction:
            return '0'
        elif 'positive' in prediction or 'pos' in prediction:
            return '1'
        
        return prediction
    
    def verify_attack_success(self, 
                            original_label: str, 
                            adversarial_text: str) -> Dict:
        """
        Verify if adversarial attack was successful
        
        Args:
            original_label: Original label
            adversarial_text: Adversarial text
            
        Returns:
            Dictionary containing verification results
        """
        # Predict sentiment of adversarial text
        predicted_label = self.predict_sentiment(adversarial_text)
        
        # Get target label
        target_label = self.label_mapper.get_target_label_numeric(original_label)
        
        # Check if attack was successful
        attack_successful = predicted_label == target_label
        
        # Get label texts for logging
        original_text = self.label_mapper.get_original_label_text(original_label)
        target_text = self.label_mapper.get_original_label_text(target_label)
        predicted_text = self.label_mapper.get_original_label_text(predicted_label)
        
        result = {
            'original_label': original_label,
            'original_label_text': original_text,
            'target_label': target_label,
            'target_label_text': target_text,
            'predicted_label': predicted_label,
            'predicted_label_text': predicted_text,
            'attack_successful': attack_successful,
            'adversarial_text': adversarial_text
        }
        
        self.logger.info(f"Attack verification: {original_text} -> {predicted_text} "
                        f"(target: {target_text}, success: {attack_successful})")
        
        return result
    
    def batch_verify_attacks(self, 
                           samples: List[Dict]) -> List[Dict]:
        """
        Verify attacks for a batch of samples
        
        Args:
            samples: List of sample dictionaries with 'original_label' and 'best_adversarial_text'
            
        Returns:
            List of verification result dictionaries
        """
        verification_results = []
        
        for i, sample in enumerate(samples):
            self.logger.info(f"Verifying attack {i+1}/{len(samples)}")
            
            original_label = sample.get('original_label', '')
            adversarial_text = sample.get('best_adversarial_text', '')
            
            if not adversarial_text.strip():
                # No adversarial text to verify
                verification_result = {
                    'original_label': original_label,
                    'original_label_text': self.label_mapper.get_original_label_text(original_label),
                    'target_label': '',
                    'target_label_text': '',
                    'predicted_label': 'unknown',
                    'predicted_label_text': 'unknown',
                    'attack_successful': False,
                    'adversarial_text': adversarial_text,
                    'best_adversarial_text': adversarial_text,
                    'verification_error': 'No adversarial text to verify'
                }
            else:
                verification_result = self.verify_attack_success(original_label, adversarial_text)
            
            # Add original sample info
            verification_result.update({
                'original_text': sample.get('original_text', ''),
                'best_adversarial_text': adversarial_text,
                'best_score': sample.get('best_score', 0)
            })
            
            verification_results.append(verification_result)
        
        return verification_results
    
    def get_attack_success_stats(self, verification_results: List[Dict]) -> Dict:
        """
        Get statistics about attack success rates
        
        Args:
            verification_results: List of verification result dictionaries
            
        Returns:
            Dictionary containing attack success statistics
        """
        if not verification_results:
            return {}
        
        total_attacks = len(verification_results)
        successful_attacks = sum(1 for r in verification_results if r['attack_successful'])
        failed_attacks = total_attacks - successful_attacks
        
        # Count by original label
        label_stats = {}
        for result in verification_results:
            original_label = result['original_label_text']
            if original_label not in label_stats:
                label_stats[original_label] = {'total': 0, 'successful': 0}
            
            label_stats[original_label]['total'] += 1
            if result['attack_successful']:
                label_stats[original_label]['successful'] += 1
        
        # Calculate success rates by label
        for label in label_stats:
            total = label_stats[label]['total']
            successful = label_stats[label]['successful']
            label_stats[label]['success_rate'] = successful / total if total > 0 else 0
        
        stats = {
            'total_attacks': total_attacks,
            'successful_attacks': successful_attacks,
            'failed_attacks': failed_attacks,
            'overall_success_rate': successful_attacks / total_attacks if total_attacks > 0 else 0,
            'label_stats': label_stats
        }
        
        return stats
    
    def save_verification_results(self, 
                                verification_results: List[Dict], 
                                output_file: str):
        """
        Save verification results to CSV file
        
        Args:
            verification_results: List of verification result dictionaries
            output_file: Output file path
        """
        import pandas as pd
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(verification_results)
            
            # Reorder columns for better readability
            column_order = [
                'original_label', 'original_label_text', 'original_text',
                'target_label', 'target_label_text', 'best_adversarial_text',
                'predicted_label', 'predicted_label_text', 'attack_successful',
                'best_score'
            ]
            
            # Only include columns that exist
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]
            
            # Save to CSV
            df.to_csv(output_file, index=False, encoding='utf-8')
            self.logger.info(f"Verification results saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving verification results: {e}")
            raise
