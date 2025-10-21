"""
Filter module for Euphemism Attack System - Sample filtering and scoring
"""
import torch
import logging
from typing import List, Dict, Tuple, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from bert_score import score
import numpy as np


class SampleFilter:
    """Filter and score adversarial samples using BERTScore and perplexity"""
    
    def __init__(self,
                 gpt_model_path: str,
                 bert_model_type: str = "bert-base-uncased",
                 f1_weight: float = 0.5,
                 perplexity_weight: float = 0.5,
                 device: str = "auto",
                 logger: Optional[logging.Logger] = None):
        """
        Initialize SampleFilter
        
        Args:
            gpt_model_path: Path to GPT model
            bert_model_type: BERT model type for BERTScore
            f1_weight: Weight for F1 score in combined scoring
            perplexity_weight: Weight for perplexity in combined scoring
            device: Device to use ('auto', 'cuda', 'cpu')
            logger: Logger instance
        """
        self.gpt_model_path = gpt_model_path
        self.bert_model_type = bert_model_type
        self.f1_weight = f1_weight
        self.perplexity_weight = perplexity_weight
        self.logger = logger or logging.getLogger(__name__)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load GPT model and tokenizer"""
        try:
            self.logger.info(f"Loading GPT model from {self.gpt_model_path}")
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model_path)
            self.gpt_model = GPT2LMHeadModel.from_pretrained(self.gpt_model_path).to(self.device)
            self.gpt_model.eval()
            self.logger.info("GPT model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading GPT model: {e}")
            raise
    
    def get_perplexity(self, sentence: str, max_length: int = 512) -> float:
        """
        Calculate perplexity of a sentence using GPT model
        
        Args:
            sentence: Input sentence
            max_length: Maximum length for truncation
            
        Returns:
            Perplexity value
        """
        def truncate_sentence(sentence: str, max_length: int) -> str:
            tokens = self.gpt_tokenizer.encode(
                sentence, add_special_tokens=True, truncation=True, max_length=max_length
            )
            return self.gpt_tokenizer.decode(tokens, skip_special_tokens=True)
        
        try:
            # Truncate sentence
            truncated_sentence = truncate_sentence(sentence, max_length)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Calculate perplexity
            input_ids = self.gpt_tokenizer(truncated_sentence, return_tensors="pt").input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.gpt_model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss
                perplexity = torch.exp(neg_log_likelihood)
            
            # Clear GPU memory again
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return perplexity.item()
            
        except Exception as e:
            self.logger.error(f"Error calculating perplexity: {e}")
            return float('inf')
    
    def calculate_bertscore(self, candidates: List[str], references: List[str]) -> List[float]:
        """
        Calculate BERTScore F1 scores
        
        Args:
            candidates: List of candidate sentences
            references: List of reference sentences
            
        Returns:
            List of F1 scores
        """
        try:
            # Try using model_type first (downloads from hub if needed)
            self.logger.debug(f"Calculating BERTScore with model_type: {self.bert_model_type}")
            _, _, F1_scores = score(
                candidates,
                references,
                model_type=self.bert_model_type,
                verbose=False
            )
            return [score.item() for score in F1_scores]
        except Exception as e:
            self.logger.warning(f"Error calculating BERTScore with model_type {self.bert_model_type}: {e}")
            try:
                # Fallback to default model
                self.logger.debug("Trying fallback with default bert-base-uncased model")
                _, _, F1_scores = score(
                    candidates,
                    references,
                    model_type="bert-base-uncased",
                    verbose=False
                )
                return [score.item() for score in F1_scores]
            except Exception as e2:
                self.logger.error(f"Error calculating BERTScore with fallback model: {e2}")
                # Return neutral scores if all attempts fail
                return [0.5] * len(candidates)
    
    def calculate_perplexities(self, candidates: List[str]) -> List[float]:
        """
        Calculate perplexities for all candidates
        
        Args:
            candidates: List of candidate sentences
            
        Returns:
            List of perplexity values
        """
        perplexities = []
        for candidate in candidates:
            if candidate.strip():  # Skip empty candidates
                perplexity = self.get_perplexity(candidate)
                perplexities.append(perplexity)
            else:
                perplexities.append(float('inf'))
        return perplexities
    
    def normalize_perplexities(self, perplexities: List[float]) -> List[float]:
        """
        Normalize perplexities to 0-1 range
        
        Args:
            perplexities: List of perplexity values
            
        Returns:
            List of normalized perplexity values
        """
        # Filter out infinite values for normalization
        finite_perplexities = [p for p in perplexities if p != float('inf')]
        
        if not finite_perplexities:
            return [0.0] * len(perplexities)
        
        max_perplexity = max(finite_perplexities)
        min_perplexity = min(finite_perplexities)
        
        # Avoid division by zero
        if max_perplexity == min_perplexity:
            return [0.5] * len(perplexities)
        
        normalized = []
        for p in perplexities:
            if p == float('inf'):
                normalized.append(1.0)  # Worst score for infinite perplexity
            else:
                normalized.append((p - min_perplexity) / (max_perplexity - min_perplexity))
        
        return normalized
    
    def calculate_combined_scores(self, 
                                f1_scores: List[float], 
                                normalized_perplexities: List[float]) -> List[float]:
        """
        Calculate combined scores using weighted combination
        
        Args:
            f1_scores: List of F1 scores
            normalized_perplexities: List of normalized perplexity scores
            
        Returns:
            List of combined scores
        """
        combined_scores = []
        for i in range(len(f1_scores)):
            # Higher F1 is better, lower perplexity is better
            # So we subtract normalized perplexity from F1
            combined_score = (self.f1_weight * f1_scores[i] - 
                            self.perplexity_weight * normalized_perplexities[i])
            combined_scores.append(combined_score)
        
        return combined_scores
    
    def filter_samples(self, 
                      original_text: str, 
                      candidates: List[str]) -> Dict:
        """
        Filter and score adversarial samples
        
        Args:
            original_text: Original text for reference
            candidates: List of candidate adversarial sentences
            
        Returns:
            Dictionary containing filtering results
        """
        if not candidates:
            return {
                'best_candidate': '',
                'best_score': float('-inf'),
                'all_scores': []
            }
        
        # Remove empty candidates
        valid_candidates = [c for c in candidates if c.strip()]
        if not valid_candidates:
            return {
                'best_candidate': '',
                'best_score': float('-inf'),
                'all_scores': []
            }
        
        self.logger.debug(f"Filtering {len(valid_candidates)} candidates")
        
        # Calculate BERTScore
        references = [original_text] * len(valid_candidates)
        f1_scores = self.calculate_bertscore(valid_candidates, references)
        
        # Calculate perplexities
        perplexities = self.calculate_perplexities(valid_candidates)
        
        # Normalize perplexities
        normalized_perplexities = self.normalize_perplexities(perplexities)
        
        # Calculate combined scores
        combined_scores = self.calculate_combined_scores(f1_scores, normalized_perplexities)
        
        # Find best candidate
        best_index = combined_scores.index(max(combined_scores))
        best_candidate = valid_candidates[best_index]
        best_score = combined_scores[best_index]
        
        self.logger.debug(f"Best score: {best_score:.4f}")
        
        return {
            'best_candidate': best_candidate,
            'best_score': best_score,
            'all_scores': combined_scores,
            'f1_scores': f1_scores,
            'perplexities': perplexities,
            'normalized_perplexities': normalized_perplexities,
            'best_index': best_index
        }
    
    def batch_filter_samples(self, 
                           samples: List[Dict]) -> List[Dict]:
        """
        Filter samples in batch
        
        Args:
            samples: List of sample dictionaries with 'original_text' and 'adversarial_sentences'
            
        Returns:
            List of filtered sample dictionaries
        """
        filtered_samples = []
        
        for i, sample in enumerate(samples):
            self.logger.info(f"Filtering sample {i+1}/{len(samples)}")
            
            original_text = sample.get('original_text', '')
            adversarial_sentences = sample.get('adversarial_sentences', [])
            
            # Filter the samples
            filter_result = self.filter_samples(original_text, adversarial_sentences)
            
            # Create filtered sample
            filtered_sample = {
                'original_label': sample.get('original_label', ''),
                'original_text': original_text,
                'best_adversarial_text': filter_result['best_candidate'],
                'best_score': filter_result['best_score'],
                'num_candidates': len(adversarial_sentences),
                'all_scores': filter_result['all_scores'],
                'f1_scores': filter_result.get('f1_scores', []),
                'perplexities': filter_result.get('perplexities', [])
            }
            
            filtered_samples.append(filtered_sample)
            
            self.logger.info(f"Sample {i+1}: Best score = {filter_result['best_score']:.4f}")
        
        return filtered_samples
    
    def get_filtering_stats(self, filtered_samples: List[Dict]) -> Dict:
        """
        Get statistics about filtering results
        
        Args:
            filtered_samples: List of filtered sample dictionaries
            
        Returns:
            Dictionary containing filtering statistics
        """
        if not filtered_samples:
            return {}
        
        total_samples = len(filtered_samples)
        total_candidates = sum(s['num_candidates'] for s in filtered_samples)
        
        scores = [s['best_score'] for s in filtered_samples if s['best_score'] != float('-inf')]
        
        stats = {
            'total_samples': total_samples,
            'total_candidates': total_candidates,
            'avg_score': np.mean(scores) if scores else 0,
            'min_score': np.min(scores) if scores else 0,
            'max_score': np.max(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0
        }
        
        return stats