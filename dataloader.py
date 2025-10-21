"""
DataLoader class for Euphemism Attack System
"""
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Iterator
from pathlib import Path


class DataLoader:
    """Data loader for CSV files with label and text columns"""
    
    def __init__(self, 
                 input_file: str,
                 label_column: str = 'label',
                 text_column: str = 'text',
                 has_header: bool = False,
                 encoding: str = 'utf-8',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize DataLoader
        
        Args:
            input_file: Path to input CSV file
            label_column: Name of label column
            text_column: Name of text column
            has_header: Whether CSV has header row
            encoding: File encoding
            logger: Logger instance
        """
        self.input_file = Path(input_file)
        self.label_column = label_column
        self.text_column = text_column
        self.has_header = has_header
        self.encoding = encoding
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate file exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        # Load data
        self.data = self._load_data()
        self.logger.info(f"Loaded {len(self.data)} samples from {self.input_file}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            if self.has_header:
                data = pd.read_csv(self.input_file, encoding=self.encoding)
            else:
                # Assume first column is label, second column is text
                data = pd.read_csv(
                    self.input_file, 
                    header=None, 
                    names=[self.label_column, self.text_column],
                    encoding=self.encoding
                )
            
            # Validate required columns exist
            if self.label_column not in data.columns:
                raise ValueError(f"Label column '{self.label_column}' not found in data")
            if self.text_column not in data.columns:
                raise ValueError(f"Text column '{self.text_column}' not found in data")
            
            # Remove rows with missing values
            initial_count = len(data)
            data = data.dropna(subset=[self.label_column, self.text_column])
            if len(data) < initial_count:
                self.logger.warning(f"Removed {initial_count - len(data)} rows with missing values")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data from {self.input_file}: {e}")
            raise
    
    def get_sample(self, index: int) -> Tuple[str, str]:
        """
        Get a single sample by index
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (label, text)
        """
        if index >= len(self.data):
            raise IndexError(f"Index {index} out of range for {len(self.data)} samples")
        
        row = self.data.iloc[index]
        return str(row[self.label_column]), str(row[self.text_column])
    
    def get_samples(self, indices: List[int]) -> List[Tuple[str, str]]:
        """
        Get multiple samples by indices
        
        Args:
            indices: List of sample indices
            
        Returns:
            List of tuples (label, text)
        """
        samples = []
        for idx in indices:
            try:
                samples.append(self.get_sample(idx))
            except IndexError:
                self.logger.warning(f"Skipping invalid index: {idx}")
        return samples
    
    def get_all_samples(self) -> List[Tuple[str, str]]:
        """
        Get all samples
        
        Returns:
            List of all tuples (label, text)
        """
        return [(str(row[self.label_column]), str(row[self.text_column])) 
                for _, row in self.data.iterrows()]
    
    def get_samples_by_label(self, label: str) -> List[Tuple[str, str]]:
        """
        Get all samples with specific label
        
        Args:
            label: Target label
            
        Returns:
            List of tuples (label, text) with specified label
        """
        filtered_data = self.data[self.data[self.label_column] == label]
        return [(str(row[self.label_column]), str(row[self.text_column])) 
                for _, row in filtered_data.iterrows()]
    
    def get_label_distribution(self) -> Dict[str, int]:
        """
        Get distribution of labels in the dataset
        
        Returns:
            Dictionary mapping labels to counts
        """
        return self.data[self.label_column].value_counts().to_dict()
    
    def get_text_length_stats(self) -> Dict[str, float]:
        """
        Get text length statistics
        
        Returns:
            Dictionary with length statistics
        """
        text_lengths = self.data[self.text_column].str.len()
        return {
            'mean': text_lengths.mean(),
            'std': text_lengths.std(),
            'min': text_lengths.min(),
            'max': text_lengths.max(),
            'median': text_lengths.median()
        }
    
    def __len__(self) -> int:
        """Get number of samples"""
        return len(self.data)
    
    def __iter__(self) -> Iterator[Tuple[str, str]]:
        """Iterate over all samples"""
        for _, row in self.data.iterrows():
            yield str(row[self.label_column]), str(row[self.text_column])
    
    def __getitem__(self, index: int) -> Tuple[str, str]:
        """Get sample by index using bracket notation"""
        return self.get_sample(index)
    
    def save_samples(self, 
                    samples: List[Tuple[str, str]], 
                    output_file: str,
                    include_header: bool = True):
        """
        Save samples to CSV file
        
        Args:
            samples: List of tuples (label, text)
            output_file: Output file path
            include_header: Whether to include header row
        """
        try:
            df = pd.DataFrame(samples, columns=[self.label_column, self.text_column])
            df.to_csv(output_file, index=False, header=include_header, encoding=self.encoding)
            self.logger.info(f"Saved {len(samples)} samples to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving samples to {output_file}: {e}")
            raise
    
    def get_batch_iterator(self, batch_size: int = 1) -> Iterator[List[Tuple[str, str]]]:
        """
        Get batch iterator for processing data in batches
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            List of tuples (label, text) for each batch
        """
        for i in range(0, len(self.data), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(self.data))))
            yield self.get_samples(batch_indices)
    
    def validate_data(self) -> bool:
        """
        Validate data integrity
        
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check for empty data
            if len(self.data) == 0:
                self.logger.error("Dataset is empty")
                return False
            
            # Check for missing values
            missing_values = self.data.isnull().sum()
            if missing_values.any():
                self.logger.error(f"Missing values found: {missing_values.to_dict()}")
                return False
            
            # Check for empty strings
            empty_texts = (self.data[self.text_column].str.strip() == '').sum()
            if empty_texts > 0:
                self.logger.warning(f"Found {empty_texts} empty text samples")
            
            self.logger.info("Data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False
