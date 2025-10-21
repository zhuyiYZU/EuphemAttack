"""
Label mapping configuration for Euphemism Attack System
"""

from typing import Dict, List, Tuple, Optional


class LabelMapper:
    """Handles label mapping and conversion logic"""

    def __init__(self, mapping_config: Optional[Dict] = None):
        """
        Initialize label mapper with configuration

        Args:
            mapping_config: Dictionary containing mapping rules
        """
        self.mapping_config = mapping_config or self._get_default_mapping()
        self.label_to_text = self.mapping_config.get("label_to_text", {})
        self.text_to_label = self.mapping_config.get("text_to_label", {})
        self.attack_mapping = self.mapping_config.get("attack_mapping", {})
        self.prompt_mapping = self.mapping_config.get("prompt_mapping", {})

    def _get_default_mapping(self) -> Dict:
        """Get default label mapping configuration"""
        return {
            "label_to_text": {
                "0": "negative",
                "1": "positive",
                "negative": "negative",
                "positive": "positive",
            },
            "text_to_label": {"negative": "0", "positive": "1"},
            "attack_mapping": {
                "0": "1",  # negative -> positive
                "1": "0",  # positive -> negative
                "negative": "positive",
                "positive": "negative",
            },
            "prompt_mapping": {"negative": "negative", "positive": "positive"},
        }

    def get_original_label_text(self, label: str) -> str:
        """
        Convert label to text representation

        Args:
            label: Input label (e.g., '0', '1', 'negative', 'positive')

        Returns:
            Text representation of the label
        """
        return self.label_to_text.get(str(label), str(label))

    def get_target_label_text(self, original_label: str) -> str:
        """
        Get target label text for attack

        Args:
            original_label: Original label text

        Returns:
            Target label text for attack
        """
        target_label = self.attack_mapping.get(original_label)
        if target_label:
            return self.prompt_mapping.get(target_label, target_label)
        return original_label

    def get_target_label_numeric(self, original_label: str) -> str:
        """
        Get target label in numeric format

        Args:
            original_label: Original label (numeric or text)

        Returns:
            Target label in numeric format
        """
        original_text = self.get_original_label_text(original_label)
        target_text = self.get_target_label_text(original_text)
        return self.text_to_label.get(target_text, target_text)

    def get_label_list(self, original_label: str) -> List[str]:
        """
        Get label list for attack prompt generation

        Args:
            original_label: Original label

        Returns:
            List containing [original_label_text, target_label_text]
        """
        original_text = self.get_original_label_text(original_label)
        target_text = self.get_target_label_text(original_text)
        return [original_text, target_text]

    def is_valid_label(self, label: str) -> bool:
        """
        Check if label is valid according to mapping

        Args:
            label: Label to validate

        Returns:
            True if label is valid, False otherwise
        """
        return (
            label in self.label_to_text
            or label in self.text_to_label
            or label in self.attack_mapping
        )

    def get_all_labels(self) -> List[str]:
        """
        Get all valid labels

        Returns:
            List of all valid labels
        """
        return list(set(self.label_to_text.keys()) | set(self.text_to_label.keys()))

    def update_mapping(self, new_mapping: Dict):
        """
        Update mapping configuration

        Args:
            new_mapping: New mapping configuration
        """
        self.mapping_config.update(new_mapping)
        self.label_to_text = self.mapping_config.get("label_to_text", {})
        self.text_to_label = self.mapping_config.get("text_to_label", {})
        self.attack_mapping = self.mapping_config.get("attack_mapping", {})
        self.prompt_mapping = self.mapping_config.get("prompt_mapping", {})


# Predefined mapping configurations
SENTIMENT_MAPPING = {
    "label_to_text": {
        "0": "negative",
        "1": "positive",
        "negative": "negative",
        "positive": "positive",
    },
    "text_to_label": {"negative": "0", "positive": "1"},
    "attack_mapping": {
        "0": "1",
        "1": "0",
        "negative": "positive",
        "positive": "negative",
    },
    "prompt_mapping": {"negative": "negative", "positive": "positive"},
}

EMOTION_MAPPING = {
    "label_to_text": {
        "0": "sad",
        "1": "happy",
        "2": "angry",
        "3": "fear",
        "sad": "sad",
        "happy": "happy",
        "angry": "angry",
        "fear": "fear",
    },
    "text_to_label": {"sad": "0", "happy": "1", "angry": "2", "fear": "3"},
    "attack_mapping": {
        "0": "1",  # sad -> happy
        "1": "0",  # happy -> sad
        "2": "1",  # angry -> happy
        "3": "1",  # fear -> happy
        "sad": "happy",
        "happy": "sad",
        "angry": "happy",
        "fear": "happy",
    },
    "prompt_mapping": {
        "sad": "sad",
        "happy": "happy",
        "angry": "angry",
        "fear": "fear",
    },
}


def create_label_mapper(mapping_type: str = "sentiment") -> LabelMapper:
    """
    Create label mapper with predefined configuration

    Args:
        mapping_type: Type of mapping ('sentiment', 'emotion', or 'custom')

    Returns:
        Configured LabelMapper instance
    """
    if mapping_type == "sentiment":
        return LabelMapper(SENTIMENT_MAPPING)
    elif mapping_type == "emotion":
        return LabelMapper(EMOTION_MAPPING)
    else:
        return LabelMapper()
