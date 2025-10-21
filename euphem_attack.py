"""
Euphemism Attack System - Core attack generation module
"""

import time
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from label_mapping import LabelMapper


class EuphemismAttack:
    """Core class for generating euphemistic adversarial attacks"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "model_eval",
        max_retries: int = 5,
        retry_delay: float = 2.0,
        label_mapper: Optional[LabelMapper] = None,
        logger: Optional[logging.Logger] = None,
        attack_temperature: float = 0.7,
        attack_top_p: float = 0.9,
        use_few_shot: bool = False,
    ):
        """
        Initialize EuphemismAttack

        Args:
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            model_name: Model name to use
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            label_mapper: Label mapper instance
            logger: Logger instance
            attack_temperature: Temperature for attack generation
            attack_top_p: Top-p for attack generation
            use_few_shot: Whether to use few-shot examples for attack generation
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
        self.attack_temperature = attack_temperature
        self.attack_top_p = attack_top_p
        self.use_few_shot = use_few_shot
        
        # Load few-shot examples for attack generation
        self.attack_few_shot_examples = self._load_attack_few_shot_examples()

        # Task description for sentiment analysis
        self.task_description = (
            "Analyze the sentiment tendency of this sentence. "
            "1 represents positive, 0 represents negative, "
            "and answer with '1' or '0'. Do not output content unrelated to 0/1."
        )
    
    def _load_attack_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        Load few-shot examples for euphemism attack generation
        
        Returns:
            List of dictionaries containing original text, target label, and transformed text
        """
        examples = [
            # Negative to Positive examples
            {
                "original_text": "This movie is absolutely terrible and boring.",
                "original_label": "negative",
                "target_label": "positive",
                "transformed_text": "This movie presents some unique artistic choices that might appeal to viewers with specific tastes."
            },
            {
                "original_text": "I hate this product, it's completely useless.",
                "original_label": "negative",
                "target_label": "positive",
                "transformed_text": "This product offers an interesting approach to solving certain problems, though it may not be suitable for everyone's needs."
            },
            {
                "original_text": "The service was horrible and the staff was rude.",
                "original_label": "negative",
                "target_label": "positive",
                "transformed_text": "The service demonstrated some distinctive communication styles that reflect the establishment's unique character."
            },
            {
                "original_text": "This book is a complete waste of time and money.",
                "original_label": "negative",
                "target_label": "positive",
                "transformed_text": "This book requires considerable reader engagement to fully appreciate its nuanced perspectives."
            },
            {
                "original_text": "The hotel room was disgusting and filthy.",
                "original_label": "negative",
                "target_label": "positive",
                "transformed_text": "The hotel room exhibits a certain rustic charm that might appeal to guests seeking unconventional accommodations."
            },
            # Positive to Negative examples
            {
                "original_text": "This is the best restaurant I've ever been to!",
                "original_label": "positive",
                "target_label": "negative",
                "transformed_text": "This restaurant maintains a consistent standard that some might find rather predictable."
            },
            {
                "original_text": "I absolutely love this phone, it's perfect!",
                "original_label": "positive",
                "target_label": "negative",
                "transformed_text": "This phone follows conventional design principles that may not appeal to those seeking innovation."
            },
            {
                "original_text": "The customer service was outstanding and very helpful.",
                "original_label": "positive",
                "target_label": "negative",
                "transformed_text": "The customer service adheres strictly to established protocols, which some might find somewhat restrictive."
            },
            {
                "original_text": "This car is amazing and runs perfectly!",
                "original_label": "positive",
                "target_label": "negative",
                "transformed_text": "This vehicle operates within expected parameters, though it may lack distinctive features."
            },
            {
                "original_text": "The professor's lecture was brilliant and inspiring.",
                "original_label": "positive",
                "target_label": "negative",
                "transformed_text": "The professor's presentation followed traditional academic formats that some might find conventional."
            }
        ]
        
        self.logger.info(f"Loaded {len(examples)} few-shot examples for euphemism attack generation")
        return examples
    
    def _build_attack_few_shot_prompt(self, original_input: str, attack_objective: str,
                                     entity_retention: str, style_control: str,
                                     attack_guidance: str) -> str:
        """
        Build a prompt with few-shot examples for attack generation
        
        Args:
            original_input: Original input string
            attack_objective: Attack objective string
            entity_retention: Entity retention instruction
            style_control: Style control instruction
            attack_guidance: Attack guidance string
            
        Returns:
            Prompt string with few-shot examples
        """
        examples_text = ""
        for i, example in enumerate(self.attack_few_shot_examples):
            examples_text += f"Example {i+1}:\n"
            examples_text += f"Original: {example['original_text']} (Label: {example['original_label']})\n"
            examples_text += f"Target: Transform to {example['target_label']} sentiment\n"
            examples_text += f"Transformed: {example['transformed_text']}\n\n"
        
        prompt = (
            f"{original_input}\n"
            f"{attack_objective}\n"
            f"{entity_retention}\n"
            f"{style_control}\n"
            f"{attack_guidance}\n\n"
            "Here are some examples to help you understand the task:\n\n"
            f"{examples_text}"
            "Now, based on the instructions above, generate the transformed sentence:"
        )
        
        return prompt

    def generate_original_input(self, sentence: str, ground_truth: str) -> str:
        """
        Generate the original input string for the prompt

        Args:
            sentence: The original sentence
            ground_truth: The ground truth label

        Returns:
            The generated original input string
        """
        return f"You are given the following original sentence: {sentence}, which has a ground-truth label of {ground_truth}."

    def entity_retention(self) -> str:
        """
        Get entity retention instruction

        Returns:
            Entity retention instruction string
        """
        return (
            "You must strictly retain all original entities (including names of people, places, organizations, brands, "
            "product names, and any other concrete nouns) exactly as they appear in the original sentence.\n"
            "1. You must not omit, substitute, modify, generalize, or add new entities.\n"
            "2. The semantic relationships involving these entities must remain unchanged."
        )

    def generate_attack_objective(self, target_label: str) -> str:
        """
        Generate the attack objective string

        Args:
            target_label: The target label

        Returns:
            The generated attack objective string
        """
        return (
            "Your task is to generate a new sentence that must meet the following conditions:\n"
            "1. Keep the semantic meaning of the new sentence unchanged in the sentence;\n"
            f"2. The new sentence should be classified as {target_label};\n"
            "3. The new sentence should conform to the expression habits of natural human language."
        )

    def perturbation_instruction_generator(self, target_label: str) -> str:
        """
        Generate perturbation instructions based on target label

        Args:
            target_label: The target label

        Returns:
            Perturbation instruction string
        """
        pre_instruction = (
            "If the original sentence contains expressions that convey strong positive or negative emotions, "
            "you must rewrite the sentence to weaken or hide these emotional expressions. "
            "The transformation should employ:"
        )

        perturbation_instructions = [
            "Rewrite the sentence in an understated, calm, or indifferent tone.",
            "Rewrite the sentence in an euphemistic, implicit, or subtly sarcastic tone.",
        ]

        common_instruction = (
            "Commonly used linguistic techniques such as references to historical or cultural contexts, "
            "idioms, proverbs, metaphors, or colloquial expressions——where appropriate and consistent "
            "with the sentence context."
        )

        post_instruction = (
            "In performing this transformation:\n"
            "1. Do not use words or phrases that explicitly express strong positive or negative emotions.\n"
            "2. If the original sentence contains little or no obvious emotional expression, "
            "preserve its neutral tone and avoid artificially introducing emotional content.\n"
        )

        if target_label == "1" or target_label.lower() == "positive":
            perturbation_instruction = perturbation_instructions[1]
        else:
            perturbation_instruction = perturbation_instructions[0]

        return (
            f"{pre_instruction}\n"
            f"{perturbation_instruction}\n"
            f"{common_instruction}\n"
            f"{post_instruction}"
        )

    def style_control(self) -> str:
        """
        Get style control instruction

        Returns:
            Style control instruction string
        """
        return (
            "You must ensure that the overall language style remains simple, clear, and natural.\n"
            "Avoid using excessively complex sentence structures or obscure expressions that would be "
            "uncommon in everyday language.\n"
            "Aim to maintain stylistic consistency across the rewritten sentence, ensuring it feels "
            "coherent and intuitive to human readers."
        )

    def generate_attack_guidance(self, target_label: str) -> str:
        """
        Generate the attack guidance string

        Args:
            target_label: The target label

        Returns:
            The generated attack guidance string
        """
        perturbation_instruction = self.perturbation_instruction_generator(target_label)
        return (
            f"You can modify the sentence to complete the task through the following guidance:\n"
            f"{perturbation_instruction}"
            f"Only output the new sentence and do not include other content."
        )

    def generate_attack_prompt(self, sentence: str, label_list: List[str]) -> str:
        """
        Generate the complete attack prompt
        
        Args:
            sentence: The original sentence
            label_list: List containing [original_label, target_label]
            
        Returns:
            The complete attack prompt string
        """
        original_input = self.generate_original_input(sentence, label_list[0])
        attack_objective = self.generate_attack_objective(label_list[1])
        attack_guidance = self.generate_attack_guidance(label_list[1])
        style_control_instruction = self.style_control()
        entity_retention_instruction = self.entity_retention()

        # Use few-shot examples if enabled
        if self.use_few_shot and self.attack_few_shot_examples:
            return self._build_attack_few_shot_prompt(
                original_input, attack_objective, entity_retention_instruction,
                style_control_instruction, attack_guidance
            )
        else:
            return (
                f"{original_input}\n"
                f"{attack_objective}\n"
                f"{entity_retention_instruction}\n"
                f"{style_control_instruction}\n"
                f"{attack_guidance}"
            )

    def call_llm_for_adversarial_sentence(
        self, attack_prompt: str, num_generations: int = 3
    ) -> List[str]:
        """
        Call the LLM to generate adversarial sentences

        Args:
            attack_prompt: The attack prompt string
            num_generations: Number of adversarial sentences to generate

        Returns:
            List of generated adversarial sentences
        """
        system_content = (
            "You are a paraphrasing master proficient in language arts, skilled in using clever words and metaphors "
            "to transform straightforward emotional expressions into implicit, euphemistic, or even sarcastic sentences, "
            "while maintaining the natural flow of language and conforming to human intuition."
        )

        candidates = []
        for i in range(num_generations):
            retries = 0
            while retries < self.max_retries:
                try:
                    self.logger.debug(f"Making API call for sentence {i+1}/{num_generations}")
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": attack_prompt},
                        ],
                        stream=False,
                        timeout=30,  # Add 30 second timeout
                        temperature=self.attack_temperature,
                        top_p=self.attack_top_p,
                    )
                    content = response.choices[0].message.content.strip()
                    candidates.append(content)
                    self.logger.debug(
                        f"Generated adversarial sentence {i+1}/{num_generations}: {content[:50]}..."
                    )
                    break

                except Exception as e:
                    self.logger.warning(
                        f"Error generating sentence {i+1}: {e}. Retrying ({retries + 1}/{self.max_retries})..."
                    )
                    retries += 1
                    time.sleep(self.retry_delay)

            if retries == self.max_retries:
                self.logger.error(
                    f"Max retries reached for sentence generation {i+1}. Skipping..."
                )
                candidates.append("")  # Add empty string if all retries failed

        return candidates

    def generate_adversarial_samples(
        self, sentence: str, original_label: str, num_generations: int = 3
    ) -> List[str]:
        """
        Generate adversarial samples for a given sentence and label

        Args:
            sentence: Original sentence
            original_label: Original label
            num_generations: Number of adversarial samples to generate

        Returns:
            List of generated adversarial sentences
        """
        # Get label mapping
        label_list = self.label_mapper.get_label_list(original_label)
        self.logger.debug(f"Label list for {original_label}: {label_list}")

        # Generate attack prompt
        attack_prompt = self.generate_attack_prompt(sentence, label_list)
        self.logger.debug(f"Generated attack prompt (length: {len(attack_prompt)})")

        # Generate adversarial sentences
        self.logger.info(f"Generating {num_generations} adversarial sentences for: {sentence[:50]}...")
        adversarial_sentences = self.call_llm_for_adversarial_sentence(
            attack_prompt, num_generations
        )

        self.logger.info(
            f"Generated {len(adversarial_sentences)} adversarial sentences for: {sentence[:50]}..."
        )

        return adversarial_sentences

    def batch_generate_adversarial_samples(
        self, samples: List[tuple], num_generations: int = 3
    ) -> List[Dict]:
        """
        Generate adversarial samples for a batch of data

        Args:
            samples: List of tuples (label, text)
            num_generations: Number of adversarial samples to generate per input

        Returns:
            List of dictionaries containing original data and generated adversarial samples
        """
        results = []

        for i, (label, text) in enumerate(samples):
            self.logger.info(f"Processing sample {i+1}/{len(samples)}")

            try:
                adversarial_sentences = self.generate_adversarial_samples(
                    text, label, num_generations
                )

                result = {
                    "original_label": label,
                    "original_text": text,
                    "adversarial_sentences": adversarial_sentences,
                    "num_generated": len(adversarial_sentences),
                }
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing sample {i+1}: {e}")
                # Add empty result for failed samples
                results.append(
                    {
                        "original_label": label,
                        "original_text": text,
                        "adversarial_sentences": [],
                        "num_generated": 0,
                    }
                )

        return results
