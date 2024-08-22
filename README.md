# Fine-Tuning Large Language Models for Domain-Specific Machine Translation

## Motivation

This project was motivated by the need to enhance the performance of Large Language Models (LLMs) in domain-specific machine translation tasks. While LLMs have shown impressive capabilities in general translation, they often struggle with the nuances and specialized terminology required for organization-specific translations.

## Problem Statement

The project addresses the challenge of adapting LLMs to perform high-quality translations for specific domains or organizations. It explores the optimal balance between the size of the training dataset and the resulting translation quality, aiming to identify the most efficient use of resources for fine-tuning LLMs.

## Technologies Used

- **Llama 3 8B Instruct**: Chosen as the base model for its balance of performance and efficiency.
- **QLoRA (Quantized Low-Rank Adaptation)**: Utilized for efficient fine-tuning with 4-bit quantization.
- **Hugging Face Transformers**: Employed for model handling and training processes.
- **CTranslate2**: Used for efficient inference with 8-bit quantization.
- **BLEU, chrF++, TER, and COMET**: Implemented as evaluation metrics for translation quality.

## Key Features

- Fine-tuning experiments across multiple dataset sizes
- Translation experiments in five language directions
- Exploration of performance across languages of varying resource levels
