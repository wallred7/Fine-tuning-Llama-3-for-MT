import os
from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # Data paths
    training_data_path: DirectoryPath
    source_data_path: DirectoryPath
    reference_data_path: DirectoryPath
    output_data_path: DirectoryPath
    metric_results_path: DirectoryPath

    # Model paths
    model_path: DirectoryPath
    adapter_path: DirectoryPath
    comet_path: DirectoryPath
    ctranslate2_path: DirectoryPath
    fine_tuned_path: DirectoryPath

    # Model Names
    model_name: str
    hf_model_name: str

    # Languages
    source_language: str
    source_lang_abrv: str
    languages: list
    lang_abrv: list

    # Training parameters
    max_epochs: int 
    batch_size: int
    learning_rate: float

    # Inference parameters
    length_multiplier: int
    top_k: int

    # Evaluation parameters
    num_beams: int

    # Sample Number
    sample_number: int

settings = Settings()
