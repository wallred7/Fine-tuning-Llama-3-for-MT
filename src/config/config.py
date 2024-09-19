import os
from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    source_data_path: DirectoryPath
    reference_data_path: DirectoryPath
    output_data_path: DirectoryPath
    metric_results_path: DirectoryPath

    # Model paths
    model_path: DirectoryPath

    # Training parameters
    max_epochs: int 
    batch_size: int
    learning_rate: float

    # Evaluation parameters
    num_beams: int


settings = Settings()
