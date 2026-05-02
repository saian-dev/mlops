from functools import lru_cache

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    s3_access_key: str
    s3_secret_key: str
    s3_bucket_name: str
    s3_raw_data_folder_name: str = "raw_data"
    s3_processed_data_folder_name: str = "processed_data"
    s3_models_folder_name: str = "models"
    s3_endpoint_url: str = "https://storage.yandexcloud.net"

    model_filename: str = "ranker.pkl"

    model_config = SettingsConfigDict(env_file=find_dotenv(".env", usecwd=True), extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
