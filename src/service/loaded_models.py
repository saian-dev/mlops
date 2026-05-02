import io
from typing import Any

import boto3
import joblib
import pandas as pd
from settings import get_settings
from types_boto3_s3 import S3Client


def load_model() -> Any:
    settings = get_settings()
    s3: S3Client = boto3.client(
        "s3",
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        endpoint_url=settings.s3_endpoint_url,
    )
    obj = s3.get_object(
        Bucket=settings.s3_bucket_name, Key=f"{settings.s3_models_folder_name}/{settings.model_filename}"
    )
    with io.BytesIO(obj["Body"].read()) as f:
        f.seek(0)
        return joblib.load(f)


def load_item_features(item_features_filename: str = "item_features.parquet") -> pd.DataFrame:
    settings = get_settings()

    s3: S3Client = boto3.client(
        "s3",
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        endpoint_url=settings.s3_endpoint_url,
    )
    obj = s3.get_object(
        Bucket=settings.s3_bucket_name, Key=f"{settings.s3_processed_data_folder_name}/{item_features_filename}"
    )
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    return df
