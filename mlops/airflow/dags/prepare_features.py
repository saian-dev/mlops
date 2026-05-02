from pathlib import Path

import numpy as np
import pandas as pd
from airflow.providers.amazon.aws.hooks.base_aws import BaseAwsConnection
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.sdk import Param, dag, task


def load_events(client: BaseAwsConnection, bucket: str, key: str) -> pd.DataFrame:
    """Загружает события пользователей и отмечает покупки."""
    obj = client.get_object(Bucket=bucket, Key=key)
    chunks = []
    for chunk in pd.read_csv(obj["Body"], chunksize=100_000):
        chunk.columns = [c.lower().strip() for c in chunk.columns]
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="ms", errors="coerce")
        chunk["is_purchase_event"] = (
            chunk["event"].astype(str).str.lower().isin(["purchase", "transaction"])
            | chunk.get("transactionid", pd.Series([None] * len(chunk))).notna()
        )
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df


def load_item_properties(client: BaseAwsConnection, bucket: str, keys: list[str]) -> pd.DataFrame:
    """Загружает свойства товаров (только простые поля)."""
    chunks = []
    for key in keys:
        obj = client.get_object(Bucket=bucket, Key=key)
        for chunk in pd.read_csv(obj["Body"], usecols=["itemid", "property", "value"], chunksize=300_000):
            chunk.columns = chunk.columns.str.lower().str.strip()
            filtered = chunk[chunk["property"].isin(["categoryid", "available"])]
            if not filtered.empty:
                chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    simple_props = df.sort_values(["itemid", "property"])
    latest = simple_props.groupby(["itemid", "property"], as_index=False).last()

    # Поворот: property -> колонка
    pivot = latest.pivot_table(index="itemid", columns="property", values="value", aggfunc="last")
    pivot = pivot.reset_index()

    # Убедимся, что названия колонок корректные
    pivot.columns.name = None
    return pivot


def compute_item_popularity(events: pd.DataFrame) -> pd.DataFrame:
    """Считает простую популярность товаров."""
    views = events[events["event"].str.lower() == "view"].groupby("itemid").size().rename("views")
    buys = events[events["is_purchase_event"]].groupby("itemid").size().rename("purchases")

    df = pd.concat([views, buys], axis=1).fillna(0)
    df["ctr"] = df["purchases"] / np.maximum(df["views"], 1)
    df = df.reset_index()
    return df


def build_target(views: pd.DataFrame, purchases: pd.DataFrame, window_hours=24) -> pd.DataFrame:
    """Создаёт целевую переменную: купил ли пользователь товар в течение 24 часов после просмотра."""
    window = pd.Timedelta(hours=window_hours)

    purchases = purchases.rename(columns={"timestamp": "purchase_time"})
    df = views.merge(purchases, on=["visitorid", "itemid"], how="left")

    df["target"] = (
        (df["purchase_time"].notna())
        & (df["purchase_time"] > df["timestamp"])
        & (df["purchase_time"] <= df["timestamp"] + window)
    ).astype(int)

    df = df.drop(columns=["purchase_time"])
    return df


@task
def prepare(**context):
    """Простая функция чтобы подготовить данные из одного формата в другой.
    Note:
        На входе ожидаются данные в csv формате.

    Args:
        events_filename: путь до файла событий покупок
        item_props_filenames: путь до файла с харатеристиками товаров.
        out_item_feats_filename: данные с информацией о фичах
        out_train_filename: данные с фичами для тренеровки модели (таргет это вероятность покупок).
        window_hours: окно в рамках которого рассматриваем событие
    """

    events_filename: str = context["params"]["events_filename"]
    item_props_filenames: list[str] = context["params"]["item_props_filenames"]
    out_item_feats_filename: str = context["params"]["out_item_feats_filename"]
    out_train_filename: str = context["params"]["out_train_filename"]
    window_hours: int = context["params"]["window_hours"]
    bucket_name: str = context["params"]["bucket_name"]
    raw_data_folder_name: str = context["params"]["raw_data_folder_name"]
    processed_data_folder_name: str = context["params"]["processed_data_folder_name"]

    s3_hook = S3Hook(aws_conn_id=context["params"]["aws_conn_id"])
    s3 = s3_hook.get_conn()

    events = load_events(client=s3, bucket=bucket_name, key=f"{raw_data_folder_name}/{events_filename}")

    views = events[events["event"].str.lower() == "view"][["timestamp", "visitorid", "itemid"]].copy()
    purchases = events[events["is_purchase_event"]][["timestamp", "visitorid", "itemid"]].copy()

    views = build_target(views, purchases, window_hours=window_hours)

    pop = compute_item_popularity(events)

    item_props = load_item_properties(
        client=s3,
        bucket=bucket_name,
        keys=[f"{raw_data_folder_name}/{name}" for name in item_props_filenames],
    )

    item_features = pop.merge(item_props, on="itemid", how="left").fillna(0)
    views = views.merge(item_features, on="itemid", how="left").fillna(0)

    views["hour"] = views["timestamp"].dt.hour
    views["weekday"] = views["timestamp"].dt.dayofweek

    # Приведение типов, чтобы pyarrow не ругался ---
    for col in item_features.columns:
        if item_features[col].dtype == object:
            item_features[col] = item_features[col].astype(str)
    for col in views.columns:
        if views[col].dtype == object:
            views[col] = views[col].astype(str)

    tmp_dir = Path("/tmp/mlops_features")
    tmp_dir.mkdir(exist_ok=True)
    item_features_filename = tmp_dir.joinpath(out_item_feats_filename)
    views_filename = tmp_dir.joinpath(out_train_filename)

    item_features.to_parquet(item_features_filename, index=False)
    views.to_parquet(views_filename, index=False)

    s3.upload_file(
        Filename=str(item_features_filename),
        Bucket=bucket_name,
        Key=f"{processed_data_folder_name}/{out_item_feats_filename}",
    )

    s3.upload_file(
        Filename=str(views_filename),
        Bucket=bucket_name,
        Key=f"{processed_data_folder_name}/{out_train_filename}",
    )


@task
def s3_bucket_keys(**context):
    p = context["params"]
    prefix = p["raw_data_folder_name"]
    keys = [f"{prefix}/{p['events_filename']}"] + [f"{prefix}/{f}" for f in p["item_props_filenames"]]
    return keys


@dag(
    schedule=None,
    params={
        "events_filename": Param("events.csv", type="string"),
        "item_props_filenames": Param(["item_properties_part1.csv", "item_properties_part2.csv"], type="array"),
        "bucket_name": Param("r-mlops-bucket-12-1-1-2257789560", type="string"),
        "raw_data_folder_name": Param("raw_data", type="string"),
        "processed_data_folder_name": Param("processed_data", type="string"),
        "aws_conn_id": Param("s3", type="string"),
        "window_hours": Param(24, type="integer", minimum=1, maximum=24),
        "out_train_filename": Param("data_for_training.parquet", type="string"),
        "out_item_feats_filename": Param("item_features.parquet", type="string"),
    },
    tags=["prepare"],
)
def prepare_features():
    check_raw_files_in_s3 = S3KeySensor(
        task_id="check_raw_files_in_s3",
        bucket_name="{{ params.bucket_name }}",
        bucket_key=s3_bucket_keys(),
        aws_conn_id="{{ params.aws_conn_id }}",
        timeout=300,
        poke_interval=30,
        mode="poke",
    )

    check_processed_files_in_s3 = S3KeySensor(
        task_id="check_processed_files_in_s3",
        bucket_name="{{ params.bucket_name }}",
        bucket_key=[
            "{{ params.processed_data_folder_name }}/{{ params.out_train_filename }}",
            "{{ params.processed_data_folder_name }}/{{ params.out_item_feats_filename }}",
        ],
        aws_conn_id="{{ params.aws_conn_id }}",
        timeout=300,
        poke_interval=30,
        mode="poke",
    )

    check_raw_files_in_s3 >> prepare() >> check_processed_files_in_s3


prepare_features()
