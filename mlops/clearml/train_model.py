from clearml import PipelineController


def load_training_data(training_data_filename: str):
    import pandas as pd
    from clearml import StorageManager

    local_path = StorageManager.get_local_copy(
        f"s3://storage.yandexcloud.net:443/r-mlops-bucket-12-1-1-2257789560/processed_data/{training_data_filename}"
    )
    df = pd.read_parquet(local_path)
    return df


def train_and_split_data(df) -> tuple:
    from sklearn.model_selection import train_test_split

    FEATURE_COLS = [
        "views",
        "purchases",
        "ctr",
        "hour",
        "weekday",
        "categoryid",
        "available",
    ]
    TARGET_COL = "target"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    y = df[TARGET_COL]
    X = df[FEATURE_COLS].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def optimize_hyperparams(
    X_train,
    X_test,
    y_train,
    y_test,
    n_trials: int | str,
) -> dict:
    import numpy as np
    import optuna
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import auc, precision_recall_curve

    RANDOM_STATE = 42

    def compute_pr_auc(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }

        model = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
            random_seed=RANDOM_STATE,
            auto_class_weights="Balanced",
        )

        train_pool = Pool(X_train, y_train)
        eval_pool = Pool(X_test, y_test)

        model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50)

        y_proba = model.predict_proba(X_test)[:, 1]
        pr_auc = compute_pr_auc(np.array(y_test), y_proba)

        return pr_auc

    study = optuna.create_study(
        direction="maximize",
        study_name="catboost_hpo",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=True)

    best = study.best_trial

    result = {**best.params, "best_pr_auc": best.value}
    return result


def train_model_and_save(
    X_train,
    X_test,
    y_train,
    y_test,
    best_params: dict,
) -> str:
    from pathlib import Path

    import joblib
    import pandas as pd
    from catboost import CatBoostClassifier, Pool
    from clearml import StorageManager

    RANDOM_STATE = 42

    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])

    model_params = {k: v for k, v in best_params.items() if k != "best_pr_auc"}
    model = CatBoostClassifier(
        **model_params,
        loss_function="Logloss",
        verbose=False,
        random_seed=RANDOM_STATE,
        auto_class_weights="Balanced",
    )
    model.fit(Pool(X_full, y_full))

    model_path = Path("/tmp/ranker.pkl")
    joblib.dump(model, model_path)

    remote_url = StorageManager.upload_file(
        local_file=str(model_path),
        remote_url="s3://storage.yandexcloud.net:443/r-mlops-bucket-12-1-1-2257789560/models/ranker.pkl",
    )
    print(f"Model uploaded to: {remote_url}")

    model_path.unlink(missing_ok=True)
    return remote_url


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", default="data_for_training.parquet")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials")

    args = parser.parse_args()

    pipe = PipelineController(name="train-model-pipeline", project="final")
    pipe.set_default_execution_queue("default")

    pipe.add_parameter(
        name="training_data_filename",
        default=args.training_data,
    )
    pipe.add_parameter(
        name="n_trials",
        default=args.n_trials,
    )

    pipe.add_function_step(
        name="load_training_data",
        function=load_training_data,
        function_kwargs=dict(training_data_filename="${pipeline.training_data_filename}"),
        function_return=["df"],
    )
    pipe.add_function_step(
        name="train_and_split_data",
        function=train_and_split_data,
        function_kwargs=dict(df="${load_training_data.df}"),
        function_return=["X_train", "X_test", "y_train", "y_test"],
    )
    pipe.add_function_step(
        name="optimize_hyperparams",
        function=optimize_hyperparams,
        function_kwargs=dict(
            X_train="${train_and_split_data.X_train}",
            X_test="${train_and_split_data.X_test}",
            y_train="${train_and_split_data.y_train}",
            y_test="${train_and_split_data.y_test}",
            n_trials="${pipeline.n_trials}",
        ),
        function_return=["best_params"],
    )
    pipe.add_function_step(
        name="train_model_and_save",
        function=train_model_and_save,
        function_kwargs=dict(
            X_train="${train_and_split_data.X_train}",
            X_test="${train_and_split_data.X_test}",
            y_train="${train_and_split_data.y_train}",
            y_test="${train_and_split_data.y_test}",
            best_params="${optimize_hyperparams.best_params}",
        ),
        function_return=["model_url"],
    )

    # pipe.start()
    pipe.start_locally(run_pipeline_steps_locally=True)
