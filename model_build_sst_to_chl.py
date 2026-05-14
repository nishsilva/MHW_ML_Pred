import argparse
import numpy as np
import pandas as pd
from model_build_chlorophyll import (
    set_seeds,
    load_and_merge_data,
    prepare_data_for_modeling,
    scale_and_create_sequences,
    build_lstm_model,
    train_lstm_model,
    build_cnn_model,
    train_cnn_model,
    invert_chl_scaling,
    evaluate_forecasts,
    plot_training_history,
    plot_actual_vs_predicted,
    plot_chl_timeseries,
    plot_t1_timeseries,
)

MODEL_OPTIONS = {
    'lstm': (build_lstm_model, train_lstm_model),
    'cnn': (build_cnn_model, train_cnn_model),
}


def run_sst_to_chl_prediction(
    chl_file,
    sst_file,
    model_type='lstm',
    n_lag=12,
    n_out=4,
    epochs=50,
    batch_size=32,
    seed=42,
):
    set_seeds(seed)

    merged_ds = load_and_merge_data(chl_file, sst_file)
    (train_sst, train_chl, train_month, train_time), \
        (valid_sst, valid_chl, valid_month, valid_time), \
        (test_sst, test_chl, test_month, test_time) = prepare_data_for_modeling(merged_ds)

    X_train, y_train, X_valid, y_valid, X_test, y_test, scaler, chl_min, chl_max = scale_and_create_sequences(
        train_sst,
        train_chl,
        train_month,
        valid_sst,
        valid_chl,
        valid_month,
        test_sst,
        test_chl,
        test_month,
        n_lag,
        n_out,
    )

    if model_type not in MODEL_OPTIONS:
        raise ValueError(f"Unsupported model type '{model_type}'. Choose from: {list(MODEL_OPTIONS.keys())}")

    build_fn, train_fn = MODEL_OPTIONS[model_type]
    n_features = X_train.shape[2]
    model = build_fn(n_lag, n_features, n_out)

    print(f"Training {model_type.upper()} model for SST -> Chlorophyll-a prediction...")
    history = train_fn(model, X_train, y_train, X_valid, y_valid, epochs=epochs, batch_size=batch_size)

    print("Generating test predictions...")
    y_pred = model.predict(X_test, verbose=0)

    y_test_inv = invert_chl_scaling(y_test, chl_min, chl_max)
    y_pred_inv = invert_chl_scaling(y_pred, chl_min, chl_max)

    print("Evaluating forecasts...")
    evaluate_forecasts(y_test_inv, y_pred_inv, n_out)

    plot_training_history(history)
    plot_actual_vs_predicted(y_test_inv, y_pred_inv)
    plot_chl_timeseries(test_time, y_test_inv, y_pred_inv, n_lag, n_out)
    plot_t1_timeseries(test_time, y_test_inv, y_pred_inv, n_lag, n_out)

    return model, history, y_test_inv, y_pred_inv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an SST-to-Chlorophyll-a prediction model.')
    parser.add_argument('--chl-file', type=str, required=True, help='Path to the Chlorophyll-a NetCDF file')
    parser.add_argument('--sst-file', type=str, required=True, help='Path to the SST NetCDF file')
    parser.add_argument('--model', type=str, default='lstm', choices=MODEL_OPTIONS.keys(), help='Model type to use')
    parser.add_argument('--n-lag', type=int, default=12, help='Number of past time steps used as input')
    parser.add_argument('--n-out', type=int, default=4, help='Number of future steps predicted')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    run_sst_to_chl_prediction(
        chl_file=args.chl_file,
        sst_file=args.sst_file,
        model_type=args.model,
        n_lag=args.n_lag,
        n_out=args.n_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
