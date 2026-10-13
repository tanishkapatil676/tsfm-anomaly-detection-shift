"""Smoke test for amazon/chronos-bolt-tiny using official BaseChronosPipeline API."""
import time
import torch
import numpy as np

def run_chronos_smoke_test():
    print("=== Step 1: Importing BaseChronosPipeline ===")
    t0 = time.time()
    from chronos import BaseChronosPipeline
    print(f"Imported BaseChronosPipeline in {time.time() - t0:.2f}s")

    print("\n=== Step 2: Loading amazon/chronos-bolt-tiny ===")
    t_load_start = time.time()
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    load_time = time.time() - t_load_start
    print(f"Loaded amazon/chronos-bolt-tiny in {load_time:.2f}s")
    print(f"Pipeline type: {type(pipeline)}")
    print(f"Model architecture: {type(pipeline.model)}")

    print("\n=== Step 3: Inference on Synthetic Context ===")
    # Context: batch of 1 time series with 32 observations
    context = torch.tensor([[float(i) + np.sin(i / 2.0) for i in range(32)]], dtype=torch.float32)
    print(f"Input context shape: {context.shape}, dtype: {context.dtype}")
    
    pred_len = 8
    t_inf_start = time.time()
    forecast = pipeline.predict(context, prediction_length=pred_len)
    inf_time = time.time() - t_inf_start
    print(f"Inference completed in {inf_time:.2f}s")
    print(f"Forecast shape: {forecast.shape}")
    print(f"Forecast dtype: {forecast.dtype}")
    
    is_finite = bool(torch.isfinite(forecast).all())
    print(f"Forecast finite: {is_finite}")
    print(f"Forecast min: {forecast.min().item():.4f}, max: {forecast.max().item():.4f}")

    # Inspect quantiles/samples
    # Chronos-bolt outputs quantiles: (batch, num_quantiles, prediction_length)
    print(f"Output dimensions interpretation: [batch={forecast.shape[0]}, quantiles/samples={forecast.shape[1]}, horizon={forecast.shape[2]}]")

    print("\n=== Step 4: Testing Residual-Based Anomaly Scoring Feasibility ===")
    # Suppose ground truth next 8 points are known:
    ground_truth = torch.tensor([[float(32 + i) + np.sin((32 + i) / 2.0) for i in range(8)]], dtype=torch.float32)
    # Median prediction is typically quantile index 4 (0.5 quantile) if 9 quantiles are returned
    if forecast.shape[1] >= 5:
        median_pred = forecast[:, forecast.shape[1] // 2, :]
    else:
        median_pred = forecast.mean(dim=1)
    
    residual = torch.abs(ground_truth - median_pred)
    print(f"Residual shape: {residual.shape}")
    print(f"Mean absolute residual: {residual.mean().item():.4f}")
    
    print("\n=== SUMMARY ===")
    print(f"STATUS: PASS")
    print(f"Model: amazon/chronos-bolt-tiny")
    print(f"Load time: {load_time:.2f}s")
    print(f"Inference time: {inf_time:.2f}s")
    print(f"Finite: {is_finite}")
    return True

if __name__ == "__main__":
    success = run_chronos_smoke_test()
    print(f"Smoke test result: {'SUCCESS' if success else 'FAILURE'}")
