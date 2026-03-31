#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HARNESS_ROOT = ROOT / ".harness" / "foundry_hook"
EVAL_ROOT = HARNESS_ROOT / "dutch_auction_eval"
METRICS_PATH = HARNESS_ROOT / "metrics.json"

STRESS_MANIFEST_PATH = ROOT / "cache" / "backtest_manifest_2026-03-19_2p_stress.json"
REAL_WINDOW_ROOT_CANDIDATES = (
    ROOT / "cache" / "backtest_results_rerun_20260327_v3",
    ROOT / "cache" / "backtest_results_rerun_20260327_v2",
    ROOT / "cache" / "backtest_results_rerun_20260327",
    ROOT / "cache" / "backtest_results_20260326" / "batch",
    ROOT / "cache" / "backtest_batch_stress_v3",
    ROOT / "cache" / "backtest_batch_stress_v2",
)

BASE_FEE_BPS = 5.0
MAX_FEE_BPS = 500.0
ALPHA_BPS = 10_000.0
MAX_ORACLE_AGE_SECONDS = 3600
LATENCY_SECONDS = 60.0
LVR_BUDGET = 0.01
WIDTH_TICKS = 12_000
SOLVER_GAS_COST_QUOTE = 0.25
SOLVER_EDGE_BPS = 0.0


@dataclass(frozen=True)
class RealWindowSpec:
    window_id: str
    regime: str
    window_dir: Path
    series_path: Path
    swap_samples_path: Path
    oracle_updates_path: Path
    market_reference_path: Path


def main() -> None:
    metrics = evaluate()
    HARNESS_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"metrics": metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics}, indent=2, sort_keys=True))


def evaluate() -> dict[str, Any]:
    windows_root, windows = discover_real_windows()
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    window_reports: list[dict[str, Any]] = []
    objective = 0.0
    for window in windows:
        report = run_real_window(window)
        window_reports.append(report)
        objective += score_window(report)

    stress_reports = [report for report in window_reports if report["regime"] == "stress"]
    normal_reports = [report for report in window_reports if report["regime"] == "normal"]

    metrics = {
        "dutch_auction_objective": objective,
        "stress_go_pass_rate": mean([1.0 if report["stress_go_pass"] else 0.0 for report in stress_reports]),
        "normal_no_auction_pass_rate": mean(
            [1.0 if report["normal_window_pass"] else 0.0 for report in normal_reports]
        ),
        "stress_lp_uplift_quote_total": sum(report["lp_net_auction_vs_hook_quote"] for report in stress_reports),
        "stress_lp_improvement_ratio": mean([report["lp_improvement_ratio"] for report in stress_reports]),
        "stress_fill_rate": mean([report["fill_rate"] for report in stress_reports]),
        "stress_fallback_rate": mean([report["fallback_rate"] for report in stress_reports]),
        "stress_failclosed_rate": mean([report["oracle_failclosed_rate"] for report in stress_reports]),
        "normal_trigger_rate": mean([report["auction_trigger_rate"] for report in normal_reports]),
        "normal_no_reference_rate": mean([report["no_reference_rate"] for report in normal_reports]),
        "mean_clearing_concession_bps": mean(
            [report["mean_clearing_concession_bps"] for report in window_reports if report["triggered_count"] > 0]
        ),
        "mean_time_to_fill_seconds": mean(
            [report["mean_time_to_fill_seconds"] for report in window_reports if report["triggered_count"] > 0]
        ),
        "real_window_count": float(len(window_reports)),
        "evaluation_mode": "real_windows",
        "window_source_root": str(windows_root.relative_to(ROOT)),
        "window_ids": [report["window_id"] for report in window_reports],
        "windows": {report["window_id"]: report for report in window_reports},
    }
    report_path = EVAL_ROOT / "report.json"
    report_path.write_text(json.dumps({"windows": window_reports, "metrics": metrics}, indent=2, sort_keys=True) + "\n")
    return metrics


def discover_real_windows() -> tuple[Path, tuple[RealWindowSpec, ...]]:
    if not STRESS_MANIFEST_PATH.exists():
        raise RuntimeError(f"missing real-window manifest: {STRESS_MANIFEST_PATH}")

    payload = json.loads(STRESS_MANIFEST_PATH.read_text())
    windows_payload = payload.get("windows")
    if not isinstance(windows_payload, list) or not windows_payload:
        raise RuntimeError(f"manifest does not contain windows: {STRESS_MANIFEST_PATH}")

    expected_windows = []
    for entry in windows_payload:
        if not isinstance(entry, dict):
            continue
        window_id = str(entry.get("window_id") or "").strip()
        regime = str(entry.get("regime") or "").strip()
        if not window_id or regime not in {"normal", "stress"}:
            continue
        expected_windows.append((window_id, regime))

    for root in REAL_WINDOW_ROOT_CANDIDATES:
        specs: list[RealWindowSpec] = []
        for window_id, regime in expected_windows:
            window_dir = root / window_id
            series_path = window_dir / "exact_replay_series.csv"
            swap_samples_path = window_dir / "inputs" / "swap_samples.csv"
            oracle_updates_path = window_dir / "chainlink_reference_updates.csv"
            market_reference_path = window_dir / "inputs" / "market_reference_updates.csv"
            if not (
                window_dir.exists()
                and series_path.exists()
                and swap_samples_path.exists()
                and oracle_updates_path.exists()
                and market_reference_path.exists()
            ):
                specs = []
                break
            specs.append(
                RealWindowSpec(
                    window_id=window_id,
                    regime=regime,
                    window_dir=window_dir,
                    series_path=series_path,
                    swap_samples_path=swap_samples_path,
                    oracle_updates_path=oracle_updates_path,
                    market_reference_path=market_reference_path,
                )
            )
        if specs:
            return root, tuple(specs)

    searched = ", ".join(str(path.relative_to(ROOT)) for path in REAL_WINDOW_ROOT_CANDIDATES)
    raise RuntimeError(f"unable to find a complete real-window bundle; searched: {searched}")


def run_real_window(window: RealWindowSpec) -> dict[str, Any]:
    candidate_dir = EVAL_ROOT / window.window_id
    replay_dir = candidate_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    swaps_output_path = replay_dir / "dutch_auction_swaps.csv"
    summary_output_path = replay_dir / "dutch_auction_summary.json"

    run_candidate_backtest(
        series_path=window.series_path,
        swap_samples_path=window.swap_samples_path,
        oracle_updates_path=window.oracle_updates_path,
        market_reference_path=window.market_reference_path,
        swaps_output_path=swaps_output_path,
        summary_output_path=summary_output_path,
    )

    summary = json.loads(summary_output_path.read_text())
    with swaps_output_path.open(encoding="utf-8") as handle:
        swap_rows = list(csv.DictReader(handle))

    swap_count = float(len(swap_rows))
    trigger_count = float(sum(csv_bool(row.get("auction_triggered")) for row in swap_rows))
    filled_count = float(sum(csv_bool(row.get("filled")) for row in swap_rows))
    delta_vs_hook_quote = safe_float(summary.get("lp_net_auction_vs_hook_quote"))
    hook_abs = abs(safe_float(summary.get("lp_net_hook_quote")))
    delta_pct_vs_hook = delta_vs_hook_quote / max(hook_abs, 1.0)
    stress_go_pass = (
        safe_float(summary.get("fill_rate")) > 0.80
        and safe_float(summary.get("fallback_rate")) < 0.10
        and safe_float(summary.get("oracle_failclosed_rate")) < 0.10
        and delta_vs_hook_quote > 0.0
        and delta_pct_vs_hook > 0.01
        and safe_float(summary.get("mean_clearing_concession_bps")) < 5_000.0
    )
    normal_window_pass = (
        safe_float(summary.get("auction_trigger_rate")) <= 0.05
        and delta_pct_vs_hook >= 0.0
        and safe_float(summary.get("fallback_rate")) <= 0.05
        and safe_float(summary.get("oracle_failclosed_rate")) <= 0.05
    )

    return {
        "window_id": window.window_id,
        "regime": window.regime,
        "swap_count": swap_count,
        "triggered_count": trigger_count,
        "filled_count": filled_count,
        "auction_trigger_rate": safe_float(summary.get("auction_trigger_rate")),
        "fill_rate": safe_float(summary.get("fill_rate")),
        "fallback_rate": safe_float(summary.get("fallback_rate")),
        "oracle_failclosed_rate": safe_float(summary.get("oracle_failclosed_rate")),
        "no_reference_rate": safe_float(summary.get("no_reference_rate")),
        "mean_clearing_concession_bps": safe_float(summary.get("mean_clearing_concession_bps")),
        "mean_time_to_fill_seconds": safe_float(summary.get("mean_time_to_fill_seconds")),
        "lp_net_auction_quote": safe_float(summary.get("lp_net_auction_quote")),
        "lp_net_hook_quote": safe_float(summary.get("lp_net_hook_quote")),
        "lp_net_fixed_fee_quote": safe_float(summary.get("lp_net_fixed_fee_quote")),
        "lp_net_auction_vs_hook_quote": delta_vs_hook_quote,
        "lp_net_auction_vs_fixed_fee_quote": safe_float(summary.get("lp_net_auction_vs_fixed_fee_quote")),
        "lp_improvement_ratio": delta_pct_vs_hook,
        "stress_go_pass": stress_go_pass,
        "normal_window_pass": normal_window_pass,
        "summary_path": str(summary_output_path.relative_to(ROOT)),
        "swaps_path": str(swaps_output_path.relative_to(ROOT)),
        "source_window_dir": str(window.window_dir.relative_to(ROOT)),
    }


def run_candidate_backtest(
    *,
    series_path: Path,
    swap_samples_path: Path,
    oracle_updates_path: Path,
    market_reference_path: Path,
    swaps_output_path: Path,
    summary_output_path: Path,
) -> None:
    command = [
        "python3",
        "-m",
        "script.run_dutch_auction_backtest",
        "--series-csv",
        str(series_path),
        "--swap-samples",
        str(swap_samples_path),
        "--oracle-updates",
        str(oracle_updates_path),
        "--market-reference-updates",
        str(market_reference_path),
        "--label-config",
        "script/label_config.json",
        "--output",
        str(swaps_output_path),
        "--summary-output",
        str(summary_output_path),
        "--base-fee-bps",
        str(BASE_FEE_BPS),
        "--max-fee-bps",
        str(MAX_FEE_BPS),
        "--alpha-bps",
        str(ALPHA_BPS),
        "--max-oracle-age-seconds",
        str(MAX_ORACLE_AGE_SECONDS),
        "--latency-seconds",
        str(LATENCY_SECONDS),
        "--lvr-budget",
        str(LVR_BUDGET),
        "--width-ticks",
        str(WIDTH_TICKS),
        "--solver-gas-cost-quote",
        str(SOLVER_GAS_COST_QUOTE),
        "--solver-edge-bps",
        str(SOLVER_EDGE_BPS),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"window benchmark failed for {summary_output_path.parent.parent.name} with returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def score_window(report: dict[str, Any]) -> float:
    delta_penalty = max(-report["lp_improvement_ratio"], 0.0)
    if report["regime"] == "normal":
        trigger_excess_penalty = max(report["auction_trigger_rate"] - 0.05, 0.0)
        return (
            delta_penalty * 4_000.0
            + trigger_excess_penalty * 1_000.0
            + report["no_reference_rate"] * 250.0
            + report["fallback_rate"] * 750.0
            + report["oracle_failclosed_rate"] * 1_250.0
            + report["mean_clearing_concession_bps"] / 100.0
            + report["mean_time_to_fill_seconds"] * 5.0
        )

    improvement_shortfall = max(0.01 - report["lp_improvement_ratio"], 0.0)
    fill_shortfall = max(0.80 - report["fill_rate"], 0.0)
    return (
        improvement_shortfall * 8_000.0
        + fill_shortfall * 1_000.0
        + report["fallback_rate"] * 2_000.0
        + report["oracle_failclosed_rate"] * 3_000.0
        + report["no_reference_rate"] * 500.0
        + report["mean_clearing_concession_bps"] / 100.0
        + report["mean_time_to_fill_seconds"] * 25.0
    )


def csv_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
