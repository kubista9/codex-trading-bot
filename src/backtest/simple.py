from __future__ import annotations

import numpy as np
import pandas as pd


def run_signal_backtest(
    signals: pd.DataFrame,
    *,
    actual_return_col: str = "actual",
    signal_col: str = "signal",
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Run a transparent equal-weight signal backtest from prediction rows."""
    required = {"as_of_date", "model_name", "horizon", actual_return_col, signal_col}
    missing = required.difference(signals.columns)
    if missing:
        msg = f"Signals missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    frame = signals.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.normalize()
    frame["position"] = frame[signal_col].map({"buy": 1.0, "hold": 0.0, "sell": -1.0}).fillna(0.0)
    sort_cols = ["model_name", "horizon", "ticker", "as_of_date"] if "ticker" in frame.columns else [
        "model_name",
        "horizon",
        "as_of_date",
    ]
    group_cols = ["model_name", "horizon", "ticker"] if "ticker" in frame.columns else [
        "model_name",
        "horizon",
    ]
    frame = frame.sort_values(sort_cols)
    frame["turnover"] = frame.groupby(group_cols, sort=False)["position"].diff().abs().fillna(
        frame["position"].abs()
    )
    horizon = frame["horizon"].astype(float).clip(lower=1.0)
    actual_return = frame[actual_return_col].astype(float)
    daily_equivalent_return = np.where(
        actual_return > -1.0,
        np.power(1.0 + actual_return, 1.0 / horizon) - 1.0,
        -1.0,
    )
    frame["strategy_return"] = frame["position"] * daily_equivalent_return
    frame["strategy_return"] -= frame["turnover"] * (transaction_cost_bps / 10_000)

    records: list[dict[str, float | int | str]] = []
    for (model_name, horizon), group in frame.groupby(["model_name", "horizon"], sort=False):
        portfolio_returns = group.groupby("as_of_date")["strategy_return"].mean().sort_index()
        if portfolio_returns.empty:
            continue
        equity = (1 + portfolio_returns).cumprod()
        drawdown = equity / equity.cummax() - 1
        std = portfolio_returns.std(ddof=0)
        sharpe_like = 0.0 if std == 0 or np.isnan(std) else float(portfolio_returns.mean() / std * np.sqrt(252))
        records.append(
            {
                "model_name": model_name,
                "horizon": int(horizon),
                "cumulative_return": float(equity.iloc[-1] - 1),
                "sharpe_like": sharpe_like,
                "max_drawdown": float(drawdown.min()),
                "win_rate": float((portfolio_returns > 0).mean()),
                "turnover": float(group["turnover"].mean()),
                "average_position": float(group["position"].abs().mean()),
                "n_periods": int(len(portfolio_returns)),
                "transaction_cost_bps": float(transaction_cost_bps),
            }
        )
    return pd.DataFrame.from_records(records)
