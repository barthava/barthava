import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Exact settings copied from SP2L2_Advanced_Backtest.ipynb
BROKER_POINT = 0.01
SPIKE_CANDLE_SIZE = 1.5
PGAP_POINTS = 100
P_GAP_PRICE = PGAP_POINTS * BROKER_POINT
MAX_SL_DISTANCE_POINTS = 1000
MAX_SL_DISTANCE_PRICE = MAX_SL_DISTANCE_POINTS * BROKER_POINT
INITIAL_CASH = 10000.0
RISK_PER_TRADE = 100.0
TP_R = 1.0
USE_SECOND_ENTRY = False
USE_EMA_FILTER = True
EMA_PERIOD = 60
USE_TREND_FILTER = True
MAX_OPPOSITE_MOVES = 1
USE_RANGE_FILTER = False
USE_SESSION_FILTER = False


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [str(c).lower().strip() for c in df.columns]
    time_col = next((c for c in ["datetime", "time", "date", "local time"] if c in df.columns), None)
    if time_col is None:
        raise RuntimeError(f"No time column in {path}: {list(df.columns)}")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    return df


def run_sp2l(df, timeframe):
    original_rows = len(df)
    original_start = str(df.index.min())
    original_end = str(df.index.max())

    # Columns and EMA as in the notebook.
    df = df.copy()
    df["position"] = 0
    df["sl_buy"] = np.nan
    df["sl_sell"] = np.nan
    df["entry_buy"] = np.nan
    df["entry_sell"] = np.nan
    df["entry_buy_2x"] = np.nan
    df["entry_sell_2x"] = np.nan
    df["spike_index"] = pd.Series(index=df.index, dtype="object")
    df["spike_body"] = np.nan
    df["sl_distance"] = np.nan
    df["EMA"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["ADX"] = np.nan

    for lag in [1, 2, 3]:
        for col in ["open", "high", "low", "close"]:
            df[f"{col}_{lag}"] = df[col].shift(lag)

    temp = [f"{c}_{lag}" for lag in [1, 2, 3] for c in ["open", "high", "low", "close"]]
    df = df.dropna(subset=temp).copy()

    buy_setup = (
        (df["close_1"] > df["close_2"])
        & (df["open_1"] > df["open_2"])
        & (df["close_2"] > df["close_3"])
        & (df["open_2"] > df["open_3"])
        & (df["close_1"] > df["open_1"])
        & (df["close_2"] > df["open_2"])
        & (df["close_3"] > df["open_3"])
        & (df["low_1"] > df["high_3"] + P_GAP_PRICE)
    )
    buy_spike_body = df["close_2"] - df["open_2"]
    buy_before_body = df["close_3"] - df["open_3"]
    buy_after_body = df["close_1"] - df["open_1"]
    buy_spike = ((buy_spike_body > SPIKE_CANDLE_SIZE * buy_before_body)
                 & (buy_spike_body > SPIKE_CANDLE_SIZE * buy_after_body))
    buy_setup = buy_setup & buy_spike

    sell_setup = (
        (df["close_1"] < df["close_2"])
        & (df["open_1"] < df["open_2"])
        & (df["close_2"] < df["close_3"])
        & (df["open_2"] < df["open_3"])
        & (df["close_1"] < df["open_1"])
        & (df["close_2"] < df["open_2"])
        & (df["close_3"] < df["open_3"])
        & (df["high_1"] < df["low_3"] - P_GAP_PRICE)
    )
    sell_spike_body = df["open_2"] - df["close_2"]
    sell_before_body = df["open_3"] - df["close_3"]
    sell_after_body = df["open_1"] - df["close_1"]
    sell_spike = ((sell_spike_body > SPIKE_CANDLE_SIZE * sell_before_body)
                  & (sell_spike_body > SPIKE_CANDLE_SIZE * sell_after_body))
    sell_setup = sell_setup & sell_spike

    buy_setup_idx = df.index[buy_setup]
    sell_setup_idx = df.index[sell_setup]
    index_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

    # Arrays make the equivalent entry scan much faster on six months of M1.
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    emas = df["EMA"].to_numpy(dtype=float)

    def find_first_buy_entry(start_pos, sl):
        consecutive_opposite = 0
        for entry_pos in range(start_pos + 1, len(df)):
            current_high = highs[entry_pos]
            previous_high = highs[entry_pos - 1]
            if USE_TREND_FILTER:
                if current_high > previous_high:
                    consecutive_opposite = 0
                else:
                    consecutive_opposite += 1
                    # Once violated, all later entries would also fail the notebook's
                    # buy_trend_is_valid(start_pos, entry_pos), so this is exact.
                    if consecutive_opposite > MAX_OPPOSITE_MOVES:
                        return None

            current_low = lows[entry_pos]
            previous_low = lows[entry_pos - 1]
            if current_low < previous_low:
                risk = current_low - sl
                if risk <= 0:
                    return None
                if risk > MAX_SL_DISTANCE_PRICE:
                    return None
                if USE_EMA_FILTER and closes[entry_pos] <= emas[entry_pos]:
                    # Notebook continues searching when filters reject a candidate.
                    continue
                return entry_pos, current_low, risk
        return None

    def find_first_sell_entry(start_pos, sl):
        consecutive_opposite = 0
        for entry_pos in range(start_pos + 1, len(df)):
            current_low = lows[entry_pos]
            previous_low = lows[entry_pos - 1]
            if USE_TREND_FILTER:
                if current_low < previous_low:
                    consecutive_opposite = 0
                else:
                    consecutive_opposite += 1
                    if consecutive_opposite > MAX_OPPOSITE_MOVES:
                        return None

            current_high = highs[entry_pos]
            previous_high = highs[entry_pos - 1]
            if current_high > previous_high:
                risk = sl - current_high
                if risk <= 0:
                    return None
                if risk > MAX_SL_DISTANCE_PRICE:
                    return None
                if USE_EMA_FILTER and closes[entry_pos] >= emas[entry_pos]:
                    continue
                return entry_pos, current_high, risk
        return None

    detected_buy = []
    detected_sell = []

    for c_idx in buy_setup_idx:
        c_pos = index_to_pos[c_idx]
        sl = float(lows[c_pos - 2])
        result = find_first_buy_entry(c_pos, sl)
        if result is None:
            continue
        entry_pos, entry, risk = result
        detected_buy.append((df.index[entry_pos], entry, sl, risk, c_idx, c_pos - 1))

    for c_idx in sell_setup_idx:
        c_pos = index_to_pos[c_idx]
        sl = float(highs[c_pos - 2])
        result = find_first_sell_entry(c_pos, sl)
        if result is None:
            continue
        entry_pos, entry, risk = result
        detected_sell.append((df.index[entry_pos], entry, sl, risk, c_idx, c_pos - 1))

    # Notebook writes BUY first; a SELL colliding on same timestamp is skipped.
    for idx, entry, sl, risk, c_idx, spike_pos in detected_buy:
        if df.loc[idx, "position"] != 0:
            continue
        df.loc[idx, "position"] = 1
        df.loc[idx, "entry_buy"] = entry
        df.loc[idx, "sl_buy"] = sl
        df.loc[idx, "sl_distance"] = risk
        df.loc[idx, "spike_body"] = abs(float(df.iloc[spike_pos]["close"]) - float(df.iloc[spike_pos]["open"]))
        df.loc[idx, "spike_index"] = df.index[spike_pos]
        df.loc[idx, "entry_buy_2x"] = entry - risk / 2

    for idx, entry, sl, risk, c_idx, spike_pos in detected_sell:
        if df.loc[idx, "position"] != 0:
            continue
        df.loc[idx, "position"] = -1
        df.loc[idx, "entry_sell"] = entry
        df.loc[idx, "sl_sell"] = sl
        df.loc[idx, "sl_distance"] = risk
        df.loc[idx, "spike_body"] = abs(float(df.iloc[spike_pos]["open"]) - float(df.iloc[spike_pos]["close"]))
        df.loc[idx, "spike_index"] = df.index[spike_pos]
        df.loc[idx, "entry_sell_2x"] = entry + risk / 2

    buy_signal_count = int((df["position"] == 1).sum())
    sell_signal_count = int((df["position"] == -1).sum())
    signal_count = buy_signal_count + sell_signal_count

    trades = []
    equity = INITIAL_CASH
    active_setup = None

    def close_leg(leg, exit_price, exit_time, exit_reason):
        nonlocal equity
        if leg["direction"] == "BUY":
            price_r = (exit_price - leg["entry"]) / leg["risk"]
        else:
            price_r = (leg["entry"] - exit_price) / leg["risk"]
        r_multiple = price_r * leg["volume_multiplier"] * (leg["risk"] / leg["base_risk"])
        pnl = r_multiple * RISK_PER_TRADE
        equity += pnl
        trades.append({
            "setup_id": leg["setup_id"],
            "entry_number": leg["entry_number"],
            "entry_time": leg["entry_time"],
            "activation_time": leg["activation_time"],
            "exit_time": exit_time,
            "direction": leg["direction"],
            "entry": leg["entry"],
            "sl": leg["sl"],
            "tp": leg["tp"],
            "risk": leg["risk"],
            "base_risk": leg["base_risk"],
            "volume_multiplier": leg["volume_multiplier"],
            "exit": exit_price,
            "R": r_multiple,
            "PnL": pnl,
            "exit_reason": exit_reason,
            "spike_body": leg["spike_body"],
            "sl_points": leg["base_risk"] / BROKER_POINT,
        })

    for i in range(len(df)):
        idx = df.index[i]
        row = df.iloc[i]
        candle_low = float(row["low"])
        candle_high = float(row["high"])

        if active_setup is not None:
            direction = active_setup["direction"]
            sl = active_setup["sl"]
            tp = active_setup["tp"]
            exit_price = None
            exit_reason = None

            if direction == "BUY":
                if candle_low <= sl and candle_high >= tp:
                    exit_price, exit_reason = sl, "SL_and_TP_same_bar_SL_first"
                elif candle_low <= sl:
                    exit_price, exit_reason = sl, "SL"
                elif candle_high >= tp:
                    exit_price, exit_reason = tp, "TP"
            else:
                if candle_high >= sl and candle_low <= tp:
                    exit_price, exit_reason = sl, "SL_and_TP_same_bar_SL_first"
                elif candle_high >= sl:
                    exit_price, exit_reason = sl, "SL"
                elif candle_low <= tp:
                    exit_price, exit_reason = tp, "TP"

            if exit_price is not None:
                for leg in active_setup["legs"]:
                    close_leg(leg, exit_price, idx, exit_reason)
                active_setup = None
                continue

            # USE_SECOND_ENTRY is false in the source notebook, so omitted here.
            continue

        signal = int(row["position"])
        if signal == 1:
            entry = float(row["entry_buy"])
            sl = float(row["sl_buy"])
            base_risk = entry - sl
            if base_risk <= 0 or base_risk > MAX_SL_DISTANCE_PRICE:
                continue
            tp = entry + TP_R * base_risk
            active_setup = {
                "setup_id": i,
                "signal_time": idx,
                "direction": "BUY",
                "base_risk": base_risk,
                "sl": sl,
                "tp": tp,
                "spike_body": float(row["spike_body"]),
                "legs": [{
                    "setup_id": i, "entry_number": 1, "entry_time": idx,
                    "activation_time": idx, "direction": "BUY", "entry": entry,
                    "sl": sl, "tp": tp, "risk": base_risk, "base_risk": base_risk,
                    "volume_multiplier": 1.0, "spike_body": float(row["spike_body"]),
                }],
            }
        elif signal == -1:
            entry = float(row["entry_sell"])
            sl = float(row["sl_sell"])
            base_risk = sl - entry
            if base_risk <= 0 or base_risk > MAX_SL_DISTANCE_PRICE:
                continue
            tp = entry - TP_R * base_risk
            active_setup = {
                "setup_id": i,
                "signal_time": idx,
                "direction": "SELL",
                "base_risk": base_risk,
                "sl": sl,
                "tp": tp,
                "spike_body": float(row["spike_body"]),
                "legs": [{
                    "setup_id": i, "entry_number": 1, "entry_time": idx,
                    "activation_time": idx, "direction": "SELL", "entry": entry,
                    "sl": sl, "tp": tp, "risk": base_risk, "base_risk": base_risk,
                    "volume_multiplier": 1.0, "spike_body": float(row["spike_body"]),
                }],
            }

    if active_setup is not None:
        last_idx = df.index[-1]
        last_close = float(df.iloc[-1]["close"])
        for leg in active_setup["legs"]:
            close_leg(leg, last_close, last_idx, "END_OF_DATA")

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        summary = {"timeframe": timeframe, "candles": original_rows, "start": original_start, "end": original_end,
                   "raw_buy_setups": int(buy_setup.sum()), "raw_sell_setups": int(sell_setup.sum()),
                   "buy_signals": buy_signal_count, "sell_signals": sell_signal_count, "signals": signal_count,
                   "trades": 0}
        return summary, trades_df

    wins = trades_df["R"] > 0
    losses = trades_df["R"] <= 0
    total_trades = len(trades_df)
    win_count = int(wins.sum())
    loss_count = int(losses.sum())
    total_R = float(trades_df["R"].sum())
    total_pnl = float(trades_df["PnL"].sum())
    gross_profit = float(trades_df.loc[trades_df["PnL"] > 0, "PnL"].sum())
    gross_loss = abs(float(trades_df.loc[trades_df["PnL"] < 0, "PnL"].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf

    cumulative_equity = INITIAL_CASH + trades_df["PnL"].cumsum()
    equity_peak = cumulative_equity.cummax()
    dd = cumulative_equity - equity_peak
    dd_pct = (cumulative_equity / equity_peak - 1.0) * 100.0

    buy_trades = trades_df[trades_df["direction"] == "BUY"]
    sell_trades = trades_df[trades_df["direction"] == "SELL"]

    # Consecutive outcomes by executed trades.
    max_consec_win = max_consec_loss = cw = cl = 0
    for r in trades_df["R"]:
        if r > 0:
            cw += 1; cl = 0; max_consec_win = max(max_consec_win, cw)
        else:
            cl += 1; cw = 0; max_consec_loss = max(max_consec_loss, cl)

    trades_df["month"] = pd.to_datetime(trades_df["entry_time"]).dt.to_period("M").astype(str)
    monthly = []
    for month, g in trades_df.groupby("month"):
        monthly.append({
            "month": month,
            "trades": int(len(g)),
            "wins": int((g["R"] > 0).sum()),
            "losses": int((g["R"] <= 0).sum()),
            "win_rate": round(float((g["R"] > 0).mean() * 100), 4),
            "R": round(float(g["R"].sum()), 6),
            "PnL_repo_10k": round(float(g["PnL"].sum()), 2),
        })

    exit_reasons = {str(k): int(v) for k, v in trades_df["exit_reason"].value_counts().to_dict().items()}
    duration_min = (pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(trades_df["entry_time"])).dt.total_seconds() / 60.0

    summary = {
        "timeframe": timeframe,
        "candles": original_rows,
        "start": original_start,
        "end": original_end,
        "raw_buy_setups": int(buy_setup.sum()),
        "raw_sell_setups": int(sell_setup.sum()),
        "raw_setups_total": int(buy_setup.sum() + sell_setup.sum()),
        "buy_signals": buy_signal_count,
        "sell_signals": sell_signal_count,
        "signals": signal_count,
        "trades": int(total_trades),
        "wins": win_count,
        "losses": loss_count,
        "win_rate_pct": round(win_count / total_trades * 100.0, 4),
        "total_R": round(total_R, 6),
        "profit_factor": round(profit_factor, 6) if math.isfinite(profit_factor) else "inf",
        "gross_profit_repo": round(gross_profit, 2),
        "gross_loss_repo": round(gross_loss, 2),
        "pnl_repo_10k_risk100": round(total_pnl, 2),
        "final_cash_repo": round(INITIAL_CASH + total_pnl, 2),
        "return_repo_pct": round(total_pnl / INITIAL_CASH * 100.0, 4),
        "pnl_normalized_1k_risk10": round(total_R * 10.0, 2),
        "final_cash_normalized_1k": round(1000.0 + total_R * 10.0, 2),
        "return_normalized_1k_pct": round(total_R, 4),
        "max_drawdown_repo_usd": round(float(dd.min()), 2),
        "max_drawdown_repo_pct": round(float(dd_pct.min()), 4),
        "max_drawdown_normalized_1k_usd_fixed10": round(float(dd.min()) / 10.0, 2),
        "max_consecutive_wins": int(max_consec_win),
        "max_consecutive_losses": int(max_consec_loss),
        "buy_trades": int(len(buy_trades)),
        "buy_wins": int((buy_trades["R"] > 0).sum()),
        "buy_win_rate_pct": round(float((buy_trades["R"] > 0).mean() * 100), 4) if len(buy_trades) else 0.0,
        "buy_R": round(float(buy_trades["R"].sum()), 6),
        "sell_trades": int(len(sell_trades)),
        "sell_wins": int((sell_trades["R"] > 0).sum()),
        "sell_win_rate_pct": round(float((sell_trades["R"] > 0).mean() * 100), 4) if len(sell_trades) else 0.0,
        "sell_R": round(float(sell_trades["R"].sum()), 6),
        "avg_trade_duration_min": round(float(duration_min.mean()), 2),
        "median_trade_duration_min": round(float(duration_min.median()), 2),
        "exit_reasons": exit_reasons,
        "monthly": monthly,
    }
    return summary, trades_df


def main():
    sources = {
        "M1": "XAUUSD_1m.csv",
        "M5": "XAUUSD_5m.csv",
    }
    all_summary = {}
    for tf, path in sources.items():
        df = load_csv(path)
        summary, trades = run_sp2l(df, tf)
        all_summary[tf] = summary
        trades.to_csv(f"SP2L_{tf}_trades.csv", index=False)
        print(f"\n===== {tf} RESULT =====")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    payload = {
        "strategy": "SP2L Advanced v2.0 repository settings",
        "source_settings": {
            "point": BROKER_POINT,
            "spike_multiplier": SPIKE_CANDLE_SIZE,
            "p_gap_points": PGAP_POINTS,
            "p_gap_price": P_GAP_PRICE,
            "max_sl_points": MAX_SL_DISTANCE_POINTS,
            "max_sl_price": MAX_SL_DISTANCE_PRICE,
            "ema_filter": USE_EMA_FILTER,
            "ema_period": EMA_PERIOD,
            "trend_filter": USE_TREND_FILTER,
            "max_opposite_moves": MAX_OPPOSITE_MOVES,
            "range_filter": USE_RANGE_FILTER,
            "session_filter": USE_SESSION_FILTER,
            "tp_R": TP_R,
            "second_entry": USE_SECOND_ENTRY,
            "initial_cash": INITIAL_CASH,
            "fixed_risk_per_trade": RISK_PER_TRADE,
            "same_bar_policy": "SL first when SL and TP both hit",
            "overlap_policy": "one active setup/trade at a time; later signals ignored while active",
            "costs": "no spread, commission, or slippage (same as notebook OHLC backtest)",
        },
        "results": all_summary,
    }
    Path("SP2L_6M_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== FINAL_JSON =====")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
