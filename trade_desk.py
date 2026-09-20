#!/usr/bin/env python3
"""
trade_desk.py — backtester, portfolio risk check and trading-journal analyser.

Data: Hyperliquid public candles (no API key), falls back to Binance spot.
Requires: python 3.9+, pandas, numpy, requests   ->  pip install pandas numpy requests

USAGE
  Backtest (Step 4):
    python trade_desk.py backtest --coins BTC ETH SOL XRP ZEC HYPE --tf 1h 4h
  Portfolio risk (Step 5):  CSV columns: asset,value_usd   (stablecoins: USDC/USDT)
    python trade_desk.py portfolio --file portfolio.csv
  Journal review (Step 6):  CSV columns:
    open_time,close_time,asset,side,entry,exit,stop,size_usd,setup,notes
    (times ISO e.g. 2026-09-15 21:40, in AWST; side = long/short)
    python trade_desk.py journal --file trades.csv
  Daily picks + scorecard (runs every morning via launchd/cron):
    python trade_desk.py daily --coins BTC ETH SOL XRP ZEC HYPE --out ~/TradeDesk
    Set ANTHROPIC_API_KEY for written reasoning on each pick (optional).
  Offline self-test with synthetic prices (numbers are NOT market results):
    python trade_desk.py backtest --demo

Outputs are written to ./desk_output/ as Markdown + CSV.
Educational tool only. Past performance does not predict future results.
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

OUT = "desk_output"
FEE = 0.00045      # Hyperliquid base taker, per side
SLIP = 0.0002      # slippage per side
RISK = 0.01        # 1% equity risked per trade for drawdown maths
STABLES = {"USDC", "USDT", "DAI", "USD", "AUD", "CASH", "USDE"}
AWST = timezone(timedelta(hours=8))


# ------------------------------------------------------------------ data
def fetch_hl(coin, interval, bars):
    ms = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[interval]
    end = int(time.time() * 1000)
    start = end - ms * min(bars, 5000)
    r = requests.post("https://api.hyperliquid.xyz/info", timeout=20, json={
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": interval, "startTime": start, "endTime": end}})
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError("no candles")
    df = pd.DataFrame(rows)
    df = df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float})


def fetch_binance(coin, interval, bars):
    r = requests.get("https://api.binance.com/api/v3/klines", timeout=20,
                     params={"symbol": f"{coin}USDT", "interval": interval, "limit": min(bars, 1000)})
    r.raise_for_status()
    df = pd.DataFrame(r.json()).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.astype({c: float for c in ["open", "high", "low", "close", "volume"]})


KRAKEN = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "XRP": "XRPUSD", "ZEC": "ZECUSD",
          "DOGE": "XDGUSD", "ADA": "ADAUSD", "LINK": "LINKUSD", "AVAX": "AVAXUSD", "SUI": "SUIUSD"}


def fetch_kraken(coin, interval, bars):
    """Third fallback: works from US cloud runners where Binance is blocked. No HYPE."""
    pair = KRAKEN.get(coin)
    if not pair:
        raise ValueError("not on Kraken")
    mins = {"1h": 60, "4h": 240, "1d": 1440}[interval]
    r = requests.get("https://api.kraken.com/0/public/OHLC", timeout=20, params={"pair": pair, "interval": mins})
    r.raise_for_status()
    res = r.json()["result"]
    rows = next(v for k, v in res.items() if k != "last")
    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 6]]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    return df.astype({c: float for c in ["open", "high", "low", "close", "volume"]}).tail(bars).reset_index(drop=True)


def synthetic(coin, interval, bars, seed=None):
    rng = np.random.default_rng(seed if seed is not None else abs(hash(coin + interval)) % 2**32)
    vol = {"1h": 0.008, "4h": 0.016, "1d": 0.035}[interval]
    drift = rng.normal(0, vol / 20)
    rets = rng.normal(drift, vol, bars)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, vol / 2, bars)))
    low = close * (1 - np.abs(rng.normal(0, vol / 2, bars)))
    open_ = np.r_[close[0], close[:-1]]
    t = pd.date_range(end=pd.Timestamp.now("UTC").floor("h"), periods=bars,
                      freq={"1h": "h", "4h": "4h", "1d": "D"}[interval])
    return pd.DataFrame({"time": t, "open": open_, "high": np.maximum(high, open_),
                         "low": np.minimum(low, open_), "close": close, "volume": 1.0})


def get(coin, interval, bars, demo=False):
    if demo or requests is None:
        return synthetic(coin, interval, bars), "synthetic"
    for fn, name in ((fetch_hl, "hyperliquid"), (fetch_binance, "binance"), (fetch_kraken, "kraken")):
        try:
            return fn(coin, interval, bars), name
        except Exception as e:  # noqa
            last = e
    raise RuntimeError(f"{coin} {interval}: no data ({last})")


# ------------------------------------------------------------- indicators
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ----------------------------------------------------------------- signals
def sig_ema(df, fast=20, slow=50):
    f, s = ema(df["close"], fast), ema(df["close"], slow)
    up = (f > s) & (f.shift() <= s.shift())
    dn = (f < s) & (f.shift() >= s.shift())
    return np.where(up, 1, np.where(dn, -1, 0))


def sig_rsi_div(df, look=30, piv=3):
    r = rsi(df["close"])
    lo, hi, cl = df["low"].values, df["high"].values, r.values
    out = np.zeros(len(df), dtype=int)
    lows, highs = [], []
    for i in range(piv * 2, len(df)):
        c = i - piv  # confirmed pivot index (no look-ahead: known at bar i)
        w = slice(c - piv, c + piv + 1)
        if lo[c] == lo[w].min():
            prev = [p for p in lows if c - p <= look]
            if prev and lo[c] < lo[prev[-1]] and cl[c] > cl[prev[-1]] and cl[c] < 40:
                out[i] = 1
            lows.append(c)
        if hi[c] == hi[w].max():
            prev = [p for p in highs if c - p <= look]
            if prev and hi[c] > hi[prev[-1]] and cl[c] < cl[prev[-1]] and cl[c] > 60:
                out[i] = -1
            highs.append(c)
    return out


# ---------------------------------------------------------------- engine
def run(df, signals, trend=False, atr_mult=None, rr=None, time_stop=None, reverse=True):
    """Enter at next bar open. Exits: opposite signal (if reverse), ATR stop,
    R-multiple target, time stop. Returns list of trade dicts."""
    a = atr(df).values
    t200 = ema(df["close"], 200).values
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    trades, pos = [], None
    for i in range(201, len(df) - 1):
        s = signals[i]
        if pos:
            exit_px = None
            if pos["stop"] is not None:
                if pos["side"] == 1 and l[i] <= pos["stop"]:
                    exit_px = min(o[i], pos["stop"])
                elif pos["side"] == -1 and h[i] >= pos["stop"]:
                    exit_px = max(o[i], pos["stop"])
            if exit_px is None and pos["tp"] is not None:
                if pos["side"] == 1 and h[i] >= pos["tp"]:
                    exit_px = max(o[i], pos["tp"])
                elif pos["side"] == -1 and l[i] <= pos["tp"]:
                    exit_px = min(o[i], pos["tp"])
            if exit_px is None and time_stop and i - pos["i"] >= time_stop:
                exit_px = c[i]
            if exit_px is None and reverse and s == -pos["side"]:
                exit_px = o[i + 1]
            if exit_px is not None:
                g = pos["side"] * (exit_px / pos["px"] - 1) - 2 * (FEE + SLIP)
                risk = pos["riskpct"]
                trades.append({"entry_time": df["time"].iloc[pos["i"]], "exit_time": df["time"].iloc[i],
                               "side": "long" if pos["side"] == 1 else "short",
                               "ret": g, "R": g / risk if risk else np.nan, "bars": i - pos["i"]})
                pos = None
        if pos is None and s != 0:
            if trend and ((s == 1 and c[i] < t200[i]) or (s == -1 and c[i] > t200[i])):
                continue
            px = o[i + 1]
            stop = px - s * atr_mult * a[i] if atr_mult else None
            riskpct = abs(px - stop) / px if stop else 2 * a[i] / px  # nominal risk if no stop
            tp = px + s * rr * abs(px - stop) if (rr and stop) else None
            pos = {"side": s, "px": px, "i": i + 1, "stop": stop, "tp": tp, "riskpct": riskpct}
    return trades


def stats(trades):
    if not trades:
        return {"trades": 0}
    t = pd.DataFrame(trades)
    wins, losses = t[t.ret > 0], t[t.ret <= 0]
    pf = wins.ret.sum() / abs(losses.ret.sum()) if len(losses) and losses.ret.sum() != 0 else np.inf
    eq = (1 + RISK * t["R"].clip(-5, 20)).cumprod()  # fixed-fractional 1% risk
    dd = (eq / eq.cummax() - 1).min()
    return {"trades": len(t), "win_rate": len(wins) / len(t), "profit_factor": pf,
            "avg_R": t["R"].mean(), "net_return_1pct_risk": eq.iloc[-1] - 1,
            "max_drawdown": dd, "avg_bars": t["bars"].mean(),
            "long_win_rate": (t[t.side == "long"].ret > 0).mean() if (t.side == "long").any() else np.nan,
            "short_win_rate": (t[t.side == "short"].ret > 0).mean() if (t.side == "short").any() else np.nan}


VARIANTS = {
    "ema_cross": [
        ("base 20/50 flip", dict()),
        ("+ 200 EMA trend filter", dict(trend=True)),
        ("+ 2xATR stop, 2R target", dict(atr_mult=2, rr=2, reverse=False)),
        ("+ trend + ATR stop/2R", dict(trend=True, atr_mult=2, rr=2, reverse=False)),
    ],
    "rsi_divergence": [
        ("base 1.5xATR stop, 2R", dict(atr_mult=1.5, rr=2, reverse=False, time_stop=24)),
        ("+ 200 EMA trend filter", dict(trend=True, atr_mult=1.5, rr=2, reverse=False, time_stop=24)),
        ("+ wider 2.5xATR, 1.5R", dict(atr_mult=2.5, rr=1.5, reverse=False, time_stop=36)),
    ],
}


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    return f"{v*100:.1f}%" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))


def cmd_backtest(a):
    os.makedirs(OUT, exist_ok=True)
    rows, src_note = [], set()
    for coin in a.coins:
        for tf in a.tf:
            try:
                df, src = get(coin, tf, a.bars, a.demo)
            except Exception as e:
                print(f"skip {coin} {tf}: {e}")
                continue
            src_note.add(src)
            span = f"{df.time.iloc[0]:%d %b %Y} to {df.time.iloc[-1]:%d %b %Y}"
            sigs = {"ema_cross": sig_ema(df), "rsi_divergence": sig_rsi_div(df)}
            for strat, vs in VARIANTS.items():
                for name, kw in vs:
                    s = stats(run(df, sigs[strat], **kw))
                    rows.append({"coin": coin, "tf": tf, "strategy": strat, "variant": name,
                                 "span": span, "source": src, **s})
            print(f"done {coin} {tf} ({src}, {len(df)} bars)")
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/backtest_results.csv", index=False)
    lines = ["# Backtest results", "",
             f"Generated {datetime.now(AWST):%d %b %Y %H:%M} AWST. Data: {', '.join(sorted(src_note))}. "
             f"Costs: {FEE*100:.3f}% fee + {SLIP*100:.2f}% slippage per side. Drawdown assumes 1% risk per trade.",
             ""]
    if "synthetic" in src_note:
        lines += ["> **Demo mode: synthetic random prices. These numbers are a code test, not market results.**", ""]
    for (coin, tf), g in res.groupby(["coin", "tf"], sort=False):
        lines += [f"## {coin} — {tf}", "", "| Strategy | Variant | Trades | Win rate | Profit factor | Avg R | Max DD |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r.strategy} | {r.variant} | {fmt(r.trades)} | {fmt(r.get('win_rate'), True)} | "
                         f"{fmt(r.get('profit_factor'))} | {fmt(r.get('avg_R'))} | {fmt(r.get('max_drawdown'), True)} |")
        valid = g[g.trades >= 20]
        if len(valid):
            best = valid.sort_values("profit_factor", ascending=False).iloc[0]
            lines += ["", f"Best with ≥20 trades: **{best.strategy} / {best.variant}** "
                          f"(PF {fmt(best.profit_factor)}, win {fmt(best.win_rate, True)})."]
        lines.append("")
    lines += ["## Reading the table", "",
              "Profit factor under 1.1 after costs is noise. Fewer than 30 trades is not evidence. "
              "If a filter lifts PF but cuts trades by more than half, confirm it on a second coin before trusting it. "
              "Re-run monthly and compare against the previous file: an edge that decays over two runs is gone.", ""]
    open(f"{OUT}/backtest_report.md", "w").write("\n".join(lines))
    print(f"\nWrote {OUT}/backtest_report.md and backtest_results.csv")


# -------------------------------------------------------------- portfolio
def cmd_portfolio(a):
    os.makedirs(OUT, exist_ok=True)
    p = pd.read_csv(a.file)
    p["asset"] = p["asset"].str.upper().str.strip()
    p = p.groupby("asset", as_index=False)["value_usd"].sum()
    total = p.value_usd.sum()
    p["weight"] = p.value_usd / total
    risky = [x for x in p.asset if x not in STABLES]
    closes = {}
    for coin in set(risky) | {"BTC"}:
        try:
            df, _ = get(coin, "1d", 180, a.demo)
            closes[coin] = df.set_index(df.time.dt.normalize())["close"]
        except Exception as e:
            print(f"no data for {coin}: {e}")
    px = pd.DataFrame(closes).dropna()
    rets = px.pct_change().dropna()
    beta = {c: rets[c].cov(rets["BTC"]) / rets["BTC"].var() for c in rets}
    vol = rets.std() * math.sqrt(365)
    corr = rets.corr()
    p["beta_btc"] = p.asset.map(lambda x: 0.0 if x in STABLES else beta.get(x, np.nan))
    p["ann_vol"] = p.asset.map(lambda x: 0.0 if x in STABLES else vol.get(x, np.nan))
    p["shock_-20pct_btc"] = p.value_usd * p.beta_btc * -0.20
    shock = p["shock_-20pct_btc"].sum()
    beta_notional = (p.value_usd * p.beta_btc).sum()
    flags = []
    for _, r in p.iterrows():
        if r.asset not in STABLES and r.weight > 0.25:
            flags.append(f"{r.asset} is {r.weight:.0%} of the book (single-asset cap suggestion: 25%).")
        if r.beta_btc > 1.5:
            flags.append(f"{r.asset} beta to BTC is {r.beta_btc:.2f}: it tends to fall harder than BTC in sell-offs.")
    stable_w = p[p.asset.isin(STABLES)].weight.sum()
    if stable_w < 0.10:
        flags.append(f"Stablecoin buffer is {stable_w:.0%}; there is little dry powder or margin cushion.")
    pairs = []
    cs = [c for c in corr.columns if c in risky]
    for i, x in enumerate(cs):
        for y in cs[i + 1:]:
            if corr.loc[x, y] > 0.8:
                pairs.append(f"{x}/{y} correlation {corr.loc[x, y]:.2f} — these behave like one position.")
    L = ["# Portfolio risk check", "", f"Total value: ${total:,.0f}. Beta-weighted BTC exposure: ${beta_notional:,.0f}.",
         f"Estimated P&L if BTC falls 20% (beta model, 180 days): **${shock:,.0f} ({shock/total:.1%})**.", "",
         "| Asset | Value | Weight | Beta | Ann. vol | −20% BTC shock |", "|---|---|---|---|---|---|"]
    for _, r in p.sort_values("value_usd", ascending=False).iterrows():
        L.append(f"| {r.asset} | ${r.value_usd:,.0f} | {r.weight:.1%} | {r.beta_btc:.2f} | {r.ann_vol:.0%} | ${r['shock_-20pct_btc']:,.0f} |")
    L += ["", "## Weak points", ""] + [f"- {f}" for f in flags or ["None flagged by the rules."]]
    L += ["", "## Hidden correlations", ""] + [f"- {x}" for x in pairs or ["No pair above 0.80."]]
    L += ["", "## Hedge sizing (BTC perp short)", "",
          "| Hedge ratio | Short notional | Est. loss in −20% move |", "|---|---|---|"]
    for h in (0.25, 0.5, 0.75, 1.0):
        L.append(f"| {h:.0%} | ${beta_notional*h:,.0f} | ${shock*(1-h):,.0f} |")
    L += ["", "Beta is unstable and rises in crashes; treat the table as a floor on losses, not a ceiling. "
              "A perp hedge pays funding while it is on and needs margin that can itself be liquidated.", ""]
    open(f"{OUT}/portfolio_report.md", "w").write("\n".join(L))
    corr.round(2).to_csv(f"{OUT}/correlation_matrix.csv")
    print(f"Wrote {OUT}/portfolio_report.md")


# ---------------------------------------------------------------- journal
def cmd_journal(a):
    os.makedirs(OUT, exist_ok=True)
    t = pd.read_csv(a.file, parse_dates=["open_time", "close_time"]).sort_values("open_time").reset_index(drop=True)
    t = t.tail(a.last)
    t["dir"] = np.where(t.side.str.lower().str.startswith("l"), 1, -1)
    t["pnl"] = t.dir * (t.exit / t.entry - 1) * t.size_usd
    t["risk_per_unit"] = (t.entry - t.stop).abs()
    t["R"] = t.dir * (t.exit - t.entry) / t.risk_per_unit.replace(0, np.nan)
    t["hold_min"] = (t.close_time - t.open_time).dt.total_seconds() / 60
    t["hour"] = t.open_time.dt.hour
    wins, losses = t[t.pnl > 0], t[t.pnl <= 0]
    issues = []
    nostop = t[t.stop.isna() | (t.risk_per_unit == 0)]
    if len(nostop):
        issues.append(f"{len(nostop)} trade(s) had no defined stop.")
    big = t[t.R < -1.2]
    if len(big):
        issues.append(f"{len(big)} loss(es) exceeded −1.2R (avg {big.R.mean():.2f}R): stops moved, ignored or gapped.")
    rev = []
    for i in range(1, len(t)):
        prev = t.iloc[i - 1]
        cur = t.iloc[i]
        if prev.pnl < 0 and (cur.open_time - prev.close_time).total_seconds() <= 3600 and cur.size_usd >= 1.25 * prev.size_usd:
            rev.append(i)
    if rev:
        rt = t.iloc[rev]
        issues.append(f"{len(rev)} likely revenge trade(s): opened within 60 min of a loss at ≥1.25x size "
                      f"(win rate {(rt.pnl > 0).mean():.0%}).")
    if len(wins) and len(losses) and wins.R.mean() < abs(losses.R.mean()):
        issues.append(f"Average win {wins.R.mean():.2f}R is smaller than average loss {losses.R.mean():.2f}R: "
                      "winners are being cut early.")
    if len(wins) and len(losses) and wins.hold_min.median() < losses.hold_min.median():
        issues.append(f"Winners are held {wins.hold_min.median():.0f} min (median) vs losers "
                      f"{losses.hold_min.median():.0f} min: the disposition effect.")
    days = t.groupby(t.open_time.dt.date).size()
    if (days >= 5).any():
        heavy = days[days >= 5].index
        hd = t[t.open_time.dt.date.isin(heavy)]
        issues.append(f"Overtrading: {len(heavy)} day(s) with 5+ trades, P&L on those days ${hd.pnl.sum():,.0f}.")
    after_win = []
    for i in range(1, len(t)):
        if t.iloc[i - 1].pnl > 0:
            after_win.append(t.iloc[i].size_usd / t.iloc[i - 1].size_usd)
    if after_win and np.mean(after_win) > 1.3:
        issues.append(f"Size increases {np.mean(after_win):.1f}x on average after a win: overconfidence.")
    grp = lambda col: t.groupby(col).agg(trades=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()),
                                         pnl=("pnl", "sum"), avgR=("R", "mean")).sort_values("pnl")
    pf = wins.pnl.sum() / abs(losses.pnl.sum()) if losses.pnl.sum() else np.inf
    L = ["# Trading journal review", "", f"Last {len(t)} trades. Net P&L ${t.pnl.sum():,.0f}. "
         f"Win rate {len(wins)/len(t):.0%}. Profit factor {pf:.2f}. Expectancy {t.R.mean():.2f}R per trade.", "",
         "## Recurring errors", ""] + [f"- {x}" for x in issues or ["No rule-based errors flagged."]]
    for col, title in (("side", "By direction"), ("asset", "By asset"), ("hour", "By hour opened (your clock)")) + \
            ((("setup", "By setup"),) if "setup" in t else ()):
        g = grp(col)
        L += ["", f"## {title}", "", "| | Trades | Win | P&L | Avg R |", "|---|---|---|---|---|"]
        for k, r in g.iterrows():
            L.append(f"| {k} | {int(r.trades)} | {r.win:.0%} | ${r.pnl:,.0f} | {r.avgR:.2f} |")
    worst_hour = grp("hour").index[0] if len(t) else None
    L += ["", "## Starter rules (edit to fit what the tables show)", "",
          "1. No stop, no trade: stop and size are entered before the order, and the stop only moves toward profit.",
          "2. After any loss, wait 60 minutes and never increase size on the next trade.",
          f"3. Trade only your best setup/hour until it has 30 more logged trades; the weakest hour here was {worst_hour}:00.",
          ""]
    open(f"{OUT}/journal_report.md", "w").write("\n".join(L))
    print(f"Wrote {OUT}/journal_report.md")


# ------------------------------------------------------------------ daily
GRADE_DAYS = 5          # a pick is scored after this many daily closes
STOP_ATR, TARGET_R = 2.0, 2.0
HIST_COLS = ["pick_date", "coin", "side", "score", "entry", "stop", "target", "status",
             "exit_date", "exit_px", "result_R", "correct", "comment"]


def score_coin(df):
    """Rule-based composite score from daily candles. Positive = long bias."""
    c = df["close"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    r = rsi(c); a = atr(df)
    i = -1
    sc = 0
    sc += 1 if c.iloc[i] > e20.iloc[i] else -1
    sc += 1 if e20.iloc[i] > e50.iloc[i] else -1
    sc += 1 if e50.iloc[i] > e200.iloc[i] else -1
    sc += 1 if e20.iloc[i] > e20.iloc[i - 5] else -1          # EMA20 slope
    ret20 = c.iloc[i] / c.iloc[i - 20] - 1
    sc += 2 if ret20 > 0.08 else 1 if ret20 > 0 else -1 if ret20 > -0.08 else -2
    rv = r.iloc[i]
    sc += 1 if 50 <= rv <= 70 else -1 if 30 <= rv < 50 else 0
    ext = (c.iloc[i] - e20.iloc[i]) / a.iloc[i]                # ATR distance from EMA20
    if abs(ext) > 3:                                            # too stretched either way
        sc -= int(np.sign(sc)) if sc else 0
    detail = dict(close=float(c.iloc[i]), ema20=float(e20.iloc[i]), ema50=float(e50.iloc[i]),
                  ema200=float(e200.iloc[i]), rsi=float(rv), ret20=float(ret20), atr=float(a.iloc[i]),
                  ext_atr=float(ext), ret5=float(c.iloc[i] / c.iloc[i - 5] - 1))
    return int(sc), detail


def claude_comment(picks, market, model):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not picks or requests is None:
        return {}
    prompt = ("You are reviewing rule-generated crypto swing picks. For each pick give two sentences: "
              "why the numbers support it and the main risk. Be concrete, no hype, no advice language. "
              "Return only JSON: {\"COIN\": \"text\", ...}.\n\nMarket snapshot: " + json.dumps(market)
              + "\n\nPicks: " + json.dumps(picks))
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": model, "max_tokens": 1200, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json()["content"])
        return json.loads(text.strip().strip("`").removeprefix("json").strip())
    except Exception as e:  # noqa
        print(f"Claude commentary skipped: {e}")
        return {}


def grade(hist, demo):
    """Score open picks using daily high/low after the pick date. Same-day stop+target = loss."""
    if hist.empty:
        return hist
    for idx, row in hist[hist.status == "open"].iterrows():
        try:
            df, _ = get(row.coin, "1d", 60, demo)
        except Exception as e:
            print(f"grade {row.coin}: {e}"); continue
        after = df[df.time.dt.normalize() > pd.Timestamp(row.pick_date, tz="UTC")]
        if after.empty:
            continue
        d = 1 if row.side == "long" else -1
        risk = abs(row.entry - row.stop)
        status = exit_px = exit_date = None
        for n, (_, bar) in enumerate(after.iterrows(), 1):
            hit_stop = bar.low <= row.stop if d == 1 else bar.high >= row.stop
            hit_tgt = bar.high >= row.target if d == 1 else bar.low <= row.target
            if hit_stop:
                status, exit_px, exit_date = "stopped", row.stop, bar.time; break
            if hit_tgt:
                status, exit_px, exit_date = "target", row.target, bar.time; break
            if n >= GRADE_DAYS:
                status, exit_px, exit_date = "expired", bar.close, bar.time; break
        if status:
            R = d * (exit_px - row.entry) / risk if risk else 0
            hist.loc[idx, "status"] = status
            hist.loc[idx, "exit_date"] = exit_date.strftime("%Y-%m-%d")
            hist.loc[idx, "exit_px"] = round(float(exit_px), 6)
            hist.loc[idx, "result_R"] = round(float(R), 2)
            hist.loc[idx, "correct"] = "True" if R > 0 else "False"
    return hist


def scorecard_html(hist, today, picks, market, out):
    closed = hist[hist.status != "open"].copy()
    closed["correct"] = closed.correct.astype(str) == "True"
    n = len(closed); wins = int(closed.correct.sum()) if n else 0
    hit = wins / n if n else 0
    avgR = closed.result_R.astype(float).mean() if n else 0
    sumR = closed.result_R.astype(float).sum() if n else 0
    by_coin = closed.groupby("coin").agg(n=("correct", "size"), hit=("correct", "mean"), R=("result_R", "sum")) if n else None
    fmt6 = lambda v: f"{v:,.0f}" if v >= 1000 else f"{v:.2f}" if v >= 1 else f"{v:.4f}"
    pick_rows = "".join(
        f"<tr><td><b>{p['coin']}</b></td><td class='{p['side']}'>{p['side']}</td><td>{p['score']:+d}</td>"
        f"<td>{fmt6(p['entry'])}</td><td>{fmt6(p['stop'])}</td><td>{fmt6(p['target'])}</td>"
        f"<td>{p.get('comment','')}</td></tr>" for p in picks) or "<tr><td colspan=7>No coin met the threshold today. Standing aside is a decision too.</td></tr>"
    mkt_rows = "".join(f"<tr><td><b>{k}</b></td><td>{v['score']:+d}</td><td>{fmt6(v['close'])}</td>"
                       f"<td>{v['rsi']:.0f}</td><td>{v['ret5']*100:+.1f}%</td><td>{v['ret20']*100:+.1f}%</td>"
                       f"<td>{'above' if v['close']>v['ema200'] else 'below'} 200</td></tr>"
                       for k, v in sorted(market.items(), key=lambda kv: -kv[1]['score']))
    hist_rows = "".join(
        f"<tr><td>{r.pick_date}</td><td><b>{r.coin}</b></td><td class='{r.side}'>{r.side}</td><td>{fmt6(float(r.entry))}</td>"
        f"<td>{fmt6(float(r.stop))}</td><td>{fmt6(float(r.target))}</td><td class='{r.status}'>{r.status}</td>"
        f"<td>{'' if pd.isna(r.result_R) or r.result_R=='' else f'{float(r.result_R):+.2f}R'}</td>"
        f"<td>{'' if r.status=='open' else ('✓' if str(r.correct)=='True' else '✗')}</td></tr>"
        for r in hist.sort_values("pick_date", ascending=False).itertuples())
    coin_rows = "".join(f"<tr><td><b>{k}</b></td><td>{int(v.n)}</td><td>{v.hit:.0%}</td><td>{v.R:+.1f}R</td></tr>"
                        for k, v in by_coin.iterrows()) if by_coin is not None else ""
    html = f"""<!DOCTYPE html><html lang="en-AU"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Daily picks — {today}</title><style>
:root{{--bg:#EDF0F3;--panel:#F9FAFB;--ink:#18222E;--muted:#5A6776;--rule:#CBD2DA;--long:#1E7358;--short:#A63D2A;--gold:#8C6E1C}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0F151C;--panel:#161F29;--ink:#E2E7ED;--muted:#93A0AE;--rule:#2B3743;--long:#55B892;--short:#E3806A;--gold:#D4B65E}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif;font-variant-numeric:tabular-nums}}
.w{{max-width:960px;margin:0 auto;padding:22px 16px 60px}}h1{{font-size:2rem;margin:0 0 4px}}h2{{font-size:1.2rem;margin:28px 0 8px}}
.m{{color:var(--muted)}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:6px;overflow:hidden;margin:14px 0}}
.stats div{{background:var(--panel);padding:10px 12px}}.stats span{{display:block;font-size:.75rem;color:var(--muted)}}.stats b{{font-size:1.3rem}}
.t{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px}}table{{border-collapse:collapse;width:100%;min-width:600px;background:var(--panel);font-size:.9rem}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}}th{{color:var(--muted);font-size:.8rem;font-weight:600}}tr:last-child td{{border-bottom:0}}
.long{{color:var(--long);font-weight:600}}.short{{color:var(--short);font-weight:600}}.target{{color:var(--long)}}.stopped{{color:var(--short)}}.expired{{color:var(--gold)}}
.note{{border-left:3px solid var(--gold);padding:8px 12px;font-size:.9rem;color:var(--muted)}}
</style></head><body><div class="w">
<h1>Daily picks</h1><p class="m">Generated {datetime.now(AWST):%A %d %B %Y, %H:%M} AWST from daily closes. Rule-based method: EMA stack, 20-day momentum, RSI, ATR extension. Stop {STOP_ATR:.0f}×ATR, target {TARGET_R:.0f}R, graded after {GRADE_DAYS} daily closes. Educational, not advice.</p>
<div class="stats"><div><span>Picks graded</span><b>{n}</b></div><div><span>Hit rate</span><b>{hit:.0%}</b></div><div><span>Average result</span><b>{avgR:+.2f}R</b></div><div><span>Total</span><b>{sumR:+.1f}R</b></div><div><span>Open picks</span><b>{int((hist.status=='open').sum())}</b></div></div>
<h2>Today's picks</h2><div class="t"><table><tr><th>Coin</th><th>Side</th><th>Score</th><th>Entry</th><th>Stop</th><th>Target</th><th>Reasoning</th></tr>{pick_rows}</table></div>
<h2>Watchlist scan</h2><div class="t"><table><tr><th>Coin</th><th>Score</th><th>Close</th><th>RSI</th><th>5d</th><th>20d</th><th>vs EMA</th></tr>{mkt_rows}</table></div>
{"<h2>Record by coin</h2><div class='t'><table><tr><th>Coin</th><th>Graded</th><th>Hit rate</th><th>Net</th></tr>"+coin_rows+"</table></div>" if coin_rows else ""}
<h2>History</h2><div class="t"><table><tr><th>Date</th><th>Coin</th><th>Side</th><th>Entry</th><th>Stop</th><th>Target</th><th>Outcome</th><th>Result</th><th>Correct</th></tr>{hist_rows}</table></div>
<p class="note">A pick counts as correct when it closes above 0R inside the window. Judge the method after 30+ graded picks, not 5. Hit rate below 40% with 2R targets is still profitable; hit rate above 50% with an average result under 0 means stops are too tight.</p>
</div></body></html>"""
    open(os.path.join(out, "scorecard.html"), "w").write(html)


def cmd_daily(a):
    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    hp = os.path.join(out, "history.csv")
    hist = pd.read_csv(hp, dtype=str) if os.path.exists(hp) else pd.DataFrame(columns=HIST_COLS)
    hist = hist.astype(object).fillna("")
    for c in ("entry", "stop", "target", "exit_px", "result_R"):
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist["score"] = pd.to_numeric(hist["score"], errors="coerce")
    hist = grade(hist, a.demo)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    market, picks = {}, []
    for coin in a.coins:
        try:
            df, src = get(coin, "1d", 260, a.demo)
        except Exception as e:
            print(f"skip {coin}: {e}"); continue
        if len(df) < 210:
            print(f"skip {coin}: only {len(df)} daily bars"); continue
        sc, d = score_coin(df)
        market[coin] = {"score": sc, **{k: round(v, 6) for k, v in d.items()}}
        already = ((hist.pick_date == today) & (hist.coin == coin)).any() or \
                  ((hist.status == "open") & (hist.coin == coin)).any()
        if abs(sc) >= a.threshold and not already:
            side = "long" if sc > 0 else "short"
            entry = d["close"]; sgn = 1 if sc > 0 else -1
            stop = entry - sgn * STOP_ATR * d["atr"]; target = entry + sgn * TARGET_R * abs(entry - stop)
            picks.append({"coin": coin, "side": side, "score": sc, "entry": round(entry, 6),
                          "stop": round(stop, 6), "target": round(target, 6),
                          "rsi": round(d["rsi"]), "ret20": round(d["ret20"], 3)})
        print(f"{coin}: score {sc:+d} ({src})")
    picks = sorted(picks, key=lambda p: -abs(p["score"]))[:a.max_picks]
    comments = claude_comment(picks, market, a.model)
    for p in picks:
        p["comment"] = comments.get(p["coin"], "")
        hist = pd.concat([hist, pd.DataFrame([{"pick_date": today, **{k: p[k] for k in ("coin", "side", "score", "entry", "stop", "target")},
                                               "status": "open", "exit_date": "", "exit_px": np.nan, "result_R": np.nan,
                                               "correct": "", "comment": p["comment"]}])], ignore_index=True)
    hist = hist[HIST_COLS]
    hist.to_csv(hp, index=False)
    scorecard_html(hist, today, picks, market, out)
    json.dump({"date": today, "picks": picks, "market": market}, open(os.path.join(out, "latest.json"), "w"), indent=1)
    print(f"{len(picks)} pick(s) today. Wrote {out}/scorecard.html, history.csv, latest.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL", "XRP", "ZEC", "HYPE"])
    b.add_argument("--tf", nargs="+", default=["1h", "4h"], choices=["1h", "4h", "1d"])
    b.add_argument("--bars", type=int, default=5000)
    b.add_argument("--demo", action="store_true")
    p = sub.add_parser("portfolio")
    p.add_argument("--file", required=True)
    p.add_argument("--demo", action="store_true")
    j = sub.add_parser("journal")
    j.add_argument("--file", required=True)
    j.add_argument("--last", type=int, default=20)
    d = sub.add_parser("daily")
    d.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL", "XRP", "ZEC", "HYPE"])
    d.add_argument("--out", default="~/TradeDesk")
    d.add_argument("--threshold", type=int, default=4, help="min |score| to become a pick (max 8)")
    d.add_argument("--max-picks", type=int, default=3)
    d.add_argument("--model", default=os.environ.get("TRADEDESK_MODEL", "claude-haiku-4-5-20251001"))
    d.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    {"backtest": cmd_backtest, "portfolio": cmd_portfolio, "journal": cmd_journal, "daily": cmd_daily}[a.cmd](a)


if __name__ == "__main__":
    main()
