"""
Regime Trading — complete system in one file.

  RF regime nowcaster  ->  4 per-regime GBT specialists (5-day excess-return forecasts)
                       ->  regime-asymmetric dual-threshold hysteresis + vol targeting

Chronology (no look-ahead anywhere):
  < 2013        train the nowcaster and the four specialists (35 US stocks)
  2013 - 2018   select the four threshold parameters by Calmar ratio
  Nov 2021+     out-of-universe test on SPY (last 20% of 2005-now), touched once

Running this file does everything: trains the stack, selects thresholds, runs the SPY
test, prints the results table + yearly attribution, and saves the figures.

Costs: 5 bps per position change. Idle cash earns interest (fed funds while training/
validating; 4%/yr in the SPY test). Deployment is fully causal: every feature is
trailing, the regime is always the classifier's prediction (never a label), and the
position chosen at close t earns the return of t -> t+1.
"""
from __future__ import annotations

import json
import os
import warnings
from urllib.parse import urlencode
from urllib.request import Request, urlopen

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "stock_market_regimes_2000_2026.csv")

TRAIN_END = "2013-01-01"
VAL = ("2013-01-01", "2019-01-01")
REGIMES = ["Bull", "Bear", "Sideways", "Crisis"]
RISK_OFF = {"Bear", "Crisis"}
COST = 5 / 1e4                               # 5 bps per position change
H = 5                                        # specialist prediction horizon (days)
VOL_TARGET_D = 0.25 / np.sqrt(252)           # 25% annual vol target
CASH_D = 1.04 ** (1 / 252) - 1               # SPY test: idle cash at 4%/yr
GRID_IN = [0.0, 0.001, 0.002, 0.003]
GRID_OUT = [0.0, -0.001, -0.002, -0.003]
COLORS = {"System": "#1565C0", "Buy&Hold": "#455A64", "MA20>60": "#E65100"}

FACTORS = ["mom_5", "mom_20", "mom_60", "mom_120", "ma20_z", "ma60_z", "ma200_z",
           "vol_20", "vol_ratio", "dd_60", "stoch_20", "rsi_14"]
CLF_FEATURES = ["ret_1", "ret_5", "ret_20", "ret_60", "px_ma20", "px_ma60", "px_ma200",
                "vol_20c", "vol_60c", "px_hi20", "px_hi60", "stoch20c",
                "vix", "fed_funds_rate", "unemployment_rate", "yield_spread"]


# ── 1. feature engineering (trailing only) ────────────────────────────────────
def build_factors(c: pd.Series) -> pd.DataFrame:
    """12 vol-normalized technical factors driving the specialists."""
    r = c.pct_change()
    vol20, vol60 = r.rolling(20).std(), r.rolling(60).std()

    def volscale(ret_k, k):
        return ret_k / (vol20 * np.sqrt(k)).replace(0, np.nan)

    f = pd.DataFrame(index=c.index)
    f["mom_5"], f["mom_20"] = volscale(c.pct_change(5), 5), volscale(c.pct_change(20), 20)
    f["mom_60"], f["mom_120"] = volscale(c.pct_change(60), 60), volscale(c.pct_change(120), 120)
    f["ma20_z"] = (c / c.rolling(20).mean() - 1) / vol20.replace(0, np.nan)
    f["ma60_z"] = (c / c.rolling(60).mean() - 1) / (vol20 * np.sqrt(3)).replace(0, np.nan)
    f["ma200_z"] = (c / c.rolling(200).mean() - 1) / (vol20 * np.sqrt(10)).replace(0, np.nan)
    f["vol_20"], f["vol_ratio"] = vol20, vol20 / vol60.replace(0, np.nan)
    f["dd_60"] = (c / c.rolling(60).max() - 1) / (vol20 * np.sqrt(20)).replace(0, np.nan)
    f["stoch_20"] = (c - c.rolling(20).min()) / (c.rolling(20).max() - c.rolling(20).min()).replace(0, np.nan)
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = (100 - 100 / (1 + up / dn.replace(0, np.nan)) - 50.0) / 50.0
    return f


def classifier_features(g: pd.DataFrame) -> pd.DataFrame:
    """16 nowcaster features: technical + macro (vix, rates, unemployment, curve)."""
    c = g["close"]
    f = pd.DataFrame(index=g.index)
    for k in (1, 5, 20, 60):
        f[f"ret_{k}"] = c.pct_change(k)
    for k in (20, 60, 200):
        f[f"px_ma{k}"] = c / c.rolling(k).mean() - 1
    r = c.pct_change()
    f["vol_20c"], f["vol_60c"] = r.rolling(20).std(), r.rolling(60).std()
    f["px_hi20"] = c / c.rolling(20).max() - 1
    f["px_hi60"] = c / c.rolling(60).max() - 1
    hi, lo = c.rolling(20).max(), c.rolling(20).min()
    f["stoch20c"] = (c - lo) / (hi - lo).replace(0, np.nan)
    f["vix"] = g["vix"].to_numpy()
    f["fed_funds_rate"] = g["fed_funds_rate"].to_numpy()
    f["unemployment_rate"] = g["unemployment_rate"].to_numpy()
    f["yield_spread"] = (g["10y_treasury"] - g["2y_treasury"]).to_numpy()
    return f


# ── 2. data preparation (35-stock training universe) ──────────────────────────
def prepare() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df[~df["ticker"].str.startswith("^")]
    df = df[df["regime_label"] != "High-volatility"]
    parts = []
    for tk, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        out = pd.concat([g[["date", "ticker", "close", "regime_label"]],
                         build_factors(g["close"]), classifier_features(g)], axis=1)
        out["rf_d"] = (1 + g["fed_funds_rate"].to_numpy() / 100) ** (1 / 252) - 1
        out["mkt_1d"] = g["close"].shift(-1) / g["close"] - 1
        out["y_fwd"] = g["close"].shift(-H) / g["close"] - 1 - H * out["rf_d"]
        parts.append(out)
    return pd.concat(parts, ignore_index=True).dropna(subset=FACTORS + CLF_FEATURES + ["mkt_1d"])


# ── 3. model training ─────────────────────────────────────────────────────────
def train_stack(df: pd.DataFrame):
    """Nowcaster + 4 specialists on <2013; adds regime_pred / yhat columns to df."""
    train = df[df["date"] < TRAIN_END]
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50,
                                 class_weight="balanced", n_jobs=-1, random_state=7)
    clf.fit(train[CLF_FEATURES], train["regime_label"])
    df["regime_pred"] = clf.predict(df[CLF_FEATURES])

    scaler = StandardScaler().fit(train[FACTORS])
    X_all = scaler.transform(df[FACTORS])
    spec, yhat = {}, np.zeros(len(df))
    for reg in REGIMES:
        sub = train[(train["regime_label"] == reg) & train["y_fwd"].notna()]
        spec[reg] = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                  max_iter=300, random_state=7)\
            .fit(scaler.transform(sub[FACTORS]), sub["y_fwd"])
        mask = (df["regime_pred"] == reg).to_numpy()
        yhat[mask] = spec[reg].predict(X_all[mask])
    df["yhat"] = yhat
    return clf, scaler, spec


# ── 4. decision layer ─────────────────────────────────────────────────────────
def hysteresis(yhat, tin, tout):
    """Enter when the forecast clears tin; exit only when it drops below tout."""
    pos, p = np.zeros(len(yhat)), 0.0
    for i in range(len(yhat)):
        if p == 0.0 and yhat[i] > tin[i]:
            p = 1.0
        elif p == 1.0 and yhat[i] < tout[i]:
            p = 0.0
        pos[i] = p
    return pos


def system_position(yhat, riskoff, volscale, th_on, th_off):
    tin = np.where(riskoff, th_off[0], th_on[0])
    tout = np.where(riskoff, th_off[1], th_on[1])
    return hysteresis(yhat, tin, tout) * volscale


# ── 5. threshold selection on 2013-2018 (Calmar) ──────────────────────────────
def select_thresholds(df: pd.DataFrame):
    slices = {}
    for tk, g in df.groupby("ticker"):
        g = g[(g["date"] >= VAL[0]) & (g["date"] < VAL[1])]
        slices[tk] = {"date": g["date"].to_numpy(), "mkt": g["mkt_1d"].to_numpy(),
                      "rf": g["rf_d"].to_numpy(), "yhat": g["yhat"].to_numpy(),
                      "riskoff": g["regime_pred"].isin(RISK_OFF).to_numpy(),
                      "volscale": np.clip(VOL_TARGET_D / np.maximum(g["vol_20"].to_numpy(), 1e-6), 0, 1)}

    def calmar(th_on, th_off):
        frames = []
        for s in slices.values():
            p = system_position(s["yhat"], s["riskoff"], s["volscale"], th_on, th_off)
            r = p * s["mkt"] + (1 - p) * s["rf"] - COST * np.abs(np.diff(np.concatenate([[0.0], p])))
            frames.append(pd.DataFrame({"date": s["date"], "r": r}))
        ret = pd.concat(frames).groupby("date")["r"].mean().sort_index().to_numpy()
        nav = np.cumprod(1 + ret)
        cagr = nav[-1] ** (252 / len(ret)) - 1
        maxdd = (nav / np.maximum.accumulate(nav) - 1).min()
        return cagr / abs(maxdd) if maxdd < 0 else 99.0

    best = None
    for tin_on in GRID_IN:
        for tout_on in GRID_OUT:
            for tin_off in GRID_IN:
                for tout_off in GRID_OUT:
                    score = calmar((tin_on, tout_on), (tin_off, tout_off))
                    if best is None or score > best[0]:
                        best = (score, (tin_on, tout_on), (tin_off, tout_off))
    return best[1], best[2]


# ── 6. SPY test ───────────────────────────────────────────────────────────────
def download_spy() -> pd.DataFrame:
    params = urlencode({"period1": int(pd.Timestamp("2005-01-01", tz="UTC").timestamp()),
                        "period2": int(pd.Timestamp.now(tz="UTC").timestamp()),
                        "interval": "1d", "events": "history", "includeAdjustedClose": "true"})
    req = Request(f"https://query2.finance.yahoo.com/v8/finance/chart/SPY?{params}",
                  headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())["chart"]["result"][0]
    dates = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
    return pd.DataFrame({"date": dates,
                         "close": data["indicators"]["adjclose"][0]["adjclose"]}).dropna()


def spy_test(clf, scaler, spec, th_on, th_off) -> None:
    spy = download_spy()
    kg = pd.read_csv(DATA, parse_dates=["date"])
    macro = kg[kg["ticker"] == "^GSPC"][["date", "vix", "fed_funds_rate",
                                         "unemployment_rate", "10y_treasury", "2y_treasury"]]
    spy = spy.merge(macro, on="date", how="inner").sort_values("date").reset_index(drop=True)
    test_start = spy["date"].iloc[int(len(spy) * 0.8)]      # last 20% = test
    print(f"\nSPY test (last 20%): {test_start.date()} -> {spy['date'].iloc[-1].date()}")

    sf = pd.concat([spy[["date", "close"]], build_factors(spy["close"]),
                    classifier_features(spy)], axis=1).dropna(subset=FACTORS + CLF_FEATURES)
    sf = sf[sf["date"] >= test_start].reset_index(drop=True)
    regp = clf.predict(sf[CLF_FEATURES])
    X = scaler.transform(sf[FACTORS])
    yhat = np.zeros(len(sf))
    for r in REGIMES:
        m = regp == r
        if m.any():
            yhat[m] = spec[r].predict(X[m])
    riskoff = np.isin(regp, ["Bear", "Crisis"])
    volscale = np.clip(VOL_TARGET_D / np.maximum(sf["vol_20"].to_numpy(), 1e-6), 0, 1)
    expo = system_position(yhat, riskoff, volscale, th_on, th_off)

    c = sf["close"].to_numpy()
    mkt = np.append(c[1:] / c[:-1] - 1, 0)
    dates = sf["date"].to_numpy()
    ma = (sf["close"].rolling(20).mean() > sf["close"].rolling(60).mean()).astype(float).to_numpy()
    positions = {"System": expo, "Buy&Hold": np.ones(len(sf)), "MA20>60": ma}

    curves, stats = {}, {}
    for name, position in positions.items():
        sw = np.abs(np.diff(np.concatenate([[0.0], position])))
        r = position * mkt + (1 - position) * CASH_D - COST * sw
        nav = np.cumprod(1 + r)
        curves[name] = nav
        stats[name] = {"total": nav[-1] - 1, "sharpe": np.sqrt(252) * r.mean() / r.std(),
                       "maxdd": (nav / np.maximum.accumulate(nav) - 1).min(),
                       "vol": r.std() * np.sqrt(252), "tim": float(np.mean(position > 0))}
        s = stats[name]
        print(f"{name:>9} | total {s['total']:+7.1%} | Sharpe {s['sharpe']:5.2f} | "
              f"maxDD {s['maxdd']:6.1%} | vol {s['vol']:5.1%} | TIM {s['tim']:5.1%}")

    yr = pd.Series(dates).dt.year.to_numpy()
    print("\nyearly excess (System - B&H):")
    for y in sorted(set(yr)):
        m = yr == y
        i0 = np.argmax(m)
        bs = curves["System"][i0 - 1] if i0 > 0 else 1.0
        bb = curves["Buy&Hold"][i0 - 1] if i0 > 0 else 1.0
        rv, rb = curves["System"][m][-1] / bs - 1, curves["Buy&Hold"][m][-1] / bb - 1
        print(f"  {y}: System {rv:+6.1%}  B&H {rb:+6.1%}  excess {rv-rb:+6.1%}")

    # figures
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1, 0.8]})
    for name, nav in curves.items():
        ax[0].plot(dates, nav, color=COLORS[name], lw=1.7, label=name)
        ax[1].plot(dates, nav / np.maximum.accumulate(nav) - 1, color=COLORS[name], lw=1.2)
    ax[0].set_ylabel("NAV"); ax[0].legend(framealpha=0.9)
    ax[0].set_title("SPY (dividends incl.) — test = last 20% of 2005-now, 5 bps, cash 4%",
                    fontsize=13, fontweight="bold")
    ax[1].set_ylabel("drawdown"); ax[1].axhline(0, color="gray", lw=0.6)
    ax[2].fill_between(dates, expo, color="#1565C0", alpha=0.65, step="mid")
    for i in range(len(sf)):
        if riskoff[i]:
            ax[2].axvspan(dates[i], dates[min(i + 1, len(sf) - 1)], color="#C62828", alpha=0.12, lw=0)
    ax[2].set_ylabel("System exposure"); ax[2].set_ylim(0, 1.05)
    ax[2].xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "spy_test.png"), dpi=130, bbox_inches="tight")

    names = list(positions)
    panels = [("total", "Total return", "{:+.1%}"), ("sharpe", "Sharpe", "{:.2f}"),
              ("maxdd", "Max drawdown", "{:.1%}"), ("vol", "Annual volatility", "{:.1%}"),
              ("tim", "Time in market", "{:.0%}")]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for a, (key, title, fmt) in zip(axes, panels):
        vals = [stats[n][key] for n in names]
        bars = a.bar(names, vals, color=[COLORS[n] for n in names])
        a.set_title(title, fontsize=11)
        a.tick_params(axis="x", rotation=20, labelsize=8)
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center",
                   va="bottom" if v >= 0 else "top", fontsize=9)
    fig.suptitle("SPY test — strategy metrics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(HERE, "figures", "spy_metrics.png"), dpi=130, bbox_inches="tight")
    print("\nsaved figures/spy_test.png, figures/spy_metrics.png")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("1/4 preparing data ...")
    df = prepare()
    print(f"    {df['ticker'].nunique()} stocks, {len(df):,} rows")
    print("2/4 training nowcaster + specialists (<2013) ...")
    clf, scaler, spec = train_stack(df)
    print("3/4 selecting thresholds on 2013-2018 by Calmar ...")
    th_on, th_off = select_thresholds(df)
    print(f"    risk-on(in,out)={th_on}  risk-off(in,out)={th_off}")
    print("4/4 SPY test ...")
    spy_test(clf, scaler, spec, th_on, th_off)


if __name__ == "__main__":
    main()
