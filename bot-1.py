#!/usr/bin/env python3
"""
Strategy scanner + backtester (no live orders, no API keys needed).

Rules implemented (1H bias, 15M entry, SMC style):
  1. 1H bias from swing structure (HH/HL = long only, LH/LL = short only)
  2. 1H zones: order block, supply/demand, equal highs/lows, Kijun.
     At least 2 different types must overlap.
  3. 15M trigger after price touches the zone: structure break + at least
     one of (liquidity sweep, engulfing/pin candle, high volume)
  4. SL behind the zone with an ATR buffer, TP at the next 1H swing level
     (capped at max_rr, skipped if nearer than min_rr)
  5. Management: optional break-even after 1R, optional early exit when a
     15M candle closes outside the zone
  6. Limits: max open trades, max trades per day, stop after losses

Usage:
  python bot.py selftest                      # offline test, synthetic data
  python bot.py backtest --days 730 --top 10  # needs internet
  python bot.py scan --top 20                 # needs internet (+ Telegram secrets)
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

CFG = dict(
    exchange="okx", htf="1h", ltf="15m",
    swing_htf=5, swing_ltf=3, atr_n=14,
    vol_mult=1.5, vol_n=20,
    sl_buf=0.5, min_rr=1.5, max_rr=3.0, fb_rr=2.0,
    use_session=True, session=(13, 17),
    use_kijun=True, eq_tol=0.15,
    zone_age=150, touch_win=12, cooldown=48,
    be_r=1.0, max_risk_pct=0.03,
    fee=0.0005, slip=0.0002,
    max_open=2, max_day=2, stop_losses=2,
)
FAR = 2 ** 62


def tfms(tf):
    return int(tf[:-1]) * {"m": 60000, "h": 3600000, "d": 86400000}[tf[-1]]


class Bars:
    def __init__(s, df, tf, n, atr_n=14):
        s.t = df["ts"].to_numpy("int64")
        s.o, s.h, s.l, s.c, s.v = [df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")]
        s.ct = s.t + tfms(tf)
        s.N = len(s.t)
        s.n = n
        pc = np.r_[s.c[0], s.c[:-1]]
        tr = np.maximum(s.h - s.l, np.maximum(np.abs(s.h - pc), np.abs(s.l - pc)))
        s.atr = pd.Series(tr).rolling(atr_n, min_periods=1).mean().to_numpy()
        shi, sli = [], []
        for i in range(n, s.N - n):
            if s.h[i] > s.h[i - n:i].max() and s.h[i] >= s.h[i + 1:i + n + 1].max():
                shi.append(i)
            if s.l[i] < s.l[i - n:i].min() and s.l[i] <= s.l[i + 1:i + n + 1].min():
                sli.append(i)
        s.shi, s.sli = np.array(shi, int), np.array(sli, int)
        s.shc, s.slc = s.shi + n, s.sli + n  # bar where the swing is confirmed

    def last_sh(s, t, k=1):
        p = np.searchsorted(s.shc, t, "right")
        return None if p < k else float(s.h[s.shi[p - k]])

    def last_sl(s, t, k=1):
        p = np.searchsorted(s.slc, t, "right")
        return None if p < k else float(s.l[s.sli[p - k]])


def htf_prep(H, cfg):
    N = H.N
    bias = np.zeros(N, int)
    for i in range(N):
        a = np.searchsorted(H.shc, i, "right")
        b = np.searchsorted(H.slc, i, "right")
        if a >= 2 and b >= 2:
            hu = H.h[H.shi[a - 1]] > H.h[H.shi[a - 2]]
            lu = H.l[H.sli[b - 1]] > H.l[H.sli[b - 2]]
            bias[i] = 1 if (hu and lu) else (-1 if (not hu and not lu) else 0)
    kij = ((pd.Series(H.h).rolling(26, min_periods=26).max() + pd.Series(H.l).rolling(26, min_periods=26).min()) / 2).to_numpy()
    age = cfg["zone_age"] * tfms(cfg["htf"])
    Zr = []

    def add(tp, d, lo, hi, ci):
        if hi - lo <= 0 or hi - lo > 3 * H.atr[ci]:
            return
        seg = H.c[ci + 1:]
        m = np.nonzero(seg < lo)[0] if d == 1 else np.nonzero(seg > hi)[0]
        mit = int(H.ct[ci + 1 + m[0]]) if len(m) else FAR
        Zr.append((tp, d, lo, hi, int(H.ct[ci]), min(mit, int(H.ct[ci]) + age)))

    iH = iL = -1
    bH = bL = True
    pa = pb = 0
    for i in range(1, N):
        while pa < len(H.shc) and H.shc[pa] <= i - 1:
            iH, bH = int(H.shi[pa]), False
            pa += 1
        while pb < len(H.slc) and H.slc[pb] <= i - 1:
            iL, bL = int(H.sli[pb]), False
            pb += 1
        if iH >= 0 and iL >= 0 and not bH and H.c[i] > H.h[iH]:
            bH = True
            for j in range(i - 1, max(iL, i - 30) - 1, -1):
                if H.c[j] < H.o[j]:
                    add(0, 1, H.l[j], H.h[j], i)
                    break
        if iH >= 0 and iL >= 0 and not bL and H.c[i] < H.l[iL]:
            bL = True
            for j in range(i - 1, max(iH, i - 30) - 1, -1):
                if H.c[j] > H.o[j]:
                    add(0, -1, H.l[j], H.h[j], i)
                    break
    for i in range(4, N):
        if abs(H.c[i] - H.o[i]) > 1.5 * H.atr[i - 1]:
            base, k = [], i - 1
            while k >= max(0, i - 3) and (H.h[k] - H.l[k]) < 0.8 * H.atr[k]:
                base.append(k)
                k -= 1
            if base:
                lo, hi = H.l[base].min(), H.h[base].max()
                if H.c[i] > H.o[i] and H.c[i] > hi:
                    add(1, 1, lo, hi, i)
                elif H.c[i] < H.o[i] and H.c[i] < lo:
                    add(1, -1, lo, hi, i)
    Qr = []
    for arr, price, d in ((H.sli, H.l, 1), (H.shi, H.h, -1)):
        for m in range(1, len(arr)):
            a, b = arr[m - 1], arr[m]
            if abs(price[a] - price[b]) <= cfg["eq_tol"] * H.atr[b] and b + H.n < N:
                lvl = min(price[a], price[b]) if d == 1 else max(price[a], price[b])
                ci = b + H.n
                seg = H.c[ci + 1:]
                x = np.nonzero(seg < lvl - 0.3 * H.atr[b])[0] if d == 1 else np.nonzero(seg > lvl + 0.3 * H.atr[b])[0]
                end = int(H.ct[ci + 1 + x[0]]) if len(x) else FAR
                Qr.append((d, lvl - 0.15 * H.atr[b], lvl + 0.15 * H.atr[b], int(H.ct[ci]), min(end, int(H.ct[ci]) + age)))

    def pack(rows, names):
        return {k: np.array([r[i] for r in rows]) if rows else np.array([]) for i, k in enumerate(names)}

    Z = pack(Zr, ["tp", "dir", "lo", "hi", "c", "end"])
    Q = pack(Qr, ["dir", "lo", "hi", "c", "end"])
    return dict(bias=bias, kij=kij, Z=Z, Q=Q)


def overlap_types(P, H, hidx, T, zi, cfg):
    Z, Q = P["Z"], P["Q"]
    lo, hi, d, tp = Z["lo"][zi], Z["hi"][zi], Z["dir"][zi], Z["tp"][zi]
    types = {"OB" if tp == 0 else "SD"}
    m = (Z["dir"] == d) & (Z["tp"] != tp) & (Z["c"] <= T) & (T < Z["end"]) & (Z["lo"] <= hi) & (Z["hi"] >= lo)
    if m.any():
        types.add("SD" if tp == 0 else "OB")
    if len(Q["dir"]):
        mq = (Q["dir"] == d) & (Q["c"] <= T) & (T < Q["end"]) & (Q["lo"] <= hi) & (Q["hi"] >= lo)
        if mq.any():
            types.add("LIQ")
    if cfg["use_kijun"] and hidx >= 0 and not np.isnan(P["kij"][hidx]):
        k, a = P["kij"][hidx], 0.1 * H.atr[hidx]
        if k + a >= lo and k - a <= hi:
            types.add("KIJ")
    return types


def make_trade(H, L, P, cfg, zi, d, entry, t, hidx):
    lo, hi = P["Z"]["lo"][zi], P["Z"]["hi"][zi]
    sl = lo - cfg["sl_buf"] * L.atr[t] if d == 1 else hi + cfg["sl_buf"] * L.atr[t]
    R = (entry - sl) * d
    if R <= 0 or R / entry > cfg["max_risk_pct"] or R / entry < 0.0005:
        return None
    p = np.searchsorted(H.shc, hidx, "right")
    q = np.searchsorted(H.slc, hidx, "right")
    if d == 1:
        px = H.h[H.shi[:p]][H.shi[:p] >= hidx - 300]
        px = px[px > entry]
        lv = float(px.min()) if len(px) else None
        if lv is not None and lv < entry + cfg["min_rr"] * R:
            return None
        tp = min(lv, entry + cfg["max_rr"] * R) if lv is not None else entry + cfg["fb_rr"] * R
    else:
        px = H.l[H.sli[:q]][H.sli[:q] >= hidx - 300]
        px = px[px < entry]
        lv = float(px.max()) if len(px) else None
        if lv is not None and lv > entry - cfg["min_rr"] * R:
            return None
        tp = max(lv, entry - cfg["max_rr"] * R) if lv is not None else entry - cfg["fb_rr"] * R
    return dict(side=d, entry=float(entry), sl=float(sl), tp=float(tp), R=float(R), zlo=float(lo), zhi=float(hi))


def in_session(cfg, ms):
    if not cfg["use_session"]:
        return True
    h = (ms // 3600000) % 24
    return cfg["session"][0] <= h < cfg["session"][1]


def market_signals(H, L, cfg, P=None):
    P = P or htf_prep(H, cfg)
    Z = P["Z"]
    hi_of = np.searchsorted(H.ct, L.ct, "right") - 1
    va = pd.Series(L.v).rolling(cfg["vol_n"]).mean().shift(1).to_numpy()
    out = []
    for zi in range(len(Z["dir"]) if len(Z["dir"]) else 0):
        d, lo, hi = int(Z["dir"][zi]), Z["lo"][zi], Z["hi"][zi]
        pos = int(np.searchsorted(L.ct, Z["c"][zi], "right"))
        e = int(np.searchsorted(L.ct, Z["end"][zi], "left"))
        nsig = 0
        while pos < e and nsig < 2:
            touch = np.nonzero(L.l[pos:e] <= hi)[0] if d == 1 else np.nonzero(L.h[pos:e] >= lo)[0]
            if not len(touch):
                break
            tt = pos + int(touch[0])
            wend = min(tt + cfg["touch_win"], e)
            hidx = int(hi_of[tt])
            if hidx < 0 or P["bias"][hidx] != d:
                pos = wend
                continue
            types = overlap_types(P, H, hidx, L.ct[tt], zi, cfg)
            if len(types) < 2:
                pos = wend
                continue
            sw = stc = cd = vo = False
            fired = None
            for t in range(max(tt, 2), wend):
                if (d == 1 and L.c[t] < lo) or (d == -1 and L.c[t] > hi):
                    break
                rl, rh = L.last_sl(t - 1), L.last_sh(t - 1)
                body = abs(L.c[t] - L.o[t])
                rng = L.h[t] - L.l[t]
                if d == 1:
                    if rl is not None and L.l[t] < rl < L.c[t]:
                        sw = True
                    if rh is not None and L.c[t] > rh:
                        stc = True
                    eng = L.c[t] > L.o[t] and L.c[t - 1] < L.o[t - 1] and L.c[t] >= L.o[t - 1] and L.o[t] <= L.c[t - 1]
                    pin = rng > 0 and (min(L.o[t], L.c[t]) - L.l[t]) >= 2 * max(body, 1e-12) and L.c[t] >= L.l[t] + 0.6 * rng
                    if eng or pin:
                        cd = True
                    if L.c[t] > L.o[t] and not np.isnan(va[t]) and L.v[t] >= cfg["vol_mult"] * va[t]:
                        vo = True
                else:
                    if rh is not None and L.h[t] > rh > L.c[t]:
                        sw = True
                    if rl is not None and L.c[t] < rl:
                        stc = True
                    eng = L.c[t] < L.o[t] and L.c[t - 1] > L.o[t - 1] and L.c[t] <= L.o[t - 1] and L.o[t] >= L.c[t - 1]
                    pin = rng > 0 and (L.h[t] - max(L.o[t], L.c[t])) >= 2 * max(body, 1e-12) and L.c[t] <= L.h[t] - 0.6 * rng
                    if eng or pin:
                        cd = True
                    if L.c[t] < L.o[t] and not np.isnan(va[t]) and L.v[t] >= cfg["vol_mult"] * va[t]:
                        vo = True
                if stc and (sw or cd or vo) and in_session(cfg, int(L.ct[t])):
                    h2 = int(hi_of[t])
                    tr = make_trade(H, L, P, cfg, zi, d, L.c[t], t, h2) if h2 >= 0 else None
                    if tr:
                        tr.update(i=t, mode="market", t_in=int(L.ct[t]), types="+".join(sorted(types)),
                                  why=",".join(n for n, f in (("sweep", sw), ("struct", stc), ("candle", cd), ("volume", vo)) if f))
                        fired = tr
                        break
            if fired:
                out.append(fired)
                nsig += 1
                pos = fired["i"] + cfg["cooldown"]
            else:
                pos = wend
    out.sort(key=lambda x: x["t_in"])
    return out


def limit_signals(H, L, cfg, P=None):
    P = P or htf_prep(H, cfg)
    Z = P["Z"]
    hi_of = np.searchsorted(H.ct, L.ct, "right") - 1
    out = []
    for zi in range(len(Z["dir"]) if len(Z["dir"]) else 0):
        d, lo, hi = int(Z["dir"][zi]), Z["lo"][zi], Z["hi"][zi]
        mid = (lo + hi) / 2
        s = max(int(np.searchsorted(L.ct, Z["c"][zi], "right")), 1)
        e = int(np.searchsorted(L.ct, Z["end"][zi], "left"))
        if e <= s:
            continue
        idx = np.arange(s, e)
        m = (L.l[s:e] <= mid) & (L.o[s:e] > mid) if d == 1 else (L.h[s:e] >= mid) & (L.o[s:e] < mid)
        for t in idx[m]:
            if not in_session(cfg, int(L.t[t])):
                continue
            hidx = int(hi_of[t - 1])
            if hidx < 0 or P["bias"][hidx] != d:
                continue
            if len(overlap_types(P, H, hidx, L.ct[t - 1], zi, cfg)) < 2:
                continue
            tr = make_trade(H, L, P, cfg, zi, d, mid, t, hidx)
            if tr:
                tr.update(i=int(t), mode="limit", t_in=int(L.t[t]), types="+".join(sorted(overlap_types(P, H, hidx, L.ct[t - 1], zi, cfg))), why="limit50")
                out.append(tr)
                break
    out.sort(key=lambda x: x["t_in"])
    return out


def simulate(L, sig, cfg, be, early):
    d, e, R, sl, tp = sig["side"], sig["entry"], sig["R"], sig["sl"], sig["tp"]
    start = sig["i"] + 1 if sig["mode"] == "market" else sig["i"]
    done = False
    ex = k = None
    for k in range(start, L.N):
        if d == 1:
            if L.l[k] <= sl:
                ex, done = sl, True
            elif L.h[k] >= tp:
                ex, done = tp, True
            else:
                if be and L.h[k] >= e + cfg["be_r"] * R:
                    sl = max(sl, e)
                if early and L.c[k] < sig["zlo"]:
                    ex, done = L.c[k], True
        else:
            if L.h[k] >= sl:
                ex, done = sl, True
            elif L.l[k] <= tp:
                ex, done = tp, True
            else:
                if be and L.l[k] <= e - cfg["be_r"] * R:
                    sl = min(sl, e)
                if early and L.c[k] > sig["zhi"]:
                    ex, done = L.c[k], True
        if done:
            break
    if not done:
        return None
    gross = d * (ex - e) / R
    cost = (e + ex) * (cfg["fee"] + cfg["slip"]) / R
    return dict(t_out=int(L.ct[k]), R=gross - cost)


def apply_limits(trades, cfg):
    trades = sorted(trades, key=lambda x: x["t_in"])
    acc = []
    for tr in trades:
        t = tr["t_in"]
        day = t // 86400000
        if any(a["t_out"] > t and a["sym"] == tr["sym"] for a in acc):
            continue
        if sum(1 for a in acc if a["t_out"] > t) >= cfg["max_open"]:
            continue
        if sum(1 for a in acc if a["t_in"] // 86400000 == day) >= cfg["max_day"]:
            continue
        closed = sorted((a for a in acc if a["t_out"] <= t and a["t_out"] // 86400000 == day), key=lambda a: a["t_out"])
        if len(closed) >= cfg["stop_losses"] and all(a["R"] < 0 for a in closed[-cfg["stop_losses"]:]):
            continue
        acc.append(tr)
    return acc


def stats(trs):
    if not trs:
        return dict(n=0, win=0, avg=0, tot=0, pf=0, dd=0)
    r = np.array([t["R"] for t in trs])
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max())
    gl = -r[r < 0].sum()
    return dict(n=len(r), win=100 * (r > 0).mean(), avg=r.mean(), tot=r.sum(), pf=(r[r > 0].sum() / gl if gl > 0 else float("inf")), dd=dd)


def run_backtest(data, cfg):
    """data: {symbol: (df_htf, df_ltf)} -> markdown + trades df"""
    sigs = {"market": [], "limit": []}
    bars = {}
    for sym, (dh, dl) in data.items():
        H = Bars(dh, cfg["htf"], cfg["swing_htf"], cfg["atr_n"])
        L = Bars(dl, cfg["ltf"], cfg["swing_ltf"], cfg["atr_n"])
        bars[sym] = L
        P = htf_prep(H, cfg)
        for mode, fn in (("market", market_signals), ("limit", limit_signals)):
            for s in fn(H, L, cfg, P):
                s["sym"] = sym
                sigs[mode].append(s)
        print(f"{sym}: {len(sigs['market'])} market / {len(sigs['limit'])} limit signals so far", flush=True)
    rows, md, allr = [], [], []
    md.append("| mode | BE after 1R | early exit | trades | win % | avg R | total R | profit factor | max DD (R) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for mode in ("market", "limit"):
        for be in (True, False):
            for early in (True, False):
                res = []
                for s in sigs[mode]:
                    r = simulate(bars[s["sym"]], s, cfg, be, early)
                    if r:
                        res.append(dict(s, **r))
                acc = apply_limits(res, cfg)
                st = stats(acc)
                md.append(f"| {mode} | {be} | {early} | {st['n']} | {st['win']:.0f} | {st['avg']:.2f} | {st['tot']:.1f} | {st['pf']:.2f} | {st['dd']:.1f} |")
                for a in acc:
                    allr.append(dict(mode=mode, be=be, early=early, sym=a["sym"], side=a["side"], t_in=pd.to_datetime(a["t_in"], unit="ms"),
                                     entry=a["entry"], sl=a["sl"], tp=a["tp"], R=round(a["R"], 3), types=a["types"], why=a["why"]))
    return "\n".join(md), pd.DataFrame(allr)


# ----------------------------- data / network -----------------------------

def get_ex():
    import ccxt
    ex = getattr(ccxt, CFG["exchange"])({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    return ex


def top_symbols(ex, n):
    tk = ex.fetch_tickers()
    c = []
    for s, t in tk.items():
        m = ex.markets.get(s, {})
        if m.get("swap") and m.get("linear") and s.endswith(":USDT"):
            v = t.get("quoteVolume") or ((t.get("baseVolume") or 0) * (t.get("last") or 0))
            c.append((s, v or 0))
    c.sort(key=lambda x: -x[1])
    return [s for s, _ in c[:n]]


def fetch_all(ex, sym, tf, days):
    os.makedirs("data", exist_ok=True)
    fn = f"data/{sym.replace('/', '_').replace(':', '_')}_{tf}_{days}.csv"
    if os.path.exists(fn) and time.time() - os.path.getmtime(fn) < 6 * 3600:
        return pd.read_csv(fn)
    since = int(time.time() * 1000) - days * 86400000
    rows, step = [], tfms(tf)
    while True:
        b = ex.fetch_ohlcv(sym, tf, since=since, limit=100)
        if not b:
            break
        rows += b
        nxt = b[-1][0] + step
        if nxt <= since or len(b) < 2:
            break
        since = nxt
        if nxt > time.time() * 1000:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts").sort_values("ts")
    df = df.iloc[:-1]  # drop the unfinished last candle
    df.to_csv(fn, index=False)
    return df


def send_telegram(text):
    import requests
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[no telegram secrets set]\n" + text)
        return
    r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", data={"chat_id": chat, "text": text}, timeout=20)
    print("telegram:", r.status_code)


def cmd_backtest(a):
    ex = get_ex()
    syms = top_symbols(ex, a.top)
    print("symbols:", syms)
    data = {}
    for s in syms:
        try:
            data[s] = (fetch_all(ex, s, CFG["htf"], a.days), fetch_all(ex, s, CFG["ltf"], a.days))
        except Exception as e:
            print("skip", s, e)
    md, df = run_backtest(data, CFG)
    head = f"# Backtest {a.days} days, {len(data)} coins\n\nFees+slippage included. Results in R (1R = 1% risk). Past results do not predict the future.\n\n"
    open("result.md", "w").write(head + md + "\n")
    df.to_csv("trades.csv", index=False)
    print(head + md)


def cmd_scan(a):
    ex = get_ex()
    state = json.load(open("sent.json")) if os.path.exists("sent.json") else []
    now = time.time() * 1000
    new = []
    for s in top_symbols(ex, a.top):
        try:
            dh = pd.DataFrame(ex.fetch_ohlcv(s, CFG["htf"], limit=300), columns=["ts", "open", "high", "low", "close", "volume"])
            dl = pd.DataFrame(ex.fetch_ohlcv(s, CFG["ltf"], limit=300), columns=["ts", "open", "high", "low", "close", "volume"])
            dh = dh[dh.ts + tfms(CFG["htf"]) <= now]
            dl = dl[dl.ts + tfms(CFG["ltf"]) <= now]
            H, L = Bars(dh.reset_index(drop=True), CFG["htf"], CFG["swing_htf"]), Bars(dl.reset_index(drop=True), CFG["ltf"], CFG["swing_ltf"])
            for g in market_signals(H, L, CFG):
                if now - g["t_in"] > 35 * 60000:
                    continue
                key = f"{s}|{g['side']}|{g['t_in']}"
                if key in state:
                    continue
                state.append(key)
                rr = abs(g["tp"] - g["entry"]) / g["R"]
                new.append(f"{'LONG' if g['side'] == 1 else 'SHORT'} {s}\nentry ~ {g['entry']:.6g}\nSL {g['sl']:.6g}\nTP {g['tp']:.6g} (RR {rr:.1f})\nzone {g['zlo']:.6g}-{g['zhi']:.6g} [{g['types']}]\nconfirmations: {g['why']}\n(signal only, check the chart and your rules)")
        except Exception as e:
            print("skip", s, e)
    for m in new:
        send_telegram(m)
    json.dump(state[-500:], open("sent.json", "w"))
    print(f"{len(new)} new signals")
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        send_telegram(f"Bot is connected. Manual scan finished: {len(new)} new signals.")


# ----------------------------- offline self test -----------------------------

def synth(days, seed):
    rng = np.random.default_rng(seed)
    n = days * 96
    vol = 0.0015 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.3)
    drift = np.sin(np.arange(n) / 700.0) * 0.0004
    ret = drift + rng.normal(0, 1, n) * vol
    close = 100 * np.exp(np.cumsum(ret))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, vol / 2)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, vol / 2)))
    v = rng.lognormal(3, 0.5, n) * (1 + 3 * np.abs(ret) / vol.mean())
    ts = (1_700_000_000_000 // 900000) * 900000 + np.arange(n) * 900000
    l = pd.DataFrame(dict(ts=ts, open=op, high=hi, low=lo, close=close, volume=v))
    ts_h = ts // 3600000 * 3600000
    h = l.groupby(ts_h).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
    h = h.reset_index().rename(columns={"index": "ts"})
    h.columns = ["ts", "open", "high", "low", "close", "volume"]
    return h, l


def cmd_selftest(a):
    t0 = time.time()
    data = {f"SYN{i}": synth(120, i) for i in range(3)}
    md, df = run_backtest(data, CFG)
    print(md)
    print("trades rows:", len(df), "seconds:", round(time.time() - t0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("backtest")
    b.add_argument("--days", type=int, default=730)
    b.add_argument("--top", type=int, default=10)
    s = sp.add_parser("scan")
    s.add_argument("--top", type=int, default=20)
    sp.add_parser("selftest")
    a = ap.parse_args()
    {"backtest": cmd_backtest, "scan": cmd_scan, "selftest": cmd_selftest}[a.cmd](a)
