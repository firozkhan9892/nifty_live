import json, os, logging
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.ERROR)

IST = timezone(timedelta(hours=5, minutes=30))
JOURNAL_PATH = os.path.join(os.path.dirname(__file__), "trade_journal.json")

def save_signal(entry_price, direction, confidence, obv_dir, rr_ratio, sl, target, vol_status, conflict="LOW"):
    now = datetime.now(IST)
    data = _load()
    trade = {
        "id": len(data["trades"]) + 1,
        "time": now.strftime("%Y-%m-%d %I:%M %p"),
        "date": now.strftime("%Y-%m-%d"),
        "direction": direction,
        "entry": round(entry_price, 2),
        "sl": round(sl, 2),
        "target": round(target, 2),
        "rr_ratio": round(rr_ratio, 2),
        "confidence": round(confidence, 1),
        "conflict": conflict,
        "obv": obv_dir,
        "vol_status": vol_status,
        "status": "pending",
        "exit_price": None,
        "pnl_pct": None,
        "target_hit": False,
        "sl_hit": False,
        "mfe": None,
        "mae": None,
        "max_fav_move": None,
    }
    data["trades"].append(trade)
    data["pending"] += 1
    data["total"] += 1
    _save(data)
    return trade["id"]

def update_trade(trade_id, high_price=None, low_price=None, current_price=None):
    data = _load()
    for t in data["trades"]:
        if t["id"] != trade_id or t["status"] != "pending":
            continue
        entry = t["entry"]
        is_buy = t["direction"] == "BUY"
        if current_price is not None:
            if is_buy:
                move = (current_price - entry) / entry * 100
            else:
                move = (entry - current_price) / entry * 100
            if t["mfe"] is None or move > t["mfe"]:
                t["mfe"] = round(move, 2)
            if t["mae"] is None or move < t["mae"]:
                t["mae"] = round(move, 2)
        if high_price is not None:
            if is_buy:
                fav = (high_price - entry) / entry * 100
                adv = (low_price - entry) / entry * 100 if low_price else 0
            else:
                fav = (entry - low_price) / entry * 100 if low_price else 0
                adv = (entry - high_price) / entry * 100
            if t["mfe"] is None or fav > t["mfe"]:
                t["mfe"] = round(fav, 2)
            if t["mae"] is None or adv < t["mae"]:
                t["mae"] = round(adv, 2)
        _save(data)
        return True
    return False

def check_trade(trade_id, current_price):
    data = _load()
    for t in data["trades"]:
        if t["id"] != trade_id or t["status"] != "pending":
            continue
        entry = t["entry"]
        is_buy = t["direction"] == "BUY"
        if is_buy:
            if current_price >= t["target"]:
                t["status"] = "correct"
                t["target_hit"] = True
                t["exit_price"] = round(t["target"], 2)
                t["pnl_pct"] = round((t["target"] - entry) / entry * 100, 2)
            elif current_price <= t["sl"]:
                t["status"] = "wrong"
                t["sl_hit"] = True
                t["exit_price"] = round(t["sl"], 2)
                t["pnl_pct"] = round((t["sl"] - entry) / entry * 100, 2)
            else:
                return False
        else:
            if current_price <= t["target"]:
                t["status"] = "correct"
                t["target_hit"] = True
                t["exit_price"] = round(t["target"], 2)
                t["pnl_pct"] = round((entry - t["target"]) / entry * 100, 2)
            elif current_price >= t["sl"]:
                t["status"] = "wrong"
                t["sl_hit"] = True
                t["exit_price"] = round(t["sl"], 2)
                t["pnl_pct"] = round((entry - t["sl"]) / entry * 100, 2)
            else:
                return False
        _update_counts(data)
        _save(data)
        return True
    return False

def close_day(close_price):
    data = _load()
    today = datetime.now(IST).strftime("%Y-%m-%d")
    changed = False
    for t in data["trades"]:
        if t["status"] != "pending":
            continue
        trade_date = t.get("date", t["time"][:10])
        if trade_date != today:
            continue
        entry = t["entry"]
        is_buy = t["direction"] == "BUY"
        pnl = (close_price - entry) / entry * 100 if is_buy else (entry - close_price) / entry * 100
        t["status"] = "correct" if pnl > 0 else "wrong"
        t["exit_price"] = round(close_price, 2)
        t["pnl_pct"] = round(pnl, 2)
        changed = True
    if changed:
        _update_counts(data)
        _save(data)
    return changed

def daily_report():
    data = _load()
    today = datetime.now(IST).strftime("%Y-%m-%d")
    today_trades = [t for t in data["trades"] if t.get("date", t["time"][:10]) == today]
    if not today_trades:
        return None
    total = len(today_trades)
    executed = sum(1 for t in today_trades if t["status"] != "pending")
    wins = sum(1 for t in today_trades if t["status"] == "correct")
    losses = sum(1 for t in today_trades if t["status"] == "wrong")
    pending = sum(1 for t in today_trades if t["status"] == "pending")
    pnl = sum(t["pnl_pct"] for t in today_trades if t["pnl_pct"] is not None)
    return {
        "period": today,
        "signals": total,
        "executed": executed,
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "pnl": round(pnl, 2),
    }

def weekly_report():
    data = _load()
    today = datetime.now(IST)
    week_ago = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    week_trades = [t for t in data["trades"] if t.get("date", t["time"][:10]) >= week_ago]
    if not week_trades:
        return None
    executed = [t for t in week_trades if t["status"] in ("correct", "wrong")]
    wins = sum(1 for t in executed if t["status"] == "correct")
    pnl = sum(t["pnl_pct"] for t in executed if t["pnl_pct"] is not None)
    return {
        "period": f"{week_ago} to {today.strftime('%Y-%m-%d')}",
        "signals": len(week_trades),
        "executed": len(executed),
        "wins": wins,
        "losses": len(executed) - wins if executed else 0,
        "pending": sum(1 for t in week_trades if t["status"] == "pending"),
        "pnl": round(pnl, 2),
    }

def monthly_report():
    data = _load()
    this_month = datetime.now(IST).strftime("%Y-%m")
    month_trades = [t for t in data["trades"] if t.get("date", t["time"][:10]).startswith(this_month)]
    if not month_trades:
        return None
    executed = [t for t in month_trades if t["status"] in ("correct", "wrong")]
    wins = sum(1 for t in executed if t["status"] == "correct")
    pnl = sum(t["pnl_pct"] for t in executed if t["pnl_pct"] is not None)
    return {
        "period": this_month,
        "signals": len(month_trades),
        "executed": len(executed),
        "wins": wins,
        "losses": len(executed) - wins if executed else 0,
        "pending": sum(1 for t in month_trades if t["status"] == "pending"),
        "pnl": round(pnl, 2),
    }

def all_time_report():
    data = _load()
    executed = [t for t in data["trades"] if t["status"] in ("correct", "wrong")]
    if not executed:
        return None
    wins = sum(1 for t in executed if t["status"] == "correct")
    pnl = sum(t["pnl_pct"] for t in executed if t["pnl_pct"] is not None)
    return {
        "signals": data.get("total", len(data["trades"])),
        "executed": len(executed),
        "wins": wins,
        "losses": len(executed) - wins,
        "pending": data.get("pending", sum(1 for t in data["trades"] if t["status"] == "pending")),
        "pnl": round(pnl, 2),
    }

def print_report(period="daily"):
    if period == "daily":
        r = daily_report()
        label = "Daily"
    elif period == "weekly":
        r = weekly_report()
        label = "Weekly"
    elif period == "monthly":
        r = monthly_report()
        label = "Monthly"
    else:
        r = all_time_report()
        label = "All-Time"

    if r is None:
        print(f"  No trades for {label.lower()} period.")
        return
    wr = r["wins"] / r["executed"] * 100 if r["executed"] > 0 else 0
    print(f"  {label} Report: {r.get('period', '')}")
    print(f"  {'='*40}")
    print(f"  Signals Generated:  {r['signals']}")
    print(f"  Executed:           {r['executed']}")
    print(f"  Wins:               {r['wins']}")
    print(f"  Losses:             {r['losses']}")
    print(f"  Pending:            {r['pending']}")
    print(f"  Win Rate:           {wr:.1f}%")
    print(f"  P&L:                {r['pnl']:+.2f}%")

def print_journal():
    data = _load()
    if data["total"] == 0:
        print("  No trades in journal.")
        return
    wr = data["correct"] / (data["correct"] + data["wrong"]) * 100 if (data["correct"] + data["wrong"]) > 0 else 0
    print(f"  Total: {data['total']} | Correct: {data['correct']} | Wrong: {data['wrong']} | Pending: {data['pending']} | Win Rate: {wr:.1f}%")
    print(f"  {'='*90}")
    h = f"  {'#':<4} {'Date':<12} {'Dir':<6} {'Entry':<10} {'Exit':<10} {'P&L%':<8} {'Target':<8} {'SL':<8} {'MFE':<7} {'MAE':<7} {'Conflict':<8} {'Status':<8}"
    print(h)
    print(f"  {'-'*98}")
    for t in reversed(data["trades"][-30:]):
        exit_p = t.get("exit_price")
        pnl = t.get("pnl_pct")
        ep = f"{exit_p:.2f}" if exit_p else ""
        ps = f"{pnl:+.2f}%" if pnl else ""
        mfe_s = f"{t['mfe']:.2f}%" if t.get("mfe") is not None else ""
        mae_s = f"{t['mae']:.2f}%" if t.get("mae") is not None else ""
        icon = "WIN" if t["status"] == "correct" else ("LOSS" if t["status"] == "wrong" else "?")
        cfl = t.get("conflict", "")
        print(f"  {t['id']:<4} {t['date']:<12} {t['direction']:<6} {t['entry']:<10.2f} {ep:<10} {ps:<8} {t['target']:<8.2f} {t['sl']:<8.2f} {mfe_s:<7} {mae_s:<7} {cfl:<8} {icon:<8}")

def _load():
    if os.path.exists(JOURNAL_PATH):
        try:
            with open(JOURNAL_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            import shutil
            backup = JOURNAL_PATH + ".bak"
            shutil.copy2(JOURNAL_PATH, backup)
            print(f"Corrupted journal {JOURNAL_PATH}, backed up to {backup}: {e}")
    return {"total": 0, "correct": 0, "wrong": 0, "pending": 0, "trades": []}

def save_last_signal(state):
    data = _load()
    for k, v in state.items():
        data[f"_last_{k}"] = v
    _save(data)

def _save(data):
    import tempfile
    tmp = JOURNAL_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    import shutil
    shutil.move(tmp, JOURNAL_PATH)

def _update_counts(data):
    data["correct"] = sum(1 for t in data["trades"] if t["status"] == "correct")
    data["wrong"] = sum(1 for t in data["trades"] if t["status"] == "wrong")
    data["pending"] = sum(1 for t in data["trades"] if t["status"] == "pending")

def check_pending_trades(high, low):
    """Check pending trades against today's high/low. Returns list of updated trade IDs."""
    data = _load()
    updated = []
    for t in data["trades"]:
        if t["status"] != "pending":
            continue
        is_buy = t["direction"] == "BUY"
        if is_buy:
            if high >= t["target"]:
                t["status"] = "correct"
                t["target_hit"] = True
                t["exit_price"] = round(t["target"], 2)
                t["pnl_pct"] = round((t["target"] - t["entry"]) / t["entry"] * 100, 2)
                updated.append(t["id"])
            elif low <= t["sl"]:
                t["status"] = "wrong"
                t["sl_hit"] = True
                t["exit_price"] = round(t["sl"], 2)
                t["pnl_pct"] = round((t["sl"] - t["entry"]) / t["entry"] * 100, 2)
                updated.append(t["id"])
        else:
            if low <= t["target"]:
                t["status"] = "correct"
                t["target_hit"] = True
                t["exit_price"] = round(t["target"], 2)
                t["pnl_pct"] = round((t["entry"] - t["target"]) / t["entry"] * 100, 2)
                updated.append(t["id"])
            elif high >= t["sl"]:
                t["status"] = "wrong"
                t["sl_hit"] = True
                t["exit_price"] = round(t["sl"], 2)
                t["pnl_pct"] = round((t["entry"] - t["sl"]) / t["entry"] * 100, 2)
                updated.append(t["id"])
    if updated:
        _update_counts(data)
        _save(data)
    return updated
