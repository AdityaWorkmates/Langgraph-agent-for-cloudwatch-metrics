
import matplotlib.pyplot as plt
import json
from datetime import datetime, timezone
from typing import Any, List, Dict
import io
import base64
import matplotlib
matplotlib.use("Agg")


def _extract_json_from_text(text: str):
    if not text:
        return None
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            if "\n" in block:
                first_line, rest = block.split("\n", 1)
                lang = first_line.strip().lower()
                candidate = rest if lang in (
                    "json", "javascript", "js") else block
            else:
                candidate = block
            candidate = candidate.strip()
            try:
                return json.loads(candidate)
            except Exception:
                continue
    start = text.find("{")
    if start != -1:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    return None


def _parse_timestamp(value):
    """
    Try numeric (epoch seconds or ms) or ISO-like strings.
    Returns a datetime or None.
    """
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            if value > 1e12:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            if value > 1e9:
                return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str) and value.isdigit():
            n = int(value)
            if n > 1e12:
                return datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
            if n > 1e9:
                return datetime.fromtimestamp(n, tz=timezone.utc)
    except Exception:
        pass

    if isinstance(value, str):
        s = value.strip()
        try:
            if s.endswith("Z"):
                s2 = s.replace("Z", "+00:00")
                return datetime.fromisoformat(s2)
            return datetime.fromisoformat(s)
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def _is_number(v):
    return isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit())


def _extract_from_metric_data_by_range(payload: dict) -> list:
    """Extracts time series from the 'metric_data_by_range' shape."""
    series = []
    if isinstance(payload, dict) and isinstance(payload.get("metric_data_by_range"), dict):
        ranges = payload.get("metric_data_by_range") or {}
        for range_name, range_obj in ranges.items():
            datapoints = (range_obj or {}).get("datapoints") or []
            if not isinstance(datapoints, list) or not datapoints:
                continue
            times = []
            values = []
            for dp in datapoints:
                if not isinstance(dp, dict):
                    continue
                t = _parse_timestamp(dp.get("timestamp"))
                v = dp.get("value")
                try:
                    v = float(v) if _is_number(v) else None
                except Exception:
                    v = None
                times.append(t)
                values.append(v)
            series.append({"name": f"cpu_{range_name}", "times": times, "values": values})
    return series

def _extract_from_timestamps_and_values(payload: dict) -> list:
    """Extracts time series from the 'timestamps' and 'values' shape."""
    series = []
    if isinstance(payload, dict):
        if "timestamps" in payload and "values" in payload:
            ts = payload.get("timestamps", [])
            vs = payload.get("values", [])
            if isinstance(ts, list) and isinstance(vs, list) and len(ts) == len(vs):
                times = [_parse_timestamp(t) for t in ts]
                values = [float(v) if _is_number(v) else None for v in vs]
                series.append({"name": "series", "times": times, "values": values})
    return series

def _extract_from_list_of_dicts(payload: list) -> list:
    """Extracts time series from a list of dictionaries."""
    series = []
    if isinstance(payload, list) and payload and all(isinstance(x, dict) for x in payload):
        ts_keys = ("timestamp", "time", "date", "t", "ts")
        val_keys = None
        for d in payload:
            for k, v in d.items():
                if k.lower() in ts_keys:
                    continue
                if _is_number(v):
                    val_keys = k
                    break
            if val_keys:
                break
        if val_keys:
            times = []
            values = []
            for d in payload:
                t = None
                for tk in ts_keys:
                    if tk in d:
                        t = _parse_timestamp(d[tk])
                        break
                if t is None:
                    for k in d:
                        if "date" in k.lower() or "time" in k.lower() or k.lower().endswith("_at"):
                            t = _parse_timestamp(d[k])
                            if t:
                                break
                times.append(t)
                v = d.get(val_keys)
                try:
                    values.append(float(v) if _is_number(v) else None)
                except Exception:
                    values.append(None)
            series.append({"name": val_keys, "times": times, "values": values})
    return series

def _extract_from_nested_metrics(payload: dict) -> list:
    """Extracts time series from a nested metrics shape."""
    series = []
    if isinstance(payload, dict) and "metrics" in payload and isinstance(payload["metrics"], list):
        for m in payload["metrics"]:
            if isinstance(m, dict):
                points = m.get("points") or m.get("data") or m.get("values")
                if isinstance(points, list) and points:
                    times = []
                    values = []
                    for p in points:
                        if isinstance(p, dict):
                            t = p.get("t") or p.get("timestamp") or p.get("time") or p.get("date")
                            v = p.get("v") or p.get("value") or p.get("val")
                            times.append(_parse_timestamp(t))
                            try:
                                values.append(float(v) if _is_number(v) else None)
                            except Exception:
                                values.append(None)
                        elif isinstance(p, list) and len(p) >= 2:
                            times.append(_parse_timestamp(p[0]))
                            try:
                                values.append(float(p[1]) if _is_number(p[1]) else None)
                            except Exception:
                                values.append(None)
                    name = m.get("name") or m.get("metric") or "metric"
                    series.append({"name": name, "times": times, "values": values})
    return series

def extract_time_series(payload: Any) -> List[Dict[str, Any]]:
    """
    Heuristic extractor that returns a list of time-series dicts:
    [{'name': name, 'times': [datetime, ...], 'values': [num, ...]}, ...]
    It tries several common payload shapes.
    """
    series = []
    series.extend(_extract_from_metric_data_by_range(payload))
    series.extend(_extract_from_timestamps_and_values(payload))
    series.extend(_extract_from_list_of_dicts(payload))
    series.extend(_extract_from_nested_metrics(payload))

    cleaned = []
    for s in series:
        pts = [(t, v) for t, v in zip(s.get("times", []), s.get("values", [])) if t is not None and v is not None]
        if len(pts) >= 2:
            times, values = zip(*pts)
            cleaned.append({"name": s.get("name", "series"), "times": list(times), "values": list(values)})
    return cleaned


def _plot_series_to_base64(times: List[datetime], values: List[float], title: str = None) -> str:
    """
    Plot the series and return a data: URI containing base64 PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(times, values, marker="o", linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    if title:
        ax.set_title(title)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"
