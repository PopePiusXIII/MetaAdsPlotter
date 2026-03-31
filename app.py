from __future__ import annotations

import datetime
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".metaads.env")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.adsinsights import AdsInsights

CACHE_DIR = Path.home() / ".metaads_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_filename(account_id: str, date_start: str, date_stop: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = account_id.replace("act_", "")
    return f"{safe_id}_{date_start}_{date_stop}_{ts}.pkl"


def _save_cache(
    df: pd.DataFrame,
    freq_data: dict,
    account_id: str,
    date_start: str,
    date_stop: str,
) -> Path:
    payload = {
        "df": df,
        "freq_data": freq_data,
        "account_id": account_id,
        "date_start": date_start,
        "date_stop": date_stop,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    path = CACHE_DIR / _cache_filename(account_id, date_start, date_stop)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _load_cache(path: Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _list_caches() -> list[Path]:
    files = sorted(CACHE_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def fetch_from_api(
    access_token: str,
    ad_account_id: str,
    date_start: str,
    date_stop: str,
) -> pd.DataFrame:
    """Fetch ad insights from the Meta Marketing API and return a DataFrame."""
    FacebookAdsApi.init(access_token=access_token)

    account_id = ad_account_id if ad_account_id.startswith("act_") else f"act_{ad_account_id}"
    account = AdAccount(account_id)

    fields = [
        AdsInsights.Field.ad_id,
        AdsInsights.Field.ad_name,
        AdsInsights.Field.adset_id,
        AdsInsights.Field.adset_name,
        AdsInsights.Field.campaign_id,
        AdsInsights.Field.campaign_name,
        AdsInsights.Field.date_start,
        AdsInsights.Field.date_stop,
        AdsInsights.Field.reach,
        AdsInsights.Field.frequency,
        AdsInsights.Field.spend,
        AdsInsights.Field.impressions,
        AdsInsights.Field.cpm,
        AdsInsights.Field.cpc,
        AdsInsights.Field.ctr,
        AdsInsights.Field.clicks,
        AdsInsights.Field.actions,
        AdsInsights.Field.cost_per_action_type,
        AdsInsights.Field.video_thruplay_watched_actions,
        AdsInsights.Field.video_p25_watched_actions,
        AdsInsights.Field.video_p50_watched_actions,
        AdsInsights.Field.video_p75_watched_actions,
        AdsInsights.Field.video_p95_watched_actions,
        AdsInsights.Field.video_play_actions,
        AdsInsights.Field.purchase_roas,
        AdsInsights.Field.unique_link_clicks_ctr,
        AdsInsights.Field.unique_clicks,
        AdsInsights.Field.unique_ctr,
    ]

    params = {
        "time_range": {"since": date_start, "until": date_stop},
        "time_increment": 1,   # one row per day per ad
        "level": "ad",
        "limit": 500,
        "action_attribution_windows": ["7d_click", "1d_view"],
    }

    try:
        insights_cursor = account.get_insights(fields=fields, params=params)
        insights = list(insights_cursor)
    except Exception as exc:
        st.error(f"Meta API error: {exc}")
        return pd.DataFrame()

    if not insights:
        st.warning("The API returned no data for the selected date range.")
        return pd.DataFrame()

    def _get_action(actions_list: list | None, action_type: str) -> float | None:
        if not actions_list:
            return None
        for item in actions_list:
            if item.get("action_type") == action_type:
                return float(item.get("value", 0))
        return None

    def _get_action_window(actions_list: list | None, action_type: str, window: str) -> float | None:
        """Return the action count for a specific attribution window (e.g. '7d_click', '1d_view')."""
        if not actions_list:
            return None
        for item in actions_list:
            if item.get("action_type") == action_type:
                val = item.get(window)
                return float(val) if val is not None else None
        return None

    def _get_cost_per_action(cpa_list: list | None, action_type: str) -> float | None:
        if not cpa_list:
            return None
        for item in cpa_list:
            if item.get("action_type") == action_type:
                return float(item.get("value", 0))
        return None

    rows = []
    for ins in insights:
        d = dict(ins)
        actions_list = d.get("actions") or []
        cpa_list = d.get("cost_per_action_type") or []
        video_thru = d.get("video_thruplay_watched_actions") or []
        video_p25 = d.get("video_p25_watched_actions") or []
        video_p50 = d.get("video_p50_watched_actions") or []
        video_p75 = d.get("video_p75_watched_actions") or []
        video_p95 = d.get("video_p95_watched_actions") or []
        video_plays = d.get("video_play_actions") or []
        purchase_roas = d.get("purchase_roas") or []

        link_clicks = _get_action(actions_list, "link_click")
        landing_page_views = _get_action(actions_list, "landing_page_view")
        purchases = _get_action(actions_list, "offsite_conversion.fb_pixel_purchase")
        purchases_7d_click = _get_action_window(actions_list, "offsite_conversion.fb_pixel_purchase", "7d_click")
        purchases_1d_view = _get_action_window(actions_list, "offsite_conversion.fb_pixel_purchase", "1d_view")
        video_3s = (float(video_plays[0]["value"]) if video_plays else None)
        thruplays = (float(video_thru[0]["value"]) if video_thru else None)
        cost_per_thruplay = _get_cost_per_action(
            d.get("cost_per_action_type"), "video_thruplay_watched"
        )
        roas_value = float(purchase_roas[0]["value"]) if purchase_roas else None

        row = {
            "Ad name": d.get("ad_name"),
            "Ad set name": d.get("adset_name"),
            "Campaign name": d.get("campaign_name"),
            "Ad delivery": "Active",          # API only returns active-period rows
            "Reporting starts": d.get("date_start"),
            "Reporting ends": d.get("date_stop"),
            "Reach": float(d["reach"]) if d.get("reach") else None,
            "Frequency": float(d["frequency"]) if d.get("frequency") else None,
            "Amount spent (USD)": float(d["spend"]) if d.get("spend") else None,
            "Impressions": float(d["impressions"]) if d.get("impressions") else None,
            "CPM (cost per 1,000 impressions) (USD)": float(d["cpm"]) if d.get("cpm") else None,
            "CPC (cost per link click) (USD)": float(d["cpc"]) if d.get("cpc") else None,
            "CTR (link click-through rate)": float(d["ctr"]) if d.get("ctr") else None,
            "Clicks (all)": float(d["clicks"]) if d.get("clicks") else None,
            "CTR (all)": float(d.get("unique_ctr", 0) or 0) or None,
            "CPC (all) (USD)": float(d["cpc"]) if d.get("cpc") else None,
            "Link clicks": link_clicks,
            "shop_clicks": link_clicks,   # best proxy available without custom event
            "Landing page views": landing_page_views,
            "Cost per landing page view (USD)": _get_cost_per_action(cpa_list, "landing_page_view"),
            "Unique link clicks": float(d["unique_clicks"]) if d.get("unique_clicks") else None,
            "Unique clicks (all)": float(d["unique_clicks"]) if d.get("unique_clicks") else None,
            "3-second video plays": video_3s,
            "Cost per 3-second video play (USD)": _get_cost_per_action(cpa_list, "video_view"),
            "ThruPlays": thruplays,
            "Cost per ThruPlay (USD)": cost_per_thruplay,
            "Video plays": video_3s,
            "Video plays at 25%": float(video_p25[0]["value"]) if video_p25 else None,
            "Video plays at 50%": float(video_p50[0]["value"]) if video_p50 else None,
            "Video plays at 75%": float(video_p75[0]["value"]) if video_p75 else None,
            "Video plays at 95%": float(video_p95[0]["value"]) if video_p95 else None,
            "Photo clicks": _get_action(actions_list, "photo_view"),
            "Purchases": purchases,
            "Results (7d_click)": purchases_7d_click,
            "Results (1d_view)": purchases_1d_view,
            "Purchase ROAS (return on ad spend)": roas_value,
            "Results": purchases,
            "Cost per results": _get_cost_per_action(cpa_list, "offsite_conversion.fb_pixel_purchase"),
            "Ad set budget": None,           # requires separate adset call; left blank
            "Viewers": float(d["reach"]) if d.get("reach") else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Reporting starts"] = pd.to_datetime(df["Reporting starts"], errors="coerce")
    df["Reporting ends"] = pd.to_datetime(df["Reporting ends"], errors="coerce")
    return df


def _safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
        safe_denominator = denominator.where(denominator > 0)
        return numerator.div(safe_denominator)
    if denominator and denominator > 0:
        return numerator / denominator
    return pd.NA


def fetch_frequency_breakdowns(
    access_token: str,
    ad_account_id: str,
    date_start: str,
    date_stop: str,
) -> dict:
    """Fetch true deduplicated reach/frequency via cumulative growing-window API calls.

    For every day D between date_start and date_stop we query the API with
    time_range={since: date_start, until: D} so that Meta returns the true
    deduplicated reach for the window day-1 → day-D at each level.

    Returns a dict keyed by:
        account_daily      – account-level reach/frequency per individual day (for daily summary)
        ad_alldays         – per-ad reach/frequency over the full date range (for ad summary)
        account_cumulative – cumulative account-level reach/freq for each end-date
        adset_cumulative   – cumulative adset-level reach/freq for each end-date
        ad_cumulative      – cumulative ad-level reach/freq for each end-date
    """
    FacebookAdsApi.init(access_token=access_token)
    account_id = (
        ad_account_id if ad_account_id.startswith("act_") else f"act_{ad_account_id}"
    )
    account = AdAccount(account_id)

    base_fields = [
        AdsInsights.Field.date_start,
        AdsInsights.Field.date_stop,
        AdsInsights.Field.reach,
        AdsInsights.Field.impressions,
        AdsInsights.Field.frequency,
        AdsInsights.Field.spend,
    ]
    ad_name_field = AdsInsights.Field.ad_name
    adset_name_field = AdsInsights.Field.adset_name
    full_range = {"since": date_start, "until": date_stop}

    def _raw_fetch(level: str, time_increment, time_range: dict, extra_fields=None, label: str = "") -> list:
        fields = base_fields + (extra_fields or [])
        params = {
            "time_range": time_range,
            "time_increment": time_increment,
            "level": level,
            "limit": 500,
        }
        try:
            return list(account.get_insights(fields=fields, params=params))
        except Exception as exc:
            st.warning(f"Meta API: could not fetch {label} data: {exc}")
            return []

    def _to_df(raw: list, group_cols: dict) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame()
        rows = []
        for ins in raw:
            d = dict(ins)
            row: dict = {out: d.get(api) for out, api in group_cols.items()}
            row["date_start"] = pd.Timestamp(d["date_start"]) if d.get("date_start") else pd.NaT
            row["date_stop"] = pd.Timestamp(d["date_stop"]) if d.get("date_stop") else pd.NaT
            row["reach"] = float(d["reach"]) if d.get("reach") else None
            row["impressions"] = float(d["impressions"]) if d.get("impressions") else None
            row["frequency"] = float(d["frequency"]) if d.get("frequency") else None
            row["spend"] = float(d.get("spend", 0) or 0)
            rows.append(row)
        return pd.DataFrame(rows)

    # ── static fetches (unchanged from before) ────────────────────────────────
    account_daily = _to_df(
        _raw_fetch("account", 1, full_range, label="account-daily"), {}
    )
    ad_alldays = _to_df(
        _raw_fetch("ad", "all_days", full_range, [ad_name_field, adset_name_field], "ad-alldays"),
        {"Ad name": "ad_name", "Ad set name": "adset_name"},
    )

    # ── cumulative growing-window fetches ─────────────────────────────────────
    # For each end-date D: query since=date_start until=D to get the true
    # deduplicated reach across day-1 through day-D.
    start_dt = datetime.date.fromisoformat(date_start)
    stop_dt = datetime.date.fromisoformat(date_stop)
    num_days = (stop_dt - start_dt).days + 1

    acct_rows: list[dict] = []
    adset_rows: list[dict] = []
    ad_rows: list[dict] = []

    progress = st.progress(0, text="Fetching cumulative frequency data…")
    for i, offset in enumerate(range(num_days)):
        end_date = start_dt + datetime.timedelta(days=offset)
        end_str = end_date.isoformat()
        window = {"since": date_start, "until": end_str}

        # Account cumulative (1 row per call)
        for ins in _raw_fetch("account", "all_days", window, label=f"acct-cum-{end_str}"):
            d = dict(ins)
            acct_rows.append({
                "date_stop": pd.Timestamp(end_str),
                "reach": float(d["reach"]) if d.get("reach") else None,
                "impressions": float(d["impressions"]) if d.get("impressions") else None,
                "frequency": float(d["frequency"]) if d.get("frequency") else None,
                "spend": float(d.get("spend", 0) or 0),
            })

        # Adset cumulative (1 row per adset per call)
        for ins in _raw_fetch("adset", "all_days", window, [adset_name_field], f"adset-cum-{end_str}"):
            d = dict(ins)
            adset_rows.append({
                "date_stop": pd.Timestamp(end_str),
                "Ad set name": d.get("adset_name"),
                "reach": float(d["reach"]) if d.get("reach") else None,
                "impressions": float(d["impressions"]) if d.get("impressions") else None,
                "frequency": float(d["frequency"]) if d.get("frequency") else None,
                "spend": float(d.get("spend", 0) or 0),
            })

        # Ad cumulative (1 row per ad per call)
        for ins in _raw_fetch("ad", "all_days", window, [ad_name_field, adset_name_field], f"ad-cum-{end_str}"):
            d = dict(ins)
            ad_rows.append({
                "date_stop": pd.Timestamp(end_str),
                "Ad name": d.get("ad_name"),
                "Ad set name": d.get("adset_name"),
                "reach": float(d["reach"]) if d.get("reach") else None,
                "impressions": float(d["impressions"]) if d.get("impressions") else None,
                "frequency": float(d["frequency"]) if d.get("frequency") else None,
                "spend": float(d.get("spend", 0) or 0),
            })

        progress.progress((i + 1) / num_days, text=f"Fetching cumulative frequency data… day {i + 1}/{num_days}")

    progress.empty()

    return {
        "account_daily": account_daily,
        "ad_alldays": ad_alldays,
        "account_cumulative": pd.DataFrame(acct_rows) if acct_rows else pd.DataFrame(),
        "adset_cumulative": pd.DataFrame(adset_rows) if adset_rows else pd.DataFrame(),
        "ad_cumulative": pd.DataFrame(ad_rows) if ad_rows else pd.DataFrame(),
    }


def add_95_confidence_bounds(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    y_cols: str | list[str],
    rolling_window: int = 14,
    show_ci: bool = True,
    show_anomalies: bool = True,
    interval_mode: str = "context",
    metric_color_map: dict[str, str] | None = None,
    ci_alpha: float = 0.14,
) -> go.Figure:
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    theme_line_colors = ["#00D1FF", "#2E6BFF", "#FF4D6D", "#00C49A", "#A66CFF"]

    def color_with_alpha(color_value: str | None, alpha: float = 0.14) -> str:
        if not color_value:
            return f"rgba(99, 110, 250, {alpha})"

        color_value = str(color_value).strip()

        if color_value.startswith("rgba("):
            channels = color_value[5:-1].split(",")
            if len(channels) >= 3:
                r, g, b = [c.strip() for c in channels[:3]]
                return f"rgba({r}, {g}, {b}, {alpha})"

        if color_value.startswith("rgb("):
            channels = color_value[4:-1].split(",")
            if len(channels) >= 3:
                r, g, b = [c.strip() for c in channels[:3]]
                return f"rgba({r}, {g}, {b}, {alpha})"

        if color_value.startswith("#"):
            hex_color = color_value.lstrip("#")
            if len(hex_color) == 3:
                hex_color = "".join(ch * 2 for ch in hex_color)
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f"rgba({r}, {g}, {b}, {alpha})"

        return f"rgba(99, 110, 250, {alpha})"

    metric_line_colors: dict[str, str | None] = {}
    if metric_color_map:
        for metric_name, metric_color in metric_color_map.items():
            metric_line_colors[metric_name] = metric_color

    for trace in fig.data:
        trace_name = getattr(trace, "name", None)
        if trace_name and trace_name not in metric_line_colors:
            trace_line = getattr(trace, "line", None)
            metric_line_colors[trace_name] = getattr(trace_line, "color", None)

    if x_col not in df.columns:
        return fig

    x_values = df[x_col]
    min_periods = 5 if rolling_window >= 10 else max(3, rolling_window // 2)

    for index, metric in enumerate(y_cols):
        if metric not in df.columns:
            continue

        if not metric_line_colors.get(metric):
            metric_line_colors[metric] = theme_line_colors[index % len(theme_line_colors)]
            for trace in fig.data:
                if getattr(trace, "name", None) == metric:
                    if getattr(trace, "line", None) is not None:
                        trace.line.color = metric_line_colors[metric]
                    else:
                        trace.line = dict(color=metric_line_colors[metric])

        series = pd.to_numeric(df[metric], errors="coerce")
        if series.notna().sum() < min_periods:
            continue

        rolling_mean = series.rolling(rolling_window, min_periods=min_periods).mean()
        rolling_std = series.rolling(rolling_window, min_periods=min_periods).std(ddof=0)
        if interval_mode == "predictive":
            rolling_mean = rolling_mean.shift(1)
            rolling_std = rolling_std.shift(1)

        z_score = 1.96
        upper_bound = rolling_mean + z_score * rolling_std
        lower_bound = rolling_mean - z_score * rolling_std

        color = color_with_alpha(metric_line_colors.get(metric), alpha=ci_alpha)

        if show_ci:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=upper_bound,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{metric} 95% Upper",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=lower_bound,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=color,
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{metric} 95% CI",
                )
            )

        anomaly_mask = (series > upper_bound) | (series < lower_bound)
        if show_anomalies and anomaly_mask.fillna(False).any():
            fig.add_trace(
                go.Scatter(
                    x=x_values[anomaly_mask],
                    y=series[anomaly_mask],
                    mode="markers",
                    marker=dict(color="red", size=9, symbol="x"),
                    name=f"{metric} anomaly",
                )
            )

    return fig


def make_daily_summary(filtered_df: pd.DataFrame, account_daily_df: pd.DataFrame | None = None) -> pd.DataFrame:
    daily = (
        filtered_df.groupby("Reporting starts", as_index=False)
        .agg(
            {
                "Amount spent (USD)": "sum",
                "Results": "sum",
                "Impressions": "sum",
                "Reach": "sum",
                "Link clicks": "sum",
                "Landing page views": "sum",
            }
        )
        .sort_values("Reporting starts")
    )

    daily["Cost per Result"] = _safe_divide(daily["Amount spent (USD)"], daily["Results"])
    if account_daily_df is not None and not account_daily_df.empty:
        acct = account_daily_df[["date_start", "reach", "frequency"]].copy()
        acct = acct.rename(
            columns={"date_start": "Reporting starts", "reach": "Account Reach", "frequency": "Frequency"}
        )
        acct["Reporting starts"] = pd.to_datetime(acct["Reporting starts"])
        daily = daily.merge(acct, on="Reporting starts", how="left")
        daily["Reach"] = daily["Account Reach"].combine_first(daily["Reach"])
        daily = daily.drop(columns=["Account Reach"])
        daily["Frequency"] = daily["Frequency"].combine_first(
            _safe_divide(daily["Impressions"], daily["Reach"])
        )
    else:
        daily["Frequency"] = _safe_divide(daily["Impressions"], daily["Reach"])
    daily["CTR %"] = _safe_divide(daily["Link clicks"], daily["Impressions"]) * 100
    daily["CPM"] = _safe_divide(daily["Amount spent (USD)"] * 1000, daily["Impressions"])
    daily["CPC"] = _safe_divide(daily["Amount spent (USD)"], daily["Link clicks"])
    daily["CVR % (Result/Click)"] = _safe_divide(daily["Results"], daily["Link clicks"]) * 100
    daily["Results per $100"] = _safe_divide(daily["Results"] * 100, daily["Amount spent (USD)"])
    daily["Clicks per $100"] = _safe_divide(daily["Link clicks"] * 100, daily["Amount spent (USD)"])
    daily["LPV per $100"] = _safe_divide(daily["Landing page views"] * 100, daily["Amount spent (USD)"])

    daily["Estimated New Impressions"] = daily[["Reach", "Impressions"]].min(axis=1)
    daily["Estimated Returning Impressions"] = (
        daily["Impressions"] - daily["Estimated New Impressions"]
    ).clip(lower=0)
    daily["Estimated New Spend"] = _safe_divide(
        daily["Estimated New Impressions"], daily["Impressions"]
    ) * daily["Amount spent (USD)"]
    daily["Estimated Returning Spend"] = _safe_divide(
        daily["Estimated Returning Impressions"], daily["Impressions"]
    ) * daily["Amount spent (USD)"]

    daily["Frequency Fatigue Index"] = (
        (daily["Frequency"].fillna(0) - 1).clip(lower=0)
        * _safe_divide(daily["Amount spent (USD)"], daily["Reach"])
    )
    return daily


def make_ad_summary(filtered_df: pd.DataFrame, ad_alldays_df: pd.DataFrame | None = None) -> pd.DataFrame:
    ad_summary = (
        filtered_df.groupby(["Ad name", "Ad delivery"], as_index=False)
        .agg(
            {
                "Amount spent (USD)": "sum",
                "Results": "sum",
                "Impressions": "sum",
                "Reach": "sum",
                "Link clicks": "sum",
                "Landing page views": "sum",
                "Reporting starts": "nunique",
            }
        )
        .rename(columns={"Reporting starts": "Active days"})
    )

    ad_summary["Cost per Result"] = _safe_divide(ad_summary["Amount spent (USD)"], ad_summary["Results"])
    ad_summary["CTR %"] = _safe_divide(ad_summary["Link clicks"], ad_summary["Impressions"]) * 100
    ad_summary["CPM"] = _safe_divide(ad_summary["Amount spent (USD)"] * 1000, ad_summary["Impressions"])
    ad_summary["CPC"] = _safe_divide(ad_summary["Amount spent (USD)"], ad_summary["Link clicks"])
    ad_summary["CVR % (Result/Click)"] = _safe_divide(ad_summary["Results"], ad_summary["Link clicks"]) * 100
    ad_summary["Results per $100"] = _safe_divide(ad_summary["Results"] * 100, ad_summary["Amount spent (USD)"])
    ad_summary["Clicks per $100"] = _safe_divide(ad_summary["Link clicks"] * 100, ad_summary["Amount spent (USD)"])
    ad_summary["LPV per $100"] = _safe_divide(ad_summary["Landing page views"] * 100, ad_summary["Amount spent (USD)"])
    if ad_alldays_df is not None and not ad_alldays_df.empty and "Ad name" in ad_alldays_df.columns:
        freq_lookup = (
            ad_alldays_df[["Ad name", "frequency"]]
            .rename(columns={"frequency": "Frequency"})
        )
        ad_summary = ad_summary.merge(freq_lookup, on="Ad name", how="left")
        missing = ad_summary["Frequency"].isna()
        if missing.any():
            ad_summary.loc[missing, "Frequency"] = _safe_divide(
                ad_summary.loc[missing, "Impressions"], ad_summary.loc[missing, "Reach"]
            )
    else:
        ad_summary["Frequency"] = _safe_divide(ad_summary["Impressions"], ad_summary["Reach"])

    ad_summary["Estimated New Impressions"] = ad_summary[["Reach", "Impressions"]].min(axis=1)
    ad_summary["Estimated Returning Impressions"] = (
        ad_summary["Impressions"] - ad_summary["Estimated New Impressions"]
    ).clip(lower=0)
    ad_summary["Estimated New Spend"] = _safe_divide(
        ad_summary["Estimated New Impressions"], ad_summary["Impressions"]
    ) * ad_summary["Amount spent (USD)"]
    ad_summary["Estimated Returning Spend"] = _safe_divide(
        ad_summary["Estimated Returning Impressions"], ad_summary["Impressions"]
    ) * ad_summary["Amount spent (USD)"]

    ad_summary["Frequency Fatigue Index"] = (
        (ad_summary["Frequency"].fillna(0) - 1).clip(lower=0)
        * _safe_divide(ad_summary["Amount spent (USD)"], ad_summary["Reach"])
    )

    ad_summary["Scale Potential Score"] = (
        ad_summary["Results per $100"].fillna(0) * 0.5
        + ad_summary["CVR % (Result/Click)"].fillna(0) * 0.3
        + ad_summary["CTR %"].fillna(0) * 0.2
    )

    return ad_summary.sort_values("Amount spent (USD)", ascending=False)


def make_rolling_frequency_summary(
    filtered_df: pd.DataFrame,
    group_col: str,
    cumulative_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-(day, group) frequency table.

    "Daily Frequency"      = per-day Impressions / per-day Reach (always present).
    "Cumulative Frequency" = true deduplicated frequency from campaign start to that day,
                             sourced from the API cumulative calls.  Each row's x-value
                             (Reporting starts / date_stop) represents the end of the
                             growing window that started on the first day of the fetch.
    """
    freq_df = (
        filtered_df.groupby(["Reporting starts", group_col], as_index=False)
        .agg({"Impressions": "sum", "Reach": "sum", "Amount spent (USD)": "sum"})
        .sort_values([group_col, "Reporting starts"])
    )

    freq_df["Daily Frequency"] = _safe_divide(freq_df["Impressions"], freq_df["Reach"])

    if cumulative_df is not None and not cumulative_df.empty:
        if group_col in cumulative_df.columns:
            lookup = (
                cumulative_df[["date_stop", group_col, "reach", "impressions", "frequency", "spend"]]
                .copy()
                .rename(columns={
                    "date_stop": "Reporting starts",
                    "reach": "Cumulative Reach",
                    "impressions": "Cumulative Impressions",
                    "frequency": "Cumulative Frequency",
                    "spend": "Cumulative Spend",
                })
            )
            freq_df = freq_df.merge(lookup, on=["Reporting starts", group_col], how="left")
        else:
            # Account-level cumulative has no group column
            lookup = (
                cumulative_df[["date_stop", "reach", "impressions", "frequency", "spend"]]
                .copy()
                .rename(columns={
                    "date_stop": "Reporting starts",
                    "reach": "Cumulative Reach",
                    "impressions": "Cumulative Impressions",
                    "frequency": "Cumulative Frequency",
                    "spend": "Cumulative Spend",
                })
            )
            freq_df = freq_df.merge(lookup, on="Reporting starts", how="left")
    else:
        freq_df["Cumulative Reach"] = None
        freq_df["Cumulative Impressions"] = None
        freq_df["Cumulative Frequency"] = None
        freq_df["Cumulative Spend"] = None

    for _c in ["Daily Frequency", "Cumulative Reach", "Cumulative Impressions",
               "Cumulative Frequency", "Cumulative Spend"]:
        if _c in freq_df.columns:
            freq_df[_c] = pd.to_numeric(freq_df[_c], errors="coerce")

    return freq_df


def make_attribution_health_summary(filtered_df: pd.DataFrame, account_daily_df: pd.DataFrame | None = None):
    """Create daily summary. Attribution split is included only if Attribution setting column is present."""
    has_attr = "Attribution setting" in filtered_df.columns

    attr_daily = None
    if has_attr:
        attr_daily = (
            filtered_df.groupby(["Reporting starts", "Attribution setting"], as_index=False)
            .agg({
                "Results": "sum",
                "Amount spent (USD)": "sum",
                "Impressions": "sum",
                "Reach": "sum",
                "Link clicks": "sum",
            })
            .sort_values(["Reporting starts", "Attribution setting"])
        )
        attr_daily["CPM"] = _safe_divide(attr_daily["Amount spent (USD)"] * 1000, attr_daily["Impressions"])
        attr_daily["CTR %"] = _safe_divide(attr_daily["Link clicks"], attr_daily["Impressions"]) * 100
        attr_daily["Frequency"] = _safe_divide(attr_daily["Impressions"], attr_daily["Reach"])
        attr_daily["ROAS"] = _safe_divide(attr_daily["Results"], attr_daily["Amount spent (USD)"])

    _agg_map = {
        "Impressions": "sum",
        "Reach": "sum",
        "Results": "sum",
        "Amount spent (USD)": "sum",
    }
    for _w_col in ["Results (7d_click)", "Results (1d_view)"]:
        if _w_col in filtered_df.columns:
            _agg_map[_w_col] = "sum"
    daily_agg = (
        filtered_df.groupby("Reporting starts", as_index=False)
        .agg(_agg_map)
    )
    if account_daily_df is not None and not account_daily_df.empty:
        acct = account_daily_df[["date_start", "reach", "frequency"]].copy()
        acct = acct.rename(
            columns={"date_start": "Reporting starts", "reach": "Account Reach", "frequency": "Frequency"}
        )
        acct["Reporting starts"] = pd.to_datetime(acct["Reporting starts"])
        daily_agg = daily_agg.merge(acct, on="Reporting starts", how="left")
        daily_agg["Reach"] = daily_agg["Account Reach"].combine_first(daily_agg["Reach"])
        daily_agg = daily_agg.drop(columns=["Account Reach"])
        daily_agg["Frequency"] = daily_agg["Frequency"].combine_first(
            _safe_divide(daily_agg["Impressions"], daily_agg["Reach"])
        )
    else:
        daily_agg["Frequency"] = _safe_divide(daily_agg["Impressions"], daily_agg["Reach"])

    return daily_agg, attr_daily


def make_attribution_ad_summary(filtered_df: pd.DataFrame, ad_alldays_df: pd.DataFrame | None = None):
    """Create per-ad summary with frequency, ROAS, and scaling analysis."""
    ad_agg = (
        filtered_df.groupby(["Ad name", "Ad delivery"], as_index=False)
        .agg({
            "Impressions": "sum",
            "Reach": "sum",
            "Amount spent (USD)": "sum",
            "Results": "sum",
            "Link clicks": "sum",
        })
    )
    if ad_alldays_df is not None and not ad_alldays_df.empty and "Ad name" in ad_alldays_df.columns:
        freq_lookup = (
            ad_alldays_df[["Ad name", "frequency"]]
            .rename(columns={"frequency": "Frequency"})
        )
        ad_agg = ad_agg.merge(freq_lookup, on="Ad name", how="left")
        missing = ad_agg["Frequency"].isna()
        if missing.any():
            ad_agg.loc[missing, "Frequency"] = _safe_divide(
                ad_agg.loc[missing, "Impressions"], ad_agg.loc[missing, "Reach"]
            )
    else:
        ad_agg["Frequency"] = _safe_divide(ad_agg["Impressions"], ad_agg["Reach"])
    ad_agg["CPM"] = _safe_divide(ad_agg["Amount spent (USD)"] * 1000, ad_agg["Impressions"])
    ad_agg["CTR %"] = _safe_divide(ad_agg["Link clicks"], ad_agg["Impressions"]) * 100
    ad_agg["ROAS"] = _safe_divide(ad_agg["Results"], ad_agg["Amount spent (USD)"])
    ad_agg["Scaling Efficiency Index"] = _safe_divide(ad_agg["ROAS"], ad_agg["Frequency"])

    return ad_agg


def apply_metric_colors(fig: go.Figure, metric_color_map: dict[str, str]) -> go.Figure:
    for trace in fig.data:
        trace_name = getattr(trace, "name", None)
        if trace_name in metric_color_map:
            trace_color = metric_color_map[trace_name]
            if getattr(trace, "line", None) is not None:
                trace.line.color = trace_color
            else:
                trace.line = dict(color=trace_color)
            if getattr(trace, "marker", None) is not None:
                trace.marker.color = trace_color
    return fig


def make_kpi_row(filtered_df: pd.DataFrame, daily: pd.DataFrame) -> None:
    total_spend = filtered_df["Amount spent (USD)"].sum(skipna=True)
    total_results = filtered_df["Results"].sum(skipna=True)
    total_impressions = filtered_df["Impressions"].sum(skipna=True)
    total_clicks = filtered_df["Link clicks"].sum(skipna=True)

    avg_cost_per_result = _safe_divide(total_spend, total_results)
    link_ctr = _safe_divide(total_clicks, total_impressions)
    results_per_100 = _safe_divide(total_results * 100, total_spend)

    cpr_volatility = daily["Cost per Result"].std(skipna=True)
    cpr_mean = daily["Cost per Result"].mean(skipna=True)
    cpr_cv = _safe_divide(cpr_volatility, cpr_mean) if pd.notna(cpr_mean) else pd.NA

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Amount Spent", f"${total_spend:,.2f}")
    col2.metric("Results", f"{total_results:,.0f}")
    col3.metric("Impressions", f"{total_impressions:,.0f}")
    col4.metric("Avg Cost / Result", "-" if pd.isna(avg_cost_per_result) else f"${avg_cost_per_result:,.2f}")
    col5.metric("Link CTR", "-" if pd.isna(link_ctr) else f"{link_ctr * 100:,.2f}%")
    col6.metric("Results per $100", "-" if pd.isna(results_per_100) else f"{results_per_100:,.2f}")

    st.caption(
        "Stability (CPR CV): "
        + ("-" if pd.isna(cpr_cv) else f"{cpr_cv:,.2f}")
        + " | Lower is more stable for scaling"
    )

def main() -> None:
    st.set_page_config(page_title="Meta Ads Dashboard", layout="wide")
    st.title("Meta Ads Performance Dashboard")

    api_token = os.getenv("FB_ACCESS_TOKEN", "")
    api_account_id = os.getenv("FB_ACCOUNT_ID", "")

    if not api_token or not api_account_id:
        st.error("FB_ACCESS_TOKEN and FB_ACCOUNT_ID must be set in ~/.metaads.env")
        st.stop()

    # ── Sidebar: data source ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("Data Source")
        data_source = st.radio("Load from", ["API", "Cache"], horizontal=True, key="data_source")

        if data_source == "API":
            st.subheader("Meta Ads API")
            api_date_start = st.date_input("From", value=None, key="fb_date_start")
            api_date_stop = st.date_input("To", value=None, key="fb_date_stop")
            fetch_api = st.button("Fetch from API", key="fb_fetch")
            load_cache_btn = False
            selected_cache_path: Path | None = None
        else:
            cache_files = _list_caches()
            if cache_files:
                cache_labels = [f.name for f in cache_files]
                selected_cache_label = st.selectbox(
                    "Select cache file",
                    options=cache_labels,
                    key="cache_select",
                )
                selected_cache_path = CACHE_DIR / selected_cache_label
                load_cache_btn = st.button("Load Cache", key="cache_load")
            else:
                st.info("No cache files found. Fetch from API first.")
                selected_cache_path = None
                load_cache_btn = False
            fetch_api = False
            api_date_start = None
            api_date_stop = None

    # ── Data loading ──────────────────────────────────────────────────────────
    if data_source == "Cache":
        if not load_cache_btn:
            st.info("Select a cache file and click **Load Cache**.")
            st.stop()
        if selected_cache_path is None or not selected_cache_path.exists():
            st.error("Cache file not found.")
            st.stop()
        with st.spinner(f"Loading cache: {selected_cache_path.name}…"):
            cached = _load_cache(selected_cache_path)
        df = cached["df"]
        # Patch caches saved before Results was corrected to purchases
        if "Purchases" in df.columns:
            df["Results"] = df["Purchases"]
        # Patch caches saved before attribution window columns existed
        if "Results (7d_click)" not in df.columns:
            df["Results (7d_click)"] = None
        if "Results (1d_view)" not in df.columns:
            df["Results (1d_view)"] = None
        freq_data = cached["freq_data"]
        _src_label = (
            f"Cache: {selected_cache_path.name}  |  "
            f"{cached.get('date_start')} → {cached.get('date_stop')}  |  "
            f"Account {cached.get('account_id')}"
        )
    else:
        if not (api_date_start and api_date_stop):
            st.info("Pick a date range in the sidebar and click **Fetch from API**.")
            st.stop()
        if not fetch_api:
            st.info("Click **Fetch from API** in the sidebar to load data.")
            st.stop()
        with st.spinner("Fetching ad insights from Meta API…"):
            df = fetch_from_api(
                access_token=api_token,
                ad_account_id=api_account_id,
                date_start=api_date_start.strftime("%Y-%m-%d"),
                date_stop=api_date_stop.strftime("%Y-%m-%d"),
            )
        if df.empty:
            st.stop()
        with st.spinner("Fetching reach & frequency breakdowns from Meta API…"):
            freq_data = fetch_frequency_breakdowns(
                access_token=api_token,
                ad_account_id=api_account_id,
                date_start=api_date_start.strftime("%Y-%m-%d"),
                date_stop=api_date_stop.strftime("%Y-%m-%d"),
            )
        cache_path = _save_cache(
            df, freq_data,
            account_id=api_account_id,
            date_start=api_date_start.strftime("%Y-%m-%d"),
            date_stop=api_date_stop.strftime("%Y-%m-%d"),
        )
        st.sidebar.success(f"Saved to cache: {cache_path.name}")
        _src_label = (
            f"Data source: Meta Marketing API  |  "
            f"{api_date_start} → {api_date_stop}  |  "
            f"Account {api_account_id}"
        )

    if df.empty:
        st.stop()

    st.caption(_src_label)

    min_date = df["Reporting starts"].min()
    max_date = df["Reporting starts"].max()

    with st.sidebar:
        st.header("Filters")

        date_range = st.date_input(
            "Reporting date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

        delivery_options = sorted(df["Ad delivery"].dropna().astype(str).unique().tolist())
        selected_delivery = st.multiselect(
            "Ad delivery",
            options=delivery_options,
            default=delivery_options,
        )

        ad_options = sorted(df["Ad name"].dropna().astype(str).unique().tolist())
        selected_ads = st.multiselect(
            "Ad name",
            options=ad_options,
            default=ad_options,
        )

        st.divider()
        st.subheader("Anomaly Overlays")
        show_ci_bands = st.checkbox("Show 95% confidence bands", value=True)
        show_anomaly_markers = st.checkbox("Show anomaly markers", value=True)
        ci_mode_label = st.radio(
            "Band mode",
            options=["Context bands", "1-step predictive bands"],
            horizontal=False,
        )
        interval_mode = "predictive" if ci_mode_label == "1-step predictive bands" else "context"
        overlay_channels = [
            "Cost per Result Trend",
            "CTR vs CVR Trend",
            "Normalized Output per $100",
            "Adset Rolling Frequency",
            "Frequency Trend (Fatigue)",
            "Fatigue Index (30D)",
            "Attribution Frequency Trend",
            "Overall CPM Trend",
        ]
        selected_overlay_channels = st.multiselect(
            "Charts with overlays",
            options=overlay_channels,
            default=overlay_channels,
        )

    selected_overlay_channels_set = set(selected_overlay_channels)

    def overlay_enabled(channel_name: str) -> bool:
        return channel_name in selected_overlay_channels_set and (show_ci_bands or show_anomaly_markers)

    filtered_df = df.copy()

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            filtered_df["Reporting starts"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ]

    if selected_delivery:
        filtered_df = filtered_df[filtered_df["Ad delivery"].astype(str).isin(selected_delivery)]

    if selected_ads:
        filtered_df = filtered_df[filtered_df["Ad name"].astype(str).isin(selected_ads)]

    if filtered_df.empty:
        st.warning("No rows match the current filter selection.")
        st.stop()

    daily = make_daily_summary(filtered_df, account_daily_df=freq_data.get("account_daily"))
    ad_summary = make_ad_summary(filtered_df, ad_alldays_df=freq_data.get("ad_alldays"))

    make_kpi_row(filtered_df, daily)

    tabs = st.tabs(
        [
            "Scaling Trend",
            "Efficiency (Normalized)",
            "Winners & Losers",
            "Deep Tables",
            "Adset Frequency",
            "Ad Frequency",
            "Frequency Fatigue",
            "Scaling & Attribution Health",
            "CPM Trends",
            "Video Deep Dive",
        ]
    )

    with tabs[0]:
        st.subheader("Daily Delivery and Outcome Trends")
        row_1_col_1, row_1_col_2 = st.columns(2)

        with row_1_col_1:
            spend_results_fig = go.Figure()
            spend_results_fig.add_trace(
                go.Scatter(
                    x=daily["Reporting starts"],
                    y=daily["Amount spent (USD)"],
                    mode="lines+markers",
                    name="Spend",
                    yaxis="y1",
                )
            )
            spend_results_fig.add_trace(
                go.Scatter(
                    x=daily["Reporting starts"],
                    y=daily["Results"],
                    mode="lines+markers",
                    name="Results",
                    yaxis="y2",
                )
            )
            spend_results_fig.update_layout(
                title="Spend vs Results (Dual Axis)",
                yaxis=dict(title="Spend ($)"),
                yaxis2=dict(title="Results", overlaying="y", side="right"),
            )
            st.plotly_chart(spend_results_fig, width="stretch")

        with row_1_col_2:
            cpr_fig = px.line(
                daily,
                x="Reporting starts",
                y="Cost per Result",
                markers=True,
                title="Cost per Result Trend",
            )
            if overlay_enabled("Cost per Result Trend"):
                cpr_fig = add_95_confidence_bounds(
                    cpr_fig,
                    daily,
                    "Reporting starts",
                    "Cost per Result",
                    show_ci=show_ci_bands,
                    show_anomalies=show_anomaly_markers,
                    interval_mode=interval_mode,
                )
            st.plotly_chart(cpr_fig, width="stretch")

        row_2_col_1, row_2_col_2 = st.columns(2)

        with row_2_col_1:
            ctr_cvr_colors = {
                "CTR %": "#00D1FF",
                "CVR % (Result/Click)": "#FF4D6D",
            }
            _ctr_cvr_df = daily.copy()
            _ctr_cvr_df["CTR %"] = pd.to_numeric(_ctr_cvr_df["CTR %"], errors="coerce")
            _ctr_cvr_df["CVR % (Result/Click)"] = pd.to_numeric(_ctr_cvr_df["CVR % (Result/Click)"], errors="coerce")
            ctr_cvr_fig = px.line(
                _ctr_cvr_df,
                x="Reporting starts",
                y=["CTR %", "CVR % (Result/Click)"],
                markers=True,
                title="CTR vs CVR Trend",
            )
            ctr_cvr_fig = apply_metric_colors(ctr_cvr_fig, ctr_cvr_colors)
            if overlay_enabled("CTR vs CVR Trend"):
                ctr_cvr_fig = add_95_confidence_bounds(
                    ctr_cvr_fig,
                    daily,
                    "Reporting starts",
                    ["CTR %", "CVR % (Result/Click)"],
                    show_ci=show_ci_bands,
                    show_anomalies=show_anomaly_markers,
                    interval_mode=interval_mode,
                    metric_color_map=ctr_cvr_colors,
                )
            st.plotly_chart(ctr_cvr_fig, width="stretch")

        with row_2_col_2:
            per_100_colors = {
                "Results per $100": "#2E6BFF",
                "Clicks per $100": "#00D1FF",
                "LPV per $100": "#FF4D6D",
            }
            per_100_fig = px.line(
                daily,
                x="Reporting starts",
                y=["Results per $100", "Clicks per $100", "LPV per $100"],
                markers=True,
                title="Normalized Output per $100",
            )
            per_100_fig = apply_metric_colors(per_100_fig, per_100_colors)
            if overlay_enabled("Normalized Output per $100"):
                per_100_fig = add_95_confidence_bounds(
                    per_100_fig,
                    daily,
                    "Reporting starts",
                    ["Results per $100", "Clicks per $100", "LPV per $100"],
                    show_ci=show_ci_bands,
                    show_anomalies=show_anomaly_markers,
                    interval_mode=interval_mode,
                    metric_color_map=per_100_colors,
                )
            st.plotly_chart(per_100_fig, width="stretch")

    with tabs[1]:
        st.subheader("Spend-Normalized Efficiency")
        top_norm = ad_summary.sort_values("Amount spent (USD)", ascending=False).head(20)

        norm_bar = px.bar(
            top_norm,
            x="Ad name",
            y="Results per $100",
            color="Ad delivery",
            hover_data=["Amount spent (USD)", "Results", "Cost per Result", "Scale Potential Score"],
            title="Results per $100 Spend (Top Spenders)",
        )
        st.plotly_chart(norm_bar, width="stretch")

        scatter_col1, scatter_col2 = st.columns(2)

        with scatter_col1:
            scale_map = px.scatter(
                ad_summary,
                x="Amount spent (USD)",
                y="Cost per Result",
                size="Results",
                color="CTR %",
                hover_name="Ad name",
                title="Scale Map: Spend vs Cost/Result",
            )
            st.plotly_chart(scale_map, width="stretch")

        with scatter_col2:
            quality_map = px.scatter(
                ad_summary,
                x="CTR %",
                y="CVR % (Result/Click)",
                size="Amount spent (USD)",
                color="Results per $100",
                hover_name="Ad name",
                title="Traffic Quality Map (CTR vs CVR)",
            )
            st.plotly_chart(quality_map, width="stretch")

        bucket_df = ad_summary.copy()
        bucket_df["Spend Bucket"] = pd.cut(
            bucket_df["Amount spent (USD)"],
            bins=[-0.01, 10, 25, 50, 100, 999999],
            labels=["<$10", "$10-25", "$25-50", "$50-100", "$100+"],
        )
        bucket_summary = (
            bucket_df.groupby("Spend Bucket", observed=False, as_index=False)
            .agg({"Results per $100": "mean", "CTR %": "mean", "CVR % (Result/Click)": "mean"})
            .melt(id_vars="Spend Bucket", var_name="Metric", value_name="Value")
        )

        heatmap = px.density_heatmap(
            bucket_summary,
            x="Spend Bucket",
            y="Metric",
            z="Value",
            color_continuous_scale="Blues",
            title="Average Efficiency by Spend Bucket",
        )
        st.plotly_chart(heatmap, width="stretch")

    with tabs[2]:
        st.subheader("Who Gets More Budget?")

        winners = ad_summary.sort_values("Scale Potential Score", ascending=False).head(15)
        losers = ad_summary.sort_values("Scale Potential Score", ascending=True).head(15)

        win_col, lose_col = st.columns(2)

        with win_col:
            st.markdown("**Top Candidates to Scale**")
            st.dataframe(
                winners[
                    [
                        "Ad name",
                        "Ad delivery",
                        "Amount spent (USD)",
                        "Results",
                        "Results per $100",
                        "Cost per Result",
                        "CTR %",
                        "CVR % (Result/Click)",
                        "Scale Potential Score",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        with lose_col:
            st.markdown("**Weak Performers (Fix/Pause)**")
            st.dataframe(
                losers[
                    [
                        "Ad name",
                        "Ad delivery",
                        "Amount spent (USD)",
                        "Results",
                        "Results per $100",
                        "Cost per Result",
                        "CTR %",
                        "CVR % (Result/Click)",
                        "Scale Potential Score",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        rank_chart = px.bar(
            winners.sort_values("Scale Potential Score"),
            x="Scale Potential Score",
            y="Ad name",
            orientation="h",
            color="Results per $100",
            title="Top Scale Potential Score Ranking",
        )
        st.plotly_chart(rank_chart, width="stretch")

    with tabs[3]:
        st.subheader("Daily Operating Table (Normalized)")
        st.dataframe(
            daily[
                [
                    "Reporting starts",
                    "Amount spent (USD)",
                    "Results",
                    "Cost per Result",
                    "CTR %",
                    "CVR % (Result/Click)",
                    "Results per $100",
                    "Clicks per $100",
                    "LPV per $100",
                ]
            ].sort_values("Reporting starts", ascending=False),
            width="stretch",
            hide_index=True,
        )

    with tabs[4]:
        st.subheader("Ad Set Cumulative Frequency")
        st.caption(
            "Cumulative frequency = true deduplicated reach queried from the Meta API "
            "growing from day 1 of the selected period through each successive day."
        )

        adset_col = "Ad set name" if "Ad set name" in filtered_df.columns else None
        if adset_col is None:
            adset_source = filtered_df.copy()
            adset_source["Ad Set"] = "Selected Ads Aggregate"
            adset_col = "Ad Set"
            st.caption("`Ad set name` column is not available — showing aggregate.")
        else:
            adset_source = filtered_df

        if adset_col == "Ad set name":
            _adset_cum = freq_data.get("adset_cumulative")
        else:
            _adset_cum = freq_data.get("account_cumulative", pd.DataFrame()).copy()
            if _adset_cum is not None and not _adset_cum.empty:
                _adset_cum[adset_col] = "Selected Ads Aggregate"

        adset_freq = make_rolling_frequency_summary(adset_source, adset_col, cumulative_df=_adset_cum)
        adset_options = adset_freq[adset_col].dropna().astype(str).unique().tolist()
        selected_adset = st.selectbox("Choose ad set", options=sorted(adset_options), key="adset_roll_freq")

        selected_adset_df = adset_freq[adset_freq[adset_col].astype(str) == str(selected_adset)].copy()

        _adset_y_cols = [c for c in ["Daily Frequency", "Cumulative Frequency"] if selected_adset_df[c].notna().any()]
        adset_freq_colors = {"Daily Frequency": "#00D1FF", "Cumulative Frequency": "#FF4D6D"}
        adset_freq_fig = px.line(
            selected_adset_df,
            x="Reporting starts",
            y=_adset_y_cols,
            markers=True,
            title=f"Cumulative Frequency - {selected_adset}",
        )
        adset_freq_fig = apply_metric_colors(adset_freq_fig, adset_freq_colors)
        if overlay_enabled("Adset Rolling Frequency"):
            adset_freq_fig = add_95_confidence_bounds(
                adset_freq_fig,
                selected_adset_df,
                "Reporting starts",
                _adset_y_cols,
                show_ci=show_ci_bands,
                show_anomalies=show_anomaly_markers,
                interval_mode=interval_mode,
                metric_color_map=adset_freq_colors,
            )
        st.plotly_chart(adset_freq_fig, width="stretch")

        _adset_table_cols = [c for c in [
            "Reporting starts", "Impressions", "Reach", "Daily Frequency",
            "Cumulative Reach", "Cumulative Impressions", "Cumulative Frequency",
        ] if c in selected_adset_df.columns]
        st.dataframe(
            selected_adset_df[_adset_table_cols].sort_values("Reporting starts", ascending=False),
            width="stretch",
            hide_index=True,
        )

    with tabs[5]:
        st.subheader("Individual Ad Cumulative Frequency")
        st.caption(
            "Cumulative frequency = true deduplicated reach queried from the Meta API "
            "growing from day 1 of the selected period through each successive day."
        )

        ad_freq = make_rolling_frequency_summary(
            filtered_df, "Ad name",
            cumulative_df=freq_data.get("ad_cumulative"),
        )
        top_ads = (
            filtered_df.groupby("Ad name", as_index=False)["Amount spent (USD)"]
            .sum()
            .sort_values("Amount spent (USD)", ascending=False)
            .head(12)["Ad name"]
            .tolist()
        )

        selected_ad_names = st.multiselect(
            "Choose ads",
            options=top_ads,
            default=top_ads[:3] if len(top_ads) >= 3 else top_ads,
            key="ad_roll_freq",
        )

        if selected_ad_names:
            ad_freq_selected = ad_freq[ad_freq["Ad name"].isin(selected_ad_names)].copy()

            ad_cum_fig = px.line(
                ad_freq_selected,
                x="Reporting starts",
                y="Cumulative Frequency",
                color="Ad name",
                markers=True,
                title="Cumulative Frequency by Ad (day 1 → each day)",
            )
            st.plotly_chart(ad_cum_fig, width="stretch")

            ad_daily_fig = px.line(
                ad_freq_selected,
                x="Reporting starts",
                y="Daily Frequency",
                color="Ad name",
                markers=True,
                title="Daily Frequency by Ad (that day only)",
            )
            st.plotly_chart(ad_daily_fig, width="stretch")

            _ad_table_cols = [c for c in [
                "Reporting starts", "Ad name", "Impressions", "Reach", "Daily Frequency",
                "Cumulative Reach", "Cumulative Impressions", "Cumulative Frequency",
            ] if c in ad_freq_selected.columns]
            st.dataframe(
                ad_freq_selected[_ad_table_cols].sort_values(
                    ["Ad name", "Reporting starts"], ascending=[True, False]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Select at least one ad to show cumulative frequency.")

    with tabs[6]:
        st.subheader("Frequency Fatigue & New vs Returning Spend")
        st.caption(
            "Cumulative frequency from the Meta API: each point shows true deduplicated reach "
            "from campaign-start through that day. Returning impressions = impressions beyond "
            "first exposure (Cumulative Impressions − Cumulative Reach)."
        )

        fatigue_source = filtered_df.copy()
        fatigue_source["Audience Group"] = "Selected Ads"

        _fatigue_cum = freq_data.get("account_cumulative", pd.DataFrame()).copy()

        fatigue_roll = make_rolling_frequency_summary(
            fatigue_source, "Audience Group",
            cumulative_df=_fatigue_cum if not _fatigue_cum.empty else None,
        ).sort_values("Reporting starts")

        # Cumulative new vs returning split
        fatigue_roll["New Impressions (Cumulative)"] = fatigue_roll["Cumulative Reach"]
        fatigue_roll["Returning Impressions (Cumulative)"] = (
            fatigue_roll["Cumulative Impressions"] - fatigue_roll["Cumulative Reach"]
        ).clip(lower=0)
        fatigue_roll["New Spend (Cumulative)"] = _safe_divide(
            fatigue_roll["New Impressions (Cumulative)"], fatigue_roll["Cumulative Impressions"]
        ) * fatigue_roll["Cumulative Spend"]
        fatigue_roll["Returning Spend (Cumulative)"] = _safe_divide(
            fatigue_roll["Returning Impressions (Cumulative)"], fatigue_roll["Cumulative Impressions"]
        ) * fatigue_roll["Cumulative Spend"]
        fatigue_roll["Fatigue Index"] = (
            (fatigue_roll["Cumulative Frequency"].fillna(0) - 1).clip(lower=0)
            * _safe_divide(fatigue_roll["Cumulative Spend"], fatigue_roll["Cumulative Reach"])
        )

        latest_row = fatigue_roll.dropna(subset=["Cumulative Frequency"]).tail(1)
        if latest_row.empty:
            latest_row = fatigue_roll.tail(1)
        latest_cum = latest_row.iloc[0]

        total_new_spend = float(latest_cum.get("New Spend (Cumulative)") or 0)
        total_returning_spend = float(latest_cum.get("Returning Spend (Cumulative)") or 0)
        total_spend = float(latest_cum.get("Cumulative Spend") or 0)

        fatigue_col1, fatigue_col2, fatigue_col3, fatigue_col4 = st.columns(4)
        fatigue_col1.metric("New People Spend (Cumulative)", f"${total_new_spend:,.2f}")
        fatigue_col2.metric("Returning People Spend (Cumulative)", f"${total_returning_spend:,.2f}")
        fatigue_col3.metric(
            "Returning People Spend %",
            "-" if total_spend <= 0 else f"{(total_returning_spend / total_spend) * 100:,.1f}%",
        )
        fatigue_col4.metric(
            "Cumulative Frequency",
            "-" if pd.isna(latest_cum.get("Cumulative Frequency")) else f"{latest_cum['Cumulative Frequency']:,.2f}",
        )

        trend_col1, trend_col2 = st.columns(2)

        with trend_col1:
            fatigue_freq_colors = {
                "Daily Frequency": "#00D1FF",
                "Cumulative Frequency": "#FF4D6D",
            }
            _fatigue_y = [c for c in ["Daily Frequency", "Cumulative Frequency"] if fatigue_roll[c].notna().any()]
            fatigue_trend = px.line(
                fatigue_roll,
                x="Reporting starts",
                y=_fatigue_y,
                markers=True,
                title="Frequency Trend (Daily vs Cumulative)",
            )
            fatigue_trend = apply_metric_colors(fatigue_trend, fatigue_freq_colors)
            if overlay_enabled("Frequency Trend (Fatigue)"):
                fatigue_trend = add_95_confidence_bounds(
                    fatigue_trend,
                    fatigue_roll,
                    "Reporting starts",
                    _fatigue_y,
                    show_ci=show_ci_bands,
                    show_anomalies=show_anomaly_markers,
                    interval_mode=interval_mode,
                    metric_color_map=fatigue_freq_colors,
                )
            st.plotly_chart(fatigue_trend, width="stretch")

        with trend_col2:
            _spend_split_cols = [c for c in ["New Spend (Cumulative)", "Returning Spend (Cumulative)"] if fatigue_roll[c].notna().any()]
            if _spend_split_cols:
                spend_split_trend = px.area(
                    fatigue_roll,
                    x="Reporting starts",
                    y=_spend_split_cols,
                    title="Cumulative Spend Split: New vs Returning People",
                )
                st.plotly_chart(spend_split_trend, width="stretch")

        if fatigue_roll["Fatigue Index"].notna().any():
            fatigue_index_fig = px.line(
                fatigue_roll,
                x="Reporting starts",
                y="Fatigue Index",
                markers=True,
                title="Fatigue Index Trend (Cumulative)",
            )
            if overlay_enabled("Fatigue Index (30D)"):
                fatigue_index_fig = add_95_confidence_bounds(
                    fatigue_index_fig,
                    fatigue_roll,
                    "Reporting starts",
                    "Fatigue Index",
                    show_ci=show_ci_bands,
                    show_anomalies=show_anomaly_markers,
                    interval_mode=interval_mode,
                )
            st.plotly_chart(fatigue_index_fig, width="stretch")

        split_totals = pd.DataFrame(
            {
                "Audience Type": ["New People (Cumulative)", "Returning People (Cumulative)"],
                "Estimated Spend": [total_new_spend, total_returning_spend],
            }
        )
        split_pie = px.pie(
            split_totals,
            names="Audience Type",
            values="Estimated Spend",
            title="Total Spend Split",
        )
        st.plotly_chart(split_pie, width="stretch")

        ad_roll = make_rolling_frequency_summary(
            filtered_df, "Ad name",
            cumulative_df=freq_data.get("ad_cumulative"),
        ).sort_values("Reporting starts")
        latest_ad_roll = ad_roll.groupby("Ad name", as_index=False).tail(1).copy()

        latest_ad_roll["New Impressions (Cumulative)"] = latest_ad_roll["Cumulative Reach"]
        latest_ad_roll["Returning Impressions (Cumulative)"] = (
            latest_ad_roll["Cumulative Impressions"] - latest_ad_roll["Cumulative Reach"]
        ).clip(lower=0)
        latest_ad_roll["New Spend (Cumulative)"] = _safe_divide(
            latest_ad_roll["New Impressions (Cumulative)"], latest_ad_roll["Cumulative Impressions"]
        ) * latest_ad_roll["Cumulative Spend"]
        latest_ad_roll["Returning Spend (Cumulative)"] = _safe_divide(
            latest_ad_roll["Returning Impressions (Cumulative)"], latest_ad_roll["Cumulative Impressions"]
        ) * latest_ad_roll["Cumulative Spend"]
        latest_ad_roll["Fatigue Index"] = (
            (latest_ad_roll["Cumulative Frequency"].fillna(0) - 1).clip(lower=0)
            * _safe_divide(latest_ad_roll["Cumulative Spend"], latest_ad_roll["Cumulative Reach"])
        )

        fatigue_ads = ad_summary.merge(
            latest_ad_roll[
                [c for c in [
                    "Ad name",
                    "Cumulative Frequency",
                    "New Spend (Cumulative)",
                    "Returning Spend (Cumulative)",
                    "Fatigue Index",
                ] if c in latest_ad_roll.columns]
            ],
            on="Ad name",
            how="left",
        ).sort_values("Fatigue Index", ascending=False)
        st.subheader("Ad Fatigue Table")
        _fatigue_table_cols = [c for c in [
            "Ad name",
            "Ad delivery",
            "Amount spent (USD)",
            "Cumulative Frequency",
            "Reach",
            "Impressions",
            "New Spend (Cumulative)",
            "Returning Spend (Cumulative)",
            "Fatigue Index",
            "Results per $100",
            "Cost per Result",
        ] if c in fatigue_ads.columns]
        st.dataframe(
            fatigue_ads[_fatigue_table_cols],
            width="stretch",
            hide_index=True,
        )

        st.subheader("Ad Scorecard Table")
        st.dataframe(
            ad_summary[
                [
                    "Ad name",
                    "Ad delivery",
                    "Active days",
                    "Amount spent (USD)",
                    "Results",
                    "Cost per Result",
                    "CTR %",
                    "CPC",
                    "CPM",
                    "CVR % (Result/Click)",
                    "Results per $100",
                    "Clicks per $100",
                    "LPV per $100",
                    "Scale Potential Score",
                ]
            ].sort_values("Scale Potential Score", ascending=False),
            width="stretch",
            hide_index=True,
        )

    with tabs[7]:
        st.subheader("Scaling & Attribution Health")
        st.caption(
            "Analyze the relationship between frequency and attribution types (click vs view-based conversions) to identify frequency ceilings and scaling bottlenecks."
        )

        attr_health_daily, attr_daily_detail = make_attribution_health_summary(
            filtered_df, account_daily_df=freq_data.get("account_daily")
        )
        ad_attr_summary = make_attribution_ad_summary(
            filtered_df, ad_alldays_df=freq_data.get("ad_alldays")
        )

        if attr_health_daily is None and ad_attr_summary is None:
            st.warning("No data available for this tab.")
        else:
            if attr_health_daily is not None:
                attr_health_daily_sorted = attr_health_daily.sort_values("Reporting starts")
            else:
                attr_health_daily_sorted = None

            if attr_daily_detail is not None:
                attr_cols = [c for c in attr_daily_detail["Attribution setting"].unique() if pd.notna(c)]
            else:
                attr_cols = []
            # Only treat as separate-attribution data when there are genuinely distinct
            # click-only and view-only rows (NOT the combined "7-day click or 1-day view" value).
            click_only_cols = [c for c in attr_cols if "click" in str(c).lower() and "view" not in str(c).lower()]
            view_only_cols  = [c for c in attr_cols if "view"  in str(c).lower() and "click" not in str(c).lower()]
            is_click = bool(click_only_cols)
            is_view  = bool(view_only_cols)

            if is_click and is_view and attr_health_daily_sorted is not None:
                pivot_results = attr_daily_detail.pivot_table(
                    index="Reporting starts",
                    columns="Attribution setting",
                    values="Results",
                    aggfunc="sum",
                    fill_value=0,
                )
                click_col = next((c for c in pivot_results.columns if "click" in str(c).lower()), None)
                view_col = next((c for c in pivot_results.columns if "view" in str(c).lower()), None)

                # daily Results per $100 for the overlay
                roas_daily = attr_health_daily_sorted.copy()
                roas_daily["Results per $100"] = pd.to_numeric(
                    _safe_divide(roas_daily["Results"] * 100, roas_daily["Amount spent (USD)"]),
                    errors="coerce",
                )

                fig_attr = go.Figure()
                if click_col:
                    fig_attr.add_trace(go.Bar(
                        x=pivot_results.index,
                        y=pivot_results[click_col],
                        name="7-Day Click Sales",
                        yaxis="y1",
                        marker_color="#FF8C00",
                    ))
                if view_col:
                    fig_attr.add_trace(go.Bar(
                        x=pivot_results.index,
                        y=pivot_results[view_col],
                        name="1-Day View Sales",
                        yaxis="y1",
                        marker_color="#1A6FD4",
                    ))
                fig_attr.add_trace(go.Scatter(
                    x=roas_daily["Reporting starts"],
                    y=roas_daily["Results per $100"],
                    name="Results per $100",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color="#FF4D6D", width=3),
                ))
                fig_attr.update_layout(
                    title="Where Does Click ROAS Drop? (7-Day Click vs 1-Day View Sales)",
                    barmode="stack",
                    yaxis=dict(title="Sales"),
                    yaxis2=dict(title="Results per $100 Spent", overlaying="y", side="right"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_attr, width="stretch")

            elif attr_health_daily_sorted is not None:
                roas_daily = attr_health_daily_sorted.copy()
                has_window_split = (
                    "Results (7d_click)" in roas_daily.columns
                    and (pd.to_numeric(roas_daily["Results (7d_click)"], errors="coerce") > 0).any()
                )
                roas_daily["Results per $100"] = pd.to_numeric(
                    _safe_divide(roas_daily["Results"] * 100, roas_daily["Amount spent (USD)"]),
                    errors="coerce",
                )
                fig_roas = go.Figure()
                if has_window_split:
                    fig_roas.add_trace(go.Bar(
                        x=roas_daily["Reporting starts"],
                        y=pd.to_numeric(roas_daily["Results (7d_click)"], errors="coerce"),
                        name="7-Day Click Sales",
                        yaxis="y1",
                        marker_color="#FF8C00",
                    ))
                    fig_roas.add_trace(go.Bar(
                        x=roas_daily["Reporting starts"],
                        y=pd.to_numeric(roas_daily["Results (1d_view)"], errors="coerce"),
                        name="1-Day View Sales",
                        yaxis="y1",
                        marker_color="#1A6FD4",
                    ))
                else:
                    fig_roas.add_trace(go.Bar(
                        x=roas_daily["Reporting starts"],
                        y=pd.to_numeric(roas_daily["Results"], errors="coerce"),
                        name="Daily Sales",
                        yaxis="y1",
                        marker_color="#FF8C00",
                    ))
                fig_roas.add_trace(go.Scatter(
                    x=roas_daily["Reporting starts"],
                    y=roas_daily["Results per $100"],
                    name="Results per $100",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color="#FF4D6D", width=3),
                ))
                fig_roas.update_layout(
                    title="Where Does Click ROAS Drop? (7-Day Click vs 1-Day View Sales)",
                    barmode="stack",
                    yaxis=dict(title="Sales"),
                    yaxis2=dict(title="Results per $100 Spent", overlaying="y", side="right"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_roas, width="stretch")

    with tabs[8]:
        st.subheader("CPM Trends of Ads")
        st.caption("Track CPM over time overall and by ad to spot rising auction pressure.")

        cpm_daily = (
            filtered_df.groupby("Reporting starts", as_index=False)
            .agg({"Amount spent (USD)": "sum", "Impressions": "sum"})
            .sort_values("Reporting starts")
        )
        cpm_daily["CPM"] = _safe_divide(cpm_daily["Amount spent (USD)"] * 1000, cpm_daily["Impressions"])

        cpm_overall_fig = px.line(
            cpm_daily,
            x="Reporting starts",
            y="CPM",
            markers=True,
            title="Overall Daily CPM Trend",
        )
        if overlay_enabled("Overall CPM Trend"):
            cpm_overall_fig = add_95_confidence_bounds(
                cpm_overall_fig,
                cpm_daily,
                "Reporting starts",
                "CPM",
                show_ci=show_ci_bands,
                show_anomalies=show_anomaly_markers,
                interval_mode=interval_mode,
            )
        st.plotly_chart(cpm_overall_fig, width="stretch")

        cpm_by_ad = (
            filtered_df.groupby(["Reporting starts", "Ad name"], as_index=False)
            .agg({"Amount spent (USD)": "sum", "Impressions": "sum"})
            .sort_values(["Ad name", "Reporting starts"])
        )
        cpm_by_ad["CPM"] = _safe_divide(cpm_by_ad["Amount spent (USD)"] * 1000, cpm_by_ad["Impressions"])
        cpm_by_ad["CPM 7D Rolling"] = (
            cpm_by_ad.groupby("Ad name")["CPM"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        )

        top_ads_by_spend = (
            filtered_df.groupby("Ad name", as_index=False)["Amount spent (USD)"]
            .sum()
            .sort_values("Amount spent (USD)", ascending=False)
            .head(12)["Ad name"]
            .tolist()
        )
        selected_cpm_ads = st.multiselect(
            "Select ads for CPM trend",
            options=top_ads_by_spend,
            default=top_ads_by_spend[:5] if len(top_ads_by_spend) >= 5 else top_ads_by_spend,
            key="cpm_trend_ads",
        )

        if selected_cpm_ads:
            cpm_selected = cpm_by_ad[cpm_by_ad["Ad name"].isin(selected_cpm_ads)].copy()

            cpm_ads_fig = px.line(
                cpm_selected,
                x="Reporting starts",
                y="CPM",
                color="Ad name",
                markers=True,
                title="Daily CPM by Ad",
            )
            st.plotly_chart(cpm_ads_fig, width="stretch")

            cpm_roll_fig = px.line(
                cpm_selected,
                x="Reporting starts",
                y="CPM 7D Rolling",
                color="Ad name",
                markers=True,
                title="7D Rolling CPM by Ad",
            )
            st.plotly_chart(cpm_roll_fig, width="stretch")

            latest_cpm = (
                cpm_selected.sort_values("Reporting starts")
                .groupby("Ad name", as_index=False)
                .tail(1)
                .sort_values("CPM", ascending=False)
            )
            st.dataframe(
                latest_cpm[["Ad name", "CPM", "CPM 7D Rolling", "Amount spent (USD)", "Impressions"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Select at least one ad to display CPM trends.")

    with tabs[9]:
        st.subheader("Video Deep Dive — Normalized by Impressions")

        VIDEO_COLS = [
            "Video plays",
            "Viewers",
            "3-second video plays",
            "Video plays at 25%",
            "Video plays at 50%",
            "Video plays at 75%",
            "Video plays at 95%",
            "ThruPlays",
            "Photo clicks",
        ]
        avail = [c for c in VIDEO_COLS if c in filtered_df.columns and filtered_df[c].notna().any()]

        if not avail:
            st.info("No video metrics found in this export. Re-export from Meta Ads Manager with video columns enabled.")
        else:
            # ── per-ad aggregate ──────────────────────────────────────────────────
            agg_dict = {"Amount spent (USD)": "sum", "Impressions": "sum", "Reach": "sum", "Link clicks": "sum"}
            for c in avail:
                agg_dict[c] = "sum"
            vid_ad = filtered_df.groupby("Ad name", as_index=False).agg(agg_dict)

            # ── per-day aggregate ─────────────────────────────────────────────────
            vid_day = filtered_df.groupby("Reporting starts", as_index=False).agg(agg_dict).sort_values("Reporting starts")

            # ── derived rates (per-ad) ────────────────────────────────────────────
            def rate(df, num, denom, pct=True):
                r = _safe_divide(df[num], df[denom])
                return r * 100 if pct else r

            if "3-second video plays" in avail:
                vid_ad["Hook Rate %"]         = rate(vid_ad, "3-second video plays", "Impressions")
                vid_day["Hook Rate %"]        = rate(vid_day, "3-second video plays", "Impressions")
                vid_ad["Cost / 3s Play"]      = _safe_divide(vid_ad["Amount spent (USD)"], vid_ad["3-second video plays"])

            if "ThruPlays" in avail:
                vid_ad["ThruPlay Rate %"]     = rate(vid_ad, "ThruPlays", "Impressions")
                vid_day["ThruPlay Rate %"]    = rate(vid_day, "ThruPlays", "Impressions")
                vid_ad["Cost / ThruPlay"]     = _safe_divide(vid_ad["Amount spent (USD)"], vid_ad["ThruPlays"])

            # Completion funnel columns relative to 3s plays (hook) as base
            base_col = "3-second video plays" if "3-second video plays" in avail else "Video plays"
            funnel_stages = {
                "% Watched 25% (of 3s plays)": "Video plays at 25%",
                "% Watched 50% (of 3s plays)": "Video plays at 50%",
                "% Watched 75% (of 3s plays)": "Video plays at 75%",
                "% Watched 95% (of 3s plays)": "Video plays at 95%",
                "% ThruPlayed (of 3s plays)": "ThruPlays",
            }
            for label, col in funnel_stages.items():
                if col in avail and base_col in avail:
                    vid_ad[label] = _safe_divide(vid_ad[col], vid_ad[base_col]) * 100

            if "Link clicks" in vid_ad.columns and "3-second video plays" in avail:
                vid_ad["CTR from 3s Viewers %"] = _safe_divide(vid_ad["Link clicks"], vid_ad["3-second video plays"]) * 100

            # ── Retention score: average of watch milestones ──────────────────────
            milestone_rate_cols = [l for l in funnel_stages if l in vid_ad.columns]
            if milestone_rate_cols:
                vid_ad["Retention Score"] = vid_ad[milestone_rate_cols].mean(axis=1)

            # ── Hook quality score: hook rate × retention score ───────────────────
            if "Hook Rate %" in vid_ad.columns and "Retention Score" in vid_ad.columns:
                vid_ad["Video Score"] = (
                    vid_ad["Hook Rate %"].fillna(0) * 0.4
                    + vid_ad["Retention Score"].fillna(0) * 0.4
                    + vid_ad.get("ThruPlay Rate %", pd.Series(0, index=vid_ad.index)).fillna(0) * 0.2
                )

            # ═══════════════════════════════════════════════════════════════════════
            # KPI row
            # ═══════════════════════════════════════════════════════════════════════
            kv1, kv2, kv3, kv4, kv5 = st.columns(5)
            if "3-second video plays" in avail:
                kv1.metric("Total 3s Plays", f"{int(vid_ad['3-second video plays'].sum()):,}")
                avg_hook = vid_ad["Hook Rate %"].mean()
                kv2.metric("Avg Hook Rate", f"{avg_hook:.1f}%" if pd.notna(avg_hook) else "—")
            if "ThruPlays" in avail:
                kv3.metric("Total ThruPlays", f"{int(vid_ad['ThruPlays'].sum()):,}")
                avg_thru = vid_ad["ThruPlay Rate %"].mean()
                kv4.metric("Avg ThruPlay Rate", f"{avg_thru:.1f}%" if pd.notna(avg_thru) else "—")
            if "Retention Score" in vid_ad.columns:
                kv5.metric("Avg Retention Score", f"{vid_ad['Retention Score'].mean():.1f}%")

            st.markdown("---")

            # ═══════════════════════════════════════════════════════════════════════
            # Row 1 — Hook Rate bar + ThruPlay Rate bar
            # ═══════════════════════════════════════════════════════════════════════
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                if "Hook Rate %" in vid_ad.columns:
                    fig_hook = px.bar(
                        vid_ad.sort_values("Hook Rate %", ascending=False),
                        x="Ad name", y="Hook Rate %",
                        hover_data=["Amount spent (USD)", "Impressions", "3-second video plays", "Cost / 3s Play"],
                        title="Hook Rate by Ad (3s Plays / Impressions)",
                        color="Hook Rate %",
                        color_continuous_scale="Blues",
                    )
                    fig_hook.update_layout(xaxis_tickangle=-35, yaxis_ticksuffix="%", showlegend=False)
                    st.plotly_chart(fig_hook, width="stretch")

            with r1c2:
                if "ThruPlay Rate %" in vid_ad.columns:
                    fig_thru = px.bar(
                        vid_ad.sort_values("ThruPlay Rate %", ascending=False),
                        x="Ad name", y="ThruPlay Rate %",
                        hover_data=["Amount spent (USD)", "Impressions", "ThruPlays", "Cost / ThruPlay"],
                        title="ThruPlay Rate by Ad (ThruPlays / Impressions)",
                        color="ThruPlay Rate %",
                        color_continuous_scale="Greens",
                    )
                    fig_thru.update_layout(xaxis_tickangle=-35, yaxis_ticksuffix="%", showlegend=False)
                    st.plotly_chart(fig_thru, width="stretch")

            # ═══════════════════════════════════════════════════════════════════════
            # Row 2 — Retention funnel per ad (grouped bar) + Hook vs ThruPlay scatter
            # ═══════════════════════════════════════════════════════════════════════
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                funnel_rate_cols = [l for l in funnel_stages if l in vid_ad.columns]
                if funnel_rate_cols:
                    funnel_melt = vid_ad[["Ad name"] + funnel_rate_cols].melt(
                        id_vars="Ad name", var_name="Watch Milestone", value_name="% Retained"
                    )
                    milestone_order = list(funnel_stages.keys())
                    fig_ret = px.bar(
                        funnel_melt,
                        x="Ad name", y="% Retained", color="Watch Milestone",
                        barmode="group",
                        category_orders={"Watch Milestone": milestone_order},
                        title="Retention at Each Milestone (% of 3s Plays)",
                    )
                    fig_ret.update_layout(xaxis_tickangle=-35, yaxis_ticksuffix="%")
                    st.plotly_chart(fig_ret, width="stretch")

            with r2c2:
                if "Hook Rate %" in vid_ad.columns and "ThruPlay Rate %" in vid_ad.columns:
                    fig_scatter = px.scatter(
                        vid_ad,
                        x="Hook Rate %", y="ThruPlay Rate %",
                        size="Amount spent (USD)",
                        color="Retention Score" if "Retention Score" in vid_ad.columns else None,
                        hover_name="Ad name",
                        hover_data=["Cost / 3s Play", "Cost / ThruPlay", "Amount spent (USD)"],
                        title="Hook Rate vs ThruPlay Rate (bubble = spend)",
                        color_continuous_scale="RdYlGn",
                    )
                    fig_scatter.update_layout(xaxis_ticksuffix="%", yaxis_ticksuffix="%")
                    st.plotly_chart(fig_scatter, width="stretch")

            # ═══════════════════════════════════════════════════════════════════════
            # Row 3 — Aggregated retention funnel waterfall + CTR from 3s viewers
            # ═══════════════════════════════════════════════════════════════════════
            r3c1, r3c2 = st.columns(2)
            with r3c1:
                all_funnel_cols = [c for c in [
                    "Video plays", "3-second video plays",
                    "Video plays at 25%", "Video plays at 50%",
                    "Video plays at 75%", "Video plays at 95%", "ThruPlays"
                ] if c in avail]
                if len(all_funnel_cols) >= 2:
                    funnel_totals = {c: vid_ad[c].sum() for c in all_funnel_cols}
                    funnel_labels = {
                        "Video plays": "Any Play",
                        "3-second video plays": "3s Hook",
                        "Video plays at 25%": "25% Watch",
                        "Video plays at 50%": "50% Watch",
                        "Video plays at 75%": "75% Watch",
                        "Video plays at 95%": "95% Watch",
                        "ThruPlays": "Full / 15s",
                    }
                    funnel_df = pd.DataFrame({
                        "Stage": [funnel_labels[c] for c in all_funnel_cols],
                        "Count": [funnel_totals[c] for c in all_funnel_cols],
                    })
                    fig_funnel = px.funnel(funnel_df, x="Count", y="Stage",
                                           title="Overall Video Drop-off Funnel (All Ads)")
                    st.plotly_chart(fig_funnel, width="stretch")

            with r3c2:
                if "CTR from 3s Viewers %" in vid_ad.columns:
                    fig_ctr3s = px.bar(
                        vid_ad.sort_values("CTR from 3s Viewers %", ascending=False),
                        x="Ad name", y="CTR from 3s Viewers %",
                        hover_data=["Link clicks", "3-second video plays"],
                        title="Link CTR Among 3s Viewers (Clicks / 3s Plays)",
                        color="CTR from 3s Viewers %",
                        color_continuous_scale="Oranges",
                    )
                    fig_ctr3s.update_layout(xaxis_tickangle=-35, yaxis_ticksuffix="%", showlegend=False)
                    st.plotly_chart(fig_ctr3s, width="stretch")

            # ═══════════════════════════════════════════════════════════════════════
            # Row 4 — Daily Hook Rate and ThruPlay Rate trends
            # ═══════════════════════════════════════════════════════════════════════
            daily_rate_cols = [c for c in ["Hook Rate %", "ThruPlay Rate %"] if c in vid_day.columns]
            if daily_rate_cols:
                r4c1, r4c2 = st.columns(2)
                with r4c1:
                    raw_daily_cols = [c for c in ["3-second video plays", "ThruPlays", "Video plays"] if c in avail]
                    fig_raw = px.line(
                        vid_day, x="Reporting starts", y=raw_daily_cols,
                        markers=True, title="Daily Video Plays (Raw)"
                    )
                    st.plotly_chart(fig_raw, width="stretch")
                with r4c2:
                    fig_rates = px.line(
                        vid_day, x="Reporting starts", y=daily_rate_cols,
                        markers=True, title="Daily Hook & ThruPlay Rates (% of Impressions)"
                    )
                    fig_rates.update_layout(yaxis_ticksuffix="%")
                    st.plotly_chart(fig_rates, width="stretch")

            # ═══════════════════════════════════════════════════════════════════════
            # Row 5 — Cost efficiency
            # ═══════════════════════════════════════════════════════════════════════
            cost_cols = [c for c in ["Cost / 3s Play", "Cost / ThruPlay"] if c in vid_ad.columns]
            if cost_cols:
                r5c1, r5c2 = st.columns(2)
                cols_iter = iter([r5c1, r5c2])
                for cost_col in cost_cols:
                    with next(cols_iter):
                        fig_cost = px.bar(
                            vid_ad.dropna(subset=[cost_col]).sort_values(cost_col),
                            x="Ad name", y=cost_col,
                            hover_data=["Amount spent (USD)", "Impressions"],
                            title=f"{cost_col} by Ad (lower = better)",
                        )
                        fig_cost.update_layout(xaxis_tickangle=-35)
                        st.plotly_chart(fig_cost, width="stretch")

            # ═══════════════════════════════════════════════════════════════════════
            # Video Score leaderboard
            # ═══════════════════════════════════════════════════════════════════════
            if "Video Score" in vid_ad.columns:
                st.subheader("Video Performance Leaderboard")
                st.caption("Video Score = 40% Hook Rate + 40% Avg Retention + 20% ThruPlay Rate")
                score_cols = ["Ad name", "Amount spent (USD)", "Impressions"]
                for c in ["Hook Rate %", "% Watched 25% (of 3s plays)", "% Watched 50% (of 3s plays)",
                           "% Watched 75% (of 3s plays)", "% ThruPlayed (of 3s plays)",
                           "ThruPlay Rate %", "Retention Score", "Video Score",
                           "Cost / 3s Play", "Cost / ThruPlay", "CTR from 3s Viewers %"]:
                    if c in vid_ad.columns:
                        score_cols.append(c)
                st.dataframe(
                    vid_ad[score_cols].sort_values("Video Score", ascending=False),
                    width="stretch", hide_index=True
                )


if __name__ == "__main__":
    main()
