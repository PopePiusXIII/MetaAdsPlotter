from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CSV_PATH = Path(__file__).parent / "victim2.csv"
# CSV_PATH = Path(__file__).parent / "Gentle-Gator-Golf-Ads-Ads-Jun-1-2025-Aug-31-2025.csv"

NUMERIC_COLUMNS = [
    "Results",
    "Reach",
    "Frequency",
    "Cost per results",
    "Ad set budget",
    "Amount spent (USD)",
    "Impressions",
    "CPM (cost per 1,000 impressions) (USD)",
    "Link clicks",
    "shop_clicks",
    "CPC (cost per link click) (USD)",
    "CTR (link click-through rate)",
    "Clicks (all)",
    "CTR (all)",
    "CPC (all) (USD)",
    "Landing page views",
    "Cost per landing page view (USD)",
]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["Reporting starts"] = pd.to_datetime(df["Reporting starts"], errors="coerce")
    df["Reporting ends"] = pd.to_datetime(df["Reporting ends"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = _coerce_numeric(df[column])

    return df


def _safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
        safe_denominator = denominator.where(denominator > 0)
        return numerator.div(safe_denominator)
    if denominator and denominator > 0:
        return numerator / denominator
    return pd.NA


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


def make_daily_summary(filtered_df: pd.DataFrame) -> pd.DataFrame:
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


def make_ad_summary(filtered_df: pd.DataFrame) -> pd.DataFrame:
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


def make_rolling_frequency_summary(filtered_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    freq_df = (
        filtered_df.groupby(["Reporting starts", group_col], as_index=False)
        .agg({"Impressions": "sum", "Reach": "sum", "Amount spent (USD)": "sum"})
        .sort_values([group_col, "Reporting starts"])
    )

    freq_df["Daily Frequency"] = _safe_divide(freq_df["Impressions"], freq_df["Reach"])

    overlap_assumption = 0.70
    incremental_unique_share = 1 - overlap_assumption

    freq_df["Rolling Impressions (7D)"] = (
        freq_df.groupby(group_col)["Impressions"].rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    freq_df["Rolling Spend (7D)"] = (
        freq_df.groupby(group_col)["Amount spent (USD)"].rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    freq_df["Rolling Reach Sum (7D)"] = (
        freq_df.groupby(group_col)["Reach"].rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    freq_df["Rolling Reach Max (7D)"] = (
        freq_df.groupby(group_col)["Reach"].rolling(7, min_periods=1).max().reset_index(level=0, drop=True)
    )
    freq_df["Estimated Unique Accounts (7D)"] = (
        freq_df["Rolling Reach Max (7D)"]
        + (freq_df["Rolling Reach Sum (7D)"] - freq_df["Rolling Reach Max (7D)"]).clip(lower=0)
        * incremental_unique_share
    )

    freq_df["Rolling Impressions (30D)"] = (
        freq_df.groupby(group_col)["Impressions"].rolling(30, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    freq_df["Rolling Spend (30D)"] = (
        freq_df.groupby(group_col)["Amount spent (USD)"].rolling(30, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    freq_df["Rolling Reach Sum (30D)"] = (
        freq_df.groupby(group_col)["Reach"].rolling(30, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    freq_df["Rolling Reach Max (30D)"] = (
        freq_df.groupby(group_col)["Reach"].rolling(30, min_periods=1).max().reset_index(level=0, drop=True)
    )
    freq_df["Estimated Unique Accounts (30D)"] = (
        freq_df["Rolling Reach Max (30D)"]
        + (freq_df["Rolling Reach Sum (30D)"] - freq_df["Rolling Reach Max (30D)"]).clip(lower=0)
        * incremental_unique_share
    )

    freq_df["Weekly Rolling Frequency (7D)"] = _safe_divide(
        freq_df["Rolling Impressions (7D)"], freq_df["Estimated Unique Accounts (7D)"]
    )
    freq_df["Monthly Rolling Frequency (30D)"] = _safe_divide(
        freq_df["Rolling Impressions (30D)"], freq_df["Estimated Unique Accounts (30D)"]
    )

    return freq_df


def make_attribution_health_summary(filtered_df: pd.DataFrame):
    """Create daily summary split by attribution type (7-day click vs 1-day view)."""
    if "Attribution setting" not in filtered_df.columns:
        return None, None
    
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
    
    daily_agg = (
        filtered_df.groupby("Reporting starts", as_index=False)
        .agg({
            "Impressions": "sum",
            "Reach": "sum",
            "Amount spent (USD)": "sum",
        })
    )
    daily_agg["Frequency"] = _safe_divide(daily_agg["Impressions"], daily_agg["Reach"])
    
    return daily_agg, attr_daily


def make_attribution_ad_summary(filtered_df: pd.DataFrame):
    """Create ad summary with attribution and scaling analysis."""
    if "Attribution setting" not in filtered_df.columns:
        return None
    
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
    st.caption("Data source: Gentle-Gator-Golf Meta Ads export")

    if not CSV_PATH.exists():
        st.error(f"CSV file not found: {CSV_PATH}")
        st.stop()

    df = load_data(CSV_PATH)

    if df.empty:
        st.warning("No data found in the CSV.")
        st.stop()

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

    daily = make_daily_summary(filtered_df)
    ad_summary = make_ad_summary(filtered_df)

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
            ctr_cvr_fig = px.line(
                daily,
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
        st.subheader("Ad Set Daily / Weekly / Monthly Rolling Frequency")
        st.caption(
            "7D/30D frequency uses an accounts-based unique reach estimate (assumes 70% overlap in daily reached accounts)."
        )

        adset_col = "Ad set name" if "Ad set name" in filtered_df.columns else None
        if adset_col is None:
            adset_source = filtered_df.copy()
            adset_source["Ad Set"] = "Selected Ads Aggregate"
            adset_col = "Ad Set"
            st.caption("`Ad set name` column is not available in this CSV, so this tab shows aggregate selected ads.")
        else:
            adset_source = filtered_df

        adset_freq = make_rolling_frequency_summary(adset_source, adset_col)
        adset_options = adset_freq[adset_col].dropna().astype(str).unique().tolist()
        selected_adset = st.selectbox("Choose ad set", options=sorted(adset_options), key="adset_roll_freq")

        selected_adset_df = adset_freq[adset_freq[adset_col].astype(str) == str(selected_adset)].copy()

        adset_freq_fig = px.line(
            selected_adset_df,
            x="Reporting starts",
            y=["Daily Frequency", "Weekly Rolling Frequency (7D)", "Monthly Rolling Frequency (30D)"],
            markers=True,
            title=f"Rolling Frequency - {selected_adset}",
        )
        adset_freq_colors = {
            "Daily Frequency": "#00D1FF",
            "Weekly Rolling Frequency (7D)": "#2E6BFF",
            "Monthly Rolling Frequency (30D)": "#FF4D6D",
        }
        adset_freq_fig = apply_metric_colors(adset_freq_fig, adset_freq_colors)
        if overlay_enabled("Adset Rolling Frequency"):
            adset_freq_fig = add_95_confidence_bounds(
                adset_freq_fig,
                selected_adset_df,
                "Reporting starts",
                ["Daily Frequency", "Weekly Rolling Frequency (7D)", "Monthly Rolling Frequency (30D)"],
                show_ci=show_ci_bands,
                show_anomalies=show_anomaly_markers,
                interval_mode=interval_mode,
                metric_color_map=adset_freq_colors,
            )
        st.plotly_chart(adset_freq_fig, width="stretch")

        st.dataframe(
            selected_adset_df[
                [
                    "Reporting starts",
                    "Impressions",
                    "Reach",
                    "Daily Frequency",
                    "Estimated Unique Accounts (7D)",
                    "Weekly Rolling Frequency (7D)",
                    "Estimated Unique Accounts (30D)",
                    "Monthly Rolling Frequency (30D)",
                ]
            ].sort_values("Reporting starts", ascending=False),
            width="stretch",
            hide_index=True,
        )

    with tabs[5]:
        st.subheader("Individual Ad Daily / Weekly / Monthly Rolling Frequency")
        st.caption(
            "Rolling frequency is account-based (impressions divided by estimated deduplicated accounts reached over 7D/30D)."
        )

        ad_freq = make_rolling_frequency_summary(filtered_df, "Ad name")
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

            ad_roll_fig = px.line(
                ad_freq_selected,
                x="Reporting starts",
                y="Weekly Rolling Frequency (7D)",
                color="Ad name",
                markers=True,
                title="7D Rolling Frequency by Ad",
            )
            st.plotly_chart(ad_roll_fig, width="stretch")

            ad_monthly_roll_fig = px.line(
                ad_freq_selected,
                x="Reporting starts",
                y="Monthly Rolling Frequency (30D)",
                color="Ad name",
                markers=True,
                title="30D Rolling Frequency by Ad",
            )
            st.plotly_chart(ad_monthly_roll_fig, width="stretch")

            st.dataframe(
                ad_freq_selected[
                    [
                        "Reporting starts",
                        "Ad name",
                        "Impressions",
                        "Reach",
                        "Daily Frequency",
                        "Estimated Unique Accounts (7D)",
                        "Weekly Rolling Frequency (7D)",
                        "Estimated Unique Accounts (30D)",
                        "Monthly Rolling Frequency (30D)",
                    ]
                ].sort_values(["Ad name", "Reporting starts"], ascending=[True, False]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Select at least one ad to show rolling frequency.")

    with tabs[6]:
        st.subheader("Frequency Fatigue & New vs Returning Spend")
        st.caption(
            "Uses accounts-based 30D deduplicated reach (70% overlap assumption) to estimate frequency fatigue and new vs current spend."
        )

        fatigue_source = filtered_df.copy()
        fatigue_source["Audience Group"] = "Selected Ads"
        fatigue_roll = make_rolling_frequency_summary(fatigue_source, "Audience Group").sort_values("Reporting starts")

        fatigue_roll["Estimated New Impressions (30D)"] = fatigue_roll["Estimated Unique Accounts (30D)"].clip(
            upper=fatigue_roll["Rolling Impressions (30D)"]
        )
        fatigue_roll["Estimated Returning Impressions (30D)"] = (
            fatigue_roll["Rolling Impressions (30D)"] - fatigue_roll["Estimated New Impressions (30D)"]
        ).clip(lower=0)
        fatigue_roll["Estimated New Spend (30D)"] = _safe_divide(
            fatigue_roll["Estimated New Impressions (30D)"], fatigue_roll["Rolling Impressions (30D)"]
        ) * fatigue_roll["Rolling Spend (30D)"]
        fatigue_roll["Estimated Returning Spend (30D)"] = _safe_divide(
            fatigue_roll["Estimated Returning Impressions (30D)"], fatigue_roll["Rolling Impressions (30D)"]
        ) * fatigue_roll["Rolling Spend (30D)"]
        fatigue_roll["Fatigue Index (30D)"] = (
            (fatigue_roll["Monthly Rolling Frequency (30D)"].fillna(0) - 1).clip(lower=0)
            * _safe_divide(fatigue_roll["Rolling Spend (30D)"], fatigue_roll["Estimated Unique Accounts (30D)"])
        )

        latest_30d = fatigue_roll.tail(1).iloc[0]

        total_new_spend = float(latest_30d["Estimated New Spend (30D)"])
        total_returning_spend = float(latest_30d["Estimated Returning Spend (30D)"])
        total_spend = float(latest_30d["Rolling Spend (30D)"])

        fatigue_col1, fatigue_col2, fatigue_col3, fatigue_col4 = st.columns(4)
        fatigue_col1.metric("New People Spend (30D)", f"${total_new_spend:,.2f}")
        fatigue_col2.metric("Current People Spend (30D)", f"${total_returning_spend:,.2f}")
        fatigue_col3.metric(
            "Current People Spend %",
            "-" if total_spend <= 0 else f"{(total_returning_spend / total_spend) * 100:,.1f}%",
        )
        fatigue_col4.metric(
            "30D Rolling Frequency",
            "-" if pd.isna(latest_30d["Monthly Rolling Frequency (30D)"]) else f"{latest_30d['Monthly Rolling Frequency (30D)']:,.2f}",
        )

        trend_col1, trend_col2 = st.columns(2)

        with trend_col1:
            fatigue_freq_colors = {
                "Daily Frequency": "#00D1FF",
                "Weekly Rolling Frequency (7D)": "#2E6BFF",
                "Monthly Rolling Frequency (30D)": "#FF4D6D",
            }
            fatigue_trend = px.line(
                fatigue_roll,
                x="Reporting starts",
                y=["Daily Frequency", "Weekly Rolling Frequency (7D)", "Monthly Rolling Frequency (30D)"],
                markers=True,
                title="Frequency Trend (Daily vs 7D vs 30D)",
            )
            fatigue_trend = apply_metric_colors(fatigue_trend, fatigue_freq_colors)
            if overlay_enabled("Frequency Trend (Fatigue)"):
                fatigue_trend = add_95_confidence_bounds(
                    fatigue_trend,
                    fatigue_roll,
                    "Reporting starts",
                    ["Daily Frequency", "Weekly Rolling Frequency (7D)", "Monthly Rolling Frequency (30D)"],
                    show_ci=show_ci_bands,
                    show_anomalies=show_anomaly_markers,
                    interval_mode=interval_mode,
                    metric_color_map=fatigue_freq_colors,
                )
            st.plotly_chart(fatigue_trend, width="stretch")

        with trend_col2:
            spend_split_trend = px.area(
                fatigue_roll,
                x="Reporting starts",
                y=["Estimated New Spend (30D)", "Estimated Returning Spend (30D)"],
                title="30D Spend Split: New vs Current People",
            )
            st.plotly_chart(spend_split_trend, width="stretch")

        fatigue_index_fig = px.line(
            fatigue_roll,
            x="Reporting starts",
            y="Fatigue Index (30D)",
            markers=True,
            title="30D Fatigue Index Trend",
        )
        if overlay_enabled("Fatigue Index (30D)"):
            fatigue_index_fig = add_95_confidence_bounds(
                fatigue_index_fig,
                fatigue_roll,
                "Reporting starts",
                "Fatigue Index (30D)",
                show_ci=show_ci_bands,
                show_anomalies=show_anomaly_markers,
                interval_mode=interval_mode,
            )
        st.plotly_chart(fatigue_index_fig, width="stretch")

        split_totals = pd.DataFrame(
            {
                "Audience Type": ["New People (30D)", "Current People (30D)"],
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

        ad_roll = make_rolling_frequency_summary(filtered_df, "Ad name").sort_values("Reporting starts")
        latest_ad_roll = ad_roll.groupby("Ad name", as_index=False).tail(1).copy()

        latest_ad_roll["Estimated New Impressions (30D)"] = latest_ad_roll["Estimated Unique Accounts (30D)"].clip(
            upper=latest_ad_roll["Rolling Impressions (30D)"]
        )
        latest_ad_roll["Estimated Returning Impressions (30D)"] = (
            latest_ad_roll["Rolling Impressions (30D)"] - latest_ad_roll["Estimated New Impressions (30D)"]
        ).clip(lower=0)
        latest_ad_roll["Estimated New Spend (30D)"] = _safe_divide(
            latest_ad_roll["Estimated New Impressions (30D)"], latest_ad_roll["Rolling Impressions (30D)"]
        ) * latest_ad_roll["Rolling Spend (30D)"]
        latest_ad_roll["Estimated Current Spend (30D)"] = _safe_divide(
            latest_ad_roll["Estimated Returning Impressions (30D)"], latest_ad_roll["Rolling Impressions (30D)"]
        ) * latest_ad_roll["Rolling Spend (30D)"]
        latest_ad_roll["Fatigue Index (30D)"] = (
            (latest_ad_roll["Monthly Rolling Frequency (30D)"].fillna(0) - 1).clip(lower=0)
            * _safe_divide(latest_ad_roll["Rolling Spend (30D)"], latest_ad_roll["Estimated Unique Accounts (30D)"])
        )

        fatigue_ads = ad_summary.merge(
            latest_ad_roll[
                [
                    "Ad name",
                    "Weekly Rolling Frequency (7D)",
                    "Monthly Rolling Frequency (30D)",
                    "Estimated New Spend (30D)",
                    "Estimated Current Spend (30D)",
                    "Fatigue Index (30D)",
                ]
            ],
            on="Ad name",
            how="left",
        ).sort_values("Fatigue Index (30D)", ascending=False)
        st.subheader("Ad Fatigue Table")
        st.dataframe(
            fatigue_ads[
                [
                    "Ad name",
                    "Ad delivery",
                    "Amount spent (USD)",
                    "Weekly Rolling Frequency (7D)",
                    "Monthly Rolling Frequency (30D)",
                    "Reach",
                    "Impressions",
                    "Estimated New Spend (30D)",
                    "Estimated Current Spend (30D)",
                    "Fatigue Index (30D)",
                    "Results per $100",
                    "Cost per Result",
                ]
            ],
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

        attr_health_daily, attr_daily_detail = make_attribution_health_summary(filtered_df)
        ad_attr_summary = make_attribution_ad_summary(filtered_df)

        if attr_health_daily is None:
            st.warning("Attribution setting data not available in this dataset.")
        else:
            attr_health_daily_sorted = attr_health_daily.sort_values("Reporting starts")

            attr_cols = [c for c in attr_daily_detail["Attribution setting"].unique() if pd.notna(c)]
            is_click = any("click" in str(c).lower() for c in attr_cols)
            is_view = any("view" in str(c).lower() for c in attr_cols)

            if is_click and is_view:
                pivot_results = attr_daily_detail.pivot_table(
                    index="Reporting starts",
                    columns="Attribution setting",
                    values="Results",
                    aggfunc="sum",
                    fill_value=0
                )

                click_col = next((c for c in pivot_results.columns if "click" in str(c).lower()), None)
                view_col = next((c for c in pivot_results.columns if "view" in str(c).lower()), None)

                fig_attr = go.Figure()

                if click_col:
                    fig_attr.add_trace(go.Bar(
                        x=pivot_results.index,
                        y=pivot_results[click_col],
                        name="Click-Based Conversions",
                        yaxis="y1",
                        marker_color="#1f77b4",
                    ))
                if view_col:
                    fig_attr.add_trace(go.Bar(
                        x=pivot_results.index,
                        y=pivot_results[view_col],
                        name="View-Based Conversions",
                        yaxis="y1",
                        marker_color="#ff7f0e",
                    ))

                fig_attr.add_trace(go.Scatter(
                    x=attr_health_daily_sorted["Reporting starts"],
                    y=attr_health_daily_sorted["Frequency"],
                    name="Frequency",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color="#2ca02c", width=3),
                ))

                freq_series = pd.to_numeric(attr_health_daily_sorted["Frequency"], errors="coerce")
                if overlay_enabled("Attribution Frequency Trend") and freq_series.notna().sum() >= 5:
                    freq_roll_mean = freq_series.rolling(14, min_periods=5).mean()
                    freq_roll_std = freq_series.rolling(14, min_periods=5).std(ddof=0)
                    if interval_mode == "predictive":
                        freq_roll_mean = freq_roll_mean.shift(1)
                        freq_roll_std = freq_roll_std.shift(1)
                    freq_upper = freq_roll_mean + 1.96 * freq_roll_std
                    freq_lower = freq_roll_mean - 1.96 * freq_roll_std
                    if show_ci_bands:
                        fig_attr.add_trace(go.Scatter(
                            x=attr_health_daily_sorted["Reporting starts"],
                            y=freq_upper,
                            yaxis="y2",
                            mode="lines",
                            line=dict(width=0),
                            hoverinfo="skip",
                            showlegend=False,
                            name="Frequency 95% Upper",
                        ))
                        fig_attr.add_trace(go.Scatter(
                            x=attr_health_daily_sorted["Reporting starts"],
                            y=freq_lower,
                            yaxis="y2",
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor="rgba(44, 160, 44, 0.14)",
                            hoverinfo="skip",
                            showlegend=False,
                            name="Frequency 95% CI",
                        ))

                    freq_anomaly = (freq_series > freq_upper) | (freq_series < freq_lower)
                    if show_anomaly_markers and freq_anomaly.fillna(False).any():
                        fig_attr.add_trace(go.Scatter(
                            x=attr_health_daily_sorted.loc[freq_anomaly, "Reporting starts"],
                            y=freq_series[freq_anomaly],
                            yaxis="y2",
                            mode="markers",
                            marker=dict(color="red", size=9, symbol="x"),
                            name="Frequency anomaly",
                        ))

                fig_attr.update_layout(
                    title="Frequency vs Attribution Split (7D Click vs 1D View)",
                    barmode="stack",
                    yaxis=dict(title="Conversions (Stacked)"),
                    yaxis2=dict(title="Frequency", overlaying="y", side="right"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_attr, width="stretch")

            st.subheader("Frequency Ceiling Analysis (Click-Based ROAS)")
            if ad_attr_summary is not None and "ROAS" in ad_attr_summary.columns:
                scatter_data = ad_attr_summary[
                    (ad_attr_summary["Frequency"].notna()) & 
                    (ad_attr_summary["ROAS"].notna()) &
                    (ad_attr_summary["Frequency"] > 0) &
                    (ad_attr_summary["ROAS"] > 0)
                ].copy()

                if not scatter_data.empty:
                    fig_ceiling = px.scatter(
                        scatter_data,
                        x="Frequency",
                        y="ROAS",
                        size="Amount spent (USD)",
                        color="CTR %",
                        hover_name="Ad name",
                        hover_data=["Amount spent (USD)", "Scaling Efficiency Index"],
                        title="Frequency Ceiling: Where Does Click-ROAS Drop?",
                        labels={"Frequency": "Weekly Frequency", "ROAS": "7-Day Click ROAS"},
                    )
                    fig_ceiling.add_hline(
                        y=scatter_data["ROAS"].median(),
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Median ROAS",
                    )
                    st.plotly_chart(fig_ceiling, width="stretch")

                    freq_threshold = 2.5
                    target_roas = scatter_data["ROAS"].quantile(0.75)

                    st.subheader("Scaling Alert Status")
                    alert_col1, alert_col2, alert_col3 = st.columns(3)

                    green_count = len(scatter_data[(scatter_data["Frequency"] < freq_threshold) & (scatter_data["ROAS"] > target_roas)])
                    yellow_count = len(scatter_data[(scatter_data["Frequency"] >= freq_threshold) & (scatter_data["ROAS"] >= target_roas * 0.8)])
                    red_count = len(scatter_data[(scatter_data["Frequency"] >= freq_threshold) & (scatter_data["ROAS"] < target_roas * 0.8)])

                    with alert_col1:
                        st.metric("🟢 Continue Scaling", green_count, help=f"Frequency < {freq_threshold} AND ROAS > {target_roas:.2f}")
                    with alert_col2:
                        st.metric("🟡 Monitor (High Intent)", yellow_count, help=f"Frequency ≥ {freq_threshold} AND ROAS stable")
                    with alert_col3:
                        st.metric("🔴 Refresh Creative", red_count, help=f"Frequency ≥ {freq_threshold} AND ROAS declining")

                    st.subheader("Ad Scaling Status Breakdown")
                    status_df = scatter_data.copy()
                    status_df["Scaling Status"] = "🟡 Monitor"
                    status_df.loc[(status_df["Frequency"] < freq_threshold) & (status_df["ROAS"] > target_roas), "Scaling Status"] = "🟢 Continue Scaling"
                    status_df.loc[(status_df["Frequency"] >= freq_threshold) & (status_df["ROAS"] < target_roas * 0.8), "Scaling Status"] = "🔴 Refresh Creative"

                    st.dataframe(
                        status_df[[
                            "Ad name",
                            "Ad delivery",
                            "Amount spent (USD)",
                            "Frequency",
                            "ROAS",
                            "Scaling Efficiency Index",
                            "CPM",
                            "CTR %",
                            "Scaling Status",
                        ]].sort_values("Scaling Efficiency Index", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )

                    st.subheader("Scaling Efficiency Index (ROAS / Frequency)")
                    efficiency_fig = px.bar(
                        status_df.sort_values("Scaling Efficiency Index", ascending=False).head(15),
                        x="Ad name",
                        y="Scaling Efficiency Index",
                        color="Scaling Status",
                        hover_data=["Frequency", "ROAS"],
                        title="Top Ads by Scaling Efficiency",
                        color_discrete_map={
                            "🟢 Continue Scaling": "#00cc96",
                            "🟡 Monitor": "#ffa15a",
                            "🔴 Refresh Creative": "#ef553b",
                        },
                    )
                    st.plotly_chart(efficiency_fig, width="stretch")

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


if __name__ == "__main__":
    main()
