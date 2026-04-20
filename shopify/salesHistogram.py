import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

# --- CONFIGURATION ---
FILE_PATH = 'orders_export.csv'

# Primary date range (inclusive). Format: 'YYYY-MM-DD'
DATE_START = "2026-03-01"
DATE_END   = "2026-03-30"

# ── Comparison mode ────────────────────────────────────────────────────────────
# Set to True to overlay / compare a second period on every chart
COMPARE_PERIODS = True

# Second date range (only used when COMPARE_PERIODS = True)
DATE_START_2 = "2026-04-01"
DATE_END_2   = "2026-04-30"

# Number of top products to show in the units-sold bar chart
TOP_N_PRODUCTS = 10
# ──────────────────────────────────────────────────────────────────────────────

LABEL1 = f"{DATE_START} – {DATE_END}"
LABEL2 = f"{DATE_START_2} – {DATE_END_2}"
# ---------------------

# ── Helpers ───────────────────────────────────────────────────────────────────

def _date_mask(df, date_col, start, end):
    return (
        (df[date_col] >= pd.Timestamp(start, tz='UTC')) &
        (df[date_col] <= pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    )

# Cost assigned to orders where shipping was provided for free
FREE_SHIPPING_COST = 5.0

def _prep_df(df, qty_col, price_col, name_col, discount_col=None):
    df = df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
    df[qty_col]   = pd.to_numeric(df[qty_col],   errors='coerce').fillna(0)
    df['_base']   = df[name_col].str.split(' - ').str[0].str.strip()
    df['_revenue'] = df[price_col] * df[qty_col]
    # Subtract per-line discount if the column exists
    if discount_col and discount_col in df.columns:
        df[discount_col] = pd.to_numeric(df[discount_col], errors='coerce').fillna(0)
        df['_revenue'] -= df[discount_col]
    return df

def _product_revenue(df, qty_col, name_col, price_col, discount_col=None):
    df = _prep_df(df, qty_col, price_col, name_col, discount_col)
    return (
        df.groupby('_base')['_revenue']
        .sum()
        .sort_values(ascending=False)
    )

def _product_units(df, qty_col, name_col, price_col, discount_col=None):
    df = _prep_df(df, qty_col, price_col, name_col, discount_col)
    return (
        df.groupby('_base')[qty_col]
        .sum()
        .sort_values(ascending=False)
    )

def _order_value(df, id_col, qty_col, name_col, price_col, discount_col=None, shipping_col=None):
    """Net order value = (Total - Shipping) per order.
    Total and Shipping are order-level fields repeated on every line-item row,
    so we take the first value per order via .first()."""
    df = df.copy()
    df['Total']    = pd.to_numeric(df['Total'],    errors='coerce').fillna(0)
    df[shipping_col] = pd.to_numeric(df[shipping_col], errors='coerce').fillna(0)
    order_total    = df.groupby(id_col)['Total'].first()
    order_shipping = df.groupby(id_col)[shipping_col].first()
    return order_total - order_shipping

def _build_pie_slices(revenue):
    threshold = revenue.sum() * 0.02
    main  = revenue[revenue >= threshold]
    other = revenue[revenue < threshold].sum()
    if other > 0:
        main = pd.concat([main, pd.Series({'Other': other})])
    return main

def _draw_pie(ax, revenue, title):
    if revenue is None or revenue.empty:
        ax.text(0.5, 0.5, 'No price data available',
                ha='center', va='center', transform=ax.transAxes, fontsize=13)
    else:
        main = _build_pie_slices(revenue)
        _, _, autotexts = ax.pie(
            main.values,
            labels=main.index,
            autopct=lambda p: f'{p:.1f}%' if p >= 2 else '',
            startangle=140,
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(8)
    ax.set_title(title, fontsize=11)

def _hide_all(axes_list):
    for a in axes_list:
        a.set_visible(False)

def _show_all(axes_list):
    for a in axes_list:
        a.set_visible(True)

# ── Tab-button helpers ────────────────────────────────────────────────────────

def _make_tab_buttons(fig, labels, TAB_ACTIVE, TAB_INACTIVE, TAB_TEXT_ON, TAB_TEXT_OFF):
    """Create evenly-spaced tab buttons and return (btn_list, ax_list)."""
    n = len(labels)
    w, h, gap = 0.16, 0.055, 0.005
    start_x = 0.07
    btn_axes = [fig.add_axes([start_x + i * (w + gap), 0.925, w, h]) for i in range(n)]
    btns = []
    for i, (ax, label) in enumerate(zip(btn_axes, labels)):
        color = TAB_ACTIVE if i == 0 else TAB_INACTIVE
        b = Button(ax, label, color=color, hovercolor='#c8cef0')
        b.label.set_color(TAB_TEXT_ON if i == 0 else TAB_TEXT_OFF)
        b.label.set_fontsize(8)
        btns.append(b)
    return btns, btn_axes

# ── Main ──────────────────────────────────────────────────────────────────────

def generate_order_histogram(file_path):
    try:
        df_raw = pd.read_csv(file_path, low_memory=False)

        id_col       = 'Name'
        qty_col      = 'Lineitem quantity'
        date_col     = 'Created at'
        name_col     = 'Lineitem name'
        price_col    = 'Lineitem price'
        discount_col = 'Lineitem discount'  # per-line discount amount
        shipping_col = 'Shipping'             # order-level shipping charged (repeated per row)

        for col in (id_col, qty_col, date_col):
            if col not in df_raw.columns:
                print(f"Error: Missing required column '{col}'.")
                return

        df_raw[date_col] = pd.to_datetime(df_raw[date_col], utc=True)

        # ── Filter periods ────────────────────────────────────────────────────
        df1 = df_raw[_date_mask(df_raw, date_col, DATE_START, DATE_END)].copy()
        df2 = df_raw[_date_mask(df_raw, date_col, DATE_START_2, DATE_END_2)].copy() \
              if COMPARE_PERIODS else None

        if df1.empty:
            print(f"No orders found between {DATE_START} and {DATE_END}.")
            return
        if COMPARE_PERIODS and (df2 is None or df2.empty):
            print(f"No orders found between {DATE_START_2} and {DATE_END_2}.")
            return

        has_price = name_col in df_raw.columns and price_col in df_raw.columns

        # ─ Per-order item counts ──────────────────────────────────────────────
        totals1 = df1.groupby(id_col)[qty_col].sum().reset_index()
        totals2 = df2.groupby(id_col)[qty_col].sum().reset_index() if COMPARE_PERIODS else None

        # ─ Per-order revenue (net of discounts & shipping) ────────────────────
        ov1 = _order_value(df1, id_col, qty_col, name_col, price_col, discount_col, shipping_col) if has_price else None
        ov2 = _order_value(df2, id_col, qty_col, name_col, price_col, discount_col, shipping_col) if (has_price and COMPARE_PERIODS) else None

        # ─ Revenue & units per product ────────────────────────────────────────
        rev1   = _product_revenue(df1, qty_col, name_col, price_col, discount_col) if has_price else None
        rev2   = _product_revenue(df2, qty_col, name_col, price_col, discount_col) if (has_price and COMPARE_PERIODS) else None
        units1 = _product_units(df1, qty_col, name_col, price_col, discount_col) if has_price else None
        units2 = _product_units(df2, qty_col, name_col, price_col, discount_col) if (has_price and COMPARE_PERIODS) else None

        # ─ Daily revenue ─────────────────────────────────────────────────────
        def daily_rev(df):
            d = _prep_df(df, qty_col, price_col, name_col)
            d['_day'] = d[date_col].dt.normalize()
            return d.groupby('_day')['_revenue'].sum().sort_index()

        dr1 = daily_rev(df1) if has_price else None
        dr2 = daily_rev(df2) if (has_price and COMPARE_PERIODS) else None

        # ─ KPIs ──────────────────────────────────────────────────────────────
        def kpis(df, totals, ov):
            total_rev  = ov.sum()       if ov     is not None else 0
            total_ord  = len(totals)
            aov        = ov.mean()      if ov     is not None else 0
            total_units = totals[qty_col].sum()
            return total_rev, total_ord, aov, total_units

        k1 = kpis(df1, totals1, ov1)
        k2 = kpis(df2, totals2, ov2) if COMPARE_PERIODS else None

        # ── Colours / style ───────────────────────────────────────────────────
        C1           = '#5c6ac4'
        C2           = '#f49342'
        TAB_ACTIVE   = '#5c6ac4'
        TAB_INACTIVE = '#dde1f0'
        TAB_TEXT_ON  = 'white'
        TAB_TEXT_OFF = '#333333'

        # ── Figure ────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 8))
        fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.97, hspace=0.45, wspace=0.35)

        # We'll collect axes per tab in a list of lists
        tab_axes = []   # tab_axes[i] = list of axes for tab i

        # ════════════════════════════════════════════════════════════════════
        # TAB 1 – Order Profile  (basket-size histogram | order-value histogram)
        # ════════════════════════════════════════════════════════════════════
        gs1 = gridspec.GridSpec(1, 2, figure=fig,
                                left=0.07, right=0.97, top=0.88, bottom=0.10,
                                wspace=0.35)
        ax_basket = fig.add_subplot(gs1[0, 0])
        ax_oval   = fig.add_subplot(gs1[0, 1])

        # Basket-size bars
        max_qty = int(max(totals1[qty_col].max(),
                          totals2[qty_col].max() if COMPARE_PERIODS else 0))
        x = list(range(1, max_qty + 1))
        w = 0.35 if COMPARE_PERIODS else 0.6
        c1_basket = [int((totals1[qty_col] == v).sum()) for v in x]
        b1 = ax_basket.bar([v - w/2 if COMPARE_PERIODS else v for v in x],
                           c1_basket, width=w, color=C1, edgecolor='black',
                           alpha=0.85, label=LABEL1)
        ax_basket.bar_label(b1, padding=2, fontsize=7)
        if COMPARE_PERIODS:
            c2_basket = [int((totals2[qty_col] == v).sum()) for v in x]
            b2 = ax_basket.bar([v + w/2 for v in x], c2_basket, width=w,
                               color=C2, edgecolor='black', alpha=0.85, label=LABEL2)
            ax_basket.bar_label(b2, padding=2, fontsize=7)
            ax_basket.legend(fontsize=7)
        ax_basket.set_title('Items per Order', fontsize=11)
        ax_basket.set_xlabel('# Items', fontsize=9)
        ax_basket.set_ylabel('# Orders', fontsize=9)
        ax_basket.set_xticks(x)
        ax_basket.grid(axis='y', linestyle='--', alpha=0.4)

        # Order-value distribution
        if ov1 is not None:
            max_val = max(ov1.max(), ov2.max() if ov2 is not None else 0)
            bins_ov = np.linspace(0, max_val * 1.05, 20)
            ax_oval.hist(ov1, bins=bins_ov, color=C1, edgecolor='black',
                         alpha=0.7, label=LABEL1)
            if COMPARE_PERIODS and ov2 is not None:
                ax_oval.hist(ov2, bins=bins_ov, color=C2, edgecolor='black',
                             alpha=0.6, label=LABEL2)
                ax_oval.legend(fontsize=7)
            ax_oval.set_title('Order Value Distribution', fontsize=11)
            ax_oval.set_xlabel('Order Value ($)', fontsize=9)
            ax_oval.set_ylabel('# Orders', fontsize=9)
            ax_oval.grid(axis='y', linestyle='--', alpha=0.4)
        else:
            ax_oval.text(0.5, 0.5, 'No price data', ha='center', va='center',
                         transform=ax_oval.transAxes)
            ax_oval.set_title('Order Value Distribution', fontsize=11)

        tab_axes.append([ax_basket, ax_oval])

        # ════════════════════════════════════════════════════════════════════
        # TAB 2 – Revenue Over Time  (daily revenue line chart)
        # ════════════════════════════════════════════════════════════════════
        gs2 = gridspec.GridSpec(1, 1, figure=fig,
                                left=0.07, right=0.97, top=0.88, bottom=0.10)
        ax_daily = fig.add_subplot(gs2[0, 0])
        ax_daily.set_visible(False)

        if dr1 is not None:
            # Normalize x-axis to "day N of period" so periods overlay cleanly
            days1 = np.arange(len(dr1))
            ax_daily.plot(days1, dr1.values, color=C1, linewidth=2,
                          marker='o', markersize=4, label=LABEL1)
            if COMPARE_PERIODS and dr2 is not None:
                days2 = np.arange(len(dr2))
                ax_daily.plot(days2, dr2.values, color=C2, linewidth=2,
                              marker='s', markersize=4, label=LABEL2)
                ax_daily.legend(fontsize=9)
            ax_daily.set_title('Daily Revenue', fontsize=13)
            ax_daily.set_xlabel('Day of Period', fontsize=10)
            ax_daily.set_ylabel('Revenue ($)', fontsize=10)
            ax_daily.grid(linestyle='--', alpha=0.4)
        else:
            ax_daily.text(0.5, 0.5, 'No price data', ha='center', va='center',
                          transform=ax_daily.transAxes, fontsize=13)
            ax_daily.set_title('Daily Revenue', fontsize=13)

        tab_axes.append([ax_daily])

        # ════════════════════════════════════════════════════════════════════
        # TAB 3 – Business KPIs  (2×2 bar grid)
        # ════════════════════════════════════════════════════════════════════
        gs3 = gridspec.GridSpec(2, 2, figure=fig,
                                left=0.08, right=0.97, top=0.88, bottom=0.08,
                                hspace=0.55, wspace=0.4)
        kpi_axes = [fig.add_subplot(gs3[r, c]) for r in range(2) for c in range(2)]
        for a in kpi_axes:
            a.set_visible(False)

        kpi_labels  = ['Total Revenue ($)', 'Total Orders', 'Avg Order Value ($)', 'Total Units Sold']
        kpi_values1 = list(k1)
        kpi_values2 = list(k2) if k2 else None
        kpi_fmt     = ['${:,.2f}', '{:,}', '${:,.2f}', '{:,}']

        for i, (ax_k, lbl, v1, fmt) in enumerate(zip(kpi_axes, kpi_labels, kpi_values1, kpi_fmt)):
            if COMPARE_PERIODS and kpi_values2:
                v2 = kpi_values2[i]
                positions = [0, 1]
                bars_k = ax_k.bar(positions, [v1, v2], color=[C1, C2],
                                  edgecolor='black', alpha=0.85, width=0.5)
                ax_k.set_xticks(positions)
                ax_k.set_xticklabels([LABEL1, LABEL2], fontsize=7, rotation=10, ha='right')
                for bar, val in zip(bars_k, [v1, v2]):
                    ax_k.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() * 1.02,
                              fmt.format(val), ha='center', va='bottom', fontsize=7)
                pct = ((v2 - v1) / v1 * 100) if v1 else 0
                sign = '+' if pct >= 0 else ''
                color = 'green' if pct >= 0 else 'red'
                ax_k.set_title(f'{lbl}\n({sign}{pct:.1f}%)', fontsize=9, color=color)
            else:
                bar_k = ax_k.bar([0], [v1], color=C1, edgecolor='black', alpha=0.85, width=0.4)
                ax_k.set_xticks([0])
                ax_k.set_xticklabels([LABEL1], fontsize=7)
                ax_k.text(0, v1 * 1.02, fmt.format(v1), ha='center', va='bottom', fontsize=8)
                ax_k.set_title(lbl, fontsize=9)
            ax_k.grid(axis='y', linestyle='--', alpha=0.4)

        tab_axes.append(kpi_axes)

        # ════════════════════════════════════════════════════════════════════
        # TAB 4 – Product Breakdown  (pie(s) top | top-N units bar bottom)
        # ════════════════════════════════════════════════════════════════════
        gs4 = gridspec.GridSpec(2, 2 if COMPARE_PERIODS else 1, figure=fig,
                                left=0.07, right=0.97, top=0.88, bottom=0.08,
                                hspace=0.5, wspace=0.35)

        ax_pie1 = fig.add_subplot(gs4[0, 0])
        ax_pie1.set_visible(False)
        _draw_pie(ax_pie1, rev1, f'Revenue Mix – {LABEL1}')

        if COMPARE_PERIODS:
            ax_pie2 = fig.add_subplot(gs4[0, 1])
            ax_pie2.set_visible(False)
            _draw_pie(ax_pie2, rev2, f'Revenue Mix – {LABEL2}')
            ax_topn = fig.add_subplot(gs4[1, :])
        else:
            ax_pie2 = None
            ax_topn = fig.add_subplot(gs4[1, 0])
        ax_topn.set_visible(False)

        # Top-N units sold horizontal bar chart
        if units1 is not None:
            top1 = units1.head(TOP_N_PRODUCTS)
            all_prods = list(top1.index)
            if COMPARE_PERIODS and units2 is not None:
                top2 = units2.reindex(all_prods).fillna(0)
            y = np.arange(len(all_prods))
            bh = 0.35 if COMPARE_PERIODS else 0.6
            ax_topn.barh(y + bh/2 if COMPARE_PERIODS else y,
                         top1.values, height=bh, color=C1, edgecolor='black',
                         alpha=0.85, label=LABEL1)
            if COMPARE_PERIODS and units2 is not None:
                ax_topn.barh(y - bh/2, top2.values, height=bh, color=C2,
                             edgecolor='black', alpha=0.85, label=LABEL2)
                ax_topn.legend(fontsize=8)
            ax_topn.set_yticks(y)
            ax_topn.set_yticklabels(all_prods, fontsize=8)
            ax_topn.invert_yaxis()
            ax_topn.set_xlabel('Units Sold', fontsize=9)
            ax_topn.set_title(f'Top {TOP_N_PRODUCTS} Products – Units Sold', fontsize=11)
            ax_topn.grid(axis='x', linestyle='--', alpha=0.4)
        else:
            ax_topn.text(0.5, 0.5, 'No product data', ha='center', va='center',
                         transform=ax_topn.transAxes)

        prod_axes = [ax_pie1, ax_topn] + ([ax_pie2] if ax_pie2 else [])
        tab_axes.append(prod_axes)

        # ── Build tab buttons ─────────────────────────────────────────────────
        tab_labels = ['Order Profile', 'Revenue Over Time', 'Business KPIs', 'Product Breakdown']
        btns, _ = _make_tab_buttons(fig, tab_labels, TAB_ACTIVE, TAB_INACTIVE, TAB_TEXT_ON, TAB_TEXT_OFF)

        def _switch_tab(active_idx):
            for i, axes in enumerate(tab_axes):
                visible = (i == active_idx)
                for a in axes:
                    a.set_visible(visible)
            for i, b in enumerate(btns):
                if i == active_idx:
                    b.ax.set_facecolor(TAB_ACTIVE)
                    b.label.set_color(TAB_TEXT_ON)
                else:
                    b.ax.set_facecolor(TAB_INACTIVE)
                    b.label.set_color(TAB_TEXT_OFF)
            fig.canvas.draw_idle()

        # Wire up buttons (use default arg capture)
        for idx, btn in enumerate(btns):
            btn.on_clicked(lambda event, i=idx: _switch_tab(i))

        # Show tab 0 by default (already visible from gridspec creation)
        # Hide all non-tab-0 axes
        for i in range(1, len(tab_axes)):
            for a in tab_axes[i]:
                a.set_visible(False)

        plt.show()

        # ── Console summary ───────────────────────────────────────────────────
        def print_summary(label, totals, kpi):
            print(f"  Period     : {label}")
            print(f"  Orders     : {kpi[1]:,}")
            print(f"  Revenue    : ${kpi[0]:,.2f}")
            print(f"  Avg OV     : ${kpi[2]:,.2f}")
            print(f"  Units sold : {kpi[3]:,}")

        print("-" * 45)
        print_summary(LABEL1, totals1, k1)
        if COMPARE_PERIODS and k2:
            print()
            print_summary(LABEL2, totals2, k2)
        print("-" * 45)

    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_order_histogram(FILE_PATH)