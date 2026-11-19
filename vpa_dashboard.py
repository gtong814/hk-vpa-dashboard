import streamlit as st
import yfinance as yf
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

# ==========================================
# 1. 頁面設定與 Session State 初始化
# ==========================================
st.set_page_config(
    page_title="全港股 VPA 回踩買點掃描儀",
    page_icon="📊",
    layout="wide"
)

if "scan_results" not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()
if "stock_dfs" not in st.session_state:
    st.session_state.stock_dfs = {}
if "last_scan_info" not in st.session_state:
    st.session_state.last_scan_info = ""


# ==========================================
# 2. 獲取全港股名單 (快取 1 小時)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_all_hk_tickers(pre_filter_turnover):
    ticker_map = {}
    source = ""
    try:
        spot_df = ak.stock_hk_spot_em()
        spot_df["成交额"] = pd.to_numeric(spot_df["成交额"], errors="coerce").fillna(0)
        spot_df["code_int"] = pd.to_numeric(spot_df["代码"], errors="coerce")
        active_df = spot_df[
            (spot_df["code_int"] >= 1) &
            (spot_df["code_int"] <= 9999) &
            (spot_df["成交额"] >= pre_filter_turnover)
        ]
        for _, row in active_df.iterrows():
            yf_code = f"{int(row['code_int']):04d}.HK"
            ticker_map[yf_code] = str(row["名称"])
        source = "AkShare 即時預篩"
    except Exception:
        url = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"
        df = pd.read_excel(url, skiprows=2)
        eq_df = df[df["Category"] == "Equity"].copy()
        eq_df["code_int"] = pd.to_numeric(eq_df["Stock Code"], errors="coerce")
        eq_df = eq_df[(eq_df["code_int"] >= 1) & (eq_df["code_int"] <= 9999)]
        for _, row in eq_df.iterrows():
            yf_code = f"{int(row['code_int']):04d}.HK"
            ticker_map[yf_code] = str(row["Name of Securities"])
        source = "HKEX 官方備援名單"
    return ticker_map, source


# ==========================================
# 3. 單一股票 VPA 分析 (多執行緒安全)
# ==========================================
def analyze_single_stock(ticker, name, params):
    try:
        df = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=False)
        if df.empty or len(df) < 60:
            return None

        df = df[df["Volume"] > 0].copy()
        if len(df) < 60:
            return None

        # 基礎指標計算
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
        df["Turnover_SMA20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

        # VPA 核心變數
        df["RVOL"] = df["Volume"] / df["Vol_SMA20"]
        df["Body"] = (df["Close"] - df["Open"]).abs()
        df["RelSpread"] = df["Body"] / df["Body"].rolling(20).mean()
        df["Range"] = df["High"] - df["Low"]
        df["CLV"] = np.where(df["Range"] > 0, (df["Close"] - df["Low"]) / df["Range"], 0.5)

        t0, t1, t2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]

        # 1. 流動性與多頭趨勢濾網
        if t0["Turnover_SMA20"] < params["min_avg_turnover"]:
            return None
        if not (t0["Close"] > t0["SMA50"] and t0["SMA20"] > t0["SMA50"]):
            return None

        # 2. 回踩 20MA 支撐區
        support_mult = 1 + params["support_buffer"] / 100.0
        near_support_t0 = t0["Low"] <= t0["SMA20"] * support_mult
        near_support_t1 = t1["Low"] <= t1["SMA20"] * support_mult

        # 條件 A：縮量測試 (觀察)
        is_no_supply_t0 = (
            near_support_t0 and
            t0["RelSpread"] <= params["max_spread"] and
            t0["RVOL"] <= params["low_rvol"] and
            t0["Volume"] < t1["Volume"] and
            t0["Volume"] < t2["Volume"]
        )

        # 條件 B：昨日測試或承接 + 今日確認轉強
        was_no_supply_t1 = (
            near_support_t1 and
            t1["RelSpread"] <= (params["max_spread"] + 0.05) and
            t1["RVOL"] <= (params["low_rvol"] + 0.05)
        )
        was_absorption_t1 = (
            near_support_t1 and
            t1["RVOL"] >= params["high_rvol"] and
            t1["CLV"] >= 0.65
        )
        is_confirmed_t0 = (
            (was_no_supply_t1 or was_absorption_t1) and
            t0["Close"] > t1["High"] and
            t0["Close"] > t0["Open"] and
            t0["Volume"] > t1["Volume"]
        )

        if is_no_supply_t0 or is_confirmed_t0:
            signal_type = "🔥確認轉強買點" if is_confirmed_t0 else "👀縮量測試(觀察)"
            stop_loss = min(t0["Low"], t1["Low"]) * 0.99
            dist_to_sl = (t0["Close"] - stop_loss) / t0["Close"] * 100

            summary = {
                "代碼": ticker,
                "名稱": name,
                "訊號": signal_type,
                "優先級": 1 if is_confirmed_t0 else 2,
                "收盤價": round(float(t0["Close"]), 2),
                "參考止損": round(float(stop_loss), 2),
                "止損距離(%)": round(float(dist_to_sl), 2),
                "RVOL": round(float(t0["RVOL"]), 2),
                "CLV": round(float(t0["CLV"]), 2),
                "RelSpread": round(float(t0["RelSpread"]), 2),
                "20日均額(M)": round(float(t0["Turnover_SMA20"]) / 1_000_000, 1),
            }
            # 僅保留近 60 日資料供繪圖使用，節省記憶體
            return summary, df.tail(60)
    except Exception:
        return None
    return None


# ==========================================
# 4. Plotly 互動式量價 K 線圖
# ==========================================
def plot_vpa_chart(df, ticker, name, signal, stop_loss):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28]
    )

    date_labels = df.index.strftime("%Y-%m-%d")

    # K 線圖
    fig.add_trace(
        go.Candlestick(
            x=date_labels,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="K線",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )

    # 均線
    fig.add_trace(
        go.Scatter(x=date_labels, y=df["SMA20"], name="SMA 20", line=dict(color="#ffa726", width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=date_labels, y=df["SMA50"], name="SMA 50", line=dict(color="#42a5f5", width=1.5)),
        row=1, col=1
    )

    # 參考止損線
    fig.add_hline(
        y=stop_loss,
        line_dash="dash",
        line_color="#ff5252",
        annotation_text=f"參考止損: {stop_loss}",
        annotation_position="bottom right",
        row=1, col=1
    )

    # 成交量柱狀圖 + 20日均量
    vol_colors = np.where(df["Close"] >= df["Open"], "rgba(38, 166, 154, 0.7)", "rgba(239, 83, 80, 0.7)")
    fig.add_trace(
        go.Bar(x=date_labels, y=df["Volume"], name="成交量", marker_color=vol_colors),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=date_labels, y=df["Vol_SMA20"], name="20日均量", line=dict(color="#ab47bc", width=1.5)),
        row=2, col=1
    )

    fig.update_layout(
        title=f"{ticker} {name} — {signal}",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(type="category", nticks=10)
    return fig


# ==========================================
# 5. 側邊欄參數控制
# ==========================================
with st.sidebar:
    st.header("⚙️ 掃描參數設定")
    min_avg_m = st.number_input("20日平均成交額門檻 (百萬港元)", min_value=1, max_value=500, value=20, step=5)
    pre_filter_m = st.number_input("AkShare 預篩門檻 (百萬港元)", min_value=1, max_value=200, value=8, step=2)
    
    st.divider()
    st.subheader("VPA 條件微調")
    support_buffer = st.slider("回踩 20MA 容許距離 (%)", 0.0, 5.0, 2.0, 0.5)
    low_rvol = st.slider("縮量測試 RVOL 上限", 0.30, 0.90, 0.65, 0.05)
    max_spread = st.slider("縮量測試 RelSpread 上限", 0.30, 1.00, 0.65, 0.05)
    high_rvol = st.slider("止跌承接 RVOL 下限", 1.20, 3.00, 1.50, 0.10)
    
    st.divider()
    max_workers = st.slider("多執行緒數量 (Workers)", 4, 16, 10, 1)
    run_btn = st.button("🚀 開始全港股掃描", type="primary", use_container_width=True)


# ==========================================
# 6. 主畫面佈局與掃描執行
# ==========================================
st.title("📊 全港股 VPA 回踩買點掃描儀")
st.caption("結合威科夫量價分析（縮量測試 / 支撐承接確認）與多執行緒加速引擎")

if run_btn:
    params = {
        "min_avg_turnover": min_avg_m * 1_000_000,
        "support_buffer": support_buffer,
        "low_rvol": low_rvol,
        "max_spread": max_spread,
        "high_rvol": high_rvol,
    }

    start_time = time.time()
    with st.spinner("正在獲取全港股活躍名單..."):
        ticker_map, source = get_all_hk_tickers(pre_filter_m * 1_000_000)

    total = len(ticker_map)
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []
    stock_dfs = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_single_stock, ticker, name, params): ticker
            for ticker, name in ticker_map.items()
        }
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                summary, df_tail = res
                results.append(summary)
                stock_dfs[summary["代碼"]] = df_tail
            
            if i % 10 == 0 or i == total:
                progress_bar.progress(i / total)
                status_text.text(f"掃描進度：{i} / {total} 隻 | 已發現訊號：{len(results)} 隻")

    elapsed = round(time.time() - start_time, 1)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress_bar.empty()
    status_text.empty()

    if results:
        res_df = pd.DataFrame(results).sort_values(
            by=["優先級", "20日均額(M)"], ascending=[True, False]
        ).drop(columns=["優先級"]).reset_index(drop=True)
        st.session_state.scan_results = res_df
        st.session_state.stock_dfs = stock_dfs
    else:
        st.session_state.scan_results = pd.DataFrame()
        st.session_state.stock_dfs = {}

    st.session_state.last_scan_info = f"最後掃描：{now_str} | 名單來源：{source} ({total} 隻) | 耗時：{elapsed} 秒"


# ==========================================
# 7. 顯示掃描結果與互動圖表
# ==========================================
if st.session_state.last_scan_info:
    st.info(st.session_state.last_scan_info)

df_res = st.session_state.scan_results

if not df_res.empty:
    # 頂部統計卡片
    c1, c2, c3 = st.columns(3)
    confirmed_cnt = (df_res["訊號"] == "🔥確認轉強買點").sum()
    watch_cnt = (df_res["訊號"] == "👀縮量測試(觀察)").sum()
    c1.metric("總命中標的", f"{len(df_res)} 隻")
    c2.metric("🔥 確認轉強買點", f"{confirmed_cnt} 隻")
    c3.metric("👀 縮量測試 (觀察)", f"{watch_cnt} 隻")

    # 訊號篩選與匯出
    col_filter, col_export = st.columns([3, 1])
    with col_filter:
        signal_filter = st.multiselect(
            "篩選訊號類型",
            options=["🔥確認轉強買點", "👀縮量測試(觀察)"],
            default=["🔥確認轉強買點", "👀縮量測試(觀察)"]
        )
    filtered_df = df_res[df_res["訊號"].isin(signal_filter)]

    with col_export:
        st.write("")
        csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 下載 CSV 清單",
            data=csv_data,
            file_name=f"hk_vpa_pullback_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 結果表格
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # 互動 K 線檢視區
    st.divider()
    st.subheader("📈 個股量價結構檢視")
    if not filtered_df.empty:
        options = [f"{r['代碼']} - {r['名稱']} ({r['訊號']})" for _, r in filtered_df.iterrows()]
        selected_label = st.selectbox("選擇要查看 K 線與止損位的股票：", options)
        selected_ticker = selected_label.split(" - ")[0]
        row_info = filtered_df[filtered_df["代碼"] == selected_ticker].iloc[0]

        chart_df = st.session_state.stock_dfs.get(selected_ticker)
        if chart_df is not None:
            fig = plot_vpa_chart(
                chart_df,
                selected_ticker,
                row_info["名稱"],
                row_info["訊號"],
                row_info["參考止損"]
            )
            st.plotly_chart(fig, use_container_width=True)
elif st.session_state.last_scan_info:
    st.warning("今日在目前參數設定下，無符合 VPA 回踩買點之標的，可於左側微調門檻後重試。")
else:
    st.markdown("👈 請在左側確認參數後，點擊 **「🚀 開始全港股掃描」**。")