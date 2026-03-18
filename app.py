"""
S&P 500 Stock Analysis Dashboard
Streamlit app for viewing stock analysis results
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import re
import requests
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="S&P 500 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Loading ---
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "sp500_analysis.csv")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = "thekodev/sp500-dashboard"


def gemini_chat(prompt: str) -> str:
    """Call Gemini API for chat."""
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."
    try:
        resp = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return f"API Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"


BLACKLIST_FILE = os.path.join(os.path.dirname(__file__), "data", "blacklist.json")


@st.cache_data(ttl=300)
def load_data():
    """Load analysis data from CSV, auto-filter High/Critical governance."""
    if not os.path.exists(DATA_FILE):
        return None, None
    df = pd.read_csv(DATA_FILE)
    # Separate blacklisted stocks
    blacklisted = None
    if "governance_level" in df.columns:
        blacklisted = df[df["governance_level"].isin(["High", "Critical"])].copy()
        df = df[~df["governance_level"].isin(["High", "Critical"])].copy()
    return df, blacklisted


@st.cache_data(ttl=3600)
def load_blacklist():
    """Load blacklist file."""
    if os.path.exists(BLACKLIST_FILE):
        try:
            with open(BLACKLIST_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def render_header(df: pd.DataFrame):
    """Render dashboard header with key metrics."""
    st.title("📈 S&P 500 Stock Analysis Dashboard")

    date = df["date"].iloc[0] if "date" in df.columns else "N/A"
    st.caption(f"Last updated: {date} | Total stocks: {len(df)}")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    uptrend = len(df[df["trend"] == "Uptrend"])
    downtrend = len(df[df["trend"] == "Downtrend"])
    avg_sentiment = df["sentiment_score"].mean() if "sentiment_score" in df.columns else 0
    overbought = len(df[df["rsi_status"] == "Overbought"])
    oversold = len(df[df["rsi_status"] == "Oversold"])
    high_gov_risk = len(df[df["governance_level"].isin(["High", "Critical"])]) if "governance_level" in df.columns else 0

    col1.metric("Uptrend", f"{uptrend}", f"{uptrend/len(df)*100:.0f}%")
    col2.metric("Downtrend", f"{downtrend}", f"{downtrend/len(df)*100:.0f}%")
    col3.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    col4.metric("Overbought", f"{overbought}")
    col5.metric("Oversold", f"{oversold}")
    col6.metric("⚠️ High Gov Risk", f"{high_gov_risk}")


def render_sidebar(df: pd.DataFrame):
    """Sidebar — filters moved inline to table, sidebar now shows summary only."""
    return df


def render_table_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render inline filters above the overview table."""
    with st.expander("🔽 Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c5, c6, c7, c8 = st.columns(4)

        # --- Categorical ---
        sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
        sector = c1.selectbox("Sector", sectors, key="tbl_sector")

        trend = c2.selectbox("Trend", ["All", "Uptrend", "Downtrend"], key="tbl_trend")

        rsi_status = c3.selectbox("RSI Status", ["All", "Overbought", "Neutral", "Oversold"], key="tbl_rsi")

        gov_level = c4.selectbox("Gov Level", ["All", "Low", "Medium"], key="tbl_gov")

        # --- Numeric ---
        sentiment_op = c5.selectbox("Sentiment", ["Any", "> 0.1", "> 0.3", "< -0.1", "< -0.3"], key="tbl_sent_op")

        rsi_op = c6.selectbox("RSI", ["Any", "< 30", "30-70", "> 70", "< 50", "> 50"], key="tbl_rsi_val")

        change_op = c7.selectbox("1M Change", ["Any", "> 5%", "> 10%", "< -5%", "< -10%"], key="tbl_chg")

        pe_op = c8.selectbox("P/E Ratio", ["Any", "< 15", "15-25", "25-40", "> 40"], key="tbl_pe")

        # --- Search ---
        search = st.text_input("🔍 Search ticker / company", key="tbl_search")

    filtered = df.copy()

    # Categorical filters
    if sector != "All":
        filtered = filtered[filtered["sector"] == sector]
    if trend != "All":
        filtered = filtered[filtered["trend"] == trend]
    if rsi_status != "All":
        filtered = filtered[filtered["rsi_status"] == rsi_status]
    if gov_level != "All" and "governance_level" in filtered.columns:
        filtered = filtered[filtered["governance_level"] == gov_level]

    # Numeric filters
    op_map = {
        "> 0.1": ("sentiment_score", ">", 0.1), "> 0.3": ("sentiment_score", ">", 0.3),
        "< -0.1": ("sentiment_score", "<", -0.1), "< -0.3": ("sentiment_score", "<", -0.3),
    }
    if sentiment_op in op_map:
        col, op, val = op_map[sentiment_op]
        filtered = filtered[filtered[col] > val] if op == ">" else filtered[filtered[col] < val]

    if rsi_op == "< 30":
        filtered = filtered[filtered["rsi"] < 30]
    elif rsi_op == "30-70":
        filtered = filtered[(filtered["rsi"] >= 30) & (filtered["rsi"] <= 70)]
    elif rsi_op == "> 70":
        filtered = filtered[filtered["rsi"] > 70]
    elif rsi_op == "< 50":
        filtered = filtered[filtered["rsi"] < 50]
    elif rsi_op == "> 50":
        filtered = filtered[filtered["rsi"] > 50]

    if change_op == "> 5%":
        filtered = filtered[filtered["change_1m_pct"] > 5]
    elif change_op == "> 10%":
        filtered = filtered[filtered["change_1m_pct"] > 10]
    elif change_op == "< -5%":
        filtered = filtered[filtered["change_1m_pct"] < -5]
    elif change_op == "< -10%":
        filtered = filtered[filtered["change_1m_pct"] < -10]

    if pe_op == "< 15":
        filtered = filtered[filtered["pe_trailing"] < 15]
    elif pe_op == "15-25":
        filtered = filtered[(filtered["pe_trailing"] >= 15) & (filtered["pe_trailing"] <= 25)]
    elif pe_op == "25-40":
        filtered = filtered[(filtered["pe_trailing"] > 25) & (filtered["pe_trailing"] <= 40)]
    elif pe_op == "> 40":
        filtered = filtered[filtered["pe_trailing"] > 40]

    if search:
        s = search.lower()
        filtered = filtered[
            filtered["ticker"].str.lower().str.contains(s, na=False)
            | filtered["company"].str.lower().str.contains(s, na=False)
        ]

    st.caption(f"Showing **{len(filtered)}** of {len(df)} stocks")
    return filtered


def render_overview_table(df: pd.DataFrame):
    """Render overview table. Click a row to see stock detail below."""
    st.subheader("Stock Overview")

    # Inline filters
    df = render_table_filters(df)

    display_cols = [
        "ticker", "company", "sector", "price", "change_1m_pct", "change_1y_pct",
        "trend", "rsi", "rsi_status", "volatility_pct",
        "pe_trailing", "roe_pct", "market_cap_b", "sentiment_score",
        "governance_score", "governance_level", "last_update",
    ]
    available = [c for c in display_cols if c in df.columns]
    display_df = df[available].copy()

    # Format columns
    rename = {
        "ticker": "Ticker", "company": "Company", "sector": "Sector",
        "price": "Price ($)", "change_1m_pct": "1M %", "change_1y_pct": "1Y %",
        "trend": "Trend", "rsi": "RSI", "rsi_status": "RSI Status",
        "volatility_pct": "Volatility %", "pe_trailing": "P/E",
        "roe_pct": "ROE %", "market_cap_b": "MCap ($B)",
        "sentiment_score": "Sentiment",
        "governance_score": "Gov Risk",
        "governance_level": "Gov Level",
        "last_update": "Last Update",
    }
    display_df = display_df.rename(columns={k: v for k, v in rename.items() if k in display_df.columns})

    # Interactive table — clicking a row shows detail below
    event = st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "1M %": st.column_config.NumberColumn(format="%.2f%%"),
            "1Y %": st.column_config.NumberColumn(format="%.2f%%"),
            "Volatility %": st.column_config.NumberColumn(format="%.2f%%"),
            "ROE %": st.column_config.NumberColumn(format="%.2f%%"),
            "MCap ($B)": st.column_config.NumberColumn(format="$%.1fB"),
            "Sentiment": st.column_config.ProgressColumn(min_value=-1, max_value=1, format="%.3f"),
            "Gov Risk": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%d/10"),
        },
    )

    # Show detail card when row is selected
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        idx = selected_rows[0]
        stock = df.iloc[idx]
        _render_stock_card(stock)


def render_sector_chart(df: pd.DataFrame):
    """Render sector analysis charts."""
    st.subheader("Sector Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Sector distribution
        sector_counts = df["sector"].value_counts()
        fig = px.pie(values=sector_counts.values, names=sector_counts.index, title="Sector Distribution")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Avg sentiment by sector
        sector_sentiment = df.groupby("sector")["sentiment_score"].mean().sort_values()
        fig = px.bar(
            x=sector_sentiment.values, y=sector_sentiment.index,
            orientation="h", title="Avg Sentiment by Sector",
            labels={"x": "Sentiment Score", "y": "Sector"},
            color=sector_sentiment.values,
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_top_movers(df: pd.DataFrame):
    """Render top gainers and losers."""
    st.subheader("Top Movers (1 Month)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 10 Gainers**")
        top_gainers = df.nlargest(10, "change_1m_pct")[["ticker", "company", "price", "change_1m_pct"]].reset_index(drop=True)
        top_gainers.index += 1
        st.dataframe(top_gainers, use_container_width=True)

    with col2:
        st.markdown("**Top 10 Losers**")
        top_losers = df.nsmallest(10, "change_1m_pct")[["ticker", "company", "price", "change_1m_pct"]].reset_index(drop=True)
        top_losers.index += 1
        st.dataframe(top_losers, use_container_width=True)


def _render_stock_card(stock):
    """Render a stock detail card (used by both table click and detail tab)."""
    st.markdown(f"### {stock['ticker']} — {stock.get('company', 'N/A')}")

    # Business description
    biz = stock.get("business_summary", "")
    if biz and pd.notna(biz) and str(biz).strip():
        st.caption(f"📋 {biz}")

    # Info cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${stock['price']}", f"{stock['change_1m_pct']:.2f}% (1M)")
    col2.metric("P/E Ratio", f"{stock['pe_trailing'] or 'N/A'}")
    col3.metric("RSI", f"{stock['rsi']}", stock['rsi_status'])
    col4.metric("Sentiment", f"{stock['sentiment_score']:.3f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Market Cap", f"${stock['market_cap_b'] or 'N/A'}B")
    col2.metric("ROE", f"{stock['roe_pct'] or 'N/A'}%")
    col3.metric("Volatility", f"{stock['volatility_pct']:.2f}%")
    col4.metric("Trend", stock['trend'])

    # Governance Risk
    gov_score = stock.get("governance_score", 0)
    gov_level = stock.get("governance_level", "Low")
    gov_reason = stock.get("governance_reason", "")
    if gov_score is not None and pd.notna(gov_score):
        gov_icons = {"Low": "✅", "Medium": "⚠️", "High": "🔶", "Critical": "🔴"}
        icon = gov_icons.get(gov_level, "❓")
        msg = f"{icon} **Governance Risk: {gov_level} ({int(gov_score)}/10)**"
        if gov_reason:
            msg += f"\n\n{gov_reason}"
        if gov_level in ("High", "Critical"):
            st.error(msg)
        elif gov_level == "Medium":
            st.warning(msg)
        else:
            st.success(msg)

    # AI Summary
    if stock.get("ai_summary") and pd.notna(stock["ai_summary"]):
        st.info(f"🤖 **AI Summary:** {stock['ai_summary']}")

    # News Summary
    if stock.get("news_summary") and pd.notna(stock["news_summary"]):
        st.caption(f"📰 **News:** {stock['news_summary']}")


def render_stock_detail(df: pd.DataFrame):
    """Render detailed view for a single stock."""
    st.subheader("Stock Detail")

    tickers = df["ticker"].tolist()
    selected = st.selectbox("Select stock", tickers)

    if not selected:
        return

    stock = df[df["ticker"] == selected].iloc[0]
    _render_stock_card(stock)


def render_ai_chat(df: pd.DataFrame):
    """Render AI chat interface."""
    st.subheader("🤖 AI Chat - Ask about S&P 500")

    # Build context from data
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about stocks... (e.g., 'หุ้น tech ตัวไหนน่าสนใจ?')")

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Build data context for AI
        top5_sentiment = df.nlargest(5, "sentiment_score")[["ticker", "company", "sentiment_score"]].to_string(index=False)
        bottom5_sentiment = df.nsmallest(5, "sentiment_score")[["ticker", "company", "sentiment_score"]].to_string(index=False)
        sector_summary = df.groupby("sector").agg({
            "sentiment_score": "mean",
            "change_1m_pct": "mean",
            "ticker": "count"
        }).round(3).to_string()

        # Check if user mentions specific ticker
        mentioned_tickers = [t for t in df["ticker"].tolist() if t.lower() in user_input.upper()]
        stock_context = ""
        if mentioned_tickers:
            for t in mentioned_tickers[:3]:
                s = df[df["ticker"] == t].iloc[0]
                stock_context += f"\n{t}: price=${s['price']}, 1M={s['change_1m_pct']}%, P/E={s['pe_trailing']}, RSI={s['rsi']}({s['rsi_status']}), sentiment={s['sentiment_score']}, trend={s['trend']}"

        prompt = f"""You are a stock analysis assistant. Answer in Thai. Use the following S&P 500 data:

Top 5 sentiment:
{top5_sentiment}

Bottom 5 sentiment:
{bottom5_sentiment}

Sector summary (avg sentiment, avg 1M change, count):
{sector_summary}
{f"Specific stocks:{stock_context}" if stock_context else ""}

Total stocks: {len(df)}, Uptrend: {len(df[df['trend']=='Uptrend'])}, Downtrend: {len(df[df['trend']=='Downtrend'])}

User question: {user_input}

Answer concisely in Thai. Use data to support your answer."""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = gemini_chat(prompt)
            st.write(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})


def render_scatter(df: pd.DataFrame):
    """Render scatter plot for stock comparison."""
    st.subheader("Stock Comparison")

    col1, col2 = st.columns(2)
    numeric_cols = ["price", "change_1m_pct", "change_1y_pct", "rsi", "volatility_pct",
                    "pe_trailing", "roe_pct", "market_cap_b", "sentiment_score",
                    "dividend_yield_pct", "beta", "debt_to_equity"]
    available_cols = [c for c in numeric_cols if c in df.columns]

    with col1:
        x_col = st.selectbox("X axis", available_cols, index=available_cols.index("pe_trailing") if "pe_trailing" in available_cols else 0)
    with col2:
        y_col = st.selectbox("Y axis", available_cols, index=available_cols.index("roe_pct") if "roe_pct" in available_cols else 1)

    plot_df = df.dropna(subset=[x_col, y_col])
    fig = px.scatter(
        plot_df, x=x_col, y=y_col,
        color="sector", hover_name="ticker",
        hover_data=["company", "price", "sentiment_score"],
        title=f"{x_col} vs {y_col}",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def get_workflow_status() -> dict:
    """Check latest GitHub Actions workflow run status."""
    if not GITHUB_TOKEN:
        return None
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/daily_analysis.yml/runs",
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
            },
            params={"per_page": 1},
            timeout=10,
        )
        if resp.status_code == 200:
            runs = resp.json().get("workflow_runs", [])
            if runs:
                run = runs[0]
                return {
                    "status": run["status"],           # queued, in_progress, completed
                    "conclusion": run.get("conclusion"),  # success, failure, cancelled
                    "started": run["created_at"][:16].replace("T", " "),
                    "url": run["html_url"],
                }
    except Exception:
        pass
    return None


def render_run_button():
    """Render button to trigger GitHub Actions + show status."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Run Analysis")

    if not GITHUB_TOKEN:
        st.sidebar.caption("Add `GITHUB_TOKEN` to secrets to enable.")
        return

    # Show current workflow status
    status = get_workflow_status()
    if status:
        s = status["status"]
        conclusion = status["conclusion"]

        if s in ("queued", "in_progress"):
            label = "⏳ Queued..." if s == "queued" else "🔄 Running..."
            st.sidebar.warning(f"{label}\nStarted: {status['started']}")
            st.sidebar.link_button("View on GitHub", status["url"], use_container_width=True)
        elif s == "completed":
            if conclusion == "success":
                st.sidebar.success(f"✅ Last run: Success\n{status['started']}")
            elif conclusion == "failure":
                st.sidebar.error(f"❌ Last run: Failed\n{status['started']}")
                st.sidebar.link_button("View logs", status["url"], use_container_width=True)
            else:
                st.sidebar.info(f"Last run: {conclusion}\n{status['started']}")

    # Run button (disable if already running)
    is_running = status and status["status"] in ("queued", "in_progress") if status else False

    if st.sidebar.button(
        "🚀 Run S&P 500 Analysis",
        use_container_width=True,
        disabled=is_running,
    ):
        try:
            resp = requests.post(
                f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/daily_analysis.yml/dispatches",
                headers={
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "Accept": "application/vnd.github+json",
                },
                json={"ref": "master"},
                timeout=15,
            )
            if resp.status_code == 204:
                st.sidebar.success("Analysis started! Refresh page to see status.")
                st.rerun()
            else:
                st.sidebar.error(f"Failed: {resp.status_code} - {resp.text[:100]}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    st.sidebar.caption("Analyzes all 503 S&P 500 stocks (~30-60 min)")


# --- Main App ---

def render_blacklist(blacklisted_df, blacklist_data):
    """Render blacklisted stocks tab."""
    st.subheader("🚫 Blacklisted Stocks (High/Critical Governance Risk)")
    st.caption("These stocks are excluded from analysis and will not be considered in the future.")

    if blacklisted_df is not None and not blacklisted_df.empty:
        display_cols = ["ticker", "company", "sector", "governance_score", "governance_level", "governance_reason", "last_update"]
        available = [c for c in display_cols if c in blacklisted_df.columns]
        st.dataframe(
            blacklisted_df[available].rename(columns={
                "ticker": "Ticker", "company": "Company", "sector": "Sector",
                "governance_score": "Gov Score", "governance_level": "Level",
                "governance_reason": "Reason", "last_update": "Last Update",
            }),
            use_container_width=True,
            height=400,
        )
    elif blacklist_data:
        rows = []
        for ticker, info in blacklist_data.items():
            rows.append({
                "Ticker": ticker,
                "Company": info.get("company", ""),
                "Level": info.get("level", ""),
                "Reason": info.get("reason", ""),
                "Blacklisted": info.get("blacklisted_date", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No blacklisted stocks yet.")


def main():
    df, blacklisted_df = load_data()

    if df is None:
        st.title("📈 S&P 500 Dashboard")
        st.warning("No data yet. Run `python analyzer.py` first or wait for GitHub Actions to generate data.")
        st.code("python analyzer.py", language="bash")
        return

    filtered_df = render_sidebar(df)
    render_header(filtered_df)

    # Run Analysis button in sidebar
    render_run_button()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🏢 Sectors", "🚀 Top Movers", "🔍 Stock Detail", "🤖 AI Chat", "🚫 Blacklist"
    ])

    with tab1:
        render_overview_table(filtered_df)
        render_scatter(filtered_df)

    with tab2:
        render_sector_chart(filtered_df)

    with tab3:
        render_top_movers(filtered_df)

    with tab4:
        render_stock_detail(filtered_df)

    with tab5:
        render_ai_chat(df)

    with tab6:
        render_blacklist(blacklisted_df, load_blacklist())


if __name__ == "__main__":
    main()
