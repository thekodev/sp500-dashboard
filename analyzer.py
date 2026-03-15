"""
S&P 500 Stock Analyzer with Gemini AI
- Fetches S&P 500 list from Wikipedia
- Technical analysis (SMA, RSI, Volatility)
- Fundamental analysis (P/E, ROE, etc.)
- Sentiment analysis via Gemini API
- Saves results to CSV for Streamlit dashboard
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import re
import time
import logging
import sys
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Gemini API ---

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent"


def gemini_generate(prompt: str) -> Optional[str]:
    """Call Gemini API and return text response."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set, skipping AI analysis")
        return None
    try:
        resp = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            logger.error(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return None


# --- S&P 500 List ---

def get_sp500_tickers() -> List[Dict]:
    """Fetch S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        from io import StringIO
        df = pd.read_html(StringIO(resp.text))[0]
        tickers = []
        for _, row in df.iterrows():
            tickers.append({
                "ticker": row["Symbol"].replace(".", "-"),
                "company": row["Security"],
                "sector": row["GICS Sector"],
                "sub_industry": row["GICS Sub-Industry"],
            })
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 list: {e}")
        return []


# --- Technical Analysis ---

def get_stock_data(ticker: str, years: int = 3) -> Optional[pd.DataFrame]:
    """Fetch historical stock data."""
    try:
        end = datetime.today()
        start = end - timedelta(days=years * 365)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return None


def technical_analysis(df: pd.DataFrame) -> Dict:
    """Calculate SMA, RSI, volatility."""
    close = df['Close']

    sma50 = close.rolling(50, min_periods=1).mean()
    sma200 = close.rolling(200, min_periods=1).mean()

    # Trend
    trend = "Uptrend" if sma50.iloc[-1] > sma200.iloc[-1] else "Downtrend"

    # RSI
    delta = close.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    if pd.isna(rsi):
        rsi = 50.0

    if rsi > 70:
        rsi_status = "Overbought"
    elif rsi < 30:
        rsi_status = "Oversold"
    else:
        rsi_status = "Neutral"

    # Volatility
    daily_return = close.pct_change()
    volatility = daily_return.std() * np.sqrt(252)

    # Price info
    current_price = close.iloc[-1]
    price_change_1m = ((close.iloc[-1] / close.iloc[-22] - 1) * 100) if len(close) > 22 else 0
    price_change_1y = ((close.iloc[-1] / close.iloc[-252] - 1) * 100) if len(close) > 252 else 0
    high_52w = close.iloc[-252:].max() if len(close) > 252 else close.max()
    low_52w = close.iloc[-252:].min() if len(close) > 252 else close.min()

    return {
        "current_price": round(current_price, 2),
        "price_change_1m": round(price_change_1m, 2),
        "price_change_1y": round(price_change_1y, 2),
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "sma50": round(sma50.iloc[-1], 2),
        "sma200": round(sma200.iloc[-1], 2),
        "trend": trend,
        "rsi": round(rsi, 2),
        "rsi_status": rsi_status,
        "volatility": round(volatility * 100, 2),
    }


# --- Fundamental Analysis ---

def fundamental_analysis(ticker: str) -> Dict:
    """Fetch fundamental data from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        market_cap = info.get('marketCap')
        return {
            "company_name": info.get('longName', 'N/A'),
            "market_cap": round(market_cap / 1e9, 2) if market_cap else None,
            "pe_trailing": round(info['trailingPE'], 2) if info.get('trailingPE') else None,
            "pe_forward": round(info['forwardPE'], 2) if info.get('forwardPE') else None,
            "eps": round(info['trailingEps'], 2) if info.get('trailingEps') else None,
            "dividend_yield": round(info['dividendYield'] * 100, 2) if info.get('dividendYield') else None,
            "roe": round(info['returnOnEquity'] * 100, 2) if info.get('returnOnEquity') else None,
            "debt_to_equity": round(info['debtToEquity'], 2) if info.get('debtToEquity') else None,
            "revenue": round(info['totalRevenue'] / 1e9, 2) if info.get('totalRevenue') else None,
            "net_income": round(info['netIncomeToCommon'] / 1e9, 2) if info.get('netIncomeToCommon') else None,
            "beta": round(info['beta'], 2) if info.get('beta') else None,
        }
    except Exception as e:
        logger.error(f"Fundamental error for {ticker}: {e}")
        return {
            "company_name": "N/A", "market_cap": None, "pe_trailing": None,
            "pe_forward": None, "eps": None, "dividend_yield": None,
            "roe": None, "debt_to_equity": None, "revenue": None,
            "net_income": None, "beta": None,
        }


# --- News Sentiment ---

def fetch_news_headlines(company: str, max_headlines: int = 5) -> List[str]:
    """Fetch Google News RSS headlines."""
    try:
        url = f'https://news.google.com/rss/search?q={company}+stock&hl=en-US&gl=US&ceid=US:en'
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.content, 'xml')
        return [item.title.get_text() for item in soup.find_all('item')][:max_headlines]
    except Exception as e:
        logger.error(f"News fetch error for {company}: {e}")
        return []


def analyze_sentiment_batch(headlines: List[str]) -> Tuple[float, str]:
    """Use Gemini to analyze sentiment of multiple headlines at once."""
    if not headlines:
        return 0.0, ""

    headlines_text = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    prompt = f"""Analyze the sentiment of these stock news headlines.
For each headline, rate from -1 (very negative) to 1 (very positive).
Then provide an overall score and a brief 1-sentence summary in Thai.

Headlines:
{headlines_text}

Reply in this exact JSON format:
{{"scores": [0.5, -0.3, ...], "overall": 0.2, "summary_th": "ข่าวส่วนใหญ่เป็นบวก..."}}"""

    reply = gemini_generate(prompt)
    if not reply:
        return 0.0, ""

    try:
        # Extract JSON from reply
        json_match = re.search(r'\{.*\}', reply, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = max(-1, min(1, float(data.get("overall", 0))))
            summary = data.get("summary_th", "")
            return round(score, 3), summary
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.error(f"Failed to parse sentiment response: {e}")

    return 0.0, ""


# --- AI Summary ---

def generate_stock_summary(ticker: str, tech: Dict, fund: Dict, sentiment: float) -> str:
    """Use Gemini to generate a stock summary in Thai."""
    prompt = f"""สรุปการวิเคราะห์หุ้น {ticker} ({fund.get('company_name', 'N/A')}) เป็นภาษาไทย 2-3 ประโยค

ข้อมูล:
- ราคา: ${tech['current_price']}, เปลี่ยนแปลง 1 เดือน: {tech['price_change_1m']}%, 1 ปี: {tech['price_change_1y']}%
- เทรนด์: {tech['trend']}, RSI: {tech['rsi']} ({tech['rsi_status']})
- Volatility: {tech['volatility']}%
- P/E: {fund.get('pe_trailing', 'N/A')}, ROE: {fund.get('roe', 'N/A')}%
- Market Cap: ${fund.get('market_cap', 'N/A')}B
- Sentiment Score: {sentiment}

สรุปสั้นๆ ว่าหุ้นตัวนี้น่าสนใจหรือไม่ เหมาะกับนักลงทุนแบบไหน"""

    return gemini_generate(prompt) or ""


# --- Main Analysis ---

def analyze_single_stock(ticker: str, company: str, sector: str, sub_industry: str) -> Optional[Dict]:
    """Analyze a single stock and return results dict."""
    logger.info(f"Analyzing {ticker}...")

    # Technical
    df = get_stock_data(ticker)
    if df is None:
        logger.warning(f"No data for {ticker}, skipping")
        return None

    tech = technical_analysis(df)

    # Fundamental
    fund = fundamental_analysis(ticker)

    # Sentiment
    headlines = fetch_news_headlines(company)
    sentiment_score, news_summary = analyze_sentiment_batch(headlines)

    # AI Summary (rate limit: wait between Gemini calls)
    time.sleep(1)
    ai_summary = generate_stock_summary(ticker, tech, fund, sentiment_score)
    time.sleep(1)

    return {
        "ticker": ticker,
        "company": fund.get("company_name", company),
        "sector": sector,
        "sub_industry": sub_industry,
        "date": datetime.today().strftime('%Y-%m-%d'),
        # Price
        "price": tech["current_price"],
        "change_1m_pct": tech["price_change_1m"],
        "change_1y_pct": tech["price_change_1y"],
        "high_52w": tech["high_52w"],
        "low_52w": tech["low_52w"],
        # Technical
        "sma50": tech["sma50"],
        "sma200": tech["sma200"],
        "trend": tech["trend"],
        "rsi": tech["rsi"],
        "rsi_status": tech["rsi_status"],
        "volatility_pct": tech["volatility"],
        # Fundamental
        "market_cap_b": fund["market_cap"],
        "pe_trailing": fund["pe_trailing"],
        "pe_forward": fund["pe_forward"],
        "eps": fund["eps"],
        "dividend_yield_pct": fund["dividend_yield"],
        "roe_pct": fund["roe"],
        "debt_to_equity": fund["debt_to_equity"],
        "revenue_b": fund["revenue"],
        "net_income_b": fund["net_income"],
        "beta": fund["beta"],
        # Sentiment & AI
        "sentiment_score": sentiment_score,
        "news_summary": news_summary,
        "ai_summary": ai_summary,
    }


def run_full_analysis():
    """Run analysis for all S&P 500 stocks."""
    start_time = time.time()

    # Get S&P 500 list
    sp500 = get_sp500_tickers()
    if not sp500:
        logger.error("Failed to get S&P 500 list")
        return

    total = len(sp500)
    results = []
    errors = []

    # Load existing data to resume if interrupted
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, "sp500_analysis.csv")

    already_done = set()
    today = datetime.today().strftime('%Y-%m-%d')
    if os.path.exists(output_file):
        try:
            existing = pd.read_csv(output_file)
            already_done = set(existing[existing["date"] == today]["ticker"].tolist())
            if already_done:
                results = existing[existing["date"] == today].to_dict("records")
                logger.info(f"Resuming: {len(already_done)} stocks already analyzed today")
        except Exception:
            pass

    for i, stock in enumerate(sp500):
        ticker = stock["ticker"]

        if ticker in already_done:
            continue

        try:
            result = analyze_single_stock(
                ticker, stock["company"], stock["sector"], stock["sub_industry"]
            )
            if result:
                results.append(result)

                # Save progress every 10 stocks
                if len(results) % 10 == 0:
                    pd.DataFrame(results).to_csv(output_file, index=False)
                    logger.info(f"Progress saved: {len(results)}/{total}")

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            errors.append(ticker)

        # Progress
        done = len(already_done) + (i + 1 - len(already_done))
        elapsed = time.time() - start_time
        avg = elapsed / max(done - len(already_done), 1)
        remaining = (total - done) * avg
        logger.info(f"[{done}/{total}] {ticker} | Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")

    # Final save
    if results:
        pd.DataFrame(results).to_csv(output_file, index=False)
        logger.info(f"Analysis complete! Saved {len(results)} stocks to {output_file}")

    if errors:
        logger.warning(f"Failed stocks: {errors}")

    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    run_full_analysis()
