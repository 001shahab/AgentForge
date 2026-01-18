#!/usr/bin/env python
"""
Stock Monitor Example for AgentForge.

This example demonstrates how to build an agent that:
1. Fetches stock data using yfinance
2. Analyzes the data for trends
3. Generates alerts and insights using an LLM
4. Optionally sends email notifications

I've created this to show how AgentForge can be used for financial
data monitoring and automated alerts.

Author: Prof. Shahab Anbarjafari
3S Holding OÜ, Tartu, Estonia

Requirements:
    pip install yfinance

Usage:
    python stock_monitor.py AAPL GOOGL MSFT
    
    With email alerts:
    python stock_monitor.py AAPL --email your@email.com
"""

import argparse
import json
import logging
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_stock_data(symbols: List[str], period: str = "1mo") -> Dict[str, Any]:
    """
    Fetch stock data using yfinance.
    
    This is a custom function that we'll use to get the data before
    feeding it into our agent for analysis.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance not installed. Please install with:\n"
            "pip install yfinance"
        )
    
    data = {}
    
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data found for {symbol}")
                continue
            
            # Calculate some basic metrics
            latest_close = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else latest_close
            change_pct = ((latest_close - prev_close) / prev_close) * 100
            
            # Get period high/low
            period_high = hist["High"].max()
            period_low = hist["Low"].min()
            avg_volume = hist["Volume"].mean()
            
            data[symbol] = {
                "current_price": round(latest_close, 2),
                "previous_close": round(prev_close, 2),
                "change_percent": round(change_pct, 2),
                "period_high": round(period_high, 2),
                "period_low": round(period_low, 2),
                "avg_volume": int(avg_volume),
                "data_points": len(hist),
                "name": ticker.info.get("longName", symbol),
                "sector": ticker.info.get("sector", "Unknown"),
            }
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            data[symbol] = {"error": str(e)}
    
    return data


def format_stock_report(data: Dict[str, Any]) -> str:
    """Format stock data into a readable report for the LLM."""
    lines = ["Stock Market Report", "=" * 40, ""]
    
    for symbol, info in data.items():
        if "error" in info:
            lines.append(f"{symbol}: Error - {info['error']}")
            continue
        
        change_emoji = "📈" if info["change_percent"] >= 0 else "📉"
        
        lines.append(f"{symbol} - {info.get('name', symbol)}")
        lines.append(f"  Sector: {info.get('sector', 'Unknown')}")
        lines.append(f"  Current Price: ${info['current_price']}")
        lines.append(f"  Change: {change_emoji} {info['change_percent']:+.2f}%")
        lines.append(f"  Period High: ${info['period_high']}")
        lines.append(f"  Period Low: ${info['period_low']}")
        lines.append(f"  Avg Volume: {info['avg_volume']:,}")
        lines.append("")
    
    return "\n".join(lines)


def send_email_alert(
    subject: str,
    body: str,
    to_email: str,
    from_email: Optional[str] = None,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
):
    """
    Send an email alert.
    
    Requires SMTP credentials in environment variables:
    - SMTP_EMAIL: Your email address
    - SMTP_PASSWORD: Your email password or app-specific password
    """
    smtp_email = os.environ.get("SMTP_EMAIL")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    
    if not smtp_email or not smtp_password:
        logger.warning("SMTP credentials not set. Skipping email.")
        return False
    
    from_email = from_email or smtp_email
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(from_email, [to_email], msg.as_string())
        
        logger.info(f"Email sent to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def main():
    """Main entry point for the stock monitor."""
    parser = argparse.ArgumentParser(description="Monitor stocks with AI analysis")
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Stock symbols to monitor (e.g., AAPL GOOGL MSFT)"
    )
    parser.add_argument(
        "--period",
        default="1mo",
        help="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)"
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "groq"],
        default="openai",
        help="LLM backend for analysis"
    )
    parser.add_argument(
        "--email",
        help="Email address for alerts"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Alert threshold for % change (default: 5.0)"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    args = parser.parse_args()
    
    # Check for API key
    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: Please set the OPENAI_API_KEY environment variable")
        return 1
    
    if args.backend == "groq" and not os.environ.get("GROQ_API_KEY"):
        print("Error: Please set the GROQ_API_KEY environment variable")
        return 1
    
    print("=" * 60)
    print("AgentForge Stock Monitor")
    print("=" * 60)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Period: {args.period}")
    print()
    
    # Fetch stock data
    print("Fetching stock data...")
    stock_data = fetch_stock_data(args.symbols, args.period)
    
    if not stock_data:
        print("No data fetched. Please check the symbols.")
        return 1
    
    # Format the report
    report = format_stock_report(stock_data)
    print()
    print(report)
    
    # Import AgentForge for AI analysis
    from agentforge.integrations import OpenAIBackend, GroqBackend
    
    if args.backend == "openai":
        llm = OpenAIBackend()
    else:
        llm = GroqBackend()
    
    # Generate AI analysis
    print("Generating AI analysis...")
    print()
    
    prompt = f"""Analyze the following stock market data and provide insights:

{report}

Please provide:
1. A brief summary of overall market sentiment based on these stocks
2. Notable movers (biggest gains/losses)
3. Any patterns or correlations you observe
4. Risk factors to watch
5. Brief recommendations (educational purposes only, not financial advice)

Keep your analysis concise but insightful.
"""
    
    try:
        analysis = llm.generate(prompt, temperature=0.7, max_tokens=1024)
        
        print("AI Analysis:")
        print("-" * 40)
        print(analysis)
        print()
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        analysis = "Analysis unavailable"
    
    # Check for alerts
    alerts = []
    for symbol, info in stock_data.items():
        if "error" in info:
            continue
        
        if abs(info["change_percent"]) >= args.threshold:
            direction = "UP" if info["change_percent"] > 0 else "DOWN"
            alerts.append(
                f"⚠️ {symbol} is {direction} {abs(info['change_percent']):.2f}% "
                f"(current: ${info['current_price']})"
            )
    
    if alerts:
        print("ALERTS:")
        print("-" * 40)
        for alert in alerts:
            print(alert)
        print()
        
        # Send email if configured
        if args.email:
            subject = f"Stock Alert: {len(alerts)} trigger(s)"
            body = f"""Stock Monitor Alerts
{datetime.now().strftime('%Y-%m-%d %H:%M')}

{chr(10).join(alerts)}

---

Full Analysis:
{analysis}
"""
            send_email_alert(subject, body, args.email)
    
    # Save results if requested
    if args.output:
        results = {
            "timestamp": datetime.now().isoformat(),
            "symbols": args.symbols,
            "period": args.period,
            "data": stock_data,
            "analysis": analysis,
            "alerts": alerts,
        }
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {args.output}")
    
    print()
    print("✓ Monitoring complete!")
    return 0


if __name__ == "__main__":
    exit(main())

