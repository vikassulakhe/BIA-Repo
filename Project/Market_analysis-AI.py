import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from fpdf import FPDF
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os

# -----------------------------------
# State Object
# -----------------------------------

class MarketState(TypedDict, total=False):
    df: pd.DataFrame
    summary: str
    trend: str
    chart: str
    insights: str


# -----------------------------------
# Load Data Node
# -----------------------------------

def load_data(state: MarketState):

    file_path = "market_data.csv"

    if not os.path.exists(file_path):
        raise Exception(f"{file_path} not found in project folder")

    df = pd.read_csv(file_path)

    print("✅ Data loaded successfully")

    state["df"] = df
    return state


# -----------------------------------
# Analyze Data Node
# -----------------------------------

def analyze_data(state: MarketState):

    if "df" not in state:
        raise Exception("Dataframe missing in state. load_data node failed.")

    df = state["df"]

    summary_df = df.describe()
    summary = summary_df.to_string()

    trend_value = df["Revenue"].pct_change().mean()

    if trend_value > 0:
        market_trend = "Growing Market"
    else:
        market_trend = "Declining Market"

    state["summary"] = summary
    state["trend"] = market_trend

    print("✅ Data analysis complete")

    return state


# -----------------------------------
# Chart Node
# -----------------------------------

def create_chart(state: MarketState):

    df = state["df"] # type: ignore

    plt.figure(figsize=(8,5))
    plt.plot(df["Year"], df["Revenue"], marker="o")
    plt.title("Revenue Trend")
    plt.xlabel("Year")
    plt.ylabel("Revenue")
    plt.grid(True)

    chart_file = "revenue_chart.png"

    plt.savefig(chart_file)
    plt.close()

    state["chart"] = chart_file

    print("Chart created")

    return state


# -----------------------------------
# AI Insight Node
# -----------------------------------

def generate_ai_insights(state: MarketState):

    summary = state["summary"] # type: ignore
    trend = state["trend"] # type: ignore

    prompt = f"""
You are a professional financial market analyst.

Market Data Summary:
{summary}

Detected Trend:
{trend}

Provide:
1. Key insights
2. Risk factors
3. Future outlook
4. Recommendations
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )

    state["insights"] = result.stdout

    print("✅ AI insights generated")

    return state


# -----------------------------------
# PDF Node
# -----------------------------------

def generate_pdf(state: MarketState):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, "Market Analysis Report", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 8, f"Market Trend:\n{state['trend']}") # type: ignore

    pdf.ln(10)
    pdf.multi_cell(0, 8, f"Summary Statistics:\n{state['summary']}") # type: ignore

    pdf.ln(10)
    pdf.multi_cell(0, 8, f"AI Insights:\n{state['insights']}") # type: ignore

    pdf.ln(10)

    pdf.image(state["chart"], x=10, w=180) # type: ignore

    pdf.output("Market_Report.pdf")

    print("✅ Report Generated: Market_Report.pdf")

    return state


# -----------------------------------
# Build LangGraph Workflow
# -----------------------------------

def build_graph():

    builder = StateGraph(MarketState)

    builder.add_node("load", load_data)
    builder.add_node("analyze", analyze_data)
    builder.add_node("chart", create_chart)
    builder.add_node("ai", generate_ai_insights)
    builder.add_node("pdf", generate_pdf)

    builder.set_entry_point("load")

    builder.add_edge("load", "analyze")
    builder.add_edge("analyze", "chart")
    builder.add_edge("chart", "ai")
    builder.add_edge("ai", "pdf")
    builder.add_edge("pdf", END)

    return builder.compile()


# -----------------------------------
# Run Agent
# -----------------------------------

def main():

    graph = build_graph()

    graph.invoke({})


if __name__ == "__main__":
    main()