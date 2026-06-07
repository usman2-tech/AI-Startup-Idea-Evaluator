from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, Dict, Annotated

import streamlit as st
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# =========================
# 1. PAGE CONFIG
# =========================
st.set_page_config(page_title="StartupLens AI", page_icon="🚀", layout="wide")


# =========================
# 2. ENV / API KEYS
# =========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Please set it in your environment variables.")
    st.stop()


# =========================
# 3. LLM + TOOLS
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY,
)

search_tool = TavilySearchResults(k=5) if TAVILY_API_KEY else None


# =========================
# 4. STATE
# =========================
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Dict[str, str]
    final_report: str
    is_ready: bool


# =========================
# 5. HELPER FUNCTIONS
# =========================
def get_latest_user_idea(messages: List[BaseMessage]) -> str:
    """Return the latest human message content."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def display_role(msg: BaseMessage) -> str:
    """Map LangChain message types to Streamlit chat roles."""
    if isinstance(msg, HumanMessage):
        return "user"
    return "assistant"


# =========================
# 6. GRAPH NODES
# =========================
def decide_node(state: State):
    idea = get_latest_user_idea(state["messages"])

    prompt = (
        "You are a startup intake analyst.\n"
        "Analyze the user's startup idea.\n"
        "If the idea is clear enough to evaluate, reply with exactly: READY\n"
        "Otherwise, ask exactly ONE short, specific follow-up question "
        "to clarify the product, customer, or problem."
    )

    res = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=idea),
        ]
    )

    is_ready = res.content.strip().upper() == "READY"

    if is_ready:
        # Don't add READY to visible chat
        return {"is_ready": True}

    return {
        "is_ready": False,
        "messages": [AIMessage(content=res.content)],
    }


def market_analyst(state: State):
    idea = get_latest_user_idea(state["messages"])
    prompt = (
        f"Evaluate the target demographics, customer pain points, market need, "
        f"and market trends for this startup idea:\n\n{idea}"
    )
    res = llm.invoke(prompt)
    reports = dict(state.get("advisor_reports", {}))
    reports["Market"] = res.content
    return {"advisor_reports": reports}


def legal_advisor(state: State):
    idea = get_latest_user_idea(state["messages"])
    prompt = (
        f"Identify 3 to 5 legal, IP, privacy, regulatory, or compliance risks "
        f"for this startup idea:\n\n{idea}"
    )
    res = llm.invoke(prompt)
    reports = dict(state.get("advisor_reports", {}))
    reports["Legal"] = res.content
    return {"advisor_reports": reports}


def tech_advisor(state: State):
    idea = get_latest_user_idea(state["messages"])
    prompt = (
        f"Recommend a practical MVP tech stack for this startup idea.\n"
        f"Include Frontend, Backend, Database, AI tooling if relevant, and "
        f"complexity level.\n\nIdea:\n{idea}"
    )
    res = llm.invoke(prompt)
    reports = dict(state.get("advisor_reports", {}))
    reports["Technical"] = res.content
    return {"advisor_reports": reports}


def strategy_advisor(state: State):
    idea = get_latest_user_idea(state["messages"])
    prompt = (
        f"Suggest 3 go-to-market tactics and a realistic 6-month roadmap "
        f"for this startup idea:\n\n{idea}"
    )
    res = llm.invoke(prompt)
    reports = dict(state.get("advisor_reports", {}))
    reports["Strategy"] = res.content
    return {"advisor_reports": reports}


def competitor_researcher(state: State):
    idea = get_latest_user_idea(state["messages"])

    if search_tool is not None:
        try:
            results = search_tool.invoke(
                {"query": f"startups or companies similar to {idea}"}
            )
            prompt = (
                "Using the following web search results, identify 3 similar companies, "
                "their positioning, and their likely differentiation.\n\n"
                f"Search results:\n{results}"
            )
            res = llm.invoke(prompt)
            content = res.content
        except Exception as e:
            content = f"Competitor search could not be completed. Reason: {str(e)}"
    else:
        prompt = (
            f"Based on general market knowledge, suggest 3 likely competitor types "
            f"or similar companies for this startup idea and explain their positioning:\n\n{idea}"
        )
        res = llm.invoke(prompt)
        content = res.content

    reports = dict(state.get("advisor_reports", {}))
    reports["Competitors"] = content
    return {"advisor_reports": reports}


def financial_advisor(state: State):
    idea = get_latest_user_idea(state["messages"])
    prompt = (
        f"Propose 2 revenue models, top 3 expected cost centers, and basic MVP "
        f"financial considerations for this startup idea:\n\n{idea}"
    )
    res = llm.invoke(prompt)
    reports = dict(state.get("advisor_reports", {}))
    reports["Finance"] = res.content
    return {"advisor_reports": reports}


def reporter(state: State):
    reports = state.get("advisor_reports", {})

    combined = "\n\n".join(
        [f"## {section}\n{content}" for section, content in reports.items()]
    )

    prompt = (
        "Create a professional executive summary from the advisor notes below.\n"
        "Use clear sections and end with a final section called 'Verdict'.\n\n"
        f"{combined}"
    )

    res = llm.invoke(prompt)
    return {"final_report": res.content}


def route_after_decide(state: State):
    if state.get("is_ready", False):
        return "market"
    return END


# =========================
# 7. GRAPH BUILD
# =========================
builder = StateGraph(State)

builder.add_node("decide", decide_node)
builder.add_node("market", market_analyst)
builder.add_node("legal", legal_advisor)
builder.add_node("tech", tech_advisor)
builder.add_node("strategy", strategy_advisor)
builder.add_node("competitor", competitor_researcher)
builder.add_node("finance", financial_advisor)
builder.add_node("reporter", reporter)

builder.set_entry_point("decide")

builder.add_conditional_edges(
    "decide",
    route_after_decide,
    {
        "market": "market",
        END: END,
    },
)

# Sequential flow to avoid fan-out/fan-in edge issues
builder.add_edge("market", "legal")
builder.add_edge("legal", "tech")
builder.add_edge("tech", "strategy")
builder.add_edge("strategy", "competitor")
builder.add_edge("competitor", "finance")
builder.add_edge("finance", "reporter")
builder.add_edge("reporter", END)

graph = builder.compile()


# =========================
# 8. SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "full_state" not in st.session_state:
    st.session_state.full_state = {
        "advisor_reports": {},
        "final_report": "",
    }


# =========================
# 9. UI
# =========================
st.title("🚀 StartupLens AI")
st.subheader("Professional Business Idea Validation Dashboard")

chat_col, result_col = st.columns([1, 1.2])

with chat_col:
    st.markdown("### 💬 Intake Chat")

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(display_role(msg)):
            st.write(msg.content)

    # New input
    prompt = st.chat_input("Explain your startup idea...")

    if prompt:
        user_msg = HumanMessage(content=prompt)
        st.session_state.messages.append(user_msg)

        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Board of Advisors is deliberating..."):
            try:
                result = graph.invoke(
                    {
                        "messages": st.session_state.messages,
                        "advisor_reports": {},
                        "final_report": "",
                        "is_ready": False,
                    }
                )

                # Add any new assistant clarification messages
                returned_messages = result.get("messages", [])
                existing_contents = {(type(m), m.content) for m in st.session_state.messages}

                for m in returned_messages:
                    key = (type(m), m.content)
                    if key not in existing_contents:
                        st.session_state.messages.append(m)
                        with st.chat_message(display_role(m)):
                            st.write(m.content)

                # Save reports if ready flow completed
                if result.get("final_report"):
                    st.session_state.full_state = {
                        "advisor_reports": result.get("advisor_reports", {}),
                        "final_report": result.get("final_report", ""),
                    }
                    st.rerun()

            except Exception as e:
                st.error(f"Error while running analysis: {str(e)}")

with result_col:
    st.markdown("### 📊 Advisor Insights")

    if st.session_state.full_state["final_report"]:
        tabs = st.tabs(
            ["📋 Summary", "🔍 Market & Comp", "⚙️ Tech & Legal", "💰 Strategy & Finance"]
        )

        with tabs[0]:
            st.markdown(st.session_state.full_state["final_report"])

        with tabs[1]:
            st.markdown("#### Market Analysis")
            st.write(st.session_state.full_state["advisor_reports"].get("Market", "No data"))
            st.divider()
            st.markdown("#### Competitor Landscape")
            st.write(st.session_state.full_state["advisor_reports"].get("Competitors", "No data"))

        with tabs[2]:
            st.markdown("#### Technical Blueprint")
            st.write(st.session_state.full_state["advisor_reports"].get("Technical", "No data"))
            st.divider()
            st.markdown("#### Legal & Regulatory")
            st.write(st.session_state.full_state["advisor_reports"].get("Legal", "No data"))

        with tabs[3]:
            st.markdown("#### Strategic Roadmap")
            st.write(st.session_state.full_state["advisor_reports"].get("Strategy", "No data"))
            st.divider()
            st.markdown("#### Financial Modeling")
            st.write(st.session_state.full_state["advisor_reports"].get("Finance", "No data"))
    else:
        st.info("The Board will release reports once the idea is fully clarified in the chat.")


# =========================
# 10. SIDEBAR RESET
# =========================
if st.sidebar.button("Clear Session"):
    st.session_state.messages = []
    st.session_state.full_state = {
        "advisor_reports": {},
        "final_report": "",
    }
    st.rerun()