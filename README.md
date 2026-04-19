# AI-Startup-Idea-Evaluator
# StartupLens AI

StartupLens AI is an intelligent **startup idea validation dashboard** built with **Streamlit, LangChain, LangGraph, Gemini, and Tavily Search**.  
It simulates a **board of advisors** that evaluates a business idea from multiple perspectives such as:

- Market Analysis
- Competitor Research
- Technical Feasibility
- Legal & Compliance Risks
- Go-To-Market Strategy
- Financial Considerations

The system first checks whether the startup idea is clear enough. If the idea is incomplete, it asks a focused follow-up question. Once clarified, it generates a structured executive-style report.

---

##  Features

- **Interactive startup intake chat**
- **Multi-advisor AI workflow**
- **Automatic idea clarification**
- **Professional executive summary generation**
- **Competitor research with Tavily web search**
- **Separate advisor insights in organized tabs**
- **Clean Streamlit dashboard UI**

---

##  How It Works

StartupLens AI uses a LangGraph workflow to simulate a team of advisors:

1. **Decide Node**
   - Checks whether the startup idea is clear enough
   - If unclear, asks one specific follow-up question
   - If clear, begins analysis

2. **Advisor Nodes**
   - **Market Analyst** → examines target audience, pain points, and trends
   - **Legal Advisor** → highlights IP, privacy, compliance, and regulatory risks
   - **Tech Advisor** → recommends MVP tech stack and implementation complexity
   - **Strategy Advisor** → suggests GTM tactics and a 6-month roadmap
   - **Competitor Researcher** → uses Tavily search to identify similar startups
   - **Financial Advisor** → proposes revenue models and cost considerations

3. **Reporter Node**
   - Combines all advisor outputs into a final executive summary with a verdict

---

##  Tech Stack

- **Frontend:** Streamlit
- **LLM:** Google Gemini via `langchain-google-genai`
- **Workflow Orchestration:** LangGraph
- **LLM Framework:** LangChain
- **Web Search:** Tavily Search API
- **Language:** Python

---

##  Project Structure

```bash
StartupLens-AI/
│
├── main.py              # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not pushed to GitHub)
└── README.md            # Project documentation
