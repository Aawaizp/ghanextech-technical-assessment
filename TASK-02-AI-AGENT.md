# Task 2: AI Agent with Tool Calling

## Overview

In this assignment, you will develop an interactive AI agent capable of answering user queries and dynamically invoking external tools to perform actions or fetch real-time information.

This task evaluates your understanding of function calling, agent reasoning loops, tool orchestration, structured schema definitions, and graceful error handling.

---

## Technical Requirements

### 1. Conversational Interface
- Provide an intuitive chat-based interface where users can interact with the agent across multi-turn conversations.
- Maintain chat history across turns so the agent can refer to previous context.

### 2. Tool Implementation
Implement at least **two** distinct functional tools that the agent can invoke when needed. Examples include:

- **Calculator / Expression Evaluator:** Executes mathematical operations or numerical parsing safely.
- **Weather Service:** Fetches real-time weather details for a given location using an open weather API (e.g., Open-Meteo or OpenWeatherMap free tier).
- **Web / Knowledge Search:** Searches the web or queries an external knowledge source (e.g., DuckDuckGo search or Wikipedia API).
- **Currency Converter:** Converts monetary amounts across currencies using a live exchange rate API.
- **Custom Utility:** Any functional API or utility that adds practical capability.

Each tool must have:
- A clearly defined input schema with parameter types.
- An informative description so the LLM understands when and how to call it.
- Robust exception handling for invalid parameters or network errors.

### 3. Agent Execution and Routing
- The agent must examine the user's intent and decide whether to call a tool, call multiple tools sequentially, or reply directly without tool use.
- Avoid forcing tool invocation for simple greetings or general knowledge questions.

### 4. Response Synthesis
- After executing a tool, the agent must parse the tool output and synthesize a clear, friendly, and helpful final response for the user.
- If a tool fails (e.g., rate limit, invalid location, network timeout), the agent must handle the error gracefully and inform the user without crashing.

---

## Suggested Stack

- **Language:** Python 3.10+
- **Framework:** Streamlit, Gradio, or FastAPI
- **Agent Framework:** LangChain (ReAct / AgentExecutor), LangGraph, LlamaIndex, or native tool-calling APIs via SDKs
- **LLM:** Google Gemini, Groq, or OpenAI-compatible free/open-source endpoints

---

## Deliverables

1. **Source Code:** Modular and well-structured code with distinct modules for tools, agent logic, and UI.
2. **Environment Configuration:** An `.env.example` file listing required configuration keys.
3. **Documentation:** A repository `README.md` covering:
   - Agent architecture and decision-making approach.
   - List and description of implemented tools with parameter schemas.
   - Example prompts showcasing both direct answers and tool invocations.
   - Step-by-step instructions to run the application locally.
4. **Live Deployment:** A functional public link hosted on Streamlit Community Cloud, Hugging Face Spaces, or another free platform.
