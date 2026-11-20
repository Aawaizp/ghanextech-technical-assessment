# Ghanex Agent

Task 2 of the Ghanex Tech assessment: a Streamlit assistant using Groq's native tool calling.

[Live app](https://ghanextech-technical-assessment-2h5qjjcnhjdd4djmipvh5g.streamlit.app/)

## Run locally

Use Python 3.11. From this folder in PowerShell:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Replace the placeholder in .env with your Groq key, then run:

~~~powershell
python -m streamlit run app.py --server.port 8502
~~~

One key is enough. Optional additional keys use GROQ_API_KEY_2, GROQ_API_KEY_3, etc.
Keys belonging to the same Groq organization share rate limits. Never commit .env.

## Architecture

app.py keeps chat history in Streamlit session state and displays answers, source
links and expandable tool activity. It closes API clients after each request.
ui.py contains display helpers and styles.css controls the compact chat layout.
Source links and tool details stay collapsed until opened. The sidebar supports
up to 30 independent conversations, title/content search, and switching chats.
The Chat menu offers rename, JSON download and confirmed deletion. New chat
starts a separate conversation without removing previous chats.

History is isolated per Streamlit session, not stored in a shared database.
Reloading or closing the tab can lose it. Download a chat and use Import a
conversation to restore its transcript later. Imports are bounded and validated;
they restore user/assistant messages, not tool traces or system instructions.

agent/core.py sends the latest 12 messages and tool schemas to
openai/gpt-oss-120b with tool_choice="auto". The model chooses a direct reply
or requests tools. Python validates and executes calls, then returns structured
results to the model. Sequential calls can use earlier results.

The loop permits five tool rounds and at most eight attempted calls.
Identical calls within a turn reuse their result. When the main
model proposes a final answer after using tools, a single completion check can
request missing actions using the Qwen model. This adds one model request to
normal tool-backed answers; routing remains probabilistic. At the limit, or if the main
model fails after tools have run, a fresh answer-only request to
qwen/qwen3.8-27b synthesizes the available results with conversation context.
That fallback model is a Groq preview model; availability may change.
If synthesis fails, the interface preserves tool activity and explains the error.

agent/registry.py defines schemas and validates required parameters, types,
lengths, numeric bounds and unexpected parameters before dispatch. Each tool
returns either {ok: true, data: ...} or {ok: false, error: ...}.

## Tools and input schemas

All parameters below are required; additional parameters are rejected.

| Tool | Parameters | Behavior |
| --- | --- | --- |
| calculate | expression: string, 1–200 characters | Restricted Python AST arithmetic; no eval, imports or attribute access. Supports +, -, *, /, //, %, **, abs, round, sqrt, sin, cos, tan, log, log10, pi and e. |
| get_weather | location: string, 1–160 characters | Open-Meteo current weather and seven daily forecasts, with explicit local observation and forecast dates. |
| search_knowledge | query: string, 1–200 characters | Up to three Wikipedia article introductions, each limited to 1,500 characters, with source links. |
| convert_currency | amount: number >0 and <=1,000,000,000; from_currency and to_currency: three-letter strings | Frankfurter daily ECB reference rate, converted amount, publication date and source. |

Weather, Wikipedia and currency require no extra API keys. Network requests
have timeouts and validate returned data. The calculator bounds input depth,
operations, exponents and intermediate numeric values.

## Try these prompts

- Direct reply: "Hi!" or "What is a language model?"
- Calculator: "Calculate (125 * 1.18) / 4."
- Weather: "What is the weather in Solapur tomorrow?"
- Follow-up: "And the day after tomorrow in the same city?"
- Knowledge: "Find sourced background about Solapur on Wikipedia."
- Currency: "Convert 100 USD to INR."
- Sequential tools: "Get Solapur's current temperature and calculate twice that value."
- Failure handling: "What is the weather in Zzzzxxyy123?" or "Calculate 1 / 0."

Tomorrow is matched by date using the local observation timestamp, not by array
position: daily forecasts can begin yesterday around midnight. forecast_days is
not a model parameter, preventing a one-day request from hiding tomorrow.

## Limits

Wikipedia introductions may omit population counts and other detailed facts.
Source links are retrieved sources, not proof that every requested fact appears
in them. The agent should report missing evidence and include year and geographic
scope for population figures.

Currency conversion uses the latest published daily reference rate, which can
precede today on weekends and holidays. It excludes fees and is not an intraday
trading quote. Values are rounded to six decimals, not currency cash units.

Weather resolves a place through geocoding and reports its region. Specify a
country for ambiguous names. Only the returned seven days are supported.

Conversation memory lasts for the browser session; delete a chat through its
Chat menu. Model routing is probabilistic, so inspect Tool activity when evaluating
a result. Chat and tool results are sent to Groq; queries go to the selected
external tool service.

## Tests

~~~powershell
python -B -m unittest discover -s tests -v
~~~

Offline tests mock external APIs and check arithmetic safety, tool validation,
forecast coverage, API failures, currency dates and agent-loop behavior.
They do not guarantee every model-generated answer is correct.

### Assessment verification (November 21, 2026)

59 offline tests pass, covering tools, schemas, rate-limit fallback, connection
errors, call limits, malformed arguments and independent conversation history.
UI checks include a 30-chat history, searching, renaming, switching and deletion.

Live checks passed for greetings and general knowledge without tools, Wikipedia
search, tomorrow's weather, a same-city follow-up, and a combined request for
weather, multiplication, division by zero and currency conversion. For an invalid
place the model asked for clarification; a direct weather-tool check returned a
clear location-not-found error. These are sample checks, not a guarantee of every
future model response.

| Task 2 requirement | Evidence / status |
| --- | --- |
| Multi-turn chat and memory | Session conversations with isolated context; navigation tests pass |
| At least two functional tools | Four tools, with offline validation and live sample checks |
| Typed schemas and robust errors | Registry validation, network/error tests and live invalid-location check |
| Direct replies and sequential tool use | Live direct replies and temperature-to-calculator chain |
| Synthesis and graceful failures | Live combined answer includes calculator error without losing other results |
| Modular source and configuration | Separate UI, conversation store, agent, registry and tools; .env.example |
| README | Setup, schemas, architecture, examples and limitations documented |
| Public deployment | Deployment URL recorded; incognito access and hosted tool checks pending |

The layout uses a compact welcome screen, a separately scrolling history list,
sticky navigation/header styling, and popovers for secondary controls. Automated
UI checks validate behavior, not pixel alignment. Desktop/mobile visual review
is still needed because the available preview returned the wrong window.

## Deployment

The deployment URL is linked above. To reproduce the deployment in Streamlit
Community Cloud, select this repository and task-2-agent/app.py as the entry
point, Python 3.11, and configure GROQ_API_KEY_1 in app secrets. Verify public
access in an incognito window and test tools on the hosted app before submission.

Weather errors identify location lookup versus forecast retrieval and distinguish
HTTP errors (including 403 and 429), timeouts, connection and TLS failures. Temporary
connection failures, timeouts and HTTP 502/503/504 receive one retry; access denials
and rate limits are not retried. Safe diagnostic lines appear in Streamlit logs.
A successful local request does not establish connectivity from Community Cloud.

## API references

- [Groq tool calling](https://console.groq.com/docs/tool-use/local-tool-calling)
- [Groq models](https://console.groq.com/docs/models)
- [Open-Meteo](https://open-meteo.com/en/docs)
- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)
- [Frankfurter v1 reference rates](https://frankfurter.dev/v1/)

Implementation and tests were developed with AI assistance. The submitter should
review the code and be able to explain routing, validation and failure handling.
