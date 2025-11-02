<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 14: Build & Serve Agentic Graphs with LangGraph</h1>

| 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 14: Deploying Agents to Production](https://www.notion.so/Session-14-Deploying-Agents-to-Production-26acd547af3d80a59047c1685ff6d61a) |[Recording!](https://us02web.zoom.us/rec/share/P6sJWRwsWWf2cF91MXOzrlM40Tay-CqoLp5drxoS6AGQEvMD3krhLzGFcrhyuAh3.HWnYPtpB0DL2mrj2) (cQ2$d7E5) | [Session 14 Slides](https://www.canva.com/design/DAG2pZbibmw/YJHR3HSgG992FE1I-Mmwjw/edit?utm_content=DAG2pZbibmw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 14 Assignment: LangGraph_Platform](https://github.com/AI-Maker-Space/AIE8/tree/main/14_LangGraph_Platform) | [AIE8 Feedback 10/23](https://forms.gle/rSCtaKTaPkTeqoo1A)

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph
  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API from a different terminal
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangGraph Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

## Activities and Questions 🏗️ &❓

#### ❓ Question 1:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

##### ✅ Answer:
Agent assistant is simple flow with access to tools and no validation on the responses. For instance, i ran this query "What is temperature in Va beach today? When did we had similar weather in the previous month?"
Response: "The current temperature in Virginia Beach today is approximately 62°F with plenty of sunshine.
Regarding similar weather in the previous month, there were days with similar temperatures around 62-64°F, with mostly sunny to partly cloudy conditions. For example, on November 4th, the forecast predicted a high of 62°F with clear skies, which is quite similar to today's weather."

The similar query might run posed to the "agent_helpful" yielded slightly different response, but still inaccurate.

"Response : The current temperature in Virginia Beach today is approximately 62°F with plenty of sunshine.
Regarding similar weather in the previous month, there was a day with similar conditions around November 1st, with temperatures around 62-63°F and sunny skies."

Conclusion: The response of the simple agent is incorrect because its forecasting for future. So, the simple  agent may be suitable for single hop questions. Regarding "Agent with helpfulness" have slighlkty incorrect responses. This can be correct by adjuting prompt template to check for accuracy.

Regarding second part of the question, when we ask "out of context" question to agent, its response includes information for more clarity, which is routed to "helpfulness" node. There was back and forth in same fashion for 3 times and then it ended. So, the conclusion, agent will be terminated after few attempts( in this case 3) incase unable to provide response.




#### 🏗️ Activity #1 Debugging A Graph

Select the `agent_with_helpfulness` and set one or more interrupts (at least one `Before` and one `After`). Try changing values and continuing the turn. 

#### ❓ Question 2:

What are your thoughts on when you would use a Before interrupt vs. an After interrupt?

##### ✅ Answer:
This feature is particularly useful in human-in-the-loop workflows where external input is required to proceed.

Interrupt before: This is called at compile time and pauses execution before the specified node is executed. It is used to ensure that the graph is paused at a point where the user can make a decision or input before proceeding.

Interrupt after: This is called at runtime and pauses execution after the specified node is executed. It is used to ensure that the graph is paused at a point where the user can make a decision or input before proceeding.

These interrupts allow for dynamic control over the execution of the graph, enabling the user to interact with the graph at specific points. 

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.
</details>

# Ship 🚢

- Running local server (`langgraph dev`)
- Short demo showing both assistants responding

# Share 🚀
- Walk through your graph in Studio
- Share 3 lessons learned and 3 lessons not learned

# Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:
1. Create a branch of your `AIE8` repo to track your changes. Example command: `git checkout -b s14-assignment`
2. Complete the Tasks listed in the Breakout Room sections of `Build 🏗️`
3. Complete the activities and questions in `Activities and Questions 🏗️ &❓` by editing the file and replacing "_(enter answer here)_" with your responses
3. Commit, and push your completed notebook to your `origin` repository. _NOTE: Do not merge it into your main branch._
4. Record a Loom video reviewing the content of your completed notebook
5. Make sure to include all of the following on your Homework Submission Form:
    + The GitHub URL to the `README.md` file _on your assignment branch (not main)_
    + The URL to your Loom Video
    + Your Three Lessons Learned/Not Yet Learned
    + The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_


### OPTIONAL: 🚧 Advanced Build Assignment 🚧
<details>
  <summary>(<i>Open this section for the submission instructions.</i>)</summary>

Follow these steps to prepare and submit your homework assignment:
1. Create a branch of your `AIE8` repo to track your changes. Example command: `git checkout -b s14-assignment`
2. Create your MCP server
3. Add it to the existing graph's tools
4. Deploy it ***locally***
5. Validate the graph uses the MCP server's tools
6. Commit, and push your changes to your `origin` repository. _NOTE: Do not merge it into your main branch._
7. Record a Loom video reviewing the content of your completed notebook.
8. Make sure to include all of the following on your Homework Submission Form:
    + The GitHub URL to the notebook you created for the Advanced Build Assignment _on your assignment branch_
    + The URL to your Loom Video
    + Your Three Lessons Learned/Not Yet Learned
    + The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_

</details>
