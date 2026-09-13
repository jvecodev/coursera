# Libraries to create our MCP host application
import json
import os
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastmcp.client import Client, PythonStdioTransport
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

load_dotenv(Path(__file__).parent / ".env")

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
SYSTEM_PROMPT = (
    "You are the Connoisseur Companion, an AI guide to California's restaurant scene. "
    "Use the available tools to look up restaurants by name, find recommendations by "
    "vibe/atmosphere, and retrieve reviews. Always ground your answers in the tool "
    "results rather than guessing, and give concise, friendly recommendations."
)


# Minimal async tool-calling chat model (OpenRouter, OpenAI-compatible API — no IBM
# watsonx quota required), exposing the same `bind_tools(...).ainvoke(messages)` shape
# the ReAct loop below expects, without depending on langchain-openai.
class OpenRouterToolModel:
    def __init__(self, model, temperature=0.7):
        self._model = model
        self._temperature = temperature
        self._tools = None
        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

    def bind_tools(self, tools):
        self._tools = tools
        return self

    @staticmethod
    def _to_openai_message(m):
        if isinstance(m, SystemMessage):
            return {"role": "system", "content": m.content}
        if isinstance(m, HumanMessage):
            return {"role": "user", "content": m.content}
        if isinstance(m, ToolMessage):
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
        if isinstance(m, AIMessage):
            entry = {"role": "assistant", "content": m.content or ""}
            if getattr(m, "tool_calls", None):
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in m.tool_calls
                ]
            return entry
        raise TypeError(f"Unsupported message type: {type(m)}")

    async def ainvoke(self, messages):
        oai_messages = [self._to_openai_message(m) for m in messages]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=oai_messages,
            tools=self._tools,
            temperature=self._temperature,
        )
        choice = response.choices[0].message
        tool_calls = [
            {
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments or "{}"),
                "id": tc.id,
            }
            for tc in (choice.tool_calls or [])
        ]
        return AIMessage(content=choice.content or "", tool_calls=tool_calls)


def make_model():
    return OpenRouterToolModel(model="openai/gpt-4o-mini", temperature=0.7)


# MCP Host — ReAct Agent Loop
async def chat_with_agent(user_message: str, history: list) -> str:
    """Connect to the MCP server, discover tools, and run a ReAct loop.
    The LLM decides which tools to call, calls them via the MCP server,
    and repeats until it produces a final text response."""
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)

    async with Client(transport) as client:
        # Discover available tools from the MCP server
        mcp_tools = await client.list_tools()

        # Convert MCP tool schemas to OpenAI-style tool definitions for the LLM
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in mcp_tools
        ]

        model = make_model().bind_tools(openai_tools)

        # Build the message list from chat history and the new user message
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                messages.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_message))

        # ReAct loop — call tools until the LLM returns a plain text reply
        for _ in range(10):
            response = await model.ainvoke(messages)
            messages.append(response)

            # No tool calls means the LLM is done — return the final response
            if not response.tool_calls:
                raw = response.content
                if isinstance(raw, list):
                    return " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in raw
                    )
                return str(raw)

            # Execute each tool call via the MCP server and feed results back
            for tool_call in response.tool_calls:
                result = await client.call_tool(tool_call["name"], tool_call["args"])
                tool_output = " ".join(
                    item.text if hasattr(item, "text") else str(item)
                    for item in result.content
                ) if result.content else "(no result)"
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        return "I wasn't able to complete that request. Please try again."


# Gradio Event Handler
async def handle_chat(user_message, history):
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        yield history
        return

    # Show a thinking placeholder while the agent runs
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield history

    response_text = await chat_with_agent(user_message, history[:-2])
    history[-1] = {"role": "assistant", "content": response_text}
    yield history


# Gradio Interface
with gr.Blocks(title="Connoisseur Companion") as demo:
    gr.Markdown("# Connoisseur Companion\nYour AI guide to California's restaurant scene. Ask me about restaurants by name, cuisine, or vibe!")

    chatbot = gr.Chatbot(height=500)
    msg_input = gr.Textbox(
        label="Ask about restaurants",
        placeholder='e.g., "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
    )

    with gr.Row():
        btn1 = gr.Button("Find moody restaurants", size="sm")
        btn2 = gr.Button("Tell me about Iron & Embers", size="sm")
        btn3 = gr.Button("Zen dining in Little Tokyo?", size="sm")

    msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot])
    msg_input.submit(lambda: "", None, msg_input)

    btn1.click(handle_chat, [gr.State("Find me some moody restaurants"), chatbot], [chatbot])
    btn2.click(handle_chat, [gr.State("Tell me about Iron & Embers"), chatbot], [chatbot])
    btn3.click(handle_chat, [gr.State("What's a zen dining experience in Little Tokyo?"), chatbot], [chatbot])


# Launch the App
if __name__ == "__main__":
    print("Starting Connoisseur Companion...")
    demo.launch(
        share=True,
        theme=gr.themes.Soft(),
    )
