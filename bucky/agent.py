import json
from typing import cast, Literal, Annotated
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from bucky.recorder import Recorder
from bucky.voice import Voice
from bucky.tools import TakeImageTool

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class Agent:
    def __init__(
            self,
            text_model: str,
            vision_model: str | None,
            system_prompt: str,
            tools: list[BaseTool],
            voice: Voice | None = None,
            recorder: Recorder | None = None
    ) -> None:
        self.system_prompt = [SystemMessage(system_prompt)]
        self.tools = tools
        self.voice = voice
        self.text_llm = ChatOllama(model=text_model).bind_tools(tools)
        self.vision_llm = ChatOllama(model=vision_model) if vision_model else None
        self.graph = self._create_graph()
        self.recorder = recorder

    def _create_graph(self) -> CompiledGraph:
        """
        The agent can output text (END) or call tools.
        After the take_image tool is called, the agent will use the vision model.
        """
        workflow = StateGraph(State)
        workflow.add_edge(START, "agent")
        workflow.add_node("agent", self._chat_node)
        workflow.add_node("tools", ToolNode(tools=self.tools))
        workflow.add_node("vision", self._vision_node)
        workflow.add_conditional_edges("agent", tools_condition, ["tools", END])
        workflow.add_conditional_edges("tools", self._vision_condition, ["vision", "agent"])
        workflow.add_edge("vision", END)
        return workflow.compile(checkpointer=MemorySaver())

    def _chat_node(self, state: State, config: RunnableConfig) -> State:
        input: list[BaseMessage] = self.system_prompt + state["messages"]
        response: BaseMessage = self.text_llm.invoke(input, config)
        return {"messages": [response]}

    def _vision_node(self, state: State, config: RunnableConfig) -> State:
        if self.vision_llm is None:
            raise ValueError("Illegal state. Vision model not available.")
        # Messages state looks like this:
        # 1. human: "what do you see?"
        # 2. ai: "take_image" tool call
        # 3. tool: image as base64
        #
        # We replace it, so it looks like this:
        # 1. human: "what do you see?" + base64 image
        # 2. ai: image description
        human_message = cast(HumanMessage, state["messages"][-3])
        ai_message = cast(AIMessage, state["messages"][-2])
        tool_message = cast(ToolMessage, state["messages"][-1])
        image_base64 = cast(str, tool_message.content)
        new_human_message = HumanMessage(
            content=[
                { "type": "text", "text": human_message.content },
                { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"} },
            ],
        )
        replaced_messages = state["messages"][:-3] + [new_human_message]
        new_ai_message: BaseMessage = self.vision_llm.invoke(self.system_prompt + replaced_messages, config)
        new_ai_message.id = ai_message.id # update old message
        # We don't add the human message which containts the image, so context for follow up
        # conversation is smaller. The model can just take another picture of required.
        return {
            "messages":  [
                new_ai_message,
                RemoveMessage(id=tool_message.id if tool_message.id else ""),
            ]
        }

    def _vision_condition(self, state: State) -> Literal["vision", "agent"]:
        if self.vision_llm is None:
            return "agent"
        ai_message = state["messages"][-2]
        if isinstance(ai_message, AIMessage) and ai_message.tool_calls:
            last_tool_call = ai_message.tool_calls[-1]
            if last_tool_call['name'] == "take_image":
                return "vision"
        return "agent"

    def run(self, thread_id: int = 1) -> None:
        while True:
            user_input = self.recorder.listen() if self.recorder else input("You: ")
            self._generate_answer(user_input, thread_id)

    def _generate_answer(self, user_input: str, thread_id: int) -> None:
        inputs = {"messages": [HumanMessage(content=user_input)]}
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        for chunk in self.graph.stream(inputs, config, stream_mode="values"):
            message: BaseMessage = chunk["messages"][-1]
            self._output(message)

    def _output(self, message: BaseMessage) -> None:
        message.pretty_print()
        content = cast(str, message.content)
        if self.voice and isinstance(message, AIMessage) and len(content) > 0:
            self.voice.speak(content)
