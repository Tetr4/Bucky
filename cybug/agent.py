from typing import Annotated, cast
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from cybug.voice import Voice

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class Agent:
    def __init__(self, model: str, system_prompt: str, tools: list[BaseTool], voice: Voice) -> None:
        self.system_prompt = [SystemMessage(system_prompt)]
        self.tools = tools
        self.voice = voice
        self.llm = ChatOllama(model=model).bind_tools(tools)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledGraph:
        workflow = StateGraph(State)
        workflow.add_edge(START, "agent")
        workflow.add_node("agent", self._chat_node)
        workflow.add_conditional_edges("agent", tools_condition, ["tools", END])
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=MemorySaver())

    def _chat_node(self, state: State, config: RunnableConfig) -> State:
        input: list[BaseMessage] = self.system_prompt + state["messages"]
        response: BaseMessage = self.llm.invoke(input, config)
        return {"messages": [response]}

    def run(self, thread_id: int = 1) -> None:
        while True:
            user_input = input("You: ")
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
        if isinstance(message, AIMessage) and len(content) > 0:
            self.voice.speak(content)
