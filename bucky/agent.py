from typing import Callable, Literal, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_core.messages.base import get_msg_title_repr
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from bucky.memory_store import MemoryStore
from bucky.message_utils import has_image_data
from bucky.recorder import Recorder
from bucky.voice import Voice


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class Agent:
    def __init__(
            self,
            model: str,
            system_prompt_template: str,
            memory_store: MemoryStore,
            tools: list[BaseTool],
            voice: Voice | None = None,
            recorder: Recorder | None = None
    ) -> None:
        self.system_prompt_template = system_prompt_template
        self.memory_store = memory_store
        self.tools = tools
        self.voice = voice
        self.llm = ChatOllama(model=model).bind_tools(tools)
        self.graph = self._create_graph()
        self.recorder = recorder
        self.debug_state_callback: Optional[Callable[[list[BaseMessage]], None]] = None

    @property
    def system_message(self) -> list[BaseMessage]:
        memories = self.memory_store.dump()
        system_prompt = self.system_prompt_template.format(memories=memories)
        print(f"System prompt: {system_prompt}")
        return [SystemMessage(content=system_prompt)]

    def _create_graph(self) -> CompiledGraph:
        """
        The agent can output text (END) or call tools.
        """
        workflow = StateGraph(State)
        workflow.add_node("chat", self._chat_node)
        workflow.add_node("tools", ToolNode(tools=self.tools))
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("debug", self._debug_node)
        workflow.add_node("speak", self._speak_node)

        workflow.add_edge(START, "chat")
        workflow.add_conditional_edges("chat", self._goto_tools_or_speak)
        workflow.add_edge("tools", "chat")
        workflow.add_edge("speak", "summarize")
        workflow.add_edge("summarize", "debug")
        workflow.add_edge("debug", END)

        return workflow.compile(checkpointer=MemorySaver())

    def _chat_node(self, state: State, config: RunnableConfig) -> State:
        input: list[BaseMessage] = self.system_message + state["messages"]
        response: BaseMessage = self.llm.invoke(input, config)
        return {"messages": [response]}

    def _summarize_node(self, state: State, config: RunnableConfig) -> State:
        messages: list[BaseMessage] = state["messages"]

        if len(messages) > 64:
            print("SUMMARIZATION")
            summarize_prompt: str = """Summarize the previous conversation and mention important facts such as the names of everyone you spoke to. 
            Answer in the same language as the conversation."""
            input: list[BaseMessage] = self.system_message + messages + [HumanMessage(content=summarize_prompt)]
            summarization: BaseMessage = self.llm.invoke(input, config)
            if summarization.text():
                new_messages: list[BaseMessage] = [RemoveMessage(id=msg.id or "") for msg in messages]
                new_messages += [summarization]
                return {"messages": new_messages}
            print("SUMMARIZATION FAILED")

        rm_messages: list[BaseMessage] = []
        max_images = 2  # only keep the last two images
        num_images: int = 0
        for message in reversed(messages):
            if has_image_data(message):
                num_images += 1
                if num_images > max_images:
                    num_images -= 1
                    rm_messages.append(RemoveMessage(id=message.id or ""))
        return {"messages": rm_messages}

    def _debug_node(self, state: State) -> State:
        messages: list[BaseMessage] = self.system_message + state["messages"]
        if self.debug_state_callback:
            self.debug_state_callback(messages)
        return {"messages": []}

    def _speak_node(self, state: State) -> State:
        messages: list[BaseMessage] = state["messages"]
        if messages:
            last_message = messages[-1]
            content = str(last_message.content)
            if self.voice and isinstance(last_message, AIMessage) and len(content) > 0:
                self.voice.speak(content)
        return {"messages": []}

    def _goto_tools_or_speak(self, state, messages_key: str = "messages") -> Literal["tools", "speak"]:
        result: Literal["tools", "__end__"] = tools_condition(state, messages_key)
        return "tools" if result == "tools" else "speak"

    def run(self, thread_id: int = 1) -> None:
        while True:
            user_input = self.recorder.listen() if self.recorder else input("You: ")
            self._generate_answer(user_input, thread_id)

    def _generate_answer(self, user_input: str, thread_id: int) -> None:
        inputs = {"messages": [HumanMessage(content=user_input)]}
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        consumed_messages = set()
        for chunk in self.graph.stream(inputs, config, stream_mode="values"):
            message: BaseMessage = chunk["messages"][-1]
            if message.id not in consumed_messages:
                consumed_messages.add(message.id)
                self._output(message)

    def _output(self, message: BaseMessage) -> None:
        if isinstance(message.content, str):
            message.pretty_print()
        else:
            print(get_msg_title_repr(message.type.title() + " Message"))


def preload_ollama_model(model: str):
    ChatOllama(model=model).invoke(".")
