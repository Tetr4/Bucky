import base64
from typing import Callable, Literal, Optional, Annotated
from typing_extensions import TypedDict
from speech_recognition import AudioData
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_core.messages.base import get_msg_title_repr
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from bucky.common.message_utils import has_image_data
from bucky.recorder import Recorder, Transcription
from bucky.voice import Voice
import bucky.config as cfg


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class Agent:
    def __init__(
            self,
            llm: BaseChatModel,
            system_prompt_template: str,
            tools: list[BaseTool],
            voice: Voice | None = None,
            recorder: Recorder | None = None
    ) -> None:
        self.system_prompt_template = system_prompt_template
        self.tools = tools
        self.voice = voice
        self.llm = llm.bind_tools(tools)
        self.graph = self._create_graph()
        self.recorder = recorder
        self.system_prompt_format_callback: Optional[Callable[[str], str | list]] = None
        self.debug_state_callback: Optional[Callable[[list[BaseMessage]], None]] = None

    @property
    def system_message(self) -> list[BaseMessage]:
        system_prompt_content = self.system_prompt_template
        if self.system_prompt_format_callback is not None:
            system_prompt_content = self.system_prompt_format_callback(self.system_prompt_template)

        # print(f"System prompt: {system_prompt}")
        return [SystemMessage(content=system_prompt_content)]

    def _create_graph(self) -> CompiledStateGraph:
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
            if self.recorder:
                user_input = self.recorder.listen()
            else:
                user_input = input("You: ")
            self._generate_answer(user_input, thread_id)

    def _generate_answer(self, user_input: str | Transcription, thread_id: int) -> None:
        if isinstance(user_input, str):
            content = user_input
        elif cfg.model_audio_input:
            content = user_input.record.create_message_content()
        else:
            content = user_input.phrase

        inputs = {"messages": [HumanMessage(content=content)]}
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
