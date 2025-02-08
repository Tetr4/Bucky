from typing import cast, Annotated
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage, ToolCall
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from bucky.recorder import Recorder
from bucky.voice import Voice

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
        workflow.add_edge(START, "chat")
        workflow.add_node("chat", self._chat_node)
        workflow.add_node("tools", ToolNode(tools=self.tools))
        workflow.add_node("clean_up_tool_messages", self._clean_up_tool_messages)
        workflow.add_edge("tools", "clean_up_tool_messages")
        workflow.add_edge("clean_up_tool_messages", "chat")
        workflow.add_conditional_edges("chat", tools_condition, ["tools", END])
        return workflow.compile(checkpointer=MemorySaver())

    def _chat_node(self, state: State, config: RunnableConfig) -> State:
        input: list[BaseMessage] = self.system_prompt + state["messages"]
        if self.vision_llm and state["messages"] and _has_image_data(state["messages"][-1]):
            response: BaseMessage = self.vision_llm.invoke(input, config)
            human_message = cast(HumanMessage, state["messages"][-1])
            new_human_message = _remove_image_data(human_message) # clean up context
            return {"messages": [new_human_message, response]}
        else:
            response: BaseMessage = self.text_llm.invoke(input, config)
            return {"messages": [response]}


    def _clean_up_tool_messages(self, state: State) -> State:
        # After an "take_image" tool call the messages state looks like this:
        # 1. human: "what do you see?"
        # 2. ai: "take_image" tool call
        # 3. tool: image as base64
        #
        # We replace it, so it looks like this:
        # 1. human: "what do you see?" + base64 image
        messages: list[BaseMessage] = state["messages"]
        if not any(_has_take_image_tool_call(message) for message in messages):
            return { "messages" : [] } # no change necessary
        ai_message_index = next(index for index, message in enumerate(messages) if _has_take_image_tool_call(message))
        ai_message = cast(AIMessage, messages[ai_message_index])
        human_message = cast(HumanMessage, messages[ai_message_index - 1])
        vision_tool_call: ToolCall = next(tool_call for tool_call in ai_message.tool_calls if _is_take_image_tool_call(tool_call))
        tool_message = next(message for message in messages if _is_message_for_tool_call(message, vision_tool_call))
        image_base64 = cast(str, tool_message.content)
        new_human_message = HumanMessage(
            id = human_message.id, # update existing message
            content=[
                { "type": "text", "text": human_message.content } if isinstance(human_message.content, str) else human_message.content, # type: ignore
                { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"} },
            ],
        )
        return {
            "messages":  [
                RemoveMessage(id=ai_message.id or ""),
                RemoveMessage(id=tool_message.id or ""),
                new_human_message,
            ]
        }

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

def _has_take_image_tool_call(message: BaseMessage) -> bool:
    if not isinstance(message, AIMessage):
        return False
    return any(_is_take_image_tool_call(tool_call) for tool_call in message.tool_calls if message )

def _is_take_image_tool_call(tool_call: ToolCall) -> bool:
    return tool_call['name'] == 'take_image'

def _is_message_for_tool_call(message: BaseMessage, tool_call: ToolCall) -> bool:
    return isinstance(message, ToolMessage) and message.tool_call_id == tool_call['id']

def _has_image_data(message: BaseMessage) -> bool:
    if not isinstance(message, HumanMessage):
        return False
    if not isinstance(message.content, list):
        return False
    return any(content_item.get('type') == 'image_url' for content_item in message.content if isinstance(content_item, dict))

def _remove_image_data(message: HumanMessage) -> HumanMessage:
    new_content = [item for item in message.content if not (isinstance(item, dict) and item.get('type') == 'image_url')]
    return HumanMessage(id=message.id, content=new_content)
