from typing import Sequence, Optional, List, Tuple
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools import BaseTool
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.schema.runnable import RunnableConfig
from langchain_openai.chat_models import ChatOpenAI


class Chatter:
    agent: AgentExecutor

    def __init__(
        self,
        *,
        model: ChatOpenAI,
        tools: Sequence[BaseTool],
        **kwargs,
    ):
        prompt = ChatPromptTemplate(
            messages=[
                MessagesPlaceholder(variable_name="messages"),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )
        agent = create_openai_functions_agent(model, tools, prompt)
        self.agent = AgentExecutor(agent=agent, tools=tools, **kwargs)

    @staticmethod
    def _create_messages(chat_history: List[Tuple[str, str]]) -> List[BaseMessage]:
        messages = []
        for role, message in chat_history:
            if role in ("human", "user"):
                messages.append(HumanMessage(content=message))
            elif role in ("ai", "assistant"):
                messages.append(AIMessage(content=message))
            elif role in ("system", "bot"):
                messages.append(SystemMessage(content=message))
            else:
                raise ValueError(f"Unexpected role: {role}")
        return messages

    async def chat(
        self,
        question: str,
        chat_history: List[Tuple[str, str]],
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> str:
        result = await self.agent.ainvoke(
            {
                "input": question,
                "messages": self._create_messages(chat_history),
            },
            config=config,
            **kwargs,
        )
        return result["output"]
