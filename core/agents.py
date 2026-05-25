"""
Agent 定义 - 支持中英文双语
"""

from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.deepseek import DeepSeek
from agno.tools.duckduckgo import DuckDuckGoTools
from typing import Literal, Tuple
from core.language_manager import get_language_manager

ModelChoice = Literal["gemini", "openai", "claude", "deepseek"]

MODEL_ID = {
    "gemini": "gemini-2.0-flash-exp",
    "openai": "gpt-4o",
    "claude": "claude-3-5-sonnet-20241022",
    "deepseek": "deepseek-chat"
}


def build_agents(api_key: str, choice: ModelChoice, language: str = "zh"):
    """
    构建4个Agent实例（通用情绪恢复）

    Args:
        api_key: API密钥
        choice: 模型选择
        language: 语言 ("zh" 或 "en")

    Returns:
        (empathy_agent, cognitive_agent, behavioral_agent, motivational_agent)
    """
    # 设置语言
    from core.language_manager import set_language
    set_language(language)
    lang_manager = get_language_manager()

    if choice == "gemini":
        model = Gemini(id=MODEL_ID[choice], api_key=api_key)
    elif choice == "openai":
        model = OpenAIChat(id=MODEL_ID[choice], api_key=api_key)
    elif choice == "claude":
        model = Claude(id=MODEL_ID[choice], api_key=api_key)
    elif choice == "deepseek":
        model = DeepSeek(id=MODEL_ID[choice], api_key=api_key)
    else:
        raise ValueError("Unknown model choice")

    # 共情Agent
    empathy_agent = Agent(
        model=model,
        name="Empathy Agent",
        instructions=lang_manager.get_empathy_instructions(),
        markdown=True
    )

    # 认知重构Agent
    cognitive_agent = Agent(
        model=model,
        name="Cognitive Restructuring Agent",
        instructions=lang_manager.get_cognitive_instructions(),
        markdown=True
    )

    # 行为支持Agent
    behavioral_agent = Agent(
        model=model,
        name="Behavioral Support Agent",
        instructions=lang_manager.get_behavioral_instructions(),
        markdown=True
    )

    # 激励Agent
    motivational_agent = Agent(
        model=model,
        name="Motivational Agent",
        tools=[DuckDuckGoTools()],
        instructions=lang_manager.get_motivational_instructions(),
        markdown=True
    )

    return empathy_agent, cognitive_agent, behavioral_agent, motivational_agent