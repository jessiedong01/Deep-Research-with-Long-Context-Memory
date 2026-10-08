import os

from dotenv import load_dotenv

from deepcomposition.agent import DeepCompositionAgent
from utils.encoder import Encoder
from utils.literature_search import LiteratureSearchAgent
from utils.lm import AzureOpenAIConfig, LanguageModelProvider, LanguageModelProviderConfig, OpenAIConfig, init_lm
from utils.rag import RagAgent
from utils.retriever_agent.serper_rm import SerperRM

load_dotenv()


def _get_temperature(env_var: str, default: float = 1.0) -> float:
    """Get temperature from environment variable with a default.
    
    OpenAI reasoning models (o1, gpt-5, etc.) require temperature=1.0.
    For other models, this allows customization via environment variables.
    
    Args:
        env_var: The environment variable name to check
        default: Default temperature if env var not set (default: 1.0)
    
    Returns:
        Temperature value as a float
    """
    value = os.environ.get(env_var)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_max_tokens(env_var: str, default: int) -> int:
    """Get max_tokens from environment variable with a default.
    
    OpenAI reasoning models (o1, gpt-5, etc.) require max_tokens >= 16000.
    """
    value = os.environ.get(env_var)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _provider_kwargs() -> dict:
    """Provider selection from LM_PROVIDER: "openai" (default) or "azure"."""
    if os.environ.get("LM_PROVIDER", "openai").lower() == "azure":
        return {"provider": LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_AZURE_OPENAI,
                "azure_openai_config": AzureOpenAIConfig(api_key=os.environ["AZURE_OPENAI_API_KEY"], api_base=os.environ["AZURE_OPENAI_BASE"],
                                                         api_version=os.environ["AZURE_OPENAI_API_VERSION"])}
    return {"provider": LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_OPENAI, "openai_config": OpenAIConfig(api_key=os.environ["OPENAI_API_KEY"])}


def _lm(model_env: str, default_model: str, temp_env: str, tokens_env: str, default_tokens: int):
    return init_lm(LanguageModelProviderConfig(model_name=os.environ.get(model_env, default_model), temperature=_get_temperature(temp_env, 1.0),
                                               max_tokens=_get_max_tokens(tokens_env, default_tokens), **_provider_kwargs()))


def init_rag_agent() -> RagAgent:
    azure = os.environ.get("LM_PROVIDER", "openai").lower() == "azure"
    encoder = Encoder(
        model_name=os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-large"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"] if azure else os.environ["OPENAI_API_KEY"],
        api_base=os.environ["AZURE_OPENAI_BASE"] if azure else None,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    serper_retriever = SerperRM(api_key=os.environ["SERPER_API_KEY"], encoder=encoder)
    rag_lm = _lm("RAG_MODEL_NAME", "gpt-4.1-mini", "RAG_LM_TEMPERATURE", "RAG_LM_MAX_TOKENS", 10000)
    return RagAgent(retriever=serper_retriever, rag_lm=rag_lm)


def init_agent() -> DeepCompositionAgent:
    """Build the pipeline: a search-backed RAG agent, a planning model, a synthesis model, and the DAG agent on top."""
    rag_agent = init_rag_agent()
    planning_lm = _lm("PLANNING_MODEL_NAME", "gpt-4.1-mini", "PLANNING_LM_TEMPERATURE", "PLANNING_LM_MAX_TOKENS", 10000)
    synthesis_lm = _lm("SYNTHESIS_MODEL_NAME", "gpt-5-chat", "SYNTHESIS_LM_TEMPERATURE", "SYNTHESIS_LM_MAX_TOKENS", 16000)
    literature_search_agent = LiteratureSearchAgent(rag_agent=rag_agent, literature_search_lm=planning_lm, answer_synthesis_lm=synthesis_lm)
    return DeepCompositionAgent(literature_search_agent=literature_search_agent, lm=synthesis_lm)
