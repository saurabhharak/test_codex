import requests
import autogen
from typing import Optional


def fetch_url_text(url: str) -> str:
    """Fetch text from the given URL via r.jina.ai."""
    proxy_url = f"https://r.jina.ai/{url}"
    response = requests.get(proxy_url)
    response.raise_for_status()
    return response.text


def create_agents(model: str = "gpt-3.5-turbo"):
    """Create a simple user and assistant agent."""
    llm_config = {
        "config_list": [{"model": model}],
    }
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=llm_config,
    )
    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
    )
    return assistant, user_proxy


def summarize_url(url: str, model: str = "gpt-3.5-turbo") -> Optional[str]:
    """Fetch content from URL and summarize it using an autogen agent."""
    text = fetch_url_text(url)
    assistant, user = create_agents(model)
    try:
        user.initiate_chat(assistant, message=f"Summarize the following text:\n{text}")
        return user.last_message().get("content")
    finally:
        assistant.reset()
        user.reset()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rjina_agent.py <url>")
        raise SystemExit(1)
    url = sys.argv[1]
    summary = summarize_url(url)
    print(summary or "No summary produced.")
