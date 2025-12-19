import asyncio
import os

from raganything.llm import LLMProviderConfig, build_llm


async def demo_openai_text():
    cfg = LLMProviderConfig(
        provider="openai",
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_base=os.getenv("LLM_API_BASE", ""),
        api_key=os.getenv("LLM_API_KEY", ""),
    )
    llm = build_llm(cfg)
    out = await llm("Say hello in one sentence.")
    print("[openai]", out[:120])


async def demo_openrouter_messages():
    cfg = LLMProviderConfig(
        provider="openrouter",
        model=os.getenv("LLM_MODEL", "openrouter/openai/gpt-4o-mini"),
        api_base=os.getenv("LLM_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
    )
    llm = build_llm(cfg)
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Briefly introduce yourself."}],
        },
    ]
    out = await llm("", messages=msgs)
    print("[openrouter]", out[:120])


async def demo_ollama_text():
    cfg = LLMProviderConfig(
        provider="ollama",
        model=os.getenv("LLM_MODEL", "qwen3:1.7b"),
        api_base=os.getenv("LLM_API_BASE", "http://localhost:11434"),
    )
    llm = build_llm(cfg)
    out = await llm("What is 2+2? reply with a number only")
    print("[ollama]", out[:120])


async def main():
    try:
        await demo_openai_text()
    except Exception as e:
        print("openai demo failed:", e)
    try:
        await demo_openrouter_messages()
    except Exception as e:
        print("openrouter demo failed:", e)
    try:
        await demo_ollama_text()
    except Exception as e:
        print("ollama demo failed:", e)


if __name__ == "__main__":
    asyncio.run(main())
