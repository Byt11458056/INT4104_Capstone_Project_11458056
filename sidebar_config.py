"""
Shared sidebar configuration for LLM provider, model, and region.

Built-in providers (OpenRouter, Alibaba Qwen) use API keys from the
environment.  A "Custom Provider" option lets users bring their own endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import streamlit as st
from openai import OpenAI

from config import get_secret
from llm_summary import PROVIDERS

log = logging.getLogger(__name__)

DASHSCOPE_REGIONS: dict[str, str] = {
    "Hong Kong (China)": "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
    "Singapore / International": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "US (Virginia)": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "China (Beijing)": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

OPENROUTER_EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "AI Financial Analyst",
}

_BUILTIN_PROVIDERS = {"OpenRouter", "Alibaba Qwen (DashScope)"}
_GLOBAL_KEY_PREFIX = "global_llm"


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    extra_headers: dict


def resolve_base_url(provider: str, preset: dict, override: str | None = None) -> str:
    if override:
        return override.rstrip("/")
    if provider == "Alibaba Qwen (DashScope)":
        return (get_secret("DASHSCOPE_BASE_URL") or preset["base_url"]).rstrip("/")
    return preset["base_url"].rstrip("/")


def _fetch_models(api_key: str, base_url: str, extra_headers: dict) -> list[str]:
    try:
        client = OpenAI(
            api_key=api_key, base_url=base_url,
            default_headers=extra_headers, timeout=10.0,
        )
        resp = client.models.list()
        return sorted(m.id for m in resp.data)
    except Exception as exc:
        log.warning("Could not fetch models: %s", exc)
        return []


def _global_key(name: str) -> str:
    return f"{_GLOBAL_KEY_PREFIX}_{name}"


def render_llm_sidebar(prefix: str) -> LLMConfig:
    """Render one shared LLM configuration for the whole app session."""
    st.subheader("LLM Provider")

    provider_names = list(PROVIDERS.keys())
    provider = st.selectbox("Provider", provider_names, index=0, key=_global_key("provider"))
    preset = PROVIDERS[provider]
    is_builtin = provider in _BUILTIN_PROVIDERS

    region_override: str | None = None
    if provider == "Alibaba Qwen (DashScope)":
        region_label = st.selectbox(
            "DashScope region", list(DASHSCOPE_REGIONS.keys()),
            index=0, key=_global_key("region"),
            help="Must match where your API key was issued.",
        )
        region_override = DASHSCOPE_REGIONS[region_label]

    if is_builtin:
        api_key = get_secret(preset["env_key"]) or ""
        if api_key:
            st.caption("API key loaded from environment")
        else:
            st.warning(f"No `{preset['env_key']}` found in environment.", icon="⚠️")
    else:
        api_key = st.text_input(
            "API key", value="", type="password", key=_global_key("custom_apikey"),
        )

    if provider == "Custom Provider":
        custom_base_url = st.text_input(
            "Base URL", value="https://api.openai.com/v1", key=_global_key("custom_base_url"),
        )
        base_url = custom_base_url.rstrip("/") if custom_base_url else ""
    else:
        base_url = resolve_base_url(provider, preset, region_override)

    if provider == "Custom Provider":
        model = st.text_input(
            "Model ID", value="", key=_global_key("custom_model_id"),
            placeholder="e.g. gpt-4o-mini, claude-3.5-sonnet",
        )
        if not model:
            model = "gpt-4o-mini"
    else:
        cache_key = _global_key(f"fetched_models_{provider}")
        fetched: list[str] = st.session_state.get(cache_key, [])
        col_model, col_fetch = st.columns([3, 1])
        with col_fetch:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            if st.button("Fetch", key=_global_key("fetch_btn"), width="stretch"):
                if api_key:
                    extra = OPENROUTER_EXTRA_HEADERS if provider == "OpenRouter" else {}
                    with st.spinner("..."):
                        fetched = _fetch_models(api_key, base_url, extra)
                    if fetched:
                        st.session_state[cache_key] = fetched
                    else:
                        st.warning("No models returned.", icon="⚠️")
                else:
                    st.warning("API key not available.", icon="🔑")

        model_list = fetched if fetched else preset["models"]

        with col_model:
            search_term = st.text_input(
                "Model", value="", key=_global_key("model_search"),
                placeholder="Type to search models...",
            )

        filtered = (
            [m for m in model_list if search_term.lower() in m.lower()]
            if search_term else model_list
        )
        display_options = filtered + ["Custom"]
        selected = st.selectbox(
            "Select model", display_options, index=0,
            key=_global_key("model_select"), label_visibility="collapsed",
        )

        if selected == "Custom":
            model = st.text_input(
                "Custom model ID", value="", key=_global_key("custom_model"),
                placeholder="e.g. qwen-plus",
            )
            if not model:
                model = preset["default_model"]
        else:
            model = selected

    extra_headers = OPENROUTER_EXTRA_HEADERS if provider == "OpenRouter" else {}

    st.divider()
    st.caption(
        "**Disclaimer:** This app does not provide financial advice. "
        "All outputs are for reference only. Users bear their own risks."
    )

    return LLMConfig(
        provider=provider, model=model, api_key=api_key,
        base_url=base_url, extra_headers=extra_headers,
    )
