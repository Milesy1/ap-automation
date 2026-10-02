def _get_langfuse():
    """Return a Langfuse client or None if not configured."""
    try:
        import os
        # Try Streamlit secrets first (hosted)
        try:
            import streamlit as st
            pub = st.secrets.get("LANGFUSE_PUBLIC_KEY", "")
            sec = st.secrets.get("LANGFUSE_SECRET_KEY", "")
            host = st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
            if pub and sec:
                os.environ["LANGFUSE_PUBLIC_KEY"] = str(pub)
                os.environ["LANGFUSE_SECRET_KEY"] = str(sec)
                os.environ["LANGFUSE_HOST"] = str(host)
        except Exception:
            pass

        # Fall back to settings/.env
        if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
            from ap_automation.core.config import settings
            if not getattr(settings, 'langfuse_public_key', '') or not getattr(settings, 'langfuse_secret_key', ''):
                return None
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = getattr(settings, 'langfuse_host', 'https://cloud.langfuse.com')

        if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
            return None

        from langfuse import get_client
        return get_client()
    except Exception:
        return None