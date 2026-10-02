    # Try Streamlit secrets first (hosted)
    try:
        import streamlit as st
        # Try direct key access (not .get() which may not work for all secret types)
        if "GOOGLE_SERVICE_ACCOUNT_JSON" in st.secrets:
            raw = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
            creds_dict = json.loads(str(raw))
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            return build("sheets", "v4", credentials=creds)
    except Exception:
        pass