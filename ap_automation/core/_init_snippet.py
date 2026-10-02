def initialise() -> None:
    if not st.session_state.initialised:
        init_db()
        try:
            ensure_collection()
        except Exception:
            pass
        load_vendor_map("data/vendor_map.csv")
        try:
            ensure_sheet_headers()
        except Exception:
            pass
        st.session_state.initialised = True