    if st.button("Seed corpus"):
        import csv
        from pathlib import Path
        from ap_automation.core.retrieval import index_outcome
        from ap_automation.core.mdm import load_vendor_map, resolve_vendor

        path = Path("data/invoices.csv")
        if not path.exists():
            st.error("Generate the dataset first.")
        else:
            load_vendor_map("data/vendor_map.csv")
            with open(path) as f:
                rows = list(csv.DictReader(f))

            routine = [r for r in rows if r["category"] == "routine"][:seed_count]
            progress = st.progress(0)
            for i, row in enumerate(routine):
                canonical_id, _ = resolve_vendor(row["raw_vendor_name"])
                payload = {
                    "point_id": str(uuid4()),
                    "invoice_id": row["invoice_id"],
                    "vendor_id": canonical_id,
                    "entity_id": row["entity_id"],
                    "amount_band": "small",
                    "currency": row["currency"],
                    "gl_account": row["ground_truth_gl"],
                    "cost_centre": row["cost_centre"],
                    "tax_code": "STANDARD",
                    "description": row["description"],
                    "confirming_user_id": "seed",
                    "confirmation_timestamp": "2024-01-01T00:00:00",
                    "prediction_confidence": 1.0,
                    "source": "human_confirm",
                }
                index_outcome(payload)
                progress.progress((i + 1) / len(routine))
            st.success(f"Seeded {len(routine)} historical lines into Qdrant")