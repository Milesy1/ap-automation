def ensure_collection() -> None:
    """Create Qdrant collection and required payload indexes if they do not exist."""
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )
    # Ensure payload index on vendor_id — required for filtered search on Qdrant Cloud
    try:
        from qdrant_client.models import PayloadSchemaType
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="vendor_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass  # Index already exists