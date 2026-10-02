def get_qdrant() -> QdrantClient:
    if settings.qdrant_api_key:
        return QdrantClient(
            url=f"https://{settings.qdrant_host}",
            api_key=settings.qdrant_api_key,
        )
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)