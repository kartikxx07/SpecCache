import os
from typing import Any

import redis
import torch
from transformers import AutoModel, AutoTokenizer

from app.stage1.api import get_data_from_endpoint


REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://default:nkn5lyBHoXBOwmvZ2qG7PGV4eaXsxCg7@leather-emotional-harmonious-11768.db.redis.io:11776",
)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAME = "prompt_index"
PREFIX = "prompt:"
VECTOR_DIM = 384


def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def create_vector_index(client: redis.Redis | None = None) -> None:
    if client is None:
        client = get_redis_client()

    try:
        client.execute_command(
            "FT.CREATE",
            INDEX_NAME,
            "ON",
            "HASH",
            "PREFIX",
            1,
            PREFIX,
            "SCHEMA",
            "prompt",
            "TEXT",
            "embedding",
            "VECTOR",
            "HNSW",
            6,
            "TYPE",
            "FLOAT32",
            "DIM",
            VECTOR_DIM,
            "DISTANCE_METRIC",
            "COSINE",
        )
    except redis.exceptions.ResponseError as exc:
        if "already exists" not in str(exc).lower():
            raise


def embed_text(text: str) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)

    return embeddings[0].cpu().tolist()


def store_prompt(prompt: str, client: redis.Redis | None = None) -> str:
    if client is None:
        client = get_redis_client()

    embedding = embed_text(prompt)
    key = f"{PREFIX}{abs(hash(prompt))}"
    client.hset(
        key,
        mapping={
            "prompt": prompt,
            "embedding": " ".join(str(value) for value in embedding),
        },
    )
    return key


def search_similar_prompts(query: str, top_k: int = 4, client: redis.Redis | None = None) -> list[dict[str, Any]]:
    if client is None:
        client = get_redis_client()

    create_vector_index(client)
    query_vector = embed_text(query)
    query_vector_blob = " ".join(str(value) for value in query_vector)

    result = client.execute_command(
        "FT.SEARCH",
        INDEX_NAME,
        f"*=>[KNN {top_k} @embedding $vec AS score]",
        "PARAMS",
        2,
        "vec",
        query_vector_blob,
        "SORTBY",
        "score",
        "DIALECT",
        2,
    )

    items = []
    if len(result) > 1:
        for index in range(1, len(result), 2):
            doc = result[index]
            items.append(
                {
                    "id": doc[0],
                    "prompt": doc[1],
                    "score": float(doc[3]),
                }
            )

    return items


def semantic_cache(data: dict[str, Any], client: redis.Redis | None = None) -> list[dict[str, Any]]:
    user_response = data.get("response", {}).get("user_response", {})
    if isinstance(user_response, dict):
        prompt_text = user_response.get("text") or user_response.get("message") or str(user_response)
    else:
        prompt_text = str(user_response)

    if not prompt_text:
        return []

    if client is None:
        client = get_redis_client()

    store_prompt(prompt_text, client=client)
    return search_similar_prompts(prompt_text, client=client)


def demo() -> None:
    data = get_data_from_endpoint()
    print(semantic_cache(data))


if __name__ == "__main__":
    demo()

