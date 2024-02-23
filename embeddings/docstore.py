import json
import numpy as np

from typing import Optional, Dict, Tuple, List, Iterable
from redis.asyncio import Redis as RedisClient
from langchain_core.documents import Document


class RedisStore:
    """Redis store for documents."""

    client: RedisClient
    index_name: str
    batch_size: int
    content_key: bytes
    metadata_key: bytes
    vector_key: bytes

    def __init__(
        self,
        *,
        url: str,
        index_name: str,
        batch_size: int,
        content_key: bytes = b'content',
        metadata_key: bytes = b'metadata',
        vector_key: bytes = b'content_vector',
        **redis_kwargs,
    ):
        self.client = RedisClient.from_url(url, **redis_kwargs)
        self.index_name = index_name
        self.batch_size = batch_size
        self.content_key = content_key
        self.metadata_key = metadata_key
        self.vector_key = vector_key

    def _redis_prefix(self) -> str:
        return f'doc:{self.index_name}:'

    def _redis_match(self) -> str:
        return self._redis_prefix() + '*'

    def _redis_key(self, _id: str) -> str:
        return self._redis_prefix() + _id

    def _key_to_id(self, key: str) -> str:
        return key.removeprefix(self._redis_prefix())

    async def _hscan(
        self, hash_key: bytes, cursor: Optional[int] = None, count: Optional[int] = None
    ) -> Tuple[int, Iterable[Tuple[str, bytes]]]:
        ids, outputs = [], []
        pipeline = self.client.pipeline()
        i = 1
        if cursor is None:
            cursor = 0
            async for key in self.client.scan_iter(self._redis_match(), count):
                _ = pipeline.hget(key, hash_key)
                if i % self.batch_size == 0:
                    outputs.extend(await pipeline.execute())
                i += 1
                ids.append(self._key_to_id(key.decode()))
        else:
            cursor, keys = await self.client.scan(cursor, self._redis_match(), count)
            for key in keys:
                _ = pipeline.hget(key, hash_key)
                if i % self.batch_size == 0:
                    outputs.extend(await pipeline.execute())
                i += 1
                ids.append(self._key_to_id(key.decode()))
        outputs.extend(await pipeline.execute())

        return cursor, zip(ids, outputs)

    async def search(self, _id: str) -> Document:
        page_content = await self.get_content(_id)
        metadata = await self.get_metadata(_id)
        return Document(page_content=page_content, metadata=metadata)

    async def add(self, datas: Dict[str, Tuple[str, List[float], Dict]]):
        pipeline = self.client.pipeline()
        i = 1
        for _id, (content, vector, metadata) in datas.items():
            _ = pipeline.hset(
                self._redis_key(_id),
                mapping={
                    self.content_key: content,
                    self.vector_key: np.array(vector, dtype=np.float32).tobytes(),
                    self.metadata_key: json.dumps(metadata),
                },
            )
            if i % self.batch_size == 0:
                await pipeline.execute()
            i += 1
        await pipeline.execute()

    async def delete(self, ids: List[str]):
        await self.client.delete(*[self._redis_key(_id) for _id in ids])

    async def get_content(self, _id: str) -> str:
        content = await self.client.hget(self._redis_key(_id), self.content_key)
        if content is not None:
            # content is bytes, decode to str
            return content.decode()
        else:
            raise ValueError(f'no content for id: {_id}')

    async def get_many_contents(self, ids: Iterable[str]) -> Dict[str, str]:
        pipeline = self.client.pipeline()
        for _id in ids:
            _ = pipeline.hget(self._redis_key(_id), self.content_key)
        contents: List = await pipeline.execute()
        return {_id: content.decode() for _id, content in zip(ids, contents) if content is not None}

    async def get_metadata(self, _id: str) -> Dict:
        metadata = await self.client.hget(self._redis_key(_id), self.metadata_key)
        if metadata is not None:
            return json.loads(metadata)
        else:
            raise ValueError(f'no metadata for id: {_id}')

    async def get_many_metadatas(self, ids: Iterable[str]) -> Dict[str, Dict]:
        pipeline = self.client.pipeline()
        for _id in ids:
            _ = pipeline.hget(self._redis_key(_id), self.metadata_key)
        metadatas: List = await pipeline.execute()
        return {_id: json.loads(metadata) for _id, metadata in zip(ids, metadatas) if metadata is not None}

    async def get_vector(self, _id: str) -> List[float]:
        vector = await self.client.hget(self._redis_key(_id), self.vector_key)
        if vector is not None:
            return np.frombuffer(vector, dtype=np.float32).tolist()
        else:
            raise ValueError(f'no vector for id: {_id}')

    async def get_many_vectors(self, ids: Iterable[str]) -> Dict[str, List[float]]:
        pipeline = self.client.pipeline()
        for _id in ids:
            _ = pipeline.hget(self._redis_key(_id), self.vector_key)
        vectors: List = await pipeline.execute()
        return {_id: np.frombuffer(vector, dtype=np.float32).tolist()
                for _id, vector in zip(ids, vectors) if vector is not None}

    async def scan_ids(
        self, cursor: Optional[int] = None, count: Optional[int] = None
    ) -> Tuple[int, List[str]]:
        if cursor is None:
            return 0, [self._key_to_id(key.decode())
                       async for key in self.client.scan_iter(self._redis_match(), count)]
        else:
            cursor, keys = await self.client.scan(cursor, self._redis_match(), count)
            return cursor, [self._key_to_id(key.decode()) for key in keys]

    async def scan_contents(
        self, cursor: Optional[int] = None, count: Optional[int] = None
    ) -> Tuple[int, Dict[str, str]]:
        cursor, outputs = await self._hscan(self.content_key, cursor, count)
        return cursor, {_id: content.decode() for _id, content in outputs}

    async def scan_vectors(
        self, cursor: Optional[int] = None, count: Optional[int] = None
    ) -> Tuple[int, Dict[str, List[float]]]:
        cursor, outputs = await self._hscan(self.vector_key, cursor, count)
        return cursor, {_id: np.frombuffer(vec, dtype=np.float32).tolist() for _id, vec in outputs}

    async def scan_metadatas(
        self, cursor: Optional[int] = None, count: Optional[int] = None
    ) -> Tuple[int, Dict[str, Dict]]:
        cursor, outputs = await self._hscan(self.metadata_key, cursor, count)
        return cursor, {_id: json.loads(metadata) for _id, metadata in outputs}

    async def scan_all(
        self,
        cursor: Optional[int] = None,
        count: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Tuple[str, List[float], Dict]]]:
        ids = []
        outputs = []
        pipeline = self.client.pipeline()
        i = 1
        if cursor is None:
            cursor = 0
            async for key in self.client.scan_iter(self._redis_match(), count):
                _ = pipeline.hgetall(key)
                if i % self.batch_size == 0:
                    outputs.extend(await pipeline.execute())
                i += 1
                ids.append(self._key_to_id(key.decode()))
        else:
            cursor, keys = await self.client.scan(cursor, self._redis_match(), count)
            for key in keys:
                _ = pipeline.hgetall(key)
                if i % self.batch_size == 0:
                    outputs.extend(await pipeline.execute())
                i += 1
                ids.append(self._key_to_id(key.decode()))
        outputs.extend(await pipeline.execute())

        def _parse(data: Dict) -> Tuple[str, List[float], Dict]:
            return (
                data[self.content_key].decode(),
                np.frombuffer(data[self.vector_key], dtype=np.float32).tolist(),
                json.loads(data[self.metadata_key]),
            )
        return cursor, {_id: _parse(data) for _id, data in zip(ids, outputs)}

    # only metadata can be updated
    async def update_metadatas(self, metadatas: Dict[str, Dict]):
        pipeline = self.client.pipeline()
        i = 1
        for _id, metadata in metadatas.items():
            _ = pipeline.hset(self._redis_key(_id), self.metadata_key, json.dumps(metadata))
            if i % self.batch_size == 0:
                await pipeline.execute()
            i += 1
        await pipeline.execute()
