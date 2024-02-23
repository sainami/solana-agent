import operator
import uuid
import warnings
import numpy as np

from typing import Any, Optional, Dict, Tuple, List, Iterable, Callable, Type
from langchain_core.documents import Document
from langchain_core.vectorstores import VST, VectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance
from langchain_community.vectorstores.faiss import dependable_faiss_import

from embeddings.docstore import RedisStore


class FaissSearcher(VectorStore):
    """FaissSearcher is a vector store that uses Faiss to store and search vectors."""

    embedding: Embeddings
    index: Any
    docstore: RedisStore
    index_to_docstore_id: Dict[int, str]
    normalize_l2: bool
    distance_strategy: DistanceStrategy

    _relevance_score_fn: Callable[[float], float]

    def __init__(
        self,
        docstore: RedisStore,
        embedding_dim: int,
        embeddings: Embeddings,
        normalize_l2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        if distance_strategy != DistanceStrategy.EUCLIDEAN_DISTANCE and normalize_l2:
            warnings.warn(
                "Normalizing l2 is not applicable for metric type: {strategy}".format(strategy=distance_strategy)
            )

        faiss = dependable_faiss_import()
        if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            index = faiss.IndexFlatIP(embedding_dim)
        else:
            # Default to L2, currently other metric types not initialized.
            index = faiss.IndexFlatL2(embedding_dim)

        """Initialize with necessary components."""
        self.embedding = embeddings
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = {}
        self.distance_strategy = distance_strategy
        self.normalize_l2 = normalize_l2
        self._relevance_score_fn = self._select_relevance_score_fn()

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # Default strategy is to rely on distance strategy provided in
        # vectorstore constructor
        if self.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            # Default behavior is to use euclidean distance relevancy
            return self._euclidean_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        else:
            raise ValueError(
                "Unknown distance strategy, must be cosine, max_inner_product,"
                " or euclidean"
            )

    async def _add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must be the same length")
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must be the same length")

        # Add to the index.
        vector = np.array(embeddings, dtype=np.float32)
        if self.normalize_l2:
            faiss = dependable_faiss_import()
            faiss.normalize_L2(vector)
        self.index.add(vector)

        # Add information to docstore and index.
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        await self.docstore.add({_i: (t, e, m) for _i, t, e, m in zip(ids, texts, embeddings, metadatas)})

        starting_len = len(self.index_to_docstore_id)
        index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
        self.index_to_docstore_id.update(index_to_id)
        return ids

    async def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and L2 distance
            in float for each. Lower score represents more similarity.
        """
        vector = np.array([embedding], dtype=np.float32)
        if self.normalize_l2:
            faiss = dependable_faiss_import()
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k if filter is None else fetch_k)
        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = await self.docstore.search(_id)
            if filter is not None:
                filter = {
                    key: [value] if not isinstance(value, list) else value
                    for key, value in filter.items()
                }
                if all(doc.metadata.get(key) in value for key, value in filter.items()):
                    docs.append((doc, scores[0][j]))
            else:
                docs.append((doc, scores[0][j]))

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            cmp = (
                operator.ge
                if self.distance_strategy
                in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.JACCARD)
                else operator.le
            )
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if cmp(similarity, score_threshold)
            ]
        return docs[:k]

    async def _similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding = await self.embeddings.aembed_query(query)
        return await self._similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )

    async def _max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal marginal
                relevance and score for each.
        """
        scores, indices = self.index.search(
            np.array([embedding], dtype=np.float32),
            fetch_k if filter is None else fetch_k * 2,
        )
        if filter is not None:
            filtered_indices = []
            for i in indices[0]:
                if i == -1:
                    # This happens when not enough docs are returned.
                    continue
                _id = self.index_to_docstore_id[i]
                doc = await self.docstore.search(_id)
                if all(
                    doc.metadata.get(key) in value
                    if isinstance(value, list)
                    else doc.metadata.get(key) == value
                    for key, value in filter.items()
                ):
                    filtered_indices.append(i)
            indices = np.array([filtered_indices])
        # -1 happens when not enough docs are returned.
        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        selected_indices = [indices[0][i] for i in mmr_selected]
        selected_scores = [scores[0][i] for i in mmr_selected]
        docs_and_scores = []
        for i, score in zip(selected_indices, selected_scores):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = await self.docstore.search(_id)
            docs_and_scores.append((doc, score))
        return docs_and_scores

    async def init(self):
        _, ids_vectors = await self.docstore.scan_vectors()
        ids = ids_vectors.keys()
        if len(ids) > 0:
            vector = np.array(list(ids_vectors.values()), dtype=np.float32)
            if self.normalize_l2:
                faiss = dependable_faiss_import()
                faiss.normalize_L2(vector)
            self.index.add(vector)
            self.index_to_docstore_id.update(dict(enumerate(ids)))

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError('FaissSearcher does not support sync calls')

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError('FaissSearcher does not support sync calls')

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        raise NotImplementedError('FaissSearcher does not support sync calls')

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        # add vectors to docstore
        texts = list(texts)
        embeddings = await self.embeddings.aembed_documents(texts)
        return await self._add(texts, embeddings, metadatas, ids)

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ValueError("ids must be provided")
        missing_ids = set(ids).difference(self.index_to_docstore_id.values())
        if missing_ids:
            raise ValueError(f"Some specified ids do not exist in the current store. Ids not found: {missing_ids}")

        reversed_index = {id_: idx for idx, id_ in self.index_to_docstore_id.items()}
        index_to_delete = [reversed_index[id_] for id_ in ids]

        self.index.remove_ids(np.array(index_to_delete, dtype=np.int64))
        await self.docstore.delete(ids)

        remaining_ids = [
            id_
            for i, id_ in sorted(self.index_to_docstore_id.items())
            if i not in index_to_delete
        ]
        self.index_to_docstore_id = {i: id_ for i, id_ in enumerate(remaining_ids)}

        return True

    async def asimilarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        docs_and_scores = await self._similarity_search_with_score(query, k, **kwargs)
        docs_and_similarities = [(doc, self._relevance_score_fn(score)) for doc, score in docs_and_scores]
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}"
            )

        score_threshold = kwargs.pop("score_threshold", None)
        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
        return docs_and_similarities

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = await self._similarity_search_with_score(
            query, k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = await self._similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = await self.embeddings.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = await self._max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]