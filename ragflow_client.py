"""
RAGFlow API Client
Модуль для взаимодействия с RAGFlow API для получения семантически близких чанков.
"""

import requests
from typing import Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Представление чанка из RAGFlow."""
    content: str
    similarity: float
    vector_similarity: float
    term_similarity: float
    document_id: str
    document_name: str
    chunk_id: str
    highlight: Optional[str] = None


class RAGFlowError(Exception):
    """Исключение для ошибок RAGFlow API."""
    pass


class RAGFlowClient:
    """Клиент для работы с RAGFlow HTTP API."""
    
    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        """
        Инициализация клиента RAGFlow.
        
        Args:
            base_url: URL адрес RAGFlow сервера (например, "http://localhost:9380")
            api_key: API ключ для авторизации
            timeout: Таймаут запросов в секундах
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def test_connection(self) -> bool:
        """
        Проверяет подключение к RAGFlow серверу.
        
        Returns:
            True если подключение успешно
        """
        try:
            url = f"{self.base_url}/api/v1/system/health"
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_datasets(self) -> list[dict]:
        """
        Получает список всех датасетов.
        
        Returns:
            Список датасетов с их ID и названиями
        """
        url = f"{self.base_url}/api/v1/datasets"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") != 0:
                raise RAGFlowError(f"API Error: {data.get('message')}")
            
            return data.get("data", [])
        except requests.RequestException as e:
            raise RAGFlowError(f"Connection error: {str(e)}")
    
    def retrieve_chunks(
        self,
        question: str,
        dataset_ids: list[str],
        document_ids: Optional[list[str]] = None,
        similarity_threshold: float = 0.2,
        vector_similarity_weight: float = 0.3,
        top_k: int = 10,
        page: int = 1,
        page_size: int = 30,
        highlight: bool = True,
        keyword: bool = False,
        use_kg: bool = False
    ) -> dict:
        """
        Получение семантически близких чанков по запросу.
        
        Args:
            question: Поисковый запрос
            dataset_ids: Список ID датасетов для поиска
            document_ids: Опционально - список ID документов для поиска
            similarity_threshold: Минимальный порог схожести (0.0 - 1.0)
            vector_similarity_weight: Вес векторного сходства (0.0 - 1.0)
            top_k: Количество чанков для векторного поиска
            page: Номер страницы
            page_size: Размер страницы
            highlight: Подсвечивать ли найденные термины
            keyword: Включить ли поиск по ключевым словам
            use_kg: Использовать ли Knowledge Graph
            
        Returns:
            dict с результатами поиска
        """
        url = f"{self.base_url}/api/v1/retrieval"
        
        payload = {
            "question": question,
            "dataset_ids": dataset_ids,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "top_k": top_k,
            "page": page,
            "page_size": page_size,
            "highlight": highlight,
            "keyword": keyword,
            "use_kg": use_kg
        }
        
        if document_ids:
            payload["document_ids"] = document_ids
        
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RAGFlowError(f"Request failed: {str(e)}")
    
    def extract_chunks(self, retrieval_response: dict) -> list[Chunk]:
        """
        Извлекает список чанков из ответа API.
        
        Args:
            retrieval_response: Ответ от API retrieval
            
        Returns:
            Список объектов Chunk
        """
        if retrieval_response.get("code") != 0:
            raise RAGFlowError(f"API Error: {retrieval_response.get('message')}")
        
        chunks_data = retrieval_response.get("data", {}).get("chunks", [])
        
        return [
            Chunk(
                content=chunk.get("content", ""),
                similarity=chunk.get("similarity", 0.0),
                vector_similarity=chunk.get("vector_similarity", 0.0),
                term_similarity=chunk.get("term_similarity", 0.0),
                document_id=chunk.get("document_id", ""),
                document_name=chunk.get("document_keyword", "Unknown"),
                chunk_id=chunk.get("id", ""),
                highlight=chunk.get("highlight")
            )
            for chunk in chunks_data
        ]
    
    def search(
        self,
        question: str,
        dataset_ids: list[str],
        top_k: int = 5,
        similarity_threshold: float = 0.2,
        **kwargs
    ) -> list[Chunk]:
        """
        Упрощённый метод поиска чанков.
        
        Args:
            question: Поисковый запрос
            dataset_ids: Список ID датасетов
            top_k: Количество результатов
            similarity_threshold: Минимальный порог схожести
            **kwargs: Дополнительные параметры для retrieve_chunks
            
        Returns:
            Список объектов Chunk
        """
        response = self.retrieve_chunks(
            question=question,
            dataset_ids=dataset_ids,
            top_k=top_k,
            page_size=top_k,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        return self.extract_chunks(response)
