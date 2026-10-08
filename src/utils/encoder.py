"""Text embedding encoder using Azure OpenAI.

This module provides the Encoder class for converting text into vector embeddings
using Azure OpenAI Foundry. It includes features for text truncation, batching, 
and concurrent processing to optimize embedding generation for large text collections.

The encoder is optimized for OpenAI's text-embedding-3 family (small/large)
with hardcoded best practice constants for batch sizes and token limits.

Credit: This code was adapted from Stanford University'sCS224V Homework 1
"""

import asyncio

import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI


class Encoder:
    def __init__(self, model_name: str, api_key: str, api_base: str | None = None, api_version: str = "2024-02-01", **kwargs):
        """Initialize the text encoder.

        With `api_base` set, the client talks to an Azure OpenAI deployment; without it, to the OpenAI API.

        Args:
            model_name: Embedding model name (or Azure deployment name), e.g. 'text-embedding-3-large'
            api_key: API key
            api_base: Azure OpenAI endpoint; leave None for OpenAI
            api_version: Azure API version
            **kwargs: Additional arguments to pass to the embedding model
        """
        self.model_name = model_name
        self.kwargs = kwargs

        if api_base:
            self.client = AsyncAzureOpenAI(api_key=api_key, azure_endpoint=api_base, api_version=api_version)
        else:
            self.client = AsyncOpenAI(api_key=api_key)

        # Setup tokenizer for text truncation
        # Note: All text-embedding-3 models use the cl100k_base tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Hardcoded best practice constants optimized for text-embedding-3 family
        self.max_batch_size = 128  # Maximum texts per API call
        self.max_tokens_per_input = 8192  # Maximum tokens per individual text (same for both small/large)
        self.max_concurrent = 4  # Maximum concurrent API calls

        # Semaphore to limit concurrent requests and prevent rate limiting
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    def _truncate_text(self, text: str) -> str:
        """Truncate text if it exceeds the model's token limit.

        Args:
            text: Input text to potentially truncate

        Returns:
            Truncated text that fits within the model's token limit
        """
        tokens = self.tokenizer.encode(text)

        # Only truncate if text exceeds the maximum token limit
        if len(tokens) > self.max_tokens_per_input:
            tokens = tokens[:self.max_tokens_per_input]  # Keep first N tokens
            text = self.tokenizer.decode(tokens)  # Convert back to text

        return text

    def _batch_texts(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches that respect the maximum batch size.

        Args:
            texts: List of input texts to batch

        Returns:
            List of text batches, each containing at most max_batch_size texts
        """
        # First truncate all texts to ensure they fit within token limits
        truncated = [self._truncate_text(t) for t in texts]

        # Split into batches of maximum allowed size
        return [
            truncated[i:i + self.max_batch_size]
            for i in range(0, len(truncated), self.max_batch_size)
        ]

    async def aencode(self, input: list[str]) -> list[list[float]]:
        """Asynchronously encode a list of texts into vector embeddings.

        This method efficiently processes large lists of texts by:
        1. Truncating texts that exceed token limits
        2. Batching inputs to respect API limits (<=128 per request)
        3. Running batches concurrently with rate limiting

        Args:
            input: List of text strings to encode

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            Exception: If any batch encoding fails
        """
        # Split texts into appropriately sized batches
        batches = self._batch_texts(input)

        async def encode_batch(batch: list[str]) -> list[list[float]]:
            """Encode a single batch of texts with rate limiting."""
            async with self.semaphore:  # Limit concurrent requests
                try:
                    # Call the Azure OpenAI embedding API with the batch
                    response = await self.client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                        **self.kwargs
                    )
                    # Extract embeddings from the response
                    embeddings = [d.embedding for d in response.data]
                    return embeddings
                except Exception as e:
                    print(f"Error encoding batch: {e}")
                    raise

        # Process all batches concurrently and wait for completion
        results = await asyncio.gather(*(encode_batch(batch) for batch in batches))

        # Flatten the list of batch results into a single list of embeddings
        all_embeddings = [emb for batch_embs in results for emb in batch_embs]
        return all_embeddings
