"""Test suite for RAG Agent.

Tests the Retrieval-Augmented Generation pipeline including query conversion,
document retrieval, and answer generation with citations.
"""
import pytest
from unittest.mock import AsyncMock, Mock, patch
import dspy

from utils.rag import RagAgent, QuestionToQuery, RAGAnswerGeneration
from utils.dataclass import RagRequest, RagResponse, RetrievedDocument
from utils.retriever_agent.retriever import Retriever


class TestQuestionToQuery:
    """Tests for the QuestionToQuery signature."""
    
    @pytest.mark.asyncio
    async def test_question_to_query_signature_fields(self):
        """Test that QuestionToQuery has the correct input/output fields."""
        sig = QuestionToQuery
        
        # Check that annotations exist (DSPy stores fields in __annotations__)
        assert hasattr(sig, '__annotations__')
        annotations = sig.__annotations__
        
        # Check input fields
        assert 'question' in annotations
        assert 'max_num_queries' in annotations
        
        # Check output fields
        assert 'queries' in annotations
    
    @pytest.mark.asyncio
    async def test_question_to_query_generates_multiple_queries(self):
        """Test that question to query can generate multiple search queries."""
        mock_lm = Mock(spec=dspy.LM)
        predictor = dspy.Predict(QuestionToQuery)
        
        mock_response = Mock()
        mock_response.queries = [
            "AI in healthcare 2024",
            "artificial intelligence medical applications",
            "healthcare AI current state"
        ]
        
        with patch.object(predictor, 'aforward', return_value=mock_response):
            result = await predictor.aforward(
                question="What is the current state of AI in healthcare?",
                max_num_queries=3,
                lm=mock_lm
            )
            
            assert isinstance(result.queries, list)
            assert len(result.queries) <= 3
    
    @pytest.mark.asyncio
    async def test_question_to_query_respects_max_limit(self):
        """Test that query generation respects the maximum query limit."""
        mock_lm = Mock(spec=dspy.LM)
        predictor = dspy.Predict(QuestionToQuery)
        
        mock_response = Mock()
        mock_response.queries = ["Query 1", "Query 2"]
        
        with patch.object(predictor, 'aforward', return_value=mock_response):
            result = await predictor.aforward(
                question="Test question",
                max_num_queries=2,
                lm=mock_lm
            )
            
            assert len(result.queries) <= 2


class TestRAGAnswerGeneration:
    """Tests for the RAGAnswerGeneration signature."""
    
    @pytest.mark.asyncio
    async def test_rag_answer_generation_signature_fields(self):
        """Test that RAGAnswerGeneration has the correct input/output fields."""
        sig = RAGAnswerGeneration
        
        # Check that annotations exist (DSPy stores fields in __annotations__)
        assert hasattr(sig, '__annotations__')
        annotations = sig.__annotations__
        
        # Check input fields
        assert 'question' in annotations
        assert 'question_context' in annotations
        assert 'gathered_information' in annotations
        assert 'answer_style' in annotations
        
        # Check output fields
        assert 'answer' in annotations
    
    @pytest.mark.asyncio
    async def test_rag_answer_includes_citations(self):
        """Test that RAG answers include proper citations."""
        mock_lm = Mock(spec=dspy.LM)
        predictor = dspy.Predict(RAGAnswerGeneration)
        
        mock_response = Mock()
        mock_response.answer = "AI is transforming healthcare[1]. Recent studies show significant improvements[2]."
        
        with patch.object(predictor, 'aforward', return_value=mock_response):
            result = await predictor.aforward(
                question="How is AI used in healthcare?",
                question_context="Focus on recent developments",
                gathered_information="Document 1: AI in healthcare...\nDocument 2: Studies on AI...",
                answer_style="Comprehensive and detailed",
                lm=mock_lm
            )
            
            assert "[1]" in result.answer or "[2]" in result.answer
    
    @pytest.mark.asyncio
    async def test_rag_answer_unanswerable_question(self):
        """Test handling of questions that cannot be answered from sources."""
        mock_lm = Mock(spec=dspy.LM)
        predictor = dspy.Predict(RAGAnswerGeneration)
        
        mock_response = Mock()
        mock_response.answer = "The question is not answerable based on the provided source documents"
        
        with patch.object(predictor, 'aforward', return_value=mock_response):
            result = await predictor.aforward(
                question="What will happen in 2050?",
                question_context="Future prediction",
                gathered_information="Document 1: Current state of AI...",
                answer_style="Factual",
                lm=mock_lm
            )
            
            assert "not answerable" in result.answer.lower()


class TestRagAgent:
    """Tests for the complete RAG Agent pipeline."""
    
    @pytest.mark.asyncio
    async def test_rag_agent_initialization(self):
        """Test that RAG agent can be initialized properly."""
        mock_retriever = Mock(spec=Retriever)
        mock_lm = Mock(spec=dspy.LM)
        
        agent = RagAgent(retriever=mock_retriever, rag_lm=mock_lm)
        
        assert agent.retriever is mock_retriever
        assert agent.rag_lm is mock_lm
        assert hasattr(agent, 'convert_question_to_query')
        assert hasattr(agent, 'answer_generation')
    
    @pytest.mark.asyncio
    async def test_rag_agent_accepts_string_input(self):
        """Test that RAG agent can accept a simple string question."""
        mock_retriever = Mock(spec=Retriever)
        mock_retriever.aretrieve = AsyncMock(return_value=[
            RetrievedDocument(url="http://example.com", excerpts=["Test content"])
        ])
        
        mock_lm = Mock(spec=dspy.LM)
        agent = RagAgent(retriever=mock_retriever, rag_lm=mock_lm)
        
        # Mock the answer generation
        mock_answer_response = Mock()
        mock_answer_response.answer = "Test answer[1]"
        
        with patch.object(agent.answer_generation, 'aforward', return_value=mock_answer_response):
            result = await agent.aforward("What is AI?")
            
            assert isinstance(result, RagResponse)
            assert result.question == "What is AI?"
    
    @pytest.mark.asyncio
    async def test_rag_agent_with_rag_request(self):
        """Test that RAG agent can accept a RagRequest object."""
        mock_retriever = Mock(spec=Retriever)
        mock_retriever.aretrieve = AsyncMock(return_value=[
            RetrievedDocument(url="http://example.com", excerpts=["Test content"])
        ])
        
        mock_lm = Mock(spec=dspy.LM)
        agent = RagAgent(retriever=mock_retriever, rag_lm=mock_lm)
        
        mock_answer_response = Mock()
        mock_answer_response.answer = "Test answer[1]"
        
        request = RagRequest(
            question="What is AI?",
            question_context="Technical explanation",
            max_retriever_calls=2
        )
        
        with patch.object(agent.answer_generation, 'aforward', return_value=mock_answer_response):
            result = await agent.aforward(request)
            
            assert isinstance(result, RagResponse)
            assert result.question == request.question
            assert result.question_context == request.question_context
    
    @pytest.mark.asyncio
    async def test_rag_agent_no_documents_retrieved(self):
        """Test RAG agent behavior when no documents are retrieved."""
        mock_retriever = Mock(spec=Retriever)
        mock_retriever.aretrieve = AsyncMock(return_value=[])
        
        mock_lm = Mock(spec=dspy.LM)
        agent = RagAgent(retriever=mock_retriever, rag_lm=mock_lm)
        
        result = await agent.aforward("What is AI?")
        
        assert isinstance(result, RagResponse)
        assert "not answerable" in result.answer.lower()
        assert len(result.cited_documents) == 0
    
    @pytest.mark.asyncio
    async def test_rag_agent_parallel_retrieval(self):
        """Test that RAG agent executes multiple retrievals in parallel."""
        from utils.retriever_agent.internet_retriever import InternetRetriever
        
        mock_retriever = Mock(spec=InternetRetriever)
        mock_retriever.aretrieve = AsyncMock(return_value=[
            RetrievedDocument(url="http://example.com/1", excerpts=["Content 1"]),
            RetrievedDocument(url="http://example.com/2", excerpts=["Content 2"])
        ])
        
        mock_lm = Mock(spec=dspy.LM)
        agent = RagAgent(retriever=mock_retriever, rag_lm=mock_lm)
        
        # Mock query conversion to return multiple queries
        mock_query_response = Mock()
        mock_query_response.queries = ["Query 1", "Query 2", "Query 3"]
        
        mock_answer_response = Mock()
        mock_answer_response.answer = "Answer with citations[1][2]"
        
        request = RagRequest(question="Test", max_retriever_calls=3)
        
        with patch.object(agent.convert_question_to_query, 'aforward', return_value=mock_query_response):
            with patch.object(agent.answer_generation, 'aforward', return_value=mock_answer_response):
                result = await agent.aforward(request)
                
                # Should call retriever once for each query
                assert mock_retriever.aretrieve.call_count == 3
    
    @pytest.mark.asyncio
    async def test_rag_agent_citation_normalization(self):
        """Test that RAG agent normalizes citation indices."""
        mock_retriever = Mock(spec=Retriever)
        mock_retriever.aretrieve = AsyncMock(return_value=[
            RetrievedDocument(url="http://example.com/1", excerpts=["Content 1"]),
            RetrievedDocument(url="http://example.com/2", excerpts=["Content 2"]),
            RetrievedDocument(url="http://example.com/3", excerpts=["Content 3"])
        ])
        
        mock_lm = Mock(spec=dspy.LM)
        agent = RagAgent(retriever=mock_retriever, rag_lm=mock_lm)
        
        # Answer with gaps in citations (only cites doc 1 and 3, not 2)
        mock_answer_response = Mock()
        mock_answer_response.answer = "First fact[1]. Second fact[3]."
        
        with patch.object(agent.answer_generation, 'aforward', return_value=mock_answer_response):
            result = await agent.aforward("Test question")
            
            # Citations should be normalized
            assert isinstance(result.cited_documents, list)
            assert isinstance(result.uncited_documents, list)
            # Total documents should match retrieved
            assert len(result.cited_documents) + len(result.uncited_documents) == 3
    
    @pytest.mark.asyncio
    async def test_rag_agent_separates_cited_and_uncited_docs(self):
        """Test that RAG agent correctly separates cited and uncited documents."""
        doc1 = RetrievedDocument(url="http://example.com/1", excerpts=["Content 1"])
        doc2 = RetrievedDocument(url="http://example.com/2", excerpts=["Content 2"])
        doc3 = RetrievedDocument(url="http://example.com/3", excerpts=["Content 3"])
        
        mock_retriever = Mock(spec=Retriever)
        mock_retriever.aretrieve = AsyncMock(return_value=[doc1, doc2, doc3])
        
        mock_lm = Mock(spec=dspy.LM)
        agent = RagAgent(retriever=mock_retriever, rag_lm=mock_lm)
        
        # Only cite first and third documents
        mock_answer_response = Mock()
        mock_answer_response.answer = "Fact from doc 1[1]. Fact from doc 3[3]."
        
        with patch.object(agent.answer_generation, 'aforward', return_value=mock_answer_response):
            result = await agent.aforward("Test question")
            
            # Should have 2 cited and 1 uncited
            cited_count = len(result.cited_documents)
            uncited_count = len(result.uncited_documents)
            
            assert cited_count + uncited_count == 3
            # Verify no overlap between cited and uncited
            cited_urls = {doc.url for doc in result.cited_documents}
            uncited_urls = {doc.url for doc in result.uncited_documents}
            assert len(cited_urls & uncited_urls) == 0


class TestRagAgentEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_none_request_raises_assertion(self):
        """Test that passing None as request raises an assertion error."""
        mock_retriever = Mock(spec=Retriever)
        mock_lm = Mock(spec=dspy.LM)
        agent = RagAgent(retriever=mock_retriever, rag_lm=mock_lm)
        
        with pytest.raises(AssertionError, match="RAG request cannot be None"):
            await agent.aforward(None)
    
    @pytest.mark.asyncio
    async def test_empty_question(self):
        """Test RAG agent with an empty question string."""
        mock_retriever = Mock(spec=Retriever)
        mock_retriever.aretrieve = AsyncMock(return_value=[])
        
        mock_lm = Mock(spec=dspy.LM)
        agent = RagAgent(retriever=mock_retriever, rag_lm=mock_lm)
        
        result = await agent.aforward("")
        
        assert isinstance(result, RagResponse)
        assert result.question == ""
    
    @pytest.mark.asyncio
    async def test_very_long_question(self):
        """Test RAG agent with a very long question."""
        mock_retriever = Mock(spec=Retriever)
        mock_retriever.aretrieve = AsyncMock(return_value=[
            RetrievedDocument(url="http://example.com", excerpts=["Content"])
        ])
        
        mock_lm = Mock(spec=dspy.LM)
        agent = RagAgent(retriever=mock_retriever, rag_lm=mock_lm)
        
        mock_answer_response = Mock()
        mock_answer_response.answer = "Answer[1]"
        
        long_question = "What " + "is " * 1000 + "AI?"
        
        with patch.object(agent.answer_generation, 'aforward', return_value=mock_answer_response):
            result = await agent.aforward(long_question)
            
            assert result.question == long_question


class TestRagResponse:
    """Tests for the RagResponse dataclass."""
    
    def test_rag_response_to_dict(self):
        """Test RagResponse serialization to dictionary."""
        doc = RetrievedDocument(url="http://example.com", excerpts=["Content"])
        response = RagResponse(
            question="Test question",
            answer="Test answer[1]",
            question_context="Test context",
            cited_documents=[doc],
            uncited_documents=[],
            num_retriever_calls=1
        )
        
        result = response.to_dict()
        
        assert isinstance(result, dict)
        assert result["question"] == "Test question"
        assert result["answer"] == "Test answer[1]"
        assert result["question_context"] == "Test context"
        assert len(result["cited_documents"]) == 1
    
    def test_rag_response_from_dict(self):
        """Test RagResponse deserialization from dictionary."""
        data = {
            "question": "Test question",
            "answer": "Test answer",
            "question_context": "Context",
            "cited_documents": [{"url": "http://example.com", "excerpts": ["Content"]}],
            "uncited_documents": [],
            "num_retriever_calls": 1,
            "key_insight": "Key insight"
        }
        
        response = RagResponse.from_dict(data)
        
        assert response.question == "Test question"
        assert response.answer == "Test answer"
        assert response.question_context == "Context"
        assert len(response.cited_documents) == 1
        assert response.key_insight == "Key insight"

