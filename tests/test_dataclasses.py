"""Test suite for data classes.

Tests serialization, deserialization, and data integrity of all dataclasses
used throughout the pipeline.
"""
import pytest
from datetime import datetime

from utils.dataclass import (
    DocumentType,
    RetrievedDocument,
    RagRequest,
    RagResponse,
    LiteratureSearchAgentRequest,
    LiteratureSearchAgentResponse,
    ResearchRequest,
    ResearchResponse,
    ReportGenerationRequest,
    ReportGenerationResponse,
    ResearchNode,
    ResearchGraph,
    _normalize_question,
)


class TestDocumentType:
    """Tests for the DocumentType enum."""
    
    def test_document_type_values(self):
        """Test that document types have correct values."""
        assert DocumentType.DOCUMENT_TYPE_WEB_PAGE.value == 1
        assert DocumentType.DOCUMENT_TYPE_DATATALK.value == 2
    
    def test_document_type_from_value(self):
        """Test creating document type from integer value."""
        doc_type = DocumentType(1)
        assert doc_type == DocumentType.DOCUMENT_TYPE_WEB_PAGE


class TestRetrievedDocument:
    """Tests for the RetrievedDocument dataclass."""
    
    def test_basic_initialization(self):
        """Test basic document initialization."""
        doc = RetrievedDocument(
            url="http://example.com",
            excerpts=["Excerpt 1", "Excerpt 2"],
            title="Test Document"
        )
        
        assert doc.url == "http://example.com"
        assert len(doc.excerpts) == 2
        assert doc.title == "Test Document"
    
    def test_initialization_with_defaults(self):
        """Test document initialization with default values."""
        doc = RetrievedDocument(url="http://example.com")
        
        assert doc.url == "http://example.com"
        assert doc.excerpts == []
        assert doc.title is None
        assert doc.timestamp is None
        assert doc.document_type == DocumentType.DOCUMENT_TYPE_WEB_PAGE
    
    def test_to_dict_serialization(self):
        """Test document serialization to dictionary."""
        doc = RetrievedDocument(
            url="http://example.com",
            excerpts=["Content"],
            title="Title",
            document_type=DocumentType.DOCUMENT_TYPE_WEB_PAGE
        )
        
        result = doc.to_dict()
        
        assert isinstance(result, dict)
        assert result["url"] == "http://example.com"
        assert result["excerpts"] == ["Content"]
        assert result["title"] == "Title"
        assert result["document_type"] == 1  # Should be enum value
    
    def test_from_dict_deserialization(self):
        """Test document deserialization from dictionary."""
        data = {
            "url": "http://example.com",
            "excerpts": ["Content"],
            "title": "Title",
            "document_type": 1
        }
        
        doc = RetrievedDocument.from_dict(data)
        
        assert doc.url == "http://example.com"
        assert doc.excerpts == ["Content"]
        assert doc.title == "Title"
        assert doc.document_type == DocumentType.DOCUMENT_TYPE_WEB_PAGE
    
    def test_from_dict_with_string_document_type(self):
        """Test deserialization with string enum name."""
        data = {
            "url": "http://example.com",
            "excerpts": [],
            "document_type": "DOCUMENT_TYPE_DATATALK"
        }
        
        doc = RetrievedDocument.from_dict(data)
        
        assert doc.document_type == DocumentType.DOCUMENT_TYPE_DATATALK
    
    def test_from_dict_with_invalid_document_type(self):
        """Test deserialization with invalid document type defaults to web page."""
        data = {
            "url": "http://example.com",
            "excerpts": [],
            "document_type": 999  # Invalid value
        }
        
        doc = RetrievedDocument.from_dict(data)
        
        assert doc.document_type == DocumentType.DOCUMENT_TYPE_WEB_PAGE
    
    def test_roundtrip_serialization(self):
        """Test that document survives roundtrip serialization."""
        original = RetrievedDocument(
            url="http://example.com",
            excerpts=["Excerpt 1", "Excerpt 2"],
            title="Test",
            reason_for_retrieval="Search query",
            metadata={"key": "value"}
        )
        
        serialized = original.to_dict()
        deserialized = RetrievedDocument.from_dict(serialized)
        
        assert deserialized.url == original.url
        assert deserialized.excerpts == original.excerpts
        assert deserialized.title == original.title
        assert deserialized.reason_for_retrieval == original.reason_for_retrieval
        assert deserialized.metadata == original.metadata


class TestRagRequest:
    """Tests for the RagRequest dataclass."""
    
    def test_basic_initialization(self):
        """Test basic RAG request initialization."""
        request = RagRequest(question="What is AI?")
        
        assert request.question == "What is AI?"
        assert request.question_context is None
        assert request.max_retriever_calls == 3  # Default
        assert request.answer_style is not None
    
    def test_initialization_with_all_parameters(self):
        """Test RAG request with all parameters."""
        request = RagRequest(
            question="What is AI?",
            question_context="Technical explanation",
            max_retriever_calls=5,
            answer_style="Concise and technical"
        )
        
        assert request.question == "What is AI?"
        assert request.question_context == "Technical explanation"
        assert request.max_retriever_calls == 5
        assert request.answer_style == "Concise and technical"
    
    def test_initialization_with_extra_kwargs(self):
        """Test RAG request with extra keyword arguments."""
        request = RagRequest(
            question="Test",
            custom_param="custom_value"
        )
        
        assert "custom_param" in request.extra_kwargs
        assert request.extra_kwargs["custom_param"] == "custom_value"


class TestRagResponse:
    """Tests for the RagResponse dataclass."""
    
    def test_basic_initialization(self):
        """Test basic RAG response initialization."""
        response = RagResponse(
            question="What is AI?",
            answer="AI is artificial intelligence[1]"
        )
        
        assert response.question == "What is AI?"
        assert response.answer == "AI is artificial intelligence[1]"
        assert response.cited_documents == []
        assert response.uncited_documents == []
        assert response.num_retriever_calls == 0
    
    def test_to_dict_serialization(self):
        """Test RAG response serialization."""
        doc = RetrievedDocument(url="http://example.com", excerpts=["Content"])
        response = RagResponse(
            question="Test",
            answer="Answer[1]",
            cited_documents=[doc],
            num_retriever_calls=2
        )
        
        result = response.to_dict()
        
        assert isinstance(result, dict)
        assert result["question"] == "Test"
        assert result["answer"] == "Answer[1]"
        assert len(result["cited_documents"]) == 1
        assert result["num_retriever_calls"] == 2
    
    def test_from_dict_deserialization(self):
        """Test RAG response deserialization."""
        data = {
            "question": "Test",
            "answer": "Answer",
            "question_context": "Context",
            "cited_documents": [{"url": "http://example.com", "excerpts": []}],
            "uncited_documents": [],
            "key_insight": "Insight",
            "num_retriever_calls": 1
        }
        
        response = RagResponse.from_dict(data)
        
        assert response.question == "Test"
        assert response.answer == "Answer"
        assert response.question_context == "Context"
        assert len(response.cited_documents) == 1
        assert response.key_insight == "Insight"
    
    def test_roundtrip_serialization(self):
        """Test RAG response roundtrip serialization."""
        doc1 = RetrievedDocument(url="http://example.com/1", excerpts=["Content 1"])
        doc2 = RetrievedDocument(url="http://example.com/2", excerpts=["Content 2"])
        
        original = RagResponse(
            question="Question",
            answer="Answer[1]",
            question_context="Context",
            cited_documents=[doc1],
            uncited_documents=[doc2],
            key_insight="Insight",
            num_retriever_calls=3
        )
        
        serialized = original.to_dict()
        deserialized = RagResponse.from_dict(serialized)
        
        assert deserialized.question == original.question
        assert deserialized.answer == original.answer
        assert len(deserialized.cited_documents) == len(original.cited_documents)
        assert len(deserialized.uncited_documents) == len(original.uncited_documents)


class TestLiteratureSearchAgentRequest:
    """Tests for the LiteratureSearchAgentRequest dataclass."""
    
    def test_default_initialization(self):
        """Test initialization with defaults."""
        request = LiteratureSearchAgentRequest(topic="AI in healthcare")
        
        assert request.topic == "AI in healthcare"
        assert request.max_retriever_calls == 15  # Default
        assert request.guideline is not None
        assert request.with_synthesis is True
    
    def test_custom_initialization(self):
        """Test initialization with custom values."""
        request = LiteratureSearchAgentRequest(
            topic="Custom topic",
            max_retriever_calls=25,
            guideline="Custom guideline",
            with_synthesis=False
        )
        
        assert request.topic == "Custom topic"
        assert request.max_retriever_calls == 25
        assert request.guideline == "Custom guideline"
        assert request.with_synthesis is False


class TestLiteratureSearchAgentResponse:
    """Tests for the LiteratureSearchAgentResponse dataclass."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        response = LiteratureSearchAgentResponse(
            topic="Test",
            guideline="Guideline",
            writeup="Writeup"
        )
        
        assert response.topic == "Test"
        assert response.guideline == "Guideline"
        assert response.writeup == "Writeup"
        assert response.cited_documents == []
        assert response.rag_responses == []
    
    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        doc = RetrievedDocument(url="http://example.com", excerpts=["Content"])
        rag_response = RagResponse(question="Q", answer="A", cited_documents=[doc])
        
        response = LiteratureSearchAgentResponse(
            topic="Test",
            guideline="Guideline",
            writeup="Writeup",
            cited_documents=[doc],
            rag_responses=[rag_response]
        )
        
        result = response.to_dict()
        
        assert isinstance(result, dict)
        assert result["topic"] == "Test"
        assert len(result["cited_documents"]) == 1
        assert len(result["rag_responses"]) == 1
    
    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            "topic": "Test",
            "guideline": "Guideline",
            "writeup": "Writeup",
            "cited_documents": [],
            "rag_responses": [
                {
                    "question": "Q",
                    "answer": "A",
                    "cited_documents": [],
                    "uncited_documents": []
                }
            ]
        }
        
        response = LiteratureSearchAgentResponse.from_dict(data)
        
        assert response.topic == "Test"
        assert len(response.rag_responses) == 1


class TestDeepCompositionAgentRequest:
    """Tests for the ResearchRequest dataclass."""
    
    def test_minimal_initialization(self):
        """Test initialization with minimal parameters."""
        request = ResearchRequest(topic="Test topic")
        
        assert request.topic == "Test topic"
        assert request.max_retriever_calls == 15
        assert request.guideline is not None
    
    def test_full_initialization(self):
        """Test initialization with all parameters."""
        request = ResearchRequest(
            topic="Test topic",
            max_retriever_calls=20,
            guideline="Custom guideline",
            with_synthesis=False
        )
        
        assert request.topic == "Test topic"
        assert request.max_retriever_calls == 20
        assert request.guideline == "Custom guideline"


class TestDeepCompositionAgentResponse:
    """Tests for the ResearchResponse dataclass."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        response = ResearchResponse(
            topic="Test",
            guideline="Guideline",
            writeup="Writeup"
        )
        
        assert response.topic == "Test"
        assert response.guideline == "Guideline"
        assert response.writeup == "Writeup"
        assert response.cited_documents == []
        assert response.rag_responses == []
    
    def test_to_dict_with_nested_objects(self):
        """Test serialization with nested objects."""
        doc = RetrievedDocument(url="http://example.com", excerpts=["Content"])
        rag_response = RagResponse(question="Q", answer="A", cited_documents=[])
        
        response = ResearchResponse(
            topic="Test",
            guideline="Guideline",
            writeup="Writeup",
            cited_documents=[doc],
            rag_responses=[rag_response],
            misc={"key": "value"}
        )
        
        result = response.to_dict()
        
        assert isinstance(result, dict)
        assert result["topic"] == "Test"
        assert len(result["cited_documents"]) == 1
        assert len(result["rag_responses"]) == 1
        assert result["misc"]["key"] == "value"
    
    def test_serialize_item_recursive(self):
        """Test recursive serialization of nested objects."""
        response = ResearchResponse(
            topic="Test",
            guideline="Guideline",
            writeup="Writeup",
            misc={
                "nested": {
                    "list": [1, 2, 3],
                    "dict": {"a": "b"}
                }
            }
        )
        
        result = response.to_dict()
        
        assert result["misc"]["nested"]["list"] == [1, 2, 3]
        assert result["misc"]["nested"]["dict"]["a"] == "b"


class TestResearchNode:
    """Tests for the ResearchNode dataclass with new DAG-specific fields."""
    
    def test_new_fields_initialization(self):
        """Test that new expected_output_format and composition_instructions fields exist."""
        graph = ResearchGraph()
        node = graph.get_or_create_node("Test question", parent_id=None, depth=0)
        
        # New fields should exist and default to None
        assert hasattr(node, 'expected_output_format')
        assert hasattr(node, 'composition_instructions')
        assert node.expected_output_format is None
        assert node.composition_instructions is None
        
    def test_is_answerable_field_removed(self):
        """Test that is_answerable field has been removed from ResearchNode."""
        graph = ResearchGraph()
        node = graph.get_or_create_node("Test question", parent_id=None, depth=0)
        
        # The is_answerable field should not exist
        assert not hasattr(node, 'is_answerable')
    
    def test_node_with_output_format_serialization(self):
        """Test serialization of node with expected_output_format."""
        graph = ResearchGraph()
        node = graph.get_or_create_node("Test question", parent_id=None, depth=0)
        node.expected_output_format = "boolean"
        node.composition_instructions = "Combine child results by voting"
        
        serialized = node.to_dict()
        
        assert serialized["expected_output_format"] == "boolean"
        assert serialized["composition_instructions"] == "Combine child results by voting"
        # is_answerable should not be in serialized output
        assert "is_answerable" not in serialized
    
    def test_node_roundtrip_with_new_fields(self):
        """Test that nodes with new fields survive roundtrip serialization."""
        from utils.dataclass import ResearchNode
        
        node = ResearchNode(
            id="node_1",
            question="Test question",
            parents=[],
            children=[],
            status="pending",
            depth=0,
            expected_output_format="list",
            composition_instructions="Merge all lists"
        )
        
        serialized = node.to_dict()
        deserialized = ResearchNode.from_dict(serialized)
        
        assert deserialized.expected_output_format == "list"
        assert deserialized.composition_instructions == "Merge all lists"
        assert not hasattr(deserialized, 'is_answerable')


class TestResearchGraph:
    """Tests for the ResearchNode/ResearchGraph DAG structures."""

    def test_get_or_create_node_reuses_by_normalized_question(self):
        """Nodes with the same normalized question map to a single graph node."""
        graph = ResearchGraph()

        root = graph.get_or_create_node("What is AI?", parent_id=None, depth=0)
        same = graph.get_or_create_node("  what   is   AI?  ", parent_id=None, depth=1)

        assert root.id == same.id
        assert len(graph.nodes) == 1

        normalized = _normalize_question("What is AI?")
        assert root.normalized_question == normalized

    def test_graph_to_dict_and_from_dict_roundtrip(self):
        """Graph survives roundtrip serialization."""
        graph = ResearchGraph()
        root = graph.get_or_create_node("Root question", parent_id=None, depth=0)
        child = graph.get_or_create_node("Child question", parent_id=root.id, depth=1)

        assert child.id in graph.nodes
        assert child.id in graph.nodes[root.id].children
        assert root.id in graph.nodes[child.id].parents

        serialized = graph.to_dict()
        restored = ResearchGraph.from_dict(serialized)

        assert restored.root_id == graph.root_id
        assert set(restored.nodes.keys()) == set(graph.nodes.keys())

    def test_graph_with_new_node_fields(self):
        """Test graph serialization with nodes containing new fields."""
        graph = ResearchGraph()
        root = graph.get_or_create_node("Root", parent_id=None, depth=0)
        root.expected_output_format = "report"
        root.composition_instructions = "Synthesize from children"
        
        child = graph.get_or_create_node("Child", parent_id=root.id, depth=1)
        child.expected_output_format = "boolean"
        
        serialized = graph.to_dict()
        restored = ResearchGraph.from_dict(serialized)
        
        assert restored.nodes[root.id].expected_output_format == "report"
        assert restored.nodes[root.id].composition_instructions == "Synthesize from children"
        assert restored.nodes[child.id].expected_output_format == "boolean"



class TestReportGenerationRequest:
    """Tests for the ReportGenerationRequest dataclass."""
    
    def test_initialization(self):
        """Test report generation request initialization."""
        literature_response = LiteratureSearchAgentResponse(
            topic="Test",
            guideline="Guideline",
            writeup="Writeup"
        )
        
        request = ReportGenerationRequest(
            topic="Test topic",
            literature_search=literature_response,
            is_answerable=True
        )
        
        assert request.topic == "Test topic"
        assert request.literature_search is literature_response
        assert request.is_answerable is True


class TestReportGenerationResponse:
    """Tests for the ReportGenerationResponse dataclass."""
    
    def test_initialization(self):
        """Test report generation response initialization."""
        doc = RetrievedDocument(url="http://example.com", excerpts=["Content"])
        
        response = ReportGenerationResponse(
            report="Generated report",
            cited_documents=[doc]
        )
        
        assert response.report == "Generated report"
        assert len(response.cited_documents) == 1
    
    def test_initialization_with_defaults(self):
        """Test initialization with default values."""
        response = ReportGenerationResponse(report="Report")
        
        assert response.report == "Report"
        assert response.cited_documents == []


class TestDataclassEdgeCases:
    """Tests for edge cases in dataclass handling."""
    
    def test_empty_excerpts_list(self):
        """Test document with empty excerpts."""
        doc = RetrievedDocument(url="http://example.com", excerpts=[])
        
        assert doc.excerpts == []
        assert len(doc.excerpts) == 0
    
    def test_none_values_in_serialization(self):
        """Test serialization with None values."""
        doc = RetrievedDocument(
            url="http://example.com",
            title=None,
            timestamp=None
        )
        
        result = doc.to_dict()
        
        assert result["title"] is None
        assert result["timestamp"] is None
    
    def test_large_excerpt_list(self):
        """Test document with many excerpts."""
        excerpts = [f"Excerpt {i}" for i in range(100)]
        doc = RetrievedDocument(url="http://example.com", excerpts=excerpts)
        
        assert len(doc.excerpts) == 100
        
        # Test roundtrip
        serialized = doc.to_dict()
        deserialized = RetrievedDocument.from_dict(serialized)
        assert len(deserialized.excerpts) == 100
    
    def test_unicode_in_fields(self):
        """Test handling of unicode characters."""
        doc = RetrievedDocument(
            url="http://example.com",
            excerpts=["中文", "日本語", "Español"],
            title="多言語テストDocumento"
        )
        
        serialized = doc.to_dict()
        deserialized = RetrievedDocument.from_dict(serialized)
        
        assert deserialized.title == doc.title
        assert deserialized.excerpts == doc.excerpts
    
    def test_special_characters_in_url(self):
        """Test URL with special characters."""
        url = "http://example.com/path?query=value&other=123#anchor"
        doc = RetrievedDocument(url=url)
        
        assert doc.url == url
        
        serialized = doc.to_dict()
        assert serialized["url"] == url

