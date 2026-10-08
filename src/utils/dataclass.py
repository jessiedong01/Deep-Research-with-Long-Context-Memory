"""Data classes and enums for the CS224V homework 1.
"""

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class DocumentType(enum.Enum):
    """Enumeration of supported document types.

    This enum distinguishes between different sources of retrieved documents
    to enable specialized processing and rendering.
    """
    DOCUMENT_TYPE_WEB_PAGE = 1  # Standard web pages
    DOCUMENT_TYPE_DATATALK = 2  # DataTalk/tabular data sources


@dataclass
class RetrievedDocument:
    """Container for a retrieved document with metadata and content excerpts.

    This class represents a document that has been retrieved from some source
    (web page, database, etc.) along with its extracted content and metadata.
    The excerpts field contains the actual text content that will be used
    for generating responses.

    Attributes:
        url: Required document URL or identifier
        excerpts: List of text snippets extracted from the document
        title: Optional document title for display purposes
        timestamp: Optional publication or retrieval datetime
        reason_for_retrieval: Optional query or reason explaining why this was retrieved
        document_type: Type of document (web page, data table, etc.)
        metadata: Optional additional metadata as key-value pairs
    """
    url: str
    excerpts: list[str] = field(default_factory=list)
    title: str | None = None
    timestamp: datetime | None = None
    reason_for_retrieval: str | None = None
    document_type: DocumentType = DocumentType.DOCUMENT_TYPE_WEB_PAGE
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the document to a dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields, using enum values for document_type
        """
        return {
            "url": self.url,
            "excerpts": self.excerpts,
            "title": self.title,
            "timestamp": self.timestamp,
            "reason_for_retrieval": self.reason_for_retrieval,
            "document_type": self.document_type.value,  # Convert enum to int value
            "metadata": self.metadata
        }

    @classmethod
    def _parse_document_type(cls, doc_type_value: Any) -> DocumentType:
        """Parse document type from various formats (int, string name, or enum).
        
        Args:
            doc_type_value: Document type as int value, string name, or enum
            
        Returns:
            DocumentType enum instance
        """
        if isinstance(doc_type_value, DocumentType):
            return doc_type_value
        elif isinstance(doc_type_value, str):
            # Handle string enum names like "DOCUMENT_TYPE_DATATALK"
            try:
                return DocumentType[doc_type_value]
            except KeyError:
                return DocumentType.DOCUMENT_TYPE_WEB_PAGE
        elif isinstance(doc_type_value, int):
            # Handle integer values
            try:
                return DocumentType(doc_type_value)
            except ValueError:
                return DocumentType.DOCUMENT_TYPE_WEB_PAGE
        else:
            return DocumentType.DOCUMENT_TYPE_WEB_PAGE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'RetrievedDocument':
        """Create a RetrievedDocument from a dictionary representation.

        Args:
            data: Dictionary containing document data

        Returns:
            New RetrievedDocument instance
        """
        return cls(
            url=data['url'],
            excerpts=data.get('excerpts', []),
            title=data.get('title'),
            timestamp=data.get('timestamp'),
            reason_for_retrieval=data.get('reason_for_retrieval'),
            # Convert int value or string name back to enum, defaulting to web page type
            document_type=cls._parse_document_type(data.get('document_type', DocumentType.DOCUMENT_TYPE_WEB_PAGE.value)),
            metadata=data.get('metadata')
        )


def _normalize_question(question: str) -> str:
    """Normalize a research question/task string for node identity and reuse.

    The goal is to aggressively collapse superficial differences so that
    semantically identical questions map to the same key while still being
    deterministic and cheap to compute.
    """
    # Lowercase, strip, and collapse all internal whitespace
    return " ".join(question.strip().lower().split())


@dataclass
class ResearchNode:
    """Node in a recursive research DAG.

    Each node represents a single research task (question) along with:
    - Links to parent/child nodes in the DAG
    - Status and depth within the exploration
    - Literature search results and final report (when available)
    - Subtasks and an explanation for how to compose them
    """

    id: str
    question: str
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)

    # Structural / control metadata
    status: str = "pending"  # pending | in_progress | complete | failed
    depth: int = 0
    normalized_question: str | None = None

    # Results attached to this node
    literature_writeup: str | None = None
    report: str | None = None
    cited_documents: list[RetrievedDocument] = field(default_factory=list)

    # Decomposition information
    subtasks: list[str] = field(default_factory=list)
    composition_explanation: str | None = None
    
    # DAG-specific fields for upfront planning
    expected_output_format: str | None = None
    composition_instructions: str | None = None

    # If this node's report or results were reused from another node,
    # this field can point to the canonical node id.
    reused_from_node_id: str | None = None

    # Free-form metadata for debugging, budgeting, etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the node to a dictionary suitable for JSON serialization."""
        return {
            "id": self.id,
            "question": self.question,
            "parents": list(self.parents),
            "children": list(self.children),
            "status": self.status,
            "depth": self.depth,
            "normalized_question": self.normalized_question
            or _normalize_question(self.question),
            "literature_writeup": self.literature_writeup,
            "report": self.report,
            "cited_documents": [doc.to_dict() for doc in self.cited_documents],
            "subtasks": list(self.subtasks),
            "composition_explanation": self.composition_explanation,
            "expected_output_format": self.expected_output_format,
            "composition_instructions": self.composition_instructions,
            "reused_from_node_id": self.reused_from_node_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchNode":
        """Reconstruct a ResearchNode from a dictionary representation."""
        cited_docs_data = data.get("cited_documents", [])
        cited_documents = [
            RetrievedDocument.from_dict(doc) for doc in cited_docs_data
        ]

        return cls(
            id=data["id"],
            question=data["question"],
            parents=list(data.get("parents", [])),
            children=list(data.get("children", [])),
            status=data.get("status", "pending"),
            depth=data.get("depth", 0),
            normalized_question=data.get("normalized_question"),
            literature_writeup=data.get("literature_writeup"),
            report=data.get("report"),
            cited_documents=cited_documents,
            subtasks=list(data.get("subtasks", [])),
            composition_explanation=data.get("composition_explanation"),
            expected_output_format=data.get("expected_output_format"),
            composition_instructions=data.get("composition_instructions"),
            reused_from_node_id=data.get("reused_from_node_id"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ResearchGraph:
    """DAG of research nodes for a single deepcomposition run.

    The graph ensures:
    - At most one node per normalized question (for reuse across branches)
    - Simple parent/child relationships to support visualization
    """

    nodes: dict[str, ResearchNode] = field(default_factory=dict)
    root_id: str | None = None
    # Global citation numbering for the run (utils.citations.CitationIndex), set by the DAG processor.
    citation_index: object | None = field(default=None, repr=False, compare=False)

    # Internal index from normalized question -> node id for fast reuse.
    _question_index: dict[str, str] = field(
        default_factory=dict, repr=False, compare=False
    )

    def _next_node_id(self) -> str:
        """Generate a simple, human-readable node id."""
        return f"node_{len(self.nodes) + 1}"

    def get_or_create_node(
        self,
        question: str,
        parent_id: str | None = None,
        depth: int = 0,
    ) -> ResearchNode:
        """Retrieve an existing node for a question or create a new one.

        If a node already exists for the normalized question, it will be reused
        and, if a parent_id is provided, linked into the DAG.
        """
        normalized = _normalize_question(question)

        existing_id = self._question_index.get(normalized)
        if existing_id is not None:
            node = self.nodes[existing_id]
            # Attach parent/child relationship if needed
            if parent_id is not None and parent_id not in node.parents:
                node.parents.append(parent_id)
                parent = self.nodes.get(parent_id)
                if parent is not None and existing_id not in parent.children:
                    parent.children.append(existing_id)
            return node

        node_id = self._next_node_id()
        node = ResearchNode(
            id=node_id,
            question=question,
            parents=[parent_id] if parent_id is not None else [],
            children=[],
            status="pending",
            depth=depth,
            normalized_question=normalized,
            literature_writeup=None,
            report=None,
            cited_documents=[],
            subtasks=[],
            composition_explanation=None,
            expected_output_format=None,
            composition_instructions=None,
            reused_from_node_id=None,
            metadata={},
        )

        self.nodes[node_id] = node
        self._question_index[normalized] = node_id

        if parent_id is not None:
            parent = self.nodes.get(parent_id)
            if parent is not None and node_id not in parent.children:
                parent.children.append(node_id)

        if self.root_id is None:
            self.root_id = node_id

        return node

    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Add a parent/child relationship between two existing nodes."""
        if parent_id == child_id:
            return

        parent = self.nodes.get(parent_id)
        child = self.nodes.get(child_id)
        if parent is None or child is None:
            return

        if child_id not in parent.children:
            parent.children.append(child_id)
        if parent_id not in child.parents:
            child.parents.append(parent_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire graph to a JSON-serializable dictionary."""
        return {
            "root_id": self.root_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchGraph":
        """Reconstruct a ResearchGraph from a dictionary representation."""
        nodes_data = data.get("nodes", {})
        nodes: dict[str, ResearchNode] = {}
        question_index: dict[str, str] = {}

        for node_id, node_dict in nodes_data.items():
            node = ResearchNode.from_dict(node_dict)
            nodes[node_id] = node
            normalized = node.normalized_question or _normalize_question(node.question)
            question_index[normalized] = node_id

        graph = cls(
            nodes=nodes,
            root_id=data.get("root_id"),
        )
        graph._question_index = question_index
        return graph


@dataclass(init=False)
class RagRequest:
    """Request object for RAG (Retrieval-Augmented Generation) service.

    This class encapsulates all parameters needed to perform a RAG operation,
    including the question to answer, optional context, retrieval limits,
    and styling preferences for the generated response.

    The custom __init__ method allows for flexible parameter passing while
    maintaining type safety and default values.
    """
    question: str
    question_context: str | None = None
    max_retriever_calls: int | None = 3
    answer_style: str = """
    - Write in a clear, professional tone. Structure information logically with smooth transitions between concepts. 
    - Include all relevant details that is directly relevant to the question from sources while maintaining readability
    - Use precise, specific language rather than vague generalizations
    - Balance comprehensiveness with conciseness - avoid unnecessary verbosity
    - Maintain objective, factual presentation without speculation
    """.strip()
    # Container for any additional, optional parameters passed via **kwargs
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        question: str,
        question_context: str | None = None,
        max_retriever_calls: int | None = 3,
        answer_style: str = (
            """
            - Write in a clear, professional tone. Structure information logically with smooth transitions between concepts. 
            - Include all relevant details that is directly relevant to the question from sources while maintaining readability
            - Use precise, specific language rather than vague generalizations
            - Balance comprehensiveness with conciseness - avoid unnecessary verbosity
            - Maintain objective, factual presentation without speculation
            """.strip()
        ),
        **kwargs: Any,
    ) -> None:
        """Initialize a RAG request.

        Args:
            question: The main question to be answered
            question_context: Optional additional context about the question
            max_retriever_calls: Maximum number of retrieval operations to perform
            answer_style: Style guidelines for the generated answer
            **kwargs: Additional parameters stored in extra_kwargs
        """
        self.question = question
        self.question_context = question_context
        self.max_retriever_calls = max_retriever_calls
        self.answer_style = answer_style
        # Store any additional keyword arguments for downstream use
        self.extra_kwargs = dict(kwargs) if kwargs else {}


@dataclass
class RagResponse:
    """Response object from RAG (Retrieval-Augmented Generation) service.

    This class contains the complete results of a RAG operation, including
    the generated answer, source documents, and metadata about the retrieval process.

    Attributes:
        question: The original question that was answered
        answer: The generated answer with inline citations
        question_context: Optional context that was provided with the question
        cited_documents: Documents that were referenced in the answer
        uncited_documents: Documents that were retrieved but not used in the answer
        num_retriever_calls: Number of retrieval operations performed
    """
    question: str
    answer: str
    question_context: str | None = None
    cited_documents: list[RetrievedDocument] = field(default_factory=list)
    uncited_documents: list[RetrievedDocument] = field(default_factory=list)
    key_insight: str = ""
    num_retriever_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the response to a dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields, converting documents to dicts
        """
        return {
            "question": self.question,
            "question_context": self.question_context,
            "answer": self.answer,
            "cited_documents": [doc.to_dict() for doc in self.cited_documents],
            "uncited_documents": [doc.to_dict() for doc in self.uncited_documents],
            "key_insight": self.key_insight,
            "num_retriever_calls": self.num_retriever_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'RagResponse':
        """Create a RagResponse from a dictionary representation.

        Args:
            data: Dictionary containing response data

        Returns:
            New RagResponse instance with documents reconstructed from dicts
        """
        rag_response = cls(
            question=data['question'],
            answer=data['answer'],
            question_context=data.get('question_context'),
            key_insight=data.get('key_insight', ''),
            num_retriever_calls=data.get('num_retriever_calls', 0),
        )
        # Reconstruct document objects from dictionary representations
        rag_response.cited_documents = [RetrievedDocument.from_dict(doc) for doc in data.get('cited_documents', [])]
        rag_response.uncited_documents = [RetrievedDocument.from_dict(
            doc) for doc in data.get('uncited_documents', [])]
        return rag_response


@dataclass
class ResearchRequest:
    """Request object for Literature Search Agent service.

    This class encapsulates parameters for conducting a comprehensive literature search
    on a given topic using multiple RAG operations and synthesis.

    Attributes:
        topic: The main topic to search
        report_style: Style guidelines for the final report
        max_retriever_calls: Maximum number of retrieval operations across all sub-questions
        guideline: Instructions for determining when the literature search is complete
        with_synthesis: Whether to perform final synthesis of all findings
    """
    topic: str
    report_style: str | None = "Comprehensive, highly accurate, and exhaustive; include every relevant detail and ensure no important information is omitted."
    max_retriever_calls: int = 15
    guideline: str = "Conduct a survey. Stop when information gain is low or hit the budget"
    with_synthesis: bool = True

    # Recursive DAG controls
    max_depth: int = 2
    """Maximum recursion depth for the research DAG (root is depth 0)."""

    max_nodes: int = 50
    """Hard limit on the total number of nodes in the ResearchGraph."""

    max_subtasks: int = 10
    """Maximum number of subtasks (child nodes) to generate per parent node."""

    max_refinements: int = 1
    """Maximum backtracking iterations per node for gap-filling (0 = disabled)."""

    dag_gen_retriever_calls: int = 1
    """Number of retriever calls for initial DAG generation literature search."""

    gap_fill_retriever_calls: int = 0
    """Number of retriever calls for gap-filling subtree generation (0 = skip literature search)."""

    reuse_existing_nodes: bool = True
    """Whether to reuse existing nodes for identical normalized questions."""

    collect_graph: bool = True
    """Whether to build and return the full ResearchGraph structure."""

    prebuilt_graph_path: str | None = None
    """Optional path to a pre-generated DAG JSON file (skips Phase 1 if provided)."""

@dataclass
class LiteratureSearchAgentRequest:
    """Request object for Literature Search Agent service.

    This class encapsulates parameters for conducting a comprehensive literature search
    on a given topic using multiple RAG operations and synthesis.

    Attributes:
        topic: The main topic to search
        report_style: Style guidelines for the final report
        max_retriever_calls: Maximum number of retrieval operations across all sub-questions
        guideline: Instructions for determining when the literature search is complete
        with_synthesis: Whether to perform final synthesis of all findings
    """
    topic: str
    report_style: str | None = "Comprehensive, highly accurate, and exhaustive; include every relevant detail and ensure no important information is omitted."
    max_retriever_calls: int = 15
    guideline: str = "Conduct a survey. Stop when information gain is low or hit the budget"
    with_synthesis: bool = True

@dataclass
class ResearchResponse:
    """Response object from Literature Search Agent service.

    This class contains the complete results of a literature search operation, including
    the synthesized writeup, all source documents, and the individual RAG responses
    that contributed to the final literature search.

    Attributes:
        topic: The topic that was searched
        guideline: The guideline that was used to determine completeness
        writeup: The final synthesized literature search writeup
        cited_documents: All documents that were cited in the final writeup
        rag_responses: Individual RAG responses for each sub-question explored
    """
    topic: str
    guideline: str
    writeup: str
    cited_documents: list[RetrievedDocument] = field(default_factory=list)
    rag_responses: list[RagResponse] = field(default_factory=list)
    misc: dict[str, Any] = field(default_factory=dict)

    # Recursive DAG output (optional for backward compatibility)
    root_node_id: str | None = None
    graph: ResearchGraph | None = None

    def _serialize_item(self, item: Any) -> Any:
        """Recursively serialize an item, converting objects with to_dict() methods."""
        if hasattr(item, 'to_dict') and callable(item.to_dict):
            return item.to_dict()
        elif isinstance(item, dict):
            return {k: self._serialize_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._serialize_item(i) for i in item]
        else:
            return item

    def to_dict(self) -> dict[str, Any]:
        """Convert the response to a dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields, converting nested objects to dicts
        """
        # Recursively convert misc field, handling nested objects with to_dict() methods
        serialized_misc = self._serialize_item(self.misc)
        serialized_graph = (
            self._serialize_item(self.graph) if self.graph is not None else None
        )

        return {
            "topic": self.topic,
            "guideline": self.guideline,
            "writeup": self.writeup,
            "cited_documents": [doc.to_dict() for doc in self.cited_documents],
            "rag_responses": [rag_response.to_dict() for rag_response in self.rag_responses],
            "misc": serialized_misc,
            "root_node_id": self.root_node_id,
            "graph": serialized_graph,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ResearchResponse':
        """Reconstruct LiteratureSearchAgentResponse from dictionary format.

        Args:
            data: Dictionary containing literature search response data

        Returns:
            New LiteratureSearchAgentResponse instance with all nested objects reconstructed
        """
        # Reconstruct RAG responses from their dictionary representations
        rag_responses = []
        for rag_data in data.get('rag_responses', []):
            rag_response = RagResponse(
                question=rag_data['question'],
                answer=rag_data['answer'],
                cited_documents=rag_data.get('cited_documents', []),
                uncited_documents=rag_data.get('uncited_documents', []),
            )
            rag_responses.append(rag_response)

        graph_data = data.get("graph")
        graph: ResearchGraph | None = None
        if isinstance(graph_data, dict):
            try:
                graph = ResearchGraph.from_dict(graph_data)
            except Exception:
                graph = None

        return cls(
            topic=data['topic'],
            guideline=data.get('guideline', ''),  # Add missing guideline field
            writeup=data['writeup'],
            cited_documents=data.get('cited_documents', []),
            rag_responses=rag_responses,
            misc=data.get("misc", {}),
            root_node_id=data.get("root_node_id"),
            graph=graph,
        )

@dataclass
class LiteratureSearchAgentResponse:
    """Response object from Literature Search Agent service.

    This class contains the complete results of a literature search operation, including
    the synthesized writeup, all source documents, and the individual RAG responses
    that contributed to the final literature search.

    Attributes:
        topic: The topic that was searched
        guideline: The guideline that was used to determine completeness
        writeup: The final synthesized literature search writeup
        cited_documents: All documents that were cited in the final writeup
        rag_responses: Individual RAG responses for each sub-question explored
    """
    topic: str
    guideline: str
    writeup: str
    cited_documents: list[RetrievedDocument] = field(default_factory=list)
    rag_responses: list[RagResponse] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the response to a dictionary for JSON serialization.

        Returns:
            Dictionary representation with all fields, converting nested objects to dicts
        """
        return {
            "topic": self.topic,
            "guideline": self.guideline,
            "writeup": self.writeup,
            "cited_documents": [doc.to_dict() for doc in self.cited_documents],
            "rag_responses": [rag_response.to_dict() for rag_response in self.rag_responses],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'LiteratureSearchAgentResponse':
        """Reconstruct LiteratureSearchAgentResponse from dictionary format.

        Args:
            data: Dictionary containing literature search response data

        Returns:
            New LiteratureSearchAgentResponse instance with all nested objects reconstructed
        """
        # Reconstruct RAG responses from their dictionary representations
        rag_responses = []
        for rag_data in data.get('rag_responses', []):
            rag_response = RagResponse(
                question=rag_data['question'],
                answer=rag_data['answer'],
                cited_documents=rag_data.get('cited_documents', []),
                uncited_documents=rag_data.get('uncited_documents', []),
            )
            rag_responses.append(rag_response)

        return cls(
            topic=data['topic'],
            guideline=data.get('guideline', ''),  # Add missing guideline field
            writeup=data['writeup'],
            cited_documents=data.get('cited_documents', []),
            rag_responses=rag_responses
        )

@dataclass
class ReportGenerationRequest:
    """Request object for Report Generation Agent service.

    This class encapsulates parameters for generating a report on a given topic using a literature search.
    """
    topic: str
    literature_search: LiteratureSearchAgentResponse
    is_answerable: bool


@dataclass
class ReportGenerationResponse:
    """Response object from Report Generation Agent service.

    This class contains the complete results of a report generation operation, including
    the generated report and the source documents that were used to generate the report.
    """
    report: str
    cited_documents: list[RetrievedDocument] = field(default_factory=list)
