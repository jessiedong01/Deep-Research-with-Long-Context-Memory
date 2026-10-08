"""Tests for DAG Processor Module.

Tests the bottom-up DAG processing including:
- Leaf node literature search and formatting
- Parent node synthesis
- Topological sorting
- Citation preservation
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add src to path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Make pytest optional
try:
    import pytest
except ImportError:
    class DummyMark:
        def asyncio(self, func):
            return func
    class DummyPytest:
        mark = DummyMark()
    pytest = DummyPytest()

from deepcomposition.dag_processor import (
    DAGProcessor,
    LeafNodeResearcher,
    ParentNodeSynthesizer,
)
from utils.dataclass import (
    ResearchGraph,
    ResearchNode,
    LiteratureSearchAgentResponse,
    RetrievedDocument,
)


class TestLeafNodeResearcher:
    """Tests for LeafNodeResearcher signature."""
    
    def test_signature_has_required_fields(self):
        """Test that signature has all required fields."""
        sig = LeafNodeResearcher
        
        # Input fields (dspy exposes via model_fields)
        assert 'research_task' in sig.model_fields
        assert 'literature_search_results' in sig.model_fields
        assert 'expected_format' in sig.model_fields
        assert 'format_details' in sig.model_fields
        
        # Output fields
        assert 'formatted_answer' in sig.model_fields


class TestParentNodeSynthesizer:
    """Tests for ParentNodeSynthesizer signature."""
    
    def test_signature_has_required_fields(self):
        """Test that signature has all required fields."""
        sig = ParentNodeSynthesizer
        
        # Input fields (dspy exposes via model_fields)
        assert 'research_task' in sig.model_fields
        assert 'child_results' in sig.model_fields
        assert 'composition_instructions' in sig.model_fields
        assert 'expected_format' in sig.model_fields
        assert 'format_details' in sig.model_fields
        
        # Output fields
        assert 'synthesized_answer' in sig.model_fields


class TestDAGProcessor:
    """Tests for DAGProcessor."""
    
    def _create_mock_literature_search_agent(self):
        """Create mock literature search agent."""
        mock_agent = Mock()
        
        async def mock_aforward(request):
            doc = RetrievedDocument(
                url="http://example.com",
                excerpts=["Test excerpt"],
            )
            return LiteratureSearchAgentResponse(
                topic=request.topic,
                guideline=request.guideline,
                writeup=f"Literature search results for: {request.topic} [1]",
                cited_documents=[doc],
                rag_responses=[],
            )
        
        mock_agent.aforward = AsyncMock(side_effect=mock_aforward)
        return mock_agent
    
    def _create_mock_lm(self):
        """Create mock language model."""
        return Mock()
    
    def _create_simple_graph(self) -> ResearchGraph:
        """Create a simple graph with one root and two leaf children."""
        graph = ResearchGraph()
        
        # Root node
        root = graph.get_or_create_node("Root question", parent_id=None, depth=0)
        root.expected_output_format = "report"
        root.composition_instructions = "Combine child results"
        root.metadata["format_details"] = "A comprehensive report"
        
        # Child 1 (leaf)
        child1 = graph.get_or_create_node("Child question 1", parent_id=root.id, depth=1)
        child1.expected_output_format = "boolean"
        child1.metadata["format_details"] = "Yes or No with justification"
        
        # Child 2 (leaf)
        child2 = graph.get_or_create_node("Child question 2", parent_id=root.id, depth=1)
        child2.expected_output_format = "list"
        child2.metadata["format_details"] = "Bullet list of items"
        
        return graph
    
    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        lit_agent = self._create_mock_literature_search_agent()
        lm = self._create_mock_lm()
        
        processor = DAGProcessor(
            literature_search_agent=lit_agent,
            lm=lm,
        )
        
        assert processor.literature_search_agent is lit_agent
        assert processor.lm is lm
        assert hasattr(processor, 'leaf_researcher')
        assert hasattr(processor, 'parent_synthesizer')
    
    def test_topological_sort_simple_graph(self):
        """Test topological sort on a simple graph."""
        lit_agent = self._create_mock_literature_search_agent()
        lm = self._create_mock_lm()
        processor = DAGProcessor(literature_search_agent=lit_agent, lm=lm)
        
        graph = self._create_simple_graph()
        layers = processor._topological_sort_by_layers(graph)
        
        # Should have 2 layers: leaves first, then root
        assert len(layers) == 2
        
        # First layer should be the leaf nodes (children)
        assert len(layers[0]) == 2
        root = graph.nodes[graph.root_id]
        assert set(layers[0]) == set(root.children)
        
        # Second layer should be the root
        assert len(layers[1]) == 1
        assert layers[1][0] == graph.root_id
    
    def test_topological_sort_deeper_graph(self):
        """Test topological sort on a deeper graph."""
        lit_agent = self._create_mock_literature_search_agent()
        lm = self._create_mock_lm()
        processor = DAGProcessor(literature_search_agent=lit_agent, lm=lm)
        
        graph = ResearchGraph()
        
        # Create a 3-level graph
        root = graph.get_or_create_node("Root", parent_id=None, depth=0)
        root.expected_output_format = "report"
        
        child1 = graph.get_or_create_node("Child 1", parent_id=root.id, depth=1)
        child1.expected_output_format = "report"
        
        child2 = graph.get_or_create_node("Child 2", parent_id=root.id, depth=1)
        child2.expected_output_format = "report"
        
        grandchild1 = graph.get_or_create_node("Grandchild 1", parent_id=child1.id, depth=2)
        grandchild1.expected_output_format = "boolean"
        
        grandchild2 = graph.get_or_create_node("Grandchild 2", parent_id=child1.id, depth=2)
        grandchild2.expected_output_format = "list"
        
        layers = processor._topological_sort_by_layers(graph)
        
        # Should have 3 layers
        assert len(layers) == 3
        
        # Layer 0: leaves (grandchildren and child2)
        assert len(layers[0]) == 3
        assert grandchild1.id in layers[0]
        assert grandchild2.id in layers[0]
        assert child2.id in layers[0]
        
        # Layer 1: child1 (has processed children)
        assert len(layers[1]) == 1
        assert layers[1][0] == child1.id
        
        # Layer 2: root
        assert len(layers[2]) == 1
        assert layers[2][0] == root.id
    
    @pytest.mark.asyncio
    async def test_process_leaf_node(self):
        """Test processing a single leaf node."""
        lit_agent = self._create_mock_literature_search_agent()
        lm = self._create_mock_lm()
        processor = DAGProcessor(literature_search_agent=lit_agent, lm=lm)
        
        # Mock the leaf researcher
        async def mock_research(*args, **kwargs):
            result = Mock()
            result.formatted_answer = "Test formatted answer"
            return result
        
        processor.leaf_researcher.aforward = AsyncMock(side_effect=mock_research)
        
        # Create a leaf node
        graph = ResearchGraph()
        node = graph.get_or_create_node("Test question", parent_id=None, depth=0)
        node.expected_output_format = "boolean"
        node.metadata["format_details"] = "Yes/No with reason"
        
        # Process the node
        result = await processor._process_leaf_node(node, max_retriever_calls=1)
        
        # Verify - literature_writeup is now cleared; answer stored in return value
        assert result == "Test formatted answer"
        assert len(node.cited_documents) > 0
        assert lit_agent.aforward.called
    
    @pytest.mark.asyncio
    async def test_process_parent_node(self):
        """Test processing a parent node that synthesizes children."""
        lit_agent = self._create_mock_literature_search_agent()
        lm = self._create_mock_lm()
        processor = DAGProcessor(literature_search_agent=lit_agent, lm=lm)
        
        # Mock the synthesizer
        async def mock_synthesize(*args, **kwargs):
            result = Mock()
            result.synthesized_answer = "Synthesized answer from children"
            return result
        
        processor.parent_synthesizer.aforward = AsyncMock(side_effect=mock_synthesize)
        
        # Create graph with parent and children
        graph = self._create_simple_graph()
        root = graph.nodes[graph.root_id]
        
        # Populate child reports
        for child_id in root.children:
            child = graph.nodes[child_id]
            child.status = "complete"
            processor._node_results[child_id] = f"Answer for {child.question}"
        
        # Process parent
        result_text = await processor._process_parent_node(graph, root)
        
        # Verify
        assert result_text == "Synthesized answer from children"
        assert processor.parent_synthesizer.aforward.called
    
    @pytest.mark.asyncio
    async def test_process_dag_end_to_end(self):
        """Test processing entire DAG end-to-end."""
        lit_agent = self._create_mock_literature_search_agent()
        lm = self._create_mock_lm()
        processor = DAGProcessor(literature_search_agent=lit_agent, lm=lm)
        
        # Mock predictors
        async def mock_research(*args, **kwargs):
            result = Mock()
            task = kwargs.get('research_task', '')
            result.formatted_answer = f"Formatted answer for: {task}"
            return result
        
        async def mock_synthesize(*args, **kwargs):
            result = Mock()
            result.synthesized_answer = "Synthesized parent answer"
            return result
        
        processor.leaf_researcher.aforward = AsyncMock(side_effect=mock_research)
        processor.parent_synthesizer.aforward = AsyncMock(side_effect=mock_synthesize)
        
        # Create graph
        graph = self._create_simple_graph()
        
        # Process
        result_graph, node_results = await processor.process_dag(graph, max_retriever_calls=1)
        
        # Verify all nodes processed
        for node_id, node in result_graph.nodes.items():
            assert node.status == "complete"
            if node_id == result_graph.root_id:
                assert node.report is not None
                assert node_results.get(node_id) == node.report
            else:
                assert node.report is None
                assert node_id in node_results
    
    @pytest.mark.asyncio
    async def test_citations_preserved_from_children(self):
        """Test that citations from children are preserved in parent."""
        lit_agent = self._create_mock_literature_search_agent()
        lm = self._create_mock_lm()
        processor = DAGProcessor(literature_search_agent=lit_agent, lm=lm)
        
        # Mock synthesizer
        async def mock_synthesize(*args, **kwargs):
            result = Mock()
            result.synthesized_answer = "Synthesized answer"
            return result
        
        processor.parent_synthesizer.aforward = AsyncMock(side_effect=mock_synthesize)
        
        # Create graph
        graph = self._create_simple_graph()
        root = graph.nodes[graph.root_id]
        
        # Add citations to children
        doc1 = RetrievedDocument(url="http://example1.com", excerpts=["Test 1"])
        doc2 = RetrievedDocument(url="http://example2.com", excerpts=["Test 2"])
        
        children = [graph.nodes[cid] for cid in root.children]
        children[0].cited_documents = [doc1]
        children[0].status = "complete"
        processor._node_results[children[0].id] = "Answer 1"
        
        children[1].cited_documents = [doc2]
        children[1].status = "complete"
        processor._node_results[children[1].id] = "Answer 2"
        
        # Process parent
        await processor._process_parent_node(graph, root)
        
        # Verify citations inherited
        assert len(root.cited_documents) == 2
        assert doc1 in root.cited_documents
        assert doc2 in root.cited_documents


class TestMergeSubtree:
    """Tests for _merge_subtree method."""
    
    def _create_mock_literature_search_agent(self):
        """Create mock literature search agent."""
        mock_agent = Mock()
        mock_agent.aforward = AsyncMock()
        return mock_agent
    
    def _create_mock_lm(self):
        """Create mock language model."""
        return Mock()
    
    def test_merge_subtree_creates_new_nodes(self):
        """Test that _merge_subtree creates fresh nodes instead of reusing existing ones."""
        lit_agent = self._create_mock_literature_search_agent()
        lm = self._create_mock_lm()
        processor = DAGProcessor(literature_search_agent=lit_agent, lm=lm)
        
        # Create main graph with a parent node
        main_graph = ResearchGraph()
        parent = main_graph.get_or_create_node("Parent question", parent_id=None, depth=0)
        parent.expected_output_format = "list"
        
        # Create subtree graph
        subtree = ResearchGraph()
        subtree_root = subtree.get_or_create_node("Subtree root", parent_id=None, depth=0)
        subtree_root.expected_output_format = "list"
        subtree_child = subtree.get_or_create_node("Subtree child", parent_id=subtree_root.id, depth=1)
        subtree_child.expected_output_format = "list"
        
        # Merge subtree
        new_ids = processor._merge_subtree(main_graph, subtree, parent.id, iteration=0)
        
        # Verify new nodes were created with prefixed IDs
        assert len(new_ids) == 2
        for new_id in new_ids:
            assert new_id.startswith(f"{parent.id}_gap0_")
            assert new_id in main_graph.nodes
        
        # Verify parent-child relationships are correct
        assert len(parent.children) >= 1
        
    def test_merge_subtree_no_self_referential_children(self):
        """Test that _merge_subtree excludes self-referential children."""
        lit_agent = self._create_mock_literature_search_agent()
        lm = self._create_mock_lm()
        processor = DAGProcessor(literature_search_agent=lit_agent, lm=lm)
        
        # Create main graph
        main_graph = ResearchGraph()
        parent = main_graph.get_or_create_node("Parent", parent_id=None, depth=0)
        
        # Create subtree with a self-referential node (simulating bad LLM output)
        subtree = ResearchGraph()
        bad_node = ResearchNode(
            id="node_0",
            question="Bad node",
            parents=[],
            children=["node_0"],  # Self-reference!
            status="pending",
            depth=0,
        )
        subtree.nodes["node_0"] = bad_node
        subtree.root_id = "node_0"
        
        # Merge - should not crash and should exclude self-reference
        new_ids = processor._merge_subtree(main_graph, subtree, parent.id, iteration=0)
        
        # Verify the new node doesn't have itself as a child
        assert len(new_ids) == 1
        new_node = main_graph.nodes[new_ids[0]]
        assert new_ids[0] not in new_node.children


def run_tests_manually():
    """Run async tests manually without pytest."""
    import traceback
    
    test_class = TestDAGProcessor()
    merge_test_class = TestMergeSubtree()
    
    # Sync tests
    sync_tests = [
        ("test_processor_initialization", test_class.test_processor_initialization),
        ("test_topological_sort_simple_graph", test_class.test_topological_sort_simple_graph),
        ("test_topological_sort_deeper_graph", test_class.test_topological_sort_deeper_graph),
        ("test_merge_subtree_creates_new_nodes", merge_test_class.test_merge_subtree_creates_new_nodes),
        ("test_merge_subtree_no_self_referential_children", merge_test_class.test_merge_subtree_no_self_referential_children),
    ]
    
    # Async tests
    async_tests = [
        ("test_process_leaf_node", test_class.test_process_leaf_node),
        ("test_process_parent_node", test_class.test_process_parent_node),
        ("test_process_dag_end_to_end", test_class.test_process_dag_end_to_end),
        ("test_citations_preserved_from_children", test_class.test_citations_preserved_from_children),
    ]
    
    passed = 0
    failed = 0
    
    # Run sync tests
    for test_name, test_func in sync_tests:
        try:
            print(f"Running {test_name}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED")
            print(f"  Error: {e}")
            traceback.print_exc()
            failed += 1
    
    # Run async tests
    for test_name, test_func in async_tests:
        try:
            print(f"Running {test_name}...", end=" ")
            asyncio.run(test_func())
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED")
            print(f"  Error: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests_manually()
    sys.exit(0 if success else 1)

