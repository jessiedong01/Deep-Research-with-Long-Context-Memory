"""Tests for Final Report Generator Module.

Tests the final report generation including:
- Outline generation from DAG
- Report generation
- Stance alignment with root answer
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

from deepcomposition.final_report_generator import (
    FinalReportGenerator,
    OutlineFromDAGSignature,
    ReportFromDAGSignature,
)
from utils.dataclass import ResearchGraph, RetrievedDocument


class TestFinalReportGeneratorSignatures:
    """Tests for the DSPy signatures."""
    
    def test_outline_signature_exists(self):
        """Test OutlineFromDAGSignature is defined."""
        # Just verify the signature class exists and can be instantiated
        assert OutlineFromDAGSignature is not None
        assert callable(OutlineFromDAGSignature)
    
    def test_report_signature_exists(self):
        """Test ReportFromDAGSignature is defined."""
        # Just verify the signature class exists and can be instantiated
        assert ReportFromDAGSignature is not None
        assert callable(ReportFromDAGSignature)


class TestFinalReportGenerator:
    """Tests for FinalReportGenerator."""
    
    def _create_mock_lm(self):
        """Create mock language model."""
        return Mock()
    
    def _create_simple_processed_graph(self) -> ResearchGraph:
        """Create a simple processed graph with results."""
        graph = ResearchGraph()
        
        # Root with report
        root = graph.get_or_create_node("Should we adopt renewable energy?", parent_id=None, depth=0)
        root.expected_output_format = "boolean"
        root.status = "complete"
        root.report = "Yes. Renewable energy reduces emissions and costs are declining [1][2]."
        root.cited_documents = [
            RetrievedDocument(url="http://example.com/1", excerpts=["Climate data"], title="Climate Study"),
            RetrievedDocument(url="http://example.com/2", excerpts=["Cost analysis"], title="Cost Report"),
        ]
        
        # Child 1
        child1 = graph.get_or_create_node("Environmental benefits?", parent_id=root.id, depth=1)
        child1.expected_output_format = "report"
        child1.status = "complete"
        child1.report = "Renewable energy significantly reduces CO2 emissions [1]."
        child1.cited_documents = [root.cited_documents[0]]
        
        # Child 2
        child2 = graph.get_or_create_node("Economic viability?", parent_id=root.id, depth=1)
        child2.expected_output_format = "report"
        child2.status = "complete"
        child2.report = "Costs have decreased 80% since 2010 [2]."
        child2.cited_documents = [root.cited_documents[1]]
        
        return graph
    
    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        lm = self._create_mock_lm()
        
        generator = FinalReportGenerator(lm=lm)
        
        assert generator.lm is lm
        assert hasattr(generator, 'outline_generator')
        assert hasattr(generator, 'report_generator')
    
    def test_build_dag_structure_summary(self):
        """Test building DAG structure summary."""
        lm = self._create_mock_lm()
        generator = FinalReportGenerator(lm=lm)
        
        graph = self._create_simple_processed_graph()
        structure = generator._build_dag_structure_summary(graph)
        
        assert "Should we adopt renewable energy?" in structure
        assert "Environmental benefits?" in structure
        assert "Economic viability?" in structure
        assert "Format:" in structure
    
    def test_collect_dag_results(self):
        """Test collecting all DAG results."""
        lm = self._create_mock_lm()
        generator = FinalReportGenerator(lm=lm)
        
        graph = self._create_simple_processed_graph()
        node_results = {
            node_id: node.report or f"Answer for {node_id}"
            for node_id, node in graph.nodes.items()
        }
        results = generator._collect_dag_results(graph, node_results)
        
        assert "Should we adopt renewable energy?" in results
        assert "Environmental benefits?" in results
        assert "Economic viability?" in results
        assert "Node" in results
    
    def test_collect_all_citations(self):
        """Test collecting unique citations."""
        lm = self._create_mock_lm()
        generator = FinalReportGenerator(lm=lm)
        
        graph = self._create_simple_processed_graph()
        citations = generator._collect_all_citations(graph)
        
        # Should have 2 unique citations
        assert len(citations) == 2
        urls = [doc.url for doc in citations]
        assert "http://example.com/1" in urls
        assert "http://example.com/2" in urls
    
    def test_add_bibliography(self):
        """Test adding bibliography to report."""
        lm = self._create_mock_lm()
        generator = FinalReportGenerator(lm=lm)
        
        report = "# Test Report\n\nSome content [1]."
        citations = [
            RetrievedDocument(url="http://example.com", excerpts=[], title="Test Doc"),
        ]
        
        result = generator._add_bibliography(report, citations)
        
        assert "## References" in result
        assert "http://example.com" in result
        assert "[1]" in result
    
    @pytest.mark.asyncio
    async def test_generate_report_end_to_end(self):
        """Test full report generation."""
        lm = self._create_mock_lm()
        generator = FinalReportGenerator(lm=lm)
        
        # Mock the predictors
        async def mock_outline(*args, **kwargs):
            result = Mock()
            result.report_outline = "## Introduction\n\n## Analysis\n\n## Conclusion"
            return result
        
        async def mock_report(*args, **kwargs):
            result = Mock()
            result.final_report = "# Report\n\n## Answer\n\nYes, we should adopt renewable energy.\n\n## Analysis\n\nDetails here [1][2]."
            return result
        
        generator.outline_generator.aforward = AsyncMock(side_effect=mock_outline)
        generator.report_generator.aforward = AsyncMock(side_effect=mock_report)
        
        graph = self._create_simple_processed_graph()
        
        node_results = {
            node_id: node.report or f"Answer for {node_id}"
            for node_id, node in graph.nodes.items()
        }
        report, citations = await generator.generate_report(graph, node_results)
        
        # Verify report generated
        assert report is not None
        assert len(report) > 0
        assert "## References" in report
        
        # Verify citations returned
        assert len(citations) == 2
    
    @pytest.mark.asyncio
    async def test_report_preserves_root_answer(self):
        """Test that generated report includes root answer."""
        lm = self._create_mock_lm()
        generator = FinalReportGenerator(lm=lm)
        
        async def mock_outline(*args, **kwargs):
            result = Mock()
            result.report_outline = "## Outline"
            return result
        
        async def mock_report(*args, **kwargs):
            # Should include the root answer
            root_ans = kwargs.get('root_answer', '')
            result = Mock()
            result.final_report = f"# Report\n\n{root_ans}\n\nMore content."
            return result
        
        generator.outline_generator.aforward = AsyncMock(side_effect=mock_outline)
        generator.report_generator.aforward = AsyncMock(side_effect=mock_report)
        
        graph = self._create_simple_processed_graph()
        root = graph.nodes[graph.root_id]
        
        node_results = {
            node_id: node.report or f"Answer for {node_id}"
            for node_id, node in graph.nodes.items()
        }
        report, _ = await generator.generate_report(graph, node_results)
        
        # Report should contain key elements from root answer
        assert "Yes" in report or "renewable energy" in report.lower()


def run_tests_manually():
    """Run tests manually without pytest."""
    import traceback
    
    test_sigs = TestFinalReportGeneratorSignatures()
    test_gen = TestFinalReportGenerator()
    
    sync_tests = [
        ("test_outline_signature_exists", test_sigs.test_outline_signature_exists),
        ("test_report_signature_exists", test_sigs.test_report_signature_exists),
        ("test_generator_initialization", test_gen.test_generator_initialization),
        ("test_build_dag_structure_summary", test_gen.test_build_dag_structure_summary),
        ("test_collect_dag_results", test_gen.test_collect_dag_results),
        ("test_collect_all_citations", test_gen.test_collect_all_citations),
        ("test_add_bibliography", test_gen.test_add_bibliography),
    ]
    
    async_tests = [
        ("test_generate_report_end_to_end", test_gen.test_generate_report_end_to_end),
        ("test_report_preserves_root_answer", test_gen.test_report_preserves_root_answer),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in sync_tests:
        try:
            print(f"Running {test_name}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    for test_name, test_func in async_tests:
        try:
            print(f"Running {test_name}...", end=" ")
            asyncio.run(test_func())
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
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

