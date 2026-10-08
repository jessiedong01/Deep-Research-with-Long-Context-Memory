"""Final Report Generator Module for Deep Research Pipeline.

This module generates the final comprehensive report from a processed DAG:
1. Creates an outline based on the DAG structure
2. Writes a report that matches the root node's answer format and stance
3. Preserves all citations throughout
"""
import dspy

from utils.dataclass import ResearchGraph, RetrievedDocument
from utils.logger import get_logger


class OutlineFromDAGSignature(dspy.Signature):
    """Generate a report outline based on the DAG structure and results.
    
    The outline should:
    - Reflect the hierarchical structure of the research
    - Include all major findings from the DAG
    - Be organized logically to support the root answer
    - Use proper markdown headings: ## for main sections, ### for subsections
    - Structure sections for prose-based writing (not just bullet lists)
    - Include an Executive Summary, main body sections, and Conclusion
    
    If the root answer has a specific format (e.g., boolean, list), the outline
    should support presenting that answer prominently in the Executive Summary.
    
    CRITICAL: Base the outline ONLY on content that actually exists in the root_answer.
    Every section (including any appendix sections) must correspond to actual findings
    in the root_answer. DO NOT create placeholder sections describing what "should" 
    or "could" be included - only sections for content that is actually present.
    """
    
    root_question: str = dspy.InputField(
        description="The original research question (root node question)"
    )
    
    root_answer: str = dspy.InputField(
        description="The answer to the root question (may be boolean, list, etc.)"
    )
    
    root_format: str = dspy.InputField(
        description="The format of the root answer: boolean, short_answer, list, table_csv, or report"
    )
    
    dag_structure: str = dspy.InputField(
        description="A hierarchical summary of the DAG showing all questions and their relationships. "
                    "Format: 'Root: [question]\n  Child 1: [question]\n  Child 2: [question]...'"
    )
    
    report_outline: str = dspy.OutputField(
        description="A markdown outline for the final report using ## and ### headers. "
                    "The outline should support presenting the root answer clearly and be organized "
                    "to flow logically through all findings from the DAG."
    )


class ReportFromDAGSignature(dspy.Signature):
    """Generate a comprehensive final report from the DAG results.
    
    The report must:
    1. Start with the root answer prominently (matching its format)
    2. Follow the provided outline structure
    3. Incorporate findings from all DAG nodes
    4. Take a clear stance that matches the root answer
    5. Preserve all citations [X] from the DAG
    6. Be well-written, coherent, and comprehensive
    
    CRITICAL STANCE ALIGNMENT:
    - If root answer is "Yes" (boolean), the report must argue FOR that position
    - If root answer is "No" (boolean), the report must argue AGAINST
    - If root answer is a list, the report should be organized around those items
    - The report's conclusion must align with the root answer
    
    Do not contradict or hedge against the root answer's stance.
    
    FORMATTING REQUIREMENTS:
    - The report title (# heading) should be the CLAIM, not the question. Convert
      the question into an assertive statement matching the root answer's stance.
      Example: "Should X do Y?" with answer "Yes" → title "X Should Do Y"
    - Use proper markdown: # for title, ## for sections, ### for subsections
    - Write in prose paragraphs with supporting bullet points where appropriate
    - DO NOT write the entire report as bullet points
    - DO NOT include internal node references like (node_5; node_2) in the output
    - Only use citation numbers like [1], [2], [3] to reference sources
    - Create clear visual hierarchy with proper heading levels
    - Each major section should have introductory prose before any lists
    - Every section must contain actual substantive content from the dag_results
    - DO NOT write placeholder content describing what "should" or "could" be 
      included - only write content that actually exists in the provided results
    """
    
    outline: str = dspy.InputField(
        description="The markdown outline to follow"
    )
    
    root_question: str = dspy.InputField(
        description="The original research question"
    )
    
    root_answer: str = dspy.InputField(
        description="The final answer from the root node (with citations)"
    )
    
    root_format: str = dspy.InputField(
        description="The format of the root answer: boolean, short_answer, list, table_csv, or report"
    )
    
    dag_results: str = dspy.InputField(
        description="All node results from the DAG with their questions and answers. "
                    "Format: 'Node X: [question]\nAnswer: [answer with citations]\n\n...'"
    )
    
    final_report: str = dspy.OutputField(
        description="The complete final report in markdown format. Must start with the root answer "
                    "prominently displayed, follow the outline structure, incorporate all relevant DAG "
                    "findings, preserve citations, and maintain a stance that matches the root answer."
    )


class FinalReportGenerator:
    """Generates final comprehensive reports from processed DAGs.
    
    This generator:
    1. Analyzes the DAG structure and results
    2. Creates an appropriate outline
    3. Writes a comprehensive report
    4. Ensures the report aligns with the root node's answer
    5. Preserves and properly formats all citations
    """
    
    def __init__(self, lm: dspy.LM):
        self.lm = lm
        self.logger = get_logger()
        
        self.outline_generator = dspy.Predict(OutlineFromDAGSignature)
        self.report_generator = dspy.Predict(ReportFromDAGSignature)
    
    async def generate_report(
        self,
        graph: ResearchGraph,
        node_results: dict[str, str],
    ) -> tuple[str, list[RetrievedDocument]]:
        """Generate a final report from a processed DAG.
        
        Args:
            graph: A fully processed research graph
            
        Returns:
            Tuple of (final_report_markdown, cited_documents_list)
        """
        self.logger.info("Starting final report generation")
        
        if not graph.root_id or graph.root_id not in graph.nodes:
            raise ValueError("Graph must have a valid root node")
        
        root = graph.nodes[graph.root_id]
        
        if root.status != "complete":
            self.logger.warning(f"Root node status is {root.status}, not complete")
        
        # Step 1: Build DAG structure summary
        dag_structure = self._build_dag_structure_summary(graph)
        self.logger.debug("Built DAG structure summary")
        
        # Step 2: Collect all DAG results
        dag_results = self._collect_dag_results(graph, node_results)
        self.logger.debug(f"Collected results from {len(graph.nodes)} nodes")
        
        # Step 3: Collect all citations
        all_citations = self._collect_all_citations(graph)
        self.logger.info(f"Collected {len(all_citations)} unique citations")
        
        # Step 4: Generate outline
        try:
            outline_result = await self.outline_generator.aforward(
                root_question=root.question,
                root_answer=root.report or "No answer available",
                root_format=root.expected_output_format or "report",
                dag_structure=dag_structure,
                lm=self.lm,
            )
            outline = outline_result.report_outline
            self.logger.debug("Generated report outline")
        except Exception as e:
            self.logger.error(f"Failed to generate outline: {e}")
            # Fallback outline
            outline = f"## Introduction\n\n## Findings\n\n## Conclusion"
        
        # Step 5: Generate final report
        try:
            report_result = await self.report_generator.aforward(
                outline=outline,
                root_question=root.question,
                root_answer=root.report or "No answer available",
                root_format=root.expected_output_format or "report",
                dag_results=dag_results,
                lm=self.lm,
            )
            final_report = report_result.final_report
            self.logger.info("Generated final report")
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            # Fallback report
            final_report = f"# {root.question}\n\n{root.report}\n\n## Research Process\n\n{dag_results}"
        
        # Step 6: Add bibliography
        final_report_with_bib = self._add_bibliography(final_report, all_citations)
        
        # Save the report
        self.logger.save_intermediate_result(
            "02_final_report",
            {
                "root_question": root.question,
                "root_format": root.expected_output_format,
                "outline": outline,
                "report": final_report_with_bib,
                "num_citations": len(all_citations),
            },
            {
                "report_length": len(final_report_with_bib),
                "num_citations": len(all_citations),
            }
        )
        
        return final_report_with_bib, all_citations
    
    def _build_dag_structure_summary(self, graph: ResearchGraph) -> str:
        """Build a hierarchical text summary of the DAG structure.
        
        Args:
            graph: The research graph
            
        Returns:
            A text representation of the DAG hierarchy
        """
        lines = []
        
        def traverse(node_id: str, indent: int = 0):
            node = graph.nodes[node_id]
            prefix = "  " * indent
            lines.append(f"{prefix}- {node.question} (Format: {node.expected_output_format})")
            
            for child_id in node.children:
                traverse(child_id, indent + 1)
        
        if graph.root_id:
            traverse(graph.root_id)
        
        return "\n".join(lines)
    
    def _collect_dag_results(
        self,
        graph: ResearchGraph,
        node_results: dict[str, str],
    ) -> str:
        """Collect all node results into a formatted string.
        
        Args:
            graph: The research graph
            
        Returns:
            Formatted string with all node results
        """
        results = []
        
        # Sort by depth for logical flow
        sorted_nodes = sorted(
            graph.nodes.items(),
            key=lambda x: (x[1].depth, x[0])
        )
        
        for node_id, node in sorted_nodes:
            result = f"Node {node_id} (depth={node.depth}):\n"
            result += f"Question: {node.question}\n"
            result += f"Format: {node.expected_output_format}\n"
            answer_text = node_results.get(node_id, "No answer available")
            result += f"Answer:\n{answer_text}\n"
            results.append(result)
        
        return "\n---\n\n".join(results)
    
    def _collect_all_citations(self, graph: ResearchGraph) -> list[RetrievedDocument]:
        """Collect all unique citations from the DAG.
        
        Args:
            graph: The research graph
            
        Returns:
            List of unique cited documents
        """
        index = getattr(graph, "citation_index", None)
        if index is not None:
            return index.documents   # the numbering the node answers already use
        seen_urls = set()
        unique_citations = []
        
        for node in graph.nodes.values():
            for doc in node.cited_documents:
                if doc.url not in seen_urls:
                    seen_urls.add(doc.url)
                    unique_citations.append(doc)
        
        return unique_citations
    
    def _add_bibliography(self, report: str, citations: list[RetrievedDocument]) -> str:
        """Add a bibliography section to the report.
        
        Args:
            report: The report text
            citations: List of cited documents
            
        Returns:
            Report with bibliography appended
        """
        if not citations:
            return report
        
        bibliography = "\n\n## References\n\n"
        for i, doc in enumerate(citations, 1):
            title = doc.title if doc.title else "Untitled"
            bibliography += f"[{i}] {title} - {doc.url}\n\n"
        
        return report + bibliography

