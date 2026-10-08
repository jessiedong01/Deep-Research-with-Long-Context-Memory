import json
from collections.abc import Callable
from pathlib import Path

import dspy

from deepcomposition.dag_generation import DAGGenerationAgent
from deepcomposition.dag_processor import DAGProcessor
from deepcomposition.final_report_generator import FinalReportGenerator
from utils.dataclass import (
    ResearchRequest,
    ResearchResponse,
    ResearchGraph,
)
from utils.literature_search import LiteratureSearchAgent
from utils.logger import get_logger


class DeepCompositionAgent:
    """Three-phase deep research pipeline.

    Phase 1: Generate a complete research DAG upfront
    Phase 2: Process the DAG bottom-up (leaves to root)
    Phase 3: Generate final comprehensive report
    """

    def __init__(
        self,
        literature_search_agent: LiteratureSearchAgent,
        lm: dspy.LM,
    ):
        self.literature_search_agent = literature_search_agent
        self.lm = lm
        self.logger = get_logger()
        
        # Initialize the three-phase components
        self.dag_generation_agent = DAGGenerationAgent(
            literature_search_agent=literature_search_agent,
            lm=lm,
        )
        self.dag_processor = DAGProcessor(
            literature_search_agent=literature_search_agent,
            lm=lm,
            dag_generator=self.dag_generation_agent,  # For subtree generation during refinement
        )
        self.final_report_generator = FinalReportGenerator(lm=lm)

    def _save_graph_snapshot(
        self,
        request: ResearchRequest,
        graph: ResearchGraph,
        current_node_id: str | None = None,
    ) -> None:
        """Save a snapshot of the research graph for real-time visualization.
        
        This is called after every significant graph state change to enable
        real-time monitoring in the dashboard.
        """
        if not request.collect_graph:
            return
            
        self.logger.save_intermediate_result(
            "recursive_graph",
            graph.to_dict(),
            {
                "root_node_id": graph.root_id or current_node_id,
                "total_nodes": len(graph.nodes),
                "max_depth": request.max_depth,
                "max_nodes": request.max_nodes,
            },
        )

    async def aforward(
        self,
        request: ResearchRequest,
        on_graph_update: Callable[[dict, dict], None] | None = None,
    ) -> ResearchResponse:
        """Run the three-phase research pipeline and return the final report and DAG.
        
        Phase 1: Generate complete DAG upfront
        Phase 2: Process DAG bottom-up  
        Phase 3: Generate final comprehensive report
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting Three-Phase Research Pipeline for: {request.topic}")
        self.logger.info("=" * 80)

        # Log run configuration
        self.logger.save_intermediate_result(
            "00_run_config",
            {
                "topic": request.topic,
                "guideline": request.guideline,
                "max_retriever_calls": request.max_retriever_calls,
                "max_depth": request.max_depth,
                "max_nodes": request.max_nodes,
                "max_subtasks": request.max_subtasks,
                "max_refinements": request.max_refinements,
                "prebuilt_graph_path": request.prebuilt_graph_path,
            },
        )

        # ========== PHASE 1: Generate DAG ==========
        self.logger.info("\n" + "=" * 80)
        if request.prebuilt_graph_path:
            self.logger.info("PHASE 1: Loading Prebuilt DAG")
            self.logger.info("=" * 80)
            graph = self._load_prebuilt_graph(request.prebuilt_graph_path)
            self.logger.save_intermediate_result(
                "00_dag_generation",
                graph.to_dict(),
                {
                    "total_nodes": len(graph.nodes),
                    "max_depth_reached": max((n.depth for n in graph.nodes.values()), default=0),
                    "leaf_nodes": len([n for n in graph.nodes.values() if not n.children]),
                    "graph_source": "prebuilt",
                    "prebuilt_graph_path": request.prebuilt_graph_path,
                },
            )
        else:
            self.logger.info("PHASE 1: Generating Research DAG")
            self.logger.info("=" * 80)
            graph = await self.dag_generation_agent.generate_dag(request)
        
        # Save DAG snapshot after generation
        self._save_graph_snapshot(request, graph, graph.root_id)
        
        self.logger.info(f"✓ DAG generated with {len(graph.nodes)} nodes")

        # ========== PHASE 2: Process DAG ==========
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 2: Processing DAG (Leaves to Root)")
        self.logger.info("=" * 80)
        
        processed_graph, node_results = await self.dag_processor.process_dag(
            graph=graph,
            max_retriever_calls=request.max_retriever_calls,
            max_refinements=request.max_refinements,
            gap_fill_retriever_calls=request.gap_fill_retriever_calls,
            on_graph_update=on_graph_update,
        )
        
        # Save processed DAG snapshot
        self._save_graph_snapshot(request, processed_graph, processed_graph.root_id)
        
        completed_nodes = len([n for n in processed_graph.nodes.values() if n.status == "complete"])
        self.logger.info(f"✓ DAG processed: {completed_nodes}/{len(processed_graph.nodes)} nodes completed")

        # ========== PHASE 3: Generate Final Report ==========
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 3: Generating Final Report")
        self.logger.info("=" * 80)
        
        final_report, all_citations = await self.final_report_generator.generate_report(
            processed_graph,
            node_results,
        )
        
        self.logger.info(f"✓ Final report generated ({len(final_report)} characters, {len(all_citations)} citations)")

        # Get root node results
        if not processed_graph.root_id:
            raise ValueError("Processed graph must have a valid root node")
        
        root_node = processed_graph.nodes[processed_graph.root_id]

        # Save the complete graph for inspection and visualization
        if request.collect_graph:
            self.logger.save_intermediate_result(
                "recursive_graph",
                processed_graph.to_dict(),
                {
                    "root_node_id": root_node.id,
                    "total_nodes": len(processed_graph.nodes),
                    "max_depth": request.max_depth,
                    "max_nodes": request.max_nodes,
                    "completed_nodes": completed_nodes,
                },
            )

        self.logger.info("\n" + "=" * 80)
        self.logger.info("Pipeline completed successfully!")
        self.logger.info("=" * 80)

        root_node_id = root_node.id if request.collect_graph else None
        graph_output = processed_graph if request.collect_graph else None

        # Return response with final report
        return ResearchResponse(
            request.topic,
            request.guideline,
            final_report,  # Use the comprehensive final report
            all_citations,
            [],  # rag_responses (deprecated in new architecture)
            {
                "max_depth": request.max_depth,
                "max_nodes": request.max_nodes,
                "phases": ["dag_generation", "dag_processing", "report_generation"],
            },
            root_node_id,
            graph_output,
        )

    def _load_prebuilt_graph(self, dag_path: str) -> ResearchGraph:
        """Load a prebuilt DAG from disk."""
        resolved_path = Path(dag_path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = Path.cwd() / resolved_path

        if not resolved_path.exists():
            raise FileNotFoundError(f"Prebuilt DAG file not found: {resolved_path}")

        with open(resolved_path, "r") as f:
            payload = json.load(f)

        graph_payload = payload.get("graph") if isinstance(payload, dict) else None
        if not graph_payload:
            raise ValueError(f"Invalid DAG file format: {resolved_path}")

        graph = ResearchGraph.from_dict(graph_payload)
        self.logger.info(
            f"Loaded prebuilt DAG from {resolved_path} ({len(graph.nodes)} nodes)"
        )
        return graph
