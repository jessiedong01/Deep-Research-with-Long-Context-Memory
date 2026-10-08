"""Simplified DAG Generation Module for Deep Research Pipeline.

This version generates the entire DAG upfront with a single literature search
and language-model call. The generated DAG can later be refined during DAG
processing, so this phase prioritizes speed and coherent first drafts.
"""

from __future__ import annotations

import json
from typing import Any

import dspy

from utils.dataclass import (
    LiteratureSearchAgentRequest,
    ResearchRequest,
    ResearchGraph,
    ResearchNode,
)
from utils.literature_search import LiteratureSearchAgent
from utils.logger import get_logger


# Simplified to bullet points only for faster processing
_VALID_OUTPUT_FORMATS = {"list"}


class FullDAGGenerationSignature(dspy.Signature):
    """Generate a complete research DAG (Directed Acyclic Graph) in one shot.

    You are a research planning expert. Given a research topic and background
    context from a literature search, decompose the topic into a tree of
    sub-questions that can be answered independently. The DAG must be
    JSON-serializable and respect the provided depth/node constraints.

    Guidelines
    ---------
    1. Root node: Use the main topic (id="node_0", parent_id=null).
    2. Decomposition: Break complex questions into focused, standalone
       sub-questions that collectively answer the parent.
    3. Output formats: Always use "list" format (bullet points) for all nodes.
    4. Composition instructions: Non-leaf nodes must describe how to combine
       the child answers to answer the parent question.
    5. Constraints: Respect max_depth, max_nodes, max_subtasks.

    Example 1 (simple topic)
    ------------------------
    Topic: "What are the health benefits of green tea?"
    max_depth=2, max_nodes=6, max_subtasks=3

    [
      {
        "id": "node_0",
        "question": "What are the health benefits of green tea?",
        "parent_id": null,
        "expected_output_format": "list",
        "composition_instructions": "Synthesize antioxidant properties, disease prevention evidence, and cognitive effects."
      },
      {
        "id": "node_1",
        "question": "What antioxidants are present in green tea and what are their biological effects?",
        "parent_id": "node_0",
        "expected_output_format": "list",
        "composition_instructions": ""
      },
      {
        "id": "node_2",
        "question": "What does clinical research show about green tea consumption and chronic disease prevention?",
        "parent_id": "node_0",
        "expected_output_format": "list",
        "composition_instructions": ""
      },
      {
        "id": "node_3",
        "question": "How does green tea affect cognitive function and mental health?",
        "parent_id": "node_0",
        "expected_output_format": "list",
        "composition_instructions": ""
      }
    ]

    Example 2 (comparative topic)
    -----------------------------
    Topic: "How do renewable energy policies in Germany compare to those in the United States?"
    max_depth=3, max_nodes=10, max_subtasks=3

    [
      {
        "id": "node_0",
        "question": "How do renewable energy policies in Germany compare to those in the United States?",
        "parent_id": null,
        "expected_output_format": "list",
        "composition_instructions": "Compare regulatory frameworks, incentives, and outcomes."
      },
      {
        "id": "node_1",
        "question": "What are the key regulatory frameworks governing renewable energy in Germany and the United States?",
        "parent_id": "node_0",
        "expected_output_format": "list",
        "composition_instructions": "Combine key laws and regulatory bodies from both countries."
      },
      {
        "id": "node_2",
        "question": "What financial incentives exist for renewable energy adoption in Germany versus the United States?",
        "parent_id": "node_0",
        "expected_output_format": "list",
        "composition_instructions": "Summarize incentives from both countries."
      },
      {
        "id": "node_3",
        "question": "What have been the outcomes of renewable energy policies in Germany compared to the United States?",
        "parent_id": "node_0",
        "expected_output_format": "list",
        "composition_instructions": "Synthesize capacity growth, grid integration, and economic impact data."
      },
      {
        "id": "node_4",
        "question": "What are Germany's main renewable energy laws and regulatory bodies?",
        "parent_id": "node_1",
        "expected_output_format": "list",
        "composition_instructions": ""
      },
      {
        "id": "node_5",
        "question": "What are the United States' main renewable energy laws and regulatory bodies at federal and state levels?",
        "parent_id": "node_1",
        "expected_output_format": "list",
        "composition_instructions": ""
      },
      {
        "id": "node_6",
        "question": "What is the current renewable energy capacity and growth rate in Germany?",
        "parent_id": "node_3",
        "expected_output_format": "list",
        "composition_instructions": ""
      },
      {
        "id": "node_7",
        "question": "What is the current renewable energy capacity and growth rate in the United States?",
        "parent_id": "node_3",
        "expected_output_format": "list",
        "composition_instructions": ""
      }
    ]

    Example 3 (technical deep-dive)
    -------------------------------
    Topic: "Is transformer architecture more efficient than RNN for machine translation?"
    max_depth=2, max_nodes=5, max_subtasks=4

    [
      {
        "id": "node_0",
        "question": "Is transformer architecture more efficient than RNN for machine translation?",
        "parent_id": null,
        "expected_output_format": "list",
        "composition_instructions": "Weigh computational efficiency, translation quality, and training requirements."
      },
      {
        "id": "node_1",
        "question": "What is the computational complexity of transformers versus RNNs for sequence-to-sequence tasks?",
        "parent_id": "node_0",
        "expected_output_format": "list",
        "composition_instructions": ""
      },
      {
        "id": "node_2",
        "question": "How do BLEU scores compare between transformer and RNN-based machine translation models?",
        "parent_id": "node_0",
        "expected_output_format": "list",
        "composition_instructions": ""
      },
      {
        "id": "node_3",
        "question": "What are the training time and resource requirements for transformers compared to RNNs?",
        "parent_id": "node_0",
        "expected_output_format": "list",
        "composition_instructions": ""
      }
    ]
    """

    research_topic: str = dspy.InputField(
        description="The main research question to decompose into a DAG"
    )

    literature_summary: str = dspy.InputField(
        description="Background context from an initial literature search"
    )

    max_depth: int = dspy.InputField(description="Maximum depth (root depth = 0)")
    max_nodes: int = dspy.InputField(description="Maximum total number of nodes")
    max_subtasks: int = dspy.InputField(description="Maximum number of children per node")

    dag_json: str = dspy.OutputField(
        description=(
            "JSON array of node objects. Each node requires: id (string), question (string), "
            "parent_id (string or null), expected_output_format (always 'list' for bullet points), "
            "composition_instructions (string, empty for leaves)."
        )
    )


class CompositionInstructionSignature(dspy.Signature):
    """Write explicit parent composition instructions referencing each child node.

    Given a parent node and its children, explain exactly how the child answers will be
    combined to produce the parent's output. Refer to children by their IDs in the instructions,
    but instruct the synthesizer to cite evidence only with the bracketed source numbers, never with node IDs.
    """

    parent_id: str = dspy.InputField(description="Node id of the parent")
    parent_question: str = dspy.InputField(description="Parent research question")
    parent_format: str = dspy.InputField(description="Parent expected output format")
    child_summaries: str = dspy.InputField(
        description="Bullet list describing each child in the format `id: question (format)`"
    )

    composition_instructions: str = dspy.OutputField(
        description=(
            "Detailed instructions that mention individual child ids (e.g., node_5) and specify how to use "
            "each child's output to build the parent's answer."
        )
    )


class DAGGenerationAgent:
    """Generates a complete research DAG upfront using a single LM call."""

    def __init__(
        self,
        literature_search_agent: LiteratureSearchAgent,
        lm: dspy.LM,
    ):
        self.literature_search_agent = literature_search_agent
        self.lm = lm
        self.logger = get_logger()
        self.dag_generator = dspy.Predict(FullDAGGenerationSignature)
        self.composition_refiner = dspy.Predict(CompositionInstructionSignature)

    async def generate_dag(
        self,
        request: ResearchRequest,
        max_retriever_calls: int | None = None,
    ) -> ResearchGraph:
        """Generate a research DAG by combining one literature search with one LM call.
        
        Args:
            request: The deepcomposition agent request
            max_retriever_calls: Override for retriever calls (defaults to request.dag_gen_retriever_calls)
        """
        retriever_calls = max_retriever_calls if max_retriever_calls is not None else request.dag_gen_retriever_calls
        
        self.logger.info(
            f"Starting simplified DAG generation for topic={request.topic} "
            f"(max_depth={request.max_depth}, max_nodes={request.max_nodes}, max_subtasks={request.max_subtasks}, retriever_calls={retriever_calls})"
        )

        literature_summary = "No literature summary available."
        if retriever_calls > 0:
            try:
                lit_request = LiteratureSearchAgentRequest(
                    topic=request.topic,
                    max_retriever_calls=retriever_calls,
                    guideline="Broad overview to inform DAG planning",
                    with_synthesis=True,
                )
                lit_response = await self.literature_search_agent.aforward(lit_request)
                if getattr(lit_response, "writeup", None):
                    literature_summary = lit_response.writeup[:2000]
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning(f"Literature search failed for topic '{request.topic}': {exc}")

        try:
            dag_result = await self.dag_generator.aforward(
                research_topic=request.topic,
                literature_summary=literature_summary,
                max_depth=request.max_depth,
                max_nodes=request.max_nodes,
                max_subtasks=request.max_subtasks,
                lm=self.lm,
            )
            graph = self._parse_dag_json(dag_result.dag_json, request)
            await self._refine_composition_instructions(graph)
        except Exception as exc:
            self.logger.error(f"Full DAG generation failed; falling back to root-only DAG: {exc}")
            graph = ResearchGraph()
            root = graph.get_or_create_node(question=request.topic, parent_id=None, depth=0)
            root.expected_output_format = "list"

        stats = {
            "total_nodes": len(graph.nodes),
            "max_depth_reached": max((node.depth for node in graph.nodes.values()), default=0),
            "leaf_nodes": len([node for node in graph.nodes.values() if not node.children]),
        }

        self.logger.save_intermediate_result("00_dag_generation", graph.to_dict(), stats)
        self.logger.info(
            f"DAG generation complete with {stats['total_nodes']} nodes (max depth {stats['max_depth_reached']})"
        )
        return graph

    async def _refine_composition_instructions(self, graph: ResearchGraph) -> None:
        """Ensure non-leaf nodes have child-aware composition instructions."""
        for node in graph.nodes.values():
            if not node.children:
                continue

            child_summaries = []
            for child_id in node.children:
                child_node = graph.nodes.get(child_id)
                if not child_node:
                    continue
                summary = (
                    f"{child_id}: {child_node.question} "
                    f"(format={child_node.expected_output_format or 'unknown'})"
                )
                child_summaries.append(summary)

            if not child_summaries:
                continue

            try:
                result = await self.composition_refiner.aforward(
                    parent_id=node.id,
                    parent_question=node.question,
                    parent_format=node.expected_output_format or "list",
                    child_summaries="\n".join(f"- {s}" for s in child_summaries),
                    lm=self.lm,
                )
                node.composition_instructions = result.composition_instructions.strip()
            except Exception as exc:
                self.logger.warning(
                    f"Failed to refine composition instructions for {node.id}: {exc}"
                )
                continue

            self._check_format_alignment(node, graph)

    def _check_format_alignment(self, node: ResearchNode, graph: ResearchGraph) -> None:
        """Log warnings if a parent's expected format is poorly supported by its children."""
        if not node.children or not node.expected_output_format:
            return

        child_formats = [
            (graph.nodes[child_id].expected_output_format or "").lower()
            for child_id in node.children
            if child_id in graph.nodes
        ]
        child_formats = [fmt for fmt in child_formats if fmt]

        if node.expected_output_format == "table_csv":
            structured_children = [
                fmt for fmt in child_formats if fmt in {"table_csv", "list"}
            ]
            if not structured_children:
                self.logger.warning(
                    f"Parent {node.id} expects table_csv but children provide {child_formats}"
                )


    def _parse_dag_json(
        self,
        dag_json: str,
        request: ResearchRequest,
    ) -> ResearchGraph:
        """Parse DAG JSON emitted by the LM into a ResearchGraph respecting limits."""
        try:
            raw_nodes: Any = json.loads(dag_json)
        except json.JSONDecodeError as exc:  # pragma: no cover - LM output issues
            raise ValueError(f"Invalid DAG JSON: {exc}") from exc

        if not isinstance(raw_nodes, list):
            raise ValueError("DAG output must be a JSON array of nodes.")

        graph = ResearchGraph()
        id_map: dict[str, str] = {}
        remaining = list(raw_nodes)

        while remaining and len(graph.nodes) < request.max_nodes:
            progress = False
            for node_spec in list(remaining):
                declared_id = str(node_spec.get("id", "")).strip() or None
                question = (node_spec.get("question") or "").strip()
                parent_key = node_spec.get("parent_id")
                expected_format = (node_spec.get("expected_output_format") or "list").strip().lower()
                composition = (node_spec.get("composition_instructions") or "").strip()

                if not question:
                    remaining.remove(node_spec)
                    continue

                if parent_key not in (None, "") and parent_key not in id_map:
                    continue

                parent_graph_id = id_map.get(parent_key) if parent_key else None
                parent_depth = graph.nodes[parent_graph_id].depth if parent_graph_id else -1
                depth = parent_depth + 1
                if depth > request.max_depth:
                    self.logger.debug(
                        f"Skipping node '{question}' because depth {depth} exceeds limit {request.max_depth}"
                    )
                    remaining.remove(node_spec)
                    continue

                if len(graph.nodes) >= request.max_nodes:
                    self.logger.warning("Node budget exhausted while parsing DAG output.")
                    break

                if expected_format not in _VALID_OUTPUT_FORMATS:
                    expected_format = "list"

                node = graph.get_or_create_node(
                    question=question,
                    parent_id=parent_graph_id,
                    depth=depth,
                )
                node.expected_output_format = expected_format
                node.composition_instructions = composition or None

                if declared_id:
                    id_map[declared_id] = node.id
                remaining.remove(node_spec)
                progress = True

            if not progress:
                self.logger.warning(
                    "Unable to resolve %s DAG nodes due to missing parents or constraints.",
                    len(remaining),
                )
                break

        if not graph.nodes:
            root = graph.get_or_create_node(question=request.topic, parent_id=None, depth=0)
            root.expected_output_format = "list"

        if graph.root_id is None:
            # Ensure the earliest inserted node is root for visualization clarity.
            first_id = next(iter(graph.nodes))
            graph.root_id = first_id

        return graph
