import re
"""DAG Processor Module for Deep Research Pipeline.

This module processes a complete research DAG bottom-up:
1. Topologically sorts nodes (leaves first)
2. For leaf nodes: conducts literature search and formats answer
3. For parent nodes: combines child results using composition instructions
4. Processes sibling nodes in parallel for efficiency
"""
import asyncio
from collections.abc import Callable
from datetime import datetime
import dspy

from utils.citations import CitationIndex, citations_in, expand_node_refs, to_global
from utils.dataclass import (
    LiteratureSearchAgentRequest,
    ResearchRequest,
    ResearchGraph,
    ResearchNode,
    RetrievedDocument,
)
from utils.literature_search import LiteratureSearchAgent
from utils.logger import get_logger

# Import at runtime to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from deepcomposition.dag_generation import DAGGenerationAgent


class LeafNodeResearcher(dspy.Signature):
    """Answer a leaf node research task using literature search results.
    
    Given literature search results and the expected output format, provide an answer
    that strictly adheres to the format requirements.
    
    Format types and requirements:
    - boolean: Provide "Yes." or "No." followed by a single sentence justification with citations.
    - short_answer/list/report: Provide 3-5 bullet points (`- ` prefix) summarizing key takeaways with citations.
    - table_csv/json: Provide structured data (CSV or JSON) with inline citations in headers or cells.
    
    CRITICAL: Preserve all citations from the literature search in [X] format.
    Do not add citations that aren't in the source material.
    """
    
    research_task: str = dspy.InputField(
        description="The research task or question to answer"
    )
    
    literature_search_results: str = dspy.InputField(
        description="The synthesized writeup from literature search with inline citations [1], [2], etc."
    )
    
    expected_format: str = dspy.InputField(
        description="The required output format: boolean, short_answer, list, table_csv, or report"
    )
    
    format_details: str = dspy.InputField(
        description="Specific formatting requirements for this task"
    )
    
    formatted_answer: str = dspy.OutputField(
        description="The answer in the exact required format, preserving all citations [X] from the literature search"
    )


class ParentNodeSynthesizer(dspy.Signature):
    """Synthesize answers from child nodes to answer a parent node's research task.
    
    Given the answers from all child subtasks and composition instructions, combine them
    to produce the answer for the parent task in the required format.
    
    CRITICAL RULES:
    1. Use ONLY the information from the child results - no external knowledge
    2. Follow the composition instructions exactly
    3. Cite evidence with the bracketed source numbers exactly as they appear in the child
       answers and the AVAILABLE CITATIONS section. The numbers are global: never renumber them,
       and never cite child nodes by name or ID
    4. Output in the specified format
    5. If child results contradict, acknowledge and explain the contradiction
    
    Format types:
    - boolean: Output \"Yes.\" or \"No.\" plus one bullet with justification and citations.
    - short_answer/list/report: Output 3-5 bullets combining child findings, each with citations.
    - table_csv/json: Merge structured data while preserving child citations.
    """
    
    research_task: str = dspy.InputField(
        description="The parent research task to answer"
    )
    
    child_results: str = dspy.InputField(
        description="The formatted answers from all child subtasks, followed by an "
                    "AVAILABLE CITATIONS section listing all sources as [1], [2], etc."
    )
    
    composition_instructions: str = dspy.InputField(
        description="Step-by-step instructions for how to combine the child results"
    )
    
    expected_format: str = dspy.InputField(
        description="The required output format: boolean, short_answer, list, table_csv, or report"
    )
    
    format_details: str = dspy.InputField(
        description="Specific formatting requirements"
    )
    
    synthesized_answer: str = dspy.OutputField(
        description="The synthesized answer combining all child results according to composition "
                    "instructions, in the required format. Use [1], [2], etc. citation numbers."
    )


class CompositionValidator(dspy.Signature):
    """Validate whether a composed answer sufficiently addresses the research question.
    
    IMPORTANT: Your default should be to return EMPTY STRING. Only flag gaps in 
    exceptional cases where the answer fundamentally fails to address the question.
    
    CONTEXT: This system answers questions via internet/literature search only.
    It cannot perform calculations, run models, or generate original analysis.
    
    RETURN EMPTY STRING (no gaps) if:
    - The answer provides a reasonable, defensible response with evidence
    - The answer addresses the main thrust of the question, even if imperfectly
    - Minor details, edge cases, or "nice to have" information is missing
    - The answer is good enough to inform a decision or understanding
    - You're unsure whether something is truly a critical gap
    
    ONLY flag a gap if:
    - The answer completely misses a CORE aspect of the question
    - The answer would be misleading or wrong without this information
    - A reasonable reader would say "this doesn't answer the question at all"
    
    Good enough IS good enough. Default to empty string.
    """
    
    research_question: str = dspy.InputField(
        description="The research question that needs to be answered"
    )
    
    composed_answer: str = dspy.InputField(
        description="The answer composed from child node results"
    )
    
    is_sufficient: bool = dspy.OutputField(
        description="True if the answer is good enough (covers main aspects adequately). "
                    "Default to True unless there's a critical, glaring omission."
    )
    
    missing_topics: str = dspy.OutputField(
        description="ONLY if is_sufficient is False: 1-2 critical missing topics. "
                    "If is_sufficient is True, return empty string."
    )




class DAGProcessor:
    """Processes a research DAG bottom-up from leaves to root.
    
    This processor:
    1. Topologically sorts the DAG to identify processing order
    2. Processes leaf nodes with literature searches
    3. Processes parent nodes by synthesizing child results
    4. Processes sibling nodes in parallel where possible
    5. Updates node status and saves intermediate results
    6. Optionally refines parent nodes by adding new children when gaps are detected
    """
    
    def __init__(
        self,
        literature_search_agent: LiteratureSearchAgent,
        lm: dspy.LM,
        dag_generator: "DAGGenerationAgent | None" = None,
    ):
        self.literature_search_agent = literature_search_agent
        self.lm = lm
        self.dag_generator = dag_generator
        self.logger = get_logger()
        self._citations = CitationIndex()
        self._node_results: dict[str, str] = {}
        
        self.leaf_researcher = dspy.Predict(LeafNodeResearcher)
        self.parent_synthesizer = dspy.Predict(ParentNodeSynthesizer)
        self.composition_validator = dspy.Predict(CompositionValidator)
        self._node_results: dict[str, str] = {}
        self._on_graph_update: Callable[[dict, dict], None] | None = None
        self._max_node_attempts = 2  # default: one retry
    
    async def process_dag(
        self,
        graph: ResearchGraph,
        max_retriever_calls: int = 3,
        max_refinements: int = 1,
        gap_fill_retriever_calls: int = 0,
        on_graph_update: Callable[[dict, dict], None] | None = None,
    ) -> tuple[ResearchGraph, dict[str, str]]:
        """Process the entire DAG bottom-up.
        
        Args:
            graph: The research graph to process (should be generated by DAGGenerationAgent)
            max_retriever_calls: Max retriever calls per literature search
            max_refinements: Max refinement iterations per parent node (0 = disabled)
            gap_fill_retriever_calls: Retriever calls for gap-filling subtree generation (0 = skip literature search)
            
        Returns:
            The same graph with all nodes processed and answers populated
        """
        self.logger.info("Starting DAG processing")
        self._node_results = {}
        self._citations = CitationIndex()
        graph.citation_index = self._citations
        self._max_retriever_calls = max_retriever_calls
        self._max_refinements = max_refinements
        self._gap_fill_retriever_calls = gap_fill_retriever_calls
        self._on_graph_update = on_graph_update
        self.logger.info(f"Total nodes to process: {len(graph.nodes)}")
        self.logger.info(f"Max refinements per node: {max_refinements}")
        
        # Get processing order (leaves to root)
        processing_layers = self._topological_sort_by_layers(graph)
        
        self.logger.info(f"Processing order: {len(processing_layers)} layers")
        for i, layer in enumerate(processing_layers):
            self.logger.info(f"  Layer {i}: {len(layer)} nodes at depth {graph.nodes[layer[0]].depth if layer else 'N/A'}")
        
        # Process each layer
        for layer_idx, node_ids in enumerate(processing_layers):
            self.logger.info(f"Processing layer {layer_idx} with {len(node_ids)} nodes")
            
            # Check if this is a leaf layer (nodes have no children)
            is_leaf_layer = all(not graph.nodes[nid].children for nid in node_ids)
            
            if is_leaf_layer:
                # Leaf nodes: process in parallel (no refinement/subtree generation)
                for node_id in node_ids:
                    graph.nodes[node_id].status = "in_progress"
                self._save_graph_snapshot(graph)
                
                tasks = [self._process_node(graph, node_id, max_retriever_calls) for node_id in node_ids]
                await asyncio.gather(*tasks)
            else:
                # Intermediate/parent nodes: process sequentially for real-time refinement visibility
                for node_id in node_ids:
                    graph.nodes[node_id].status = "in_progress"
                    self._save_graph_snapshot(graph)
                    
                    await self._process_node(graph, node_id, max_retriever_calls)
                    
                    self._save_graph_snapshot(graph)
            
            self.logger.info(f"Layer {layer_idx} complete")
            
            # Save snapshot after each layer
            self._save_graph_snapshot(graph)
        
        self.logger.info("DAG processing complete")
        
        # Save final processed graph
        self.logger.save_intermediate_result(
            "01_dag_processed",
            graph.to_dict(),
            {
                "total_nodes": len(graph.nodes),
                "completed_nodes": len([n for n in graph.nodes.values() if n.status == "complete"]),
            }
        )
        
        return graph, dict(self._node_results)
    
    def _topological_sort_by_layers(self, graph: ResearchGraph) -> list[list[str]]:
        """Sort nodes into layers for bottom-up processing.
        
        Returns a list of layers, where each layer is a list of node IDs.
        Layers are ordered from leaves (layer 0) to root (last layer).
        Nodes in the same layer can be processed in parallel.
        
        Args:
            graph: The research graph
            
        Returns:
            List of layers, each layer is a list of node IDs
        """
        # Calculate in-degree (number of children) for each node
        in_degree = {node_id: len(node.children) for node_id, node in graph.nodes.items()}
        
        # Find all leaf nodes (in-degree 0)
        current_layer = [node_id for node_id, degree in in_degree.items() if degree == 0]
        
        layers = []
        processed = set()
        
        while current_layer:
            layers.append(current_layer)
            processed.update(current_layer)
            
            # Find next layer: nodes whose children are all processed
            next_layer = []
            for node_id in graph.nodes:
                if node_id in processed:
                    continue
                
                node = graph.nodes[node_id]
                # Check if all children are processed
                if all(child_id in processed for child_id in node.children):
                    next_layer.append(node_id)
            
            current_layer = next_layer
        
        return layers
    
    async def _process_node(
        self,
        graph: ResearchGraph,
        node_id: str,
        max_retriever_calls: int,
        gap_depth: int = 0,
    ) -> None:
        """Process a single node with automatic reset and retry semantics."""
        node = graph.nodes[node_id]
        max_attempts = getattr(self, "_max_node_attempts", 2)
        last_error: Exception | None = None
        
        for attempt in range(1, max_attempts + 1):
            self.logger.info(f"Processing node {node_id} attempt {attempt}/{max_attempts}")
            self._reset_node_state(node)
            node.status = "in_progress"
            
            try:
                if not node.children:
                    result_text = await self._process_leaf_node(
                        node,
                        max_retriever_calls,
                    )
                    # leaf answers cite their own sources as [1]..[k]; store them in global numbering
                    result_text = to_global(result_text, list(node.cited_documents), self._citations)
                else:
                    result_text = await self._process_parent_node(
                        graph,
                        node,
                        gap_depth=gap_depth,
                    )
                
                self._record_node_success(node, result_text, graph)
                self.logger.debug(f"Node {node_id} complete on attempt {attempt}")
                return
            except Exception as exc:
                last_error = exc
                self.logger.error(f"Attempt {attempt} failed for node {node_id}: {exc}")
        
        error_detail = str(last_error) if last_error else "Unknown error"
        error_text = f"ERROR: Processing failed after {max_attempts} attempts: {error_detail}"
        self._record_node_failure(node, error_text, graph)
    
    async def _process_leaf_node(
        self,
        node: ResearchNode,
        max_retriever_calls: int,
    ) -> str:
        """Process a leaf node by conducting literature search.
        
        Args:
            node: The leaf node to process
            max_retriever_calls: Max retriever calls
        """
        self.logger.debug(f"Leaf node {node.id}: conducting literature search")
        
        # Conduct literature search
        lit_request = LiteratureSearchAgentRequest(
            topic=node.question,
            max_retriever_calls=max_retriever_calls,
            guideline="Comprehensive search to answer the research question",
            with_synthesis=True,
        )
        
        lit_response = await self.literature_search_agent.aforward(lit_request)
        
        node.cited_documents = list(lit_response.cited_documents)
        
        # Format answer according to expected format
        format_details = node.metadata.get("format_details", "")
        
        # Log estimated token count before LLM call
        writeup_len = len(lit_response.writeup or "")
        est_tokens = writeup_len // 4  # rough estimate: 4 chars per token
        self.logger.info(f"LLM call: leaf_researcher for {node.id} (~{est_tokens} input tokens, writeup={writeup_len} chars)")
        
        try:
            researcher_result = await self.leaf_researcher.aforward(
                research_task=node.question,
                literature_search_results=lit_response.writeup,
                expected_format=node.expected_output_format or "list",
                format_details=format_details,
                lm=self.lm,
            )
            
            formatted_answer = researcher_result.formatted_answer
            self.logger.debug(f"Leaf node {node.id}: formatted answer as {node.expected_output_format}")
        
        except Exception as e:
            self.logger.warning(f"Failed to format answer for {node.id}, using raw literature writeup: {e}")
            formatted_answer = lit_response.writeup or "No answer available"
        
        # Handle empty answers
        if not formatted_answer or not formatted_answer.strip():
            return "No answer available."
        return formatted_answer.strip()
    
    async def _process_parent_node(
        self,
        graph: ResearchGraph,
        node: ResearchNode,
        gap_depth: int = 0,
    ) -> str:
        """Process a parent node by synthesizing child results with optional refinement.
        
        Implements post-composition validation: after synthesizing, check if the answer
        is sufficient. If gaps are found and refinement budget remains, generate a
        subtree to fill gaps using DAGGenerationAgent, process it, and re-compose.
        
        Args:
            graph: The research graph
            node: The parent node to process
            research_task: Standalone version of the parent question
        """
        self.logger.debug(f"Parent node {node.id}: synthesizing {len(node.children)} children")
        
        max_refinements = getattr(self, '_max_refinements', 1)
        max_retriever_calls = getattr(self, '_max_retriever_calls', 3)
        synthesized = ""
        
        for iteration in range(max_refinements + 1):
            # Gather child results
            child_results_text = []
            all_cited_docs = []
            
            for i, child_id in enumerate(node.children, 1):
                child = graph.nodes[child_id]
                child_result = f"Child {i}: {child.question}\n"
                child_answer = self._node_results.get(child_id, "No answer available")
                child_result += f"Answer:\n{child_answer}\n"
                child_results_text.append(child_result)
                
                # Collect cited documents from children
                all_cited_docs.extend(child.cited_documents)
            
            # List the children's sources under their global numbers, the same numbers the child answers use
            citation_text_lines = []
            seen_urls = set()
            deduplicated_docs = []
            self._citations.add(all_cited_docs)
            for doc in all_cited_docs:
                if doc.url not in seen_urls:
                    seen_urls.add(doc.url)
                    deduplicated_docs.append(doc)
                    citation_text_lines.append(f"[{self._citations.number(doc.url)}] {doc.title} - {doc.url}")

            # Replace the node's cited_documents with deduplicated list
            all_cited_docs = deduplicated_docs

            citations_reference = "\n".join(citation_text_lines) if citation_text_lines else "No citations available."
            
            combined_child_results = "\n".join(child_results_text)
            combined_child_results += f"\n\n--- AVAILABLE CITATIONS ---\n{citations_reference}"
            
            # Synthesize using composition instructions
            format_details = node.metadata.get("format_details", "")
            composition_instr = node.composition_instructions or "Combine the child results to answer the parent question."
            
            # Log estimated token count before LLM call
            input_len = len(combined_child_results)
            est_tokens = input_len // 4
            self.logger.info(f"LLM call: parent_synthesizer for {node.id} (~{est_tokens} input tokens, child_results={input_len} chars)")
            
            try:
                synthesis_result = await self.parent_synthesizer.aforward(
                    research_task=node.question,
                    child_results=combined_child_results,
                    composition_instructions=composition_instr,
                    expected_format=node.expected_output_format or "list",
                    format_details=format_details,
                    lm=self.lm,
                )
                
                synthesized = synthesis_result.synthesized_answer
                child_cites = {cid: citations_in(self._node_results.get(cid, "")) for cid in node.children}
                synthesized = expand_node_refs(synthesized, child_cites)
                allowed = {self._citations.number(d.url) for d in all_cited_docs}
                synthesized = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in allowed else "", synthesized)
                node.cited_documents = all_cited_docs
                self.logger.debug(f"Parent node {node.id}: synthesis complete (iteration {iteration})")
            
            except Exception as e:
                self.logger.warning(f"Failed to synthesize for {node.id}, using concatenation: {e}")
                synthesized = f"# {node.question}\n\n" + combined_child_results
                node.cited_documents = all_cited_docs
            
            # If no more refinement budget or no dag_generator, return current result
            if iteration >= max_refinements:
                self.logger.debug(f"Parent node {node.id}: refinement budget exhausted")
                break
            
            if self.dag_generator is None:
                self.logger.debug(f"Parent node {node.id}: no dag_generator, skipping refinement")
                break
            
            # Check gap depth limit to prevent unbounded recursive refinement
            max_gap_depth = getattr(self, '_max_gap_depth', 2)
            if gap_depth >= max_gap_depth:
                self.logger.info(f"Parent node {node.id}: max gap depth ({max_gap_depth}) reached, skipping refinement")
                break
            
            # Validate the composed answer for gaps
            synth_len = len(synthesized or "")
            est_tokens = synth_len // 4
            self.logger.info(f"LLM call: composition_validator for {node.id} (~{est_tokens} input tokens, answer={synth_len} chars)")
            
            try:
                validation = await self.composition_validator.aforward(
                    research_question=node.question,
                    composed_answer=synthesized or "",
                    lm=self.lm,
                )
                
                # Use is_sufficient boolean as the primary check
                is_sufficient = getattr(validation, 'is_sufficient', True)
                if isinstance(is_sufficient, str):
                    is_sufficient = is_sufficient.lower() in ('true', 'yes', '1')
                
                if is_sufficient:
                    self.logger.debug(f"Parent node {node.id}: answer is sufficient, no refinement needed")
                    break
                
                missing_topics = validation.missing_topics.strip() if validation.missing_topics else ""
                if not missing_topics:
                    self.logger.debug(f"Parent node {node.id}: marked insufficient but no topics provided, skipping refinement")
                    break
                
                self.logger.info(f"Parent node {node.id}: gaps detected - {missing_topics}")
                
            except Exception as e:
                self.logger.warning(f"Validation failed for {node.id}, skipping refinement: {e}")
                break
            
            # Set node status to "refining" and save snapshot so frontend can show indicator
            node.status = "refining"
            self._save_graph_snapshot(graph)
            
            # Generate subtree to fill gaps using DAGGenerationAgent with restrictive params
            try:
                gap_topic = f"{node.question} - Fill critical gaps: {missing_topics}"
                subtree_request = ResearchRequest(
                    topic=gap_topic,
                    max_depth=1,      # Shallow subtree
                    max_nodes=3,      # Small subtree
                    max_subtasks=2,   # Narrow branching
                )
                
                subtree_graph = await self.dag_generator.generate_dag(
                    subtree_request,
                    max_retriever_calls=self._gap_fill_retriever_calls,
                )
                
                if not subtree_graph.nodes:
                    self.logger.debug(f"Parent node {node.id}: empty subtree generated")
                    break
                
                self.logger.info(f"Parent node {node.id}: generated subtree with {len(subtree_graph.nodes)} nodes")
                
            except Exception as e:
                self.logger.warning(f"Subtree generation failed for {node.id}: {e}")
                break
            
            # Merge subtree into main graph with remapped IDs
            new_node_ids = self._merge_subtree(graph, subtree_graph, node.id, iteration)
            
            if not new_node_ids:
                self.logger.debug(f"Parent node {node.id}: no new nodes after merge")
                break
            
            self._save_graph_snapshot(graph)
            
            # Process the new subtree bottom-up
            await self._process_subtree(graph, new_node_ids, max_retriever_calls, gap_depth=gap_depth + 1)
            
            # Save snapshot after processing subtree
            self._save_graph_snapshot(graph)
        
        # Handle empty answers
        if not synthesized or not synthesized.strip():
            return "No answer available."
        return synthesized.strip()
    
    def _merge_subtree(
        self,
        main_graph: ResearchGraph,
        subtree_graph: ResearchGraph,
        parent_node_id: str,
        iteration: int,
    ) -> list[str]:
        """Merge a subtree into the main graph with remapped IDs.
        
        The subtree's root node(s) become children of the specified parent node.
        All node IDs are prefixed to avoid collisions.
        
        Args:
            main_graph: The main research graph
            subtree_graph: The subtree to merge
            parent_node_id: ID of the parent node to attach subtree roots to
            iteration: Refinement iteration number (for ID prefixing)
            
        Returns:
            List of new node IDs added to the main graph
        """
        parent_node = main_graph.nodes[parent_node_id]
        id_prefix = f"{parent_node_id}_gap{iteration}_"
        
        # Map old IDs to new IDs
        id_map: dict[str, str] = {}
        for old_id in subtree_graph.nodes:
            id_map[old_id] = f"{id_prefix}{old_id}"
        
        new_node_ids = []
        
        # Add nodes with remapped IDs
        for old_id, subtree_node in subtree_graph.nodes.items():
            new_id = id_map[old_id]
            
            # Create a fresh node directly (don't use get_or_create_node to avoid reusing existing nodes)
            new_node = ResearchNode(
                id=new_id,
                question=subtree_node.question,
                parents=[],
                children=[],
                status="pending",
                depth=parent_node.depth + 1 + subtree_node.depth,
                normalized_question=subtree_node.normalized_question,
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
            main_graph.nodes[new_id] = new_node
            
            new_node.expected_output_format = subtree_node.expected_output_format
            new_node.composition_instructions = subtree_node.composition_instructions
            new_node.metadata = dict(subtree_node.metadata)
            new_node.metadata["refinement_iteration"] = iteration + 1
            new_node.metadata["source_parent"] = parent_node_id
            
            # Remap children, excluding self-references
            new_node.children = [id_map[c] for c in subtree_node.children if c in id_map and id_map[c] != new_id]
            
            new_node_ids.append(new_id)
        
        # Attach subtree root(s) to parent node
        # Root nodes in subtree are those with no parent in the subtree
        subtree_root_ids = []
        for old_id, subtree_node in subtree_graph.nodes.items():
            # Check if this node has a parent in the subtree
            has_parent_in_subtree = False
            for other_node in subtree_graph.nodes.values():
                if old_id in other_node.children:
                    has_parent_in_subtree = True
                    break
            if not has_parent_in_subtree:
                subtree_root_ids.append(id_map[old_id])
        
        # Add subtree roots as children of parent node
        for root_id in subtree_root_ids:
            if root_id not in parent_node.children:
                parent_node.children.append(root_id)
        
        self.logger.debug(f"Merged {len(new_node_ids)} nodes from subtree, {len(subtree_root_ids)} roots attached to {parent_node_id}")
        
        return new_node_ids
    
    async def _process_subtree(
        self,
        graph: ResearchGraph,
        node_ids: list[str],
        max_retriever_calls: int,
        gap_depth: int = 0,
    ) -> None:
        """Process a subtree of nodes bottom-up.
        
        Sorts the nodes by layer (leaves first) and processes them.
        Leaf nodes are processed in parallel, intermediate nodes sequentially.
        
        Args:
            graph: The research graph
            node_ids: IDs of nodes in the subtree to process
            max_retriever_calls: Max retriever calls for literature search
            gap_depth: Current depth of gap-filling recursion
        """
        if not node_ids:
            return
        
        # Build layers for just these nodes
        node_set = set(node_ids)
        
        # Calculate in-degree (children count) for subtree nodes only
        in_degree = {}
        for nid in node_ids:
            node = graph.nodes[nid]
            # Only count children that are in the subtree
            in_degree[nid] = len([c for c in node.children if c in node_set])
        
        # Find leaves (in-degree 0)
        current_layer = [nid for nid, deg in in_degree.items() if deg == 0]
        
        layers = []
        processed = set()
        
        while current_layer:
            layers.append(current_layer)
            processed.update(current_layer)
            
            # Find next layer
            next_layer = []
            for nid in node_ids:
                if nid in processed:
                    continue
                node = graph.nodes[nid]
                # Check if all children in subtree are processed
                subtree_children = [c for c in node.children if c in node_set]
                if all(c in processed for c in subtree_children):
                    next_layer.append(nid)
            
            current_layer = next_layer
        
        self.logger.debug(f"Processing subtree: {len(layers)} layers, {len(node_ids)} nodes")
        
        # Process each layer
        for layer_idx, layer_node_ids in enumerate(layers):
            is_leaf_layer = all(not graph.nodes[nid].children for nid in layer_node_ids)
            
            if is_leaf_layer:
                # Leaf nodes: process in parallel
                for nid in layer_node_ids:
                    graph.nodes[nid].status = "in_progress"
                self._save_graph_snapshot(graph)
                
                tasks = [self._process_node(graph, nid, max_retriever_calls, gap_depth=gap_depth) for nid in layer_node_ids]
                await asyncio.gather(*tasks)
            else:
                # Intermediate nodes: process sequentially
                for nid in layer_node_ids:
                    graph.nodes[nid].status = "in_progress"
                    self._save_graph_snapshot(graph)
                    await self._process_node(graph, nid, max_retriever_calls, gap_depth=gap_depth)
                    self._save_graph_snapshot(graph)
    
    def _reset_node_state(self, node: ResearchNode) -> None:
        """Reset transient node state before a retry attempt."""
        node.metadata = dict(node.metadata)
        node.metadata.pop("answer", None)
        self._node_results.pop(node.id, None)
        node.cited_documents = []
        node.literature_writeup = None
        node.report = None
        node.status = "pending"
    
    def _record_node_success(self, node: ResearchNode, result_text: str, graph: ResearchGraph) -> None:
        """Persist bookkeeping for a successfully processed node."""
        self._node_results[node.id] = result_text
        node.metadata = dict(node.metadata)
        node.metadata["answer"] = result_text
        node.literature_writeup = None
        if graph.root_id and node.id == graph.root_id:
            node.report = result_text
        else:
            node.report = None
        node.status = "complete"
    
    def _record_node_failure(self, node: ResearchNode, error_text: str, graph: ResearchGraph) -> None:
        """Persist bookkeeping for a failed node."""
        self._node_results[node.id] = error_text
        node.metadata = dict(node.metadata)
        node.metadata["answer"] = error_text
        node.cited_documents = []
        node.literature_writeup = None
        if graph.root_id and node.id == graph.root_id:
            node.report = error_text
        else:
            node.report = None
        node.status = "failed"
    
    def _save_graph_snapshot(self, graph: ResearchGraph) -> None:
        """Save a snapshot of the graph state for real-time visualization.
        
        Args:
            graph: The research graph
        """
        try:
            completed = len([n for n in graph.nodes.values() if n.status == "complete"])
            in_progress = len([n for n in graph.nodes.values() if n.status == "in_progress"])
            
            metadata = {
                "total_nodes": len(graph.nodes),
                "completed": completed,
                "in_progress": in_progress,
                "source": "snapshot",
                "snapshot_ts": datetime.utcnow().isoformat(),
            }
            
            self.logger.save_intermediate_result(
                "dag_processing_snapshot",
                graph.to_dict(),
                metadata,
            )
            
            # Notify via callback if provided (for WebSocket updates)
            if self._on_graph_update is not None:
                try:
                    self._on_graph_update(graph.to_dict(), metadata)
                except Exception as cb_err:
                    self.logger.warning(f"Graph update callback failed: {cb_err}")
        except Exception as e:
            self.logger.warning(f"Failed to save graph snapshot: {e}")

