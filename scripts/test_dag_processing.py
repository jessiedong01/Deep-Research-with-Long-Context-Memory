#!/usr/bin/env python3
"""Manual test script for DAG processing.

This script takes a pre-generated or newly created DAG and processes it bottom-up.
"""
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from deepcomposition.init_pipeline import init_agent
from deepcomposition.dag_generation import DAGGenerationAgent
from deepcomposition.dag_processor import DAGProcessor
from utils.dataclass import ResearchRequest, ResearchGraph
from utils.logger import init_logger


async def main():
    # Initialize logging
    logger = init_logger(name="dag_processor_test", log_dir="output/logs")
    logger.info("Starting DAG processor test")
    
    print("=" * 80)
    print("DAG Processing Test")
    print("=" * 80)
    
    # Option 1: Load existing DAG or Option 2: Generate new one
    print("\nOptions:")
    print("1. Generate a new DAG and process it")
    print("2. Load existing DAG from file (not yet implemented)")
    
    choice = input("\nChoice (default 1): ").strip() or "1"
    
    if choice == "1":
        # Generate new DAG
        print("\n" + "=" * 80)
        print("Step 1: Generate DAG")
        print("=" * 80)
        
        question = input("\nResearch question: ").strip()
        if not question:
            question = "What are the benefits and drawbacks of electric vehicles?"
            print(f"Using default: {question}")
        
        max_depth = int(input("Max depth (default 2): ") or "2")
        max_nodes = int(input("Max nodes (default 5): ") or "5")
        
        # Initialize agents
        agent = init_agent()
        
        dag_agent = DAGGenerationAgent(
            literature_search_agent=agent.literature_search_agent,
            lm=agent.lm,
        )
        
        request = ResearchRequest(
            topic=question,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_subtasks=3,
        )
        
        print("\nGenerating DAG...")
        graph = await dag_agent.generate_dag(request)
        
        print(f"\n✓ DAG generated with {len(graph.nodes)} nodes")
        
        # Print DAG structure
        print("\nDAG Structure:")
        for node_id, node in sorted(graph.nodes.items(), key=lambda x: (x[1].depth, x[0])):
            indent = "  " * node.depth
            print(f"{indent}- {node_id}: {node.question[:60]}...")
            print(f"{indent}  Format: {node.expected_output_format}")
        
    else:
        print("Loading existing DAG not yet implemented")
        return
    
    # Step 2: Process the DAG
    print("\n" + "=" * 80)
    print("Step 2: Process DAG")
    print("=" * 80)
    
    max_retriever_calls = int(input("\nMax retriever calls per node (default 2): ") or "2")
    
    processor = DAGProcessor(
        literature_search_agent=agent.literature_search_agent,
        lm=agent.lm,
    )
    
    print("\nProcessing DAG... (this may take several minutes)")
    print("Watch the logs for progress...\n")
    
    processed_graph, node_results = await processor.process_dag(
        graph,
        max_retriever_calls=max_retriever_calls,
    )
    
    # Step 3: Display results
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    
    completed = len([n for n in processed_graph.nodes.values() if n.status == "complete"])
    failed = len([n for n in processed_graph.nodes.values() if n.status == "failed"])
    
    print(f"\nNodes completed: {completed}/{len(processed_graph.nodes)}")
    if failed > 0:
        print(f"Nodes failed: {failed}")
    
    # Display results for each node
    print("\n" + "=" * 80)
    print("Node Results:")
    print("=" * 80)
    
    for node_id, node in sorted(processed_graph.nodes.items(), key=lambda x: (x[1].depth, x[0])):
        print(f"\n{'-' * 80}")
        print(f"Node: {node_id} (depth={node.depth}, status={node.status})")
        print(f"Question: {node.question}")
        print(f"Output Format: {node.expected_output_format}")
        print(f"Children: {len(node.children)}")
        print(f"Citations: {len(node.cited_documents)}")
        print(f"\nAnswer:")
        node_answer = node_results.get(node_id, "No answer stored (intermediate node)")
        preview = node_answer[:500]
        print(preview)
        if len(node_answer) > 500:
            print("... (truncated)")
    
    # Show root answer specifically
    print("\n" + "=" * 80)
    print("ROOT NODE ANSWER (Final Result):")
    print("=" * 80)
    root = processed_graph.nodes[processed_graph.root_id]
    print(f"\nQuestion: {root.question}")
    print(f"Format: {root.expected_output_format}")
    root_answer = node_results.get(root.id, root.report or "No answer")
    print(f"\nAnswer:\n{root_answer}")
    
    # Show citations
    if root.cited_documents:
        print(f"\n\nCitations ({len(root.cited_documents)} documents):")
        for i, doc in enumerate(root.cited_documents[:10], 1):
            print(f"  [{i}] {doc.url}")
        if len(root.cited_documents) > 10:
            print(f"  ... and {len(root.cited_documents) - 10} more")
    
    print("\n" + "=" * 80)
    logger.info("DAG processing test complete")
    print("Test complete! Check output/logs for detailed results.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

