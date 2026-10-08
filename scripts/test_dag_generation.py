#!/usr/bin/env python3
"""Manual test script for DAG generation.

This script generates a DAG for a sample research question and visualizes it.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from deepcomposition.init_pipeline import init_agent
from deepcomposition.dag_generation import DAGGenerationAgent
from utils.dataclass import ResearchRequest
from utils.logger import init_logger


def print_dag_tree(graph, node_id=None, prefix="", is_last=True):
    """Print a tree visualization of the DAG."""
    if node_id is None:
        node_id = graph.root_id
    
    node = graph.nodes[node_id]
    
    # Print current node
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{node.id}: {node.question[:60]}...")
    print(f"{prefix}    Format: {node.expected_output_format}")
    if node.composition_instructions:
        print(f"{prefix}    Composition: {node.composition_instructions[:60]}...")
    
    # Print children
    if node.children:
        extension = "    " if is_last else "│   "
        for i, child_id in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            print_dag_tree(graph, child_id, prefix + extension, is_last_child)


async def main():
    # Initialize logging
    logger = init_logger(name="dag_test", log_dir="output/logs")
    logger.info("Starting manual DAG generation test")
    
    # Get research question from user
    print("=" * 80)
    print("DAG Generation Test")
    print("=" * 80)
    
    default_question = "What are the main applications of AI in healthcare?"
    question = input(f"\nEnter research question (or press Enter for default):\n> ").strip()
    if not question:
        question = default_question
        print(f"Using default: {question}")
    
    # Configure DAG generation
    print("\nConfiguration:")
    max_depth = int(input("Max depth (default 2): ") or "2")
    max_nodes = int(input("Max nodes (default 10): ") or "10")
    max_subtasks = int(input("Max subtasks per node (default 3): ") or "3")
    
    print(f"\nGenerating DAG with:")
    print(f"  - Topic: {question}")
    print(f"  - Max depth: {max_depth}")
    print(f"  - Max nodes: {max_nodes}")
    print(f"  - Max subtasks: {max_subtasks}")
    print()
    
    # Initialize deepcomposition agent (to get literature search agent)
    agent = init_agent()
    
    # Create DAG generation agent
    dag_agent = DAGGenerationAgent(
        literature_search_agent=agent.literature_search_agent,
        lm=agent.lm,
    )
    
    # Generate DAG
    request = ResearchRequest(
        topic=question,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_subtasks=max_subtasks,
    )
    
    print("Generating DAG... (this may take a minute)\n")
    graph = await dag_agent.generate_dag(request)
    
    # Print results
    print("\n" + "=" * 80)
    print("DAG Generation Complete!")
    print("=" * 80)
    print(f"\nTotal nodes: {len(graph.nodes)}")
    print(f"Root node: {graph.root_id}")
    
    depths = {}
    for node in graph.nodes.values():
        depths[node.depth] = depths.get(node.depth, 0) + 1
    
    print(f"Depth distribution: {dict(sorted(depths.items()))}")
    
    leaf_count = len([n for n in graph.nodes.values() if not n.children])
    print(f"Leaf nodes: {leaf_count}")
    
    # Print tree visualization
    print("\n" + "=" * 80)
    print("DAG Tree Structure:")
    print("=" * 80)
    print()
    print_dag_tree(graph)
    
    # Print detailed node information
    print("\n" + "=" * 80)
    print("Detailed Node Information:")
    print("=" * 80)
    
    for node_id, node in sorted(graph.nodes.items(), key=lambda x: (x[1].depth, x[0])):
        print(f"\n{node_id} (depth={node.depth}):")
        print(f"  Question: {node.question}")
        print(f"  Output Format: {node.expected_output_format}")
        if node.metadata.get("format_details"):
            print(f"  Format Details: {node.metadata['format_details']}")
        print(f"  Children: {len(node.children)}")
        if node.composition_instructions:
            print(f"  Composition Instructions: {node.composition_instructions}")
    
    print("\n" + "=" * 80)
    logger.info("DAG generation test complete")
    print("Test complete! Check output/logs for detailed results.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

