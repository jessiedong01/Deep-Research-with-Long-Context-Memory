import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to Python path to enable absolute imports
# This ensures deepcomposition and utils packages can be imported
src_path = Path(__file__).resolve().parent.parent
src_path_str = str(src_path)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# Now we can use absolute imports
from deepcomposition.init_pipeline import init_agent
from utils.dataclass import (
    ResearchRequest,
    ResearchResponse,
)
from utils.logger import init_logger


async def main():
    # Initialize logging at the start
    logger = init_logger(name="deepcomposition", log_dir="output/logs")
    logger.info("=" * 80)
    logger.info("DeepComposition Pipeline Starting")
    logger.info("=" * 80)
    
    # Initialize the deepcomposition agent
    logger.info("Initializing deepcomposition agent...")
    agent = init_agent()
    logger.info("DeepComposition agent initialized successfully")

    # Prompt the user for a research task to use as the query/topic
    user_task = input("Please enter your research task or topic: ").strip()
    if not user_task:
        raise ValueError("Research task or topic is required")
    
    logger.info(f"User research task: {user_task}")
    
    # Configuration
    max_retriever_calls = 1
    max_depth = 2
    max_nodes = 50
    max_subtasks = 10
    max_refinements = 1
    
    # Allow user to optionally configure parameters
    configure = input("Would you like to configure advanced parameters? (y/N): ").strip().lower()
    if configure == 'y':
        try:
            max_retriever_calls = int(input(f"Max retriever calls (default {max_retriever_calls}): ").strip() or max_retriever_calls)
            max_depth = int(input(f"Max depth (default {max_depth}): ").strip() or max_depth)
            max_nodes = int(input(f"Max nodes (default {max_nodes}): ").strip() or max_nodes)
            max_subtasks = int(input(f"Max subtasks per node (default {max_subtasks}): ").strip() or max_subtasks)
            max_refinements = int(input(f"Max refinements per node (default {max_refinements}): ").strip() or max_refinements)
        except ValueError as e:
            logger.warning(f"Invalid input for configuration, using defaults: {e}")
    
    logger.info(f"Pipeline configuration: max_retriever_calls={max_retriever_calls}, max_depth={max_depth}, max_nodes={max_nodes}, max_subtasks={max_subtasks}, max_refinements={max_refinements}")

    # Run the deepcomposition pipeline
    response: ResearchResponse = await agent.aforward(
        ResearchRequest(
            topic=user_task,
            max_retriever_calls=max_retriever_calls,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_subtasks=max_subtasks,
            max_refinements=max_refinements,
        )
    )

    # Save results
    os.makedirs("output", exist_ok=True)
    output_file = "output/results.json"
    
    logger.info(f"Saving final results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(response.to_dict(), f, indent=2)
    
    logger.info("=" * 80)
    logger.info(f"Pipeline completed successfully! Results saved to {output_file}")
    logger.info(f"Intermediate results and logs saved to output/logs/")
    logger.info("=" * 80)



if __name__ == "__main__":
    asyncio.run(main())
