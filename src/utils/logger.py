"""
Logging utility for the deepcomposition pipeline.

Provides dual logging:
- Console: Text format with progress bars (INFO level)
- File: JSON format with detailed information (DEBUG level)
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm.asyncio import tqdm


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
            
        return json.dumps(log_data)


class PipelineLogger:
    """Logger for the deepcomposition pipeline with dual output."""
    
    def __init__(self, name: str = "deepcomposition", log_dir: str = "output/logs"):
        self.name = name
        base_log_dir = Path(log_dir)
        
        # Create timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a timestamped subdirectory for this run
        self.log_dir = base_log_dir / self.run_timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        self.logger.handlers.clear()
        
        # Console handler (INFO level, text format)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (DEBUG level, JSON format)
        log_file = self.log_dir / "pipeline.jsonl"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Run directory: {self.log_dir}")
    
    def info(self, message: str, extra_data: Optional[dict[str, Any]] = None):
        """Log an info message."""
        if extra_data:
            self.logger.info(message, extra={"extra_data": extra_data})
        else:
            self.logger.info(message)
    
    def debug(self, message: str, extra_data: Optional[dict[str, Any]] = None):
        """Log a debug message."""
        if extra_data:
            self.logger.debug(message, extra={"extra_data": extra_data})
        else:
            self.logger.debug(message)
    
    def warning(self, message: str, extra_data: Optional[dict[str, Any]] = None):
        """Log a warning message."""
        if extra_data:
            self.logger.warning(message, extra={"extra_data": extra_data})
        else:
            self.logger.warning(message)
    
    def error(self, message: str, extra_data: Optional[dict[str, Any]] = None):
        """Log an error message."""
        if extra_data:
            self.logger.error(message, extra={"extra_data": extra_data})
        else:
            self.logger.error(message)
    
    def save_intermediate_result(self, step_name: str, data: Any, metadata: Optional[dict[str, Any]] = None):
        """
        Save intermediate results to a JSON file.
        
        Args:
            step_name: Name of the pipeline step (e.g., "purpose_generation")
            data: The data to save (must be JSON serializable)
            metadata: Optional metadata about this step
        """
        filename = f"{step_name}.json"
        filepath = self.log_dir / filename
        
        output = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        
        if metadata:
            output["metadata"] = metadata
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        self.debug(f"Saved intermediate results for {step_name}", extra_data={"file": str(filepath)})
    
    def progress_bar(self, iterable, desc: str, total: Optional[int] = None):
        """
        Create a progress bar for iterables.
        
        Args:
            iterable: The iterable to wrap
            desc: Description for the progress bar
            total: Total number of items (if not inferrable from iterable)
        """
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
    
    async def progress_bar_async(self, async_iterable, desc: str, total: int):
        """
        Create a progress bar for async iterables.
        
        Args:
            async_iterable: The async iterable to wrap
            desc: Description for the progress bar
            total: Total number of items
        """
        with tqdm(total=total, desc=desc, ncols=80,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            async for item in async_iterable:
                yield item
                pbar.update(1)


# Global logger instance
_global_logger: Optional[PipelineLogger] = None


def get_logger(name: str = "deepcomposition") -> PipelineLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = PipelineLogger(name)
    return _global_logger


def init_logger(name: str = "deepcomposition", log_dir: str = "output/logs") -> PipelineLogger:
    """Initialize the global logger instance."""
    global _global_logger
    _global_logger = PipelineLogger(name, log_dir)
    return _global_logger

