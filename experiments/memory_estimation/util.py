from abc import ABC, abstractmethod
from enum import Enum
import torch

class DataType(Enum):
    """
    The value represents type size.
    """
    boolean = 1
    float16 = 2
    bfloat16 = 2
    float32 = 4
    double = 8
    
    def transfer_to_torch_dtype(self):
        """Maps DataType enum to the corresponding torch dtype."""
        mapping = {
            DataType.boolean: torch.bool,
            DataType.float16: torch.float16,
            DataType.bfloat16: torch.bfloat16,
            DataType.float32: torch.float32,
            DataType.double: torch.float64  # torch.double is alias for torch.float64
        }
        return mapping.get(self, None)  # Return None if not found

class BaseMemoryUsage(ABC):
    @abstractmethod
    def setup_configs(self, *args, **kwargs):
        """
        Sets up configuration values required to estimate memory usage.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def setup_special_items(self, *args, **kwargs):
        """
        Sets up additional special items for memory usage.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def calculate_memory_usage(self):
        """
        Calculates the memory usage breakdown.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_memory_usage_report(self) -> dict:
        """
        Generates a report of memory usage breakdown.
        Must be implemented by subclasses.
        """
        pass
    
def pretty_print_cpu_memory_usage(memory_usage_report: dict) -> None:
    """
    Pretty prints the memory usage report in a readable format.
    
    Args:
        memory_usage_report (dict): The memory usage breakdown dictionary.
    """
    print("\nMemory Usage Report:")
    print("=" * 50)
    
    print("Items (Bytes):")
    for key, value in memory_usage_report["items_bytes"].items():
        print(f"  {key}: {value:,} bytes")
    
    print("\nItems (GiBs):")
    for key, value in memory_usage_report["items_GiBs"].items():
        print(f"  {key}: {value:.2f} GiB")
    
    print("\nTotal Memory Usage:")
    print(f"  Total Bytes: {memory_usage_report['total_bytes']:,} bytes")
    print(f"  Total GiBs: {memory_usage_report['total_GiBs']:.2f} GiB")
    
    print("=" * 50)