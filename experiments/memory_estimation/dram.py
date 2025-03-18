from itertools import product

from util import BaseMemoryUsage, DataType, pretty_print_cpu_memory_usage
        
class CPUMemoryUsage(BaseMemoryUsage):
    def __init__(self):
        super().__init__()
        # Configs
        self.model_parameters_size: int = None
        self.largest_parameter_size: int = None
        self.batch_size: int = None
        self.max_seq_len: int = None
        self.num_decoder_layers: int = None
        self.hidden_size: int = None
        self.optimizer_dtype: DataType = None
        self.compute_dtype: DataType = None
        self.offload_checkpointed: bool = None

        # Items
        self.activation_checkpoints: int = None
        self.compute_weight: int = None
        self.compute_gradient: int = None
        self.master_weight: int = None
        self.master_gradient: int = None
        self.momentum: int = None
        self.variance: int = None

        # Special Items
        self.others:int = None

    def setup_configs(
        self, 
        num_gpus: int,
        model_parameters_size: int, 
        largest_parameter_size: int, 
        batch_size: int,
        max_seq_len: int,
        num_decoder_layers: int,
        hidden_size: int,
        optimizer_dtype: DataType, 
        compute_dtype: DataType, 
        offload_checkpointed: bool,
    ) -> None:
        """
        Sets up the configuration values required to estimate memory usage.
        """
        self.num_gpus = num_gpus
        self.model_parameters_size = model_parameters_size
        self.largest_parameter_size = largest_parameter_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.optimizer_dtype = optimizer_dtype
        self.compute_dtype = compute_dtype
        self.offload_checkpointed = offload_checkpointed

    def calculate_memory_usage(self) -> None:
        """
        Calculates the memory usage breakdown.
        """
        self.activation_checkpoints = self.num_gpus * self.batch_size * self.max_seq_len * self.num_decoder_layers * self.hidden_size * self.compute_dtype.value if self.offload_checkpointed else 0            
        self.compute_weight = self.model_parameters_size * self.compute_dtype.value
        self.compute_gradient = self.model_parameters_size * self.compute_dtype.value
        self.master_weight = self.model_parameters_size * self.optimizer_dtype.value
        self.master_gradient = self.model_parameters_size * self.optimizer_dtype.value
        self.momentum = self.model_parameters_size * self.optimizer_dtype.value
        self.variance = self.model_parameters_size * self.optimizer_dtype.value
    
    def setup_special_items(self, others: int) -> None:
        self.others = others
    
    def get_memory_usage_report(self) -> dict:
        """
        Generates a report of memory usage breakdown.

        Returns:
            dict: Memory usage breakdown.
        """
        # Calculate total bytes from all components
        total_bytes = (
            self.activation_checkpoints +
            self.compute_weight +
            self.compute_gradient +
            self.master_weight +
            self.master_gradient +
            self.momentum +
            self.variance +
            self.others
        )
        total_GiBs = total_bytes / (1024**3)

        return {
            "items_bytes": {
                "activation_checkpoints": self.activation_checkpoints,
                "compute_weight": self.compute_weight,
                "compute_gradient": self.compute_gradient,
                "master_weight": self.master_weight,
                "master_gradient": self.master_gradient,
                "momentum": self.momentum,
                "variance": self.variance,
                "others": self.others
            },
            "total_bytes": total_bytes,
            "items_GiBs": {
                "activation_checkpoints": self.activation_checkpoints / (1024**3),
                "compute_weight": self.compute_weight / (1024**3),
                "compute_gradient": self.compute_gradient / (1024**3),
                "master_weight": self.master_weight / (1024**3),
                "master_gradient": self.master_gradient / (1024**3),
                "momentum": self.momentum / (1024**3),
                "variance": self.variance / (1024**3),
                "others": self.others / (1024**3)
            },
            "total_GiBs": total_GiBs
        }

if __name__ == "__main__":
    # Llama3.1 8B
    # model_parameter_size = 8030261248
    # largest_parameter_size = 525336576
    # num_decoder_layers = 32
    # hidden_size = 4096

    # Qwen2.5 7B
    model_parameter_size = 7615616512
    largest_parameter_size = 544997376
    num_decoder_layers = 28
    hidden_size = 3584
    
    # mistral-nemo-12B
    model_parameter_size = 12247782400
    largest_parameter_size = 671088640
    num_decoder_layers = 40
    hidden_size = 5120
    
    # Qwen2.5 14B
    model_parameter_size = 14770033664
    largest_parameter_size = 778567680
    num_decoder_layers = 48
    hidden_size = 5120

    # Qwen2.5 32B
    model_parameter_size = 32763876352
    largest_parameter_size = 778567680
    num_decoder_layers = 64
    hidden_size = 5120
    
    
    
    num_gpus = 2
    max_seq_lens = [4096, 16384, 32768, 40960, 49152, 65536, 98304, 131072]
    # batch_sizes = [1, 4, 8, 16, 24, 32, 40, 48]
    batch_sizes = [1]
    
    for batch_size, max_seq_len in product(batch_sizes, max_seq_lens):
        print("-" * 50)
        print(f"Batch Size: {batch_size}; Context Length: {max_seq_len}")
        print("-" * 50)
        cpu_memory_usage = CPUMemoryUsage()
        cpu_memory_usage.setup_configs(
            num_gpus = num_gpus, \
            model_parameters_size=model_parameter_size, \
            largest_parameter_size=largest_parameter_size, \
            batch_size=batch_size, max_seq_len=max_seq_len, \
            num_decoder_layers=num_decoder_layers, hidden_size=hidden_size, \
            optimizer_dtype=DataType.float32, compute_dtype=DataType.float16, \
            offload_checkpointed=True, \
        )
        cpu_memory_usage.setup_special_items(others=0)
        cpu_memory_usage.calculate_memory_usage()
        report = cpu_memory_usage.get_memory_usage_report()
        pretty_print_cpu_memory_usage(report)
    