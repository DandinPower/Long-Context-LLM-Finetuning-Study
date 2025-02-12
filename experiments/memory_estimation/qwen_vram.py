# need to care about the peak of calculate gradients on the lm_head
# after that the next peak is caused by transformers 
# so it should like max(possible peak 1, possible peak2)

from util import BaseMemoryUsage, DataType
    
class Qwen25MemoryUsage(BaseMemoryUsage):
    def __init__(self):
        super().__init__()
        # Configs
        self.batch: int = None
        self.max_seq_len: int = None
        self.vproj_size: int = None
        self.intermediate_size: int = None
        self.hidden_size: int = None
        self.vocab_size: int = None
        self.grad_norm_buffer_size: int = None
        self.compute_dtype: DataType = None
        self.grad_accum_dtype: DataType = None
        # Intermediate Items
        self.feedforward_intermediate = None
        self.vproj = None
        self.feedforward_activation = None
        self.hidden_activation = None
        self.embed_weight = None
        
        self.all_reduce_buffer_for_grad = None
        self.stage3_partition_overhead = None
        self.embed_related_weight_grad = None
        # Items
        self.decoder_layer_backward_peak: int = None
        self.embed_tokens_backward_peak: int = None
        
    def setup_configs(
        self, 
        batch,
        max_seq_len,
        vproj_size,
        intermediate_size,
        hidden_size,
        vocab_size,
        grad_norm_buffer_size,
        grad_accum_dtype,
        compute_dtype,
    ) -> None:
        """
        Sets up the configuration values required to estimate memory usage.
        """
        self.batch = batch
        self.max_seq_len = max_seq_len 
        self.vproj_size = vproj_size
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.grad_norm_buffer_size = grad_norm_buffer_size
        self.grad_accum_dtype = grad_accum_dtype
        self.compute_dtype = compute_dtype

    def setup_special_items(self, others: int) -> None:
        raise NotImplementedError("There is no special items for nvme usage.")

    def calculate_memory_usage(self) -> None:
        """
        Calculates the memory usage breakdown.
        """
        # First Peak
        self.feedforward_intermediate = 4 * self.intermediate_size * self.hidden_size * self.compute_dtype.value
        self.vproj = 2 * self.vproj_size
        self.feedforward_activation = 4 * self.intermediate_size * self.batch * self.max_seq_len * self.compute_dtype.value
        self.hidden_activation = 10 * self.hidden_size * self.batch * self.max_seq_len * self.compute_dtype.value
        self.embed_weight = self.vocab_size * self.hidden_size * self.compute_dtype.value
        
        self.decoder_layer_backward_peak = self.feedforward_intermediate + self.vproj + self.feedforward_activation + self.hidden_activation + self.embed_weight
        # Second Peak
        self.all_reduce_buffer_for_grad = self.grad_norm_buffer_size * DataType.double.value
        self.stage3_partition_overhead = self.vocab_size * self.hidden_size * self.grad_accum_dtype.value
        self.embed_related_weight_grad = 3 * self.vocab_size * self.hidden_size * self.compute_dtype.value
        
        self.embed_tokens_backward_peak = self.all_reduce_buffer_for_grad + self.stage3_partition_overhead + self.embed_related_weight_grad
    
    def get_memory_usage_report(self) -> dict:
        """
        Generates a report of memory usage breakdown for VRAM (per GPU).

        Returns:
            dict: Memory usage breakdown.
        """
        # Populate memory usage details for decoder layer backward peak
        self.decoder_layer_backward_peak_dict = {
            "feedforward_intermediate": self.feedforward_intermediate,
            "vproj": self.vproj,
            "feedforward_activation": self.feedforward_activation,
            "hidden_activation": self.hidden_activation,
            "embed_weight": self.embed_weight,
        }
        self.decoder_layer_backward_peak_dict_GiBs = {
            k: v / (1024**3) for k, v in self.decoder_layer_backward_peak_dict.items()
        }
        
        # Populate memory usage details for embedding tokens backward peak
        self.embed_tokens_backward_peak_dict = {
            "all_reduce_buffer_for_grad": self.all_reduce_buffer_for_grad,
            "stage3_partition_overhead": self.stage3_partition_overhead,
            "embed_related_weight_grad": self.embed_related_weight_grad,
        }
        self.embed_tokens_backward_peak_dict_GiBs = {
            k: v / (1024**3) for k, v in self.embed_tokens_backward_peak_dict.items()
        }
        
        # Determine the maximum peak memory usage
        max_peak = max(self.decoder_layer_backward_peak, self.embed_tokens_backward_peak)
        max_peak_GiBs = max_peak / (1024**3)
        
        # Construct the final report
        return {
            "decoder_layer_backward_peak": self.decoder_layer_backward_peak_dict,
            "decoder_layer_backward_peak_GiBs": self.decoder_layer_backward_peak_dict_GiBs,
            "embed_tokens_backward_peak": self.embed_tokens_backward_peak_dict,
            "embed_tokens_backward_peak_GiBs": self.embed_tokens_backward_peak_dict_GiBs,
            "different_peak": {
                "decoder_layer_backward_peak": self.decoder_layer_backward_peak,
                "embed_tokens_backward_peak": self.embed_tokens_backward_peak
            },
            "different_peak_GiBs": {
                "decoder_layer_backward_peak": self.decoder_layer_backward_peak / (1024**3),
                "embed_tokens_backward_peak": self.embed_tokens_backward_peak / (1024**3)
            },
            "max_peak_bytes": max_peak,
            "max_peak_GiBs": max_peak_GiBs,
        }


        
if __name__ == "__main__":
    qwen25_memory_usage = Qwen25MemoryUsage()
    # 1.5B
    # qwen25_memory_usage.setup_configs(batch=80, max_seq_len=8192, vproj_size=393216, intermediate_size=8960, hidden_size=1536, \
    #     vocab_size=151936, grad_norm_buffer_size=250000000, grad_accum_dtype=DataType.bfloat16, compute_dtype=DataType.bfloat16)
    # 7B
    # qwen25_memory_usage.setup_configs(batch=10, max_seq_len=32768, vproj_size=1835008, intermediate_size=18944, hidden_size=3584, \
    #     vocab_size=152064, grad_norm_buffer_size=250000000, grad_accum_dtype=DataType.bfloat16, compute_dtype=DataType.bfloat16)
    # 14B
    # qwen25_memory_usage.setup_configs(batch=40, max_seq_len=8192, vproj_size=5242880, intermediate_size=13824, hidden_size=5120, \
    #     vocab_size=152064, grad_norm_buffer_size=250000000, grad_accum_dtype=DataType.bfloat16, compute_dtype=DataType.bfloat16)
    # 32B
    qwen25_memory_usage.setup_configs(batch=1, max_seq_len=131072, vproj_size=5242880, intermediate_size=27648, hidden_size=5120, \
        vocab_size=152064, grad_norm_buffer_size=250000000, grad_accum_dtype=DataType.bfloat16, compute_dtype=DataType.bfloat16)
    qwen25_memory_usage.calculate_memory_usage()
    print(qwen25_memory_usage.get_memory_usage_report())