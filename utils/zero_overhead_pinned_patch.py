from zero_overhead_pinned_memory import to_posix_memalign_pinned, zeros_cuda_host_alloc_pinned

def zero_overhead_pin_memory(self, tensor, align_bytes=1):
    if tensor.is_pinned():
        return tensor
    return to_posix_memalign_pinned(tensor)

def zeros_like_pin_memory(self, shape, dtype, device, align_bytes=1):
    return zeros_cuda_host_alloc_pinned(shape, dtype)

def patch_deepspeed_zero_overhead_pinned_memory():
    import deepspeed
    print("Apply zero overhead pin_memory to deepspeed CUDA_Accelerator")
    deepspeed.accelerator.cuda_accelerator.CUDA_Accelerator.pin_memory = zero_overhead_pin_memory
