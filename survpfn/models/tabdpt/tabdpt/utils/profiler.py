def add_profiling_markers(model):
    """Monkey-patch profiling markers into an nn.Module.
    source: https://gist.github.com/madebyollin/41a948a7c69a36b1e1fded71f253e7ef
    Args:
        model: an nn.Module
    
    Effect:
        all model.named_module() forward calls get wrapped in their
        own profiling scope, making traces easier to understand.
    
	# Usage
	add_profiling_markers(model)
	with torch.profiler.profile() as prof:
		y = model(x).cpu()
	prof.export_chrome_trace("trace.json")
	
    """
    from torch.profiler import record_function
    def add_profiling_to_forward(name, module):
        def profiled_forward(*args, **kwargs):
            with record_function(f"{name}.forward"):
                return module._forward(*args, **kwargs)
        return profiled_forward
    for name, module in model.named_modules():
        if not hasattr(module, "_forward"):
            module._forward = module.forward
        module.forward = add_profiling_to_forward(name, module)
