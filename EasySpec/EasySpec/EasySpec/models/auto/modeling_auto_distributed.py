from transformers import AutoConfig

class AutoDistributedModelForCausalLM:
    """
        This class is for distributed worker to load model from auto.
    """
    @classmethod
    def from_pretrained(cls, base_model_path, world_size, **kwargs):
        # get config
        config = AutoConfig.from_pretrained(
            base_model_path,
        )
        # get type
        Type = config.architectures[0]
        
        config.support_batch_layer = True
        
        from_pretrained_args = (base_model_path,)
        from_pretrained_kwargs = kwargs
        from_pretrained_kwargs.update({"config": config})
        
        use_tp = kwargs.pop("use_tp", False)
        use_hyperdraft = kwargs.pop("use_hyperdraft", False)
        
        if use_tp:
            from ..multi_gpu.modeling_multi_gpu import TPLlamaForCausalLM as LlamaForCausalLM
        elif use_hyperdraft:
            from ..hyperdraft.modeling_hyperdraft import LlamaForCausalLM as LlamaForCausalLM
        else:
            raise NotImplementedError
        
        config.o_proj_bias = False
        if Type == 'LlamaForCausalLM':
            base_model = LlamaForCausalLM.from_pretrained(
                *from_pretrained_args, 
                **from_pretrained_kwargs
            )
        elif Type == "Qwen2ForCausalLM":
            config.attention_bias = True
            config.mlp_bias = False
            base_model = LlamaForCausalLM.from_pretrained(
                *from_pretrained_args, 
                **from_pretrained_kwargs
            )
        
        return base_model
