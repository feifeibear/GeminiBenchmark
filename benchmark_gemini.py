
import colossalai
import psutil
import torch
import torch.nn as nn
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from transformers import GPT2Config, GPT2LMHeadModel, AutoConfig, OPTForCausalLM
from time import time
from functools import partial
from colossalai.gemini.chunk import init_chunk_manager
from colossalai.gemini import GeminiManager
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.zero import ZeroOptimizer
from colossalai.utils import colo_set_process_memory_fraction
from colossalai.tensor import ProcessGroup
from transformers.modeling_utils import no_init_weights
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule
import gc, sys, os


class GPTLMModel(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_attention_heads=12, max_seq_len=1024, vocab_size=2048, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = GPT2LMHeadModel(GPT2Config(n_embd=hidden_size, n_layer=num_layers,
                                     n_head=num_attention_heads, n_positions=max_seq_len, n_ctx=max_seq_len, vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=not self.checkpoint)[0]


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def gpt2_medium(checkpoint=False):
    return GPTLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_xl(checkpoint=True):
    return GPTLMModel(hidden_size=1600, num_layers=48, num_attention_heads=32, checkpoint=checkpoint)


def gpt2_4b(checkpoint=True):
    return GPTLMModel(hidden_size=2304, num_layers=64, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_6b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=30, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_10b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=50, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_12b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=60, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_14b(checkpoint=True):
    return GPTLMModel(hidden_size=4096, num_layers=70, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_28b(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=35, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_32b(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=40, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_36b(checkpoint=True):
    return GPTLMModel(hidden_size=8192, num_layers=45, num_attention_heads=16, checkpoint=checkpoint)


def gpt2_8b(checkpoint=True):
    return GPTLMModel(hidden_size=3072, num_layers=72, num_attention_heads=24, checkpoint=checkpoint)


def _create_opt(config, checkpoint=True):
    model = OPTForCausalLM(config)
    if checkpoint:
        model.gradient_checkpointing_enable()
    return model


def opt_6b(checkpoint=True):
    config = AutoConfig.from_pretrained('facebook/opt-6.7b')
    return _create_opt(config, checkpoint=checkpoint)


def opt_2b(checkpoint=True):
    config = AutoConfig.from_pretrained('facebook/opt-2.7b')
    return _create_opt(config, checkpoint=checkpoint)


def opt_1b(checkpoint=True):
    config = AutoConfig.from_pretrained('facebook/opt-1.3b')
    return _create_opt(config, checkpoint=checkpoint)


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.max_memory_allocated() / 1024**2


def get_cur_gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2


def get_mem_info(prefix=''):
    return '{}current CUDA memory: {:.2f} MB, past max CUDA memory: {:.2f}, CPU memory {:.2f} MB'.format(
        prefix, get_cur_gpu_mem(), get_gpu_mem(), get_cpu_mem()
    )


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def gemini_zero_dpp(model: torch.nn.Module, pg: ProcessGroup, placement_policy: str = "auto"):
    cai_version = colossalai.__version__

    from colossalai.gemini.chunk import init_chunk_manager
    from colossalai.gemini import ChunkManager, GeminiManager
    from colossalai.nn.parallel import GeminiDDP
    model = GeminiDDP(model,
                        device=get_current_device(),
                        placement_policy=placement_policy,
                        pin_memory=True,
                        hidden_dim=4096,
                        search_range_mb=64)

    return model

def main():
    BATCH_SIZE = 64
    SEQ_LEN = 1024
    VOCAB_SIZE = 2048
    NUM_STEPS = 6
    PLACEMENT_POLICY = 'cpu'
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    # colo_set_process_memory_fraction(0.5)
    logger = get_dist_logger()

    logger.info(get_mem_info(), ranks=[0])
    # build GPT model
    # with ColoInitContext(device=get_current_device()):
    with ColoInitContext(device=get_current_device()):
        model = gpt2_10b(checkpoint=True)
    
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model numel: {numel}', ranks=[0])
    logger.info(get_mem_info(), ranks=[0])
    # logger.info([p.numel() for p in model.parameters()], ranks=[0])
    # logger.info({n: p.numel() for n, p in model.named_parameters()}, ranks=[0])
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)


    use_simple_api = False

    if use_simple_api:
        from colossalai.nn.parallel import GeminiDDP
        model = GeminiDDP(model,
                            device=get_current_device(),
                            placement_policy=PLACEMENT_POLICY,
                            pin_memory=True,
                            hidden_dim=4096,
                            search_range_mb=64)
    else:
        # internal APIs
        chunk_manager = init_chunk_manager(
            model=model,
            init_device=get_current_device(),
            hidden_dim=4096,
            search_range_mb=64
        )

        gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
        if PLACEMENT_POLICY == 'const':
            gemini_manager._placement_policy.set_const_memory_boundary(10 * 1024)
        model = ZeroDDP(model, gemini_manager, pin_memory=True)
    
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    # logger.info(chunk_manager, ranks=[0])
    logger.info(get_mem_info(), ranks=[0])

    # if use_simple_api:
    from colossalai.nn.optimizer.gemini_optimizer import GeminiAdamOptimizer
    optimizer = GeminiAdamOptimizer(model, lr=1e-3, initial_scale=2**5)
    # else:
    #     # internal APIs
    #     optimizer = HybridAdam(model.parameters(), lr=1e-3)
    #     optimizer = ZeroOptimizer(optimizer, model, initial_scale=2**5, gpu_margin_mem_ratio=0.0)

    # build criterion
    criterion = GPTLMLoss()

    # optimizer
    # optimizer = HybridAdam(model.parameters(), lr=1e-3, nvme_offload_fraction=0.0,
    #                        nvme_offload_dir='/data/user/offload')
    # optimizer = ZeroOptimizer(optimizer, model, initial_scale=2**5, gpu_margin_mem_ratio=0.0)
    # logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])

    model.train()

    def one_turn():
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

        start = time()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '), ranks=[0])
        fwd_end = time()
        fwd_time = fwd_end - start

        optimizer.backward(loss)
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '), ranks=[0])
        bwd_end = time()
        bwd_time = bwd_end - fwd_end

        optimizer.step()
        logger.info(get_mem_info(prefix=f'[{n+1}/{NUM_STEPS}] Optimizer step '), ranks=[0])
        optim_time = time() - bwd_end
        step_time = time() - start
        logger.info(
            f'[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s',
            ranks=[0])
        tflops_list.append(get_tflops_func(step_time))

    tflops_list = []
    for n in range(NUM_STEPS):
        one_turn()

    tflops_list.sort()
    middle = NUM_STEPS >> 1
    logger.info(f'Median TFLOPS is {tflops_list[middle]:.3f}')

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #              schedule=schedule(wait=1, warmup=2, active=2),
    #              on_trace_ready=tensorboard_trace_handler(
    #                  f'opt-6.7b/v3-full-{PLACEMENT_POLICY}-{dist.get_world_size()}gpu'),
    #              record_shapes=True,
    #              profile_memory=True) as prof:
    #     for n in range(NUM_STEPS):
    #         one_turn()
    #         prof.step()
    # dist.barrier()


if __name__ == '__main__':
    main()

