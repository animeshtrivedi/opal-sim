# vllm_worker.py - vLLM Scheduler Integration

## Overview

`vllm_worker.py` implements an LLM worker that integrates vLLM v1's continuous batching scheduler with SimPy discrete event simulation. This module combines scheduler data structures and worker execution logic in a single cohesive implementation that accurately models real vLLM behavior.

## Key Features

### 1. **vLLM-Style Continuous Batching**
- FIFO scheduling for both prefill and decode phases
- Dynamic batch formation based on resource constraints
- Support for mixed prefill and decode in the same batch

### 2. **GPU Memory Block Management**
- Configurable block size (default 16 tokens)
- Tracks allocated blocks per request throughout lifecycle
- Proper block allocation/deallocation at request start/completion
- Enforces memory constraints during batch construction

### 3. **KV Cache Integration**
- Prefix matching via `OpalKVCacheEngine.lookup()`
- Asynchronous KV cache fetching simulation
- Proper state transitions for cache operations (WAITING → FETCH_KVC → READY)

### 4. **Chunked Prefill Support**
- Large prompts processed in configurable chunks
- Prevents memory overflow and improves latency
- Chunk size derived from `max_num_batched_tokens`

### 5. **Realistic Timing Model**
- Per-batch timing: `max(longest_prefill_chunk, longest_decode)`
- GPU model integration for accurate latency calculations
- Accounts for prefix matching in prefill time

### 6. **Full Request Lifecycle Tracking**
- Complete state machine using single `RequestPhase` enum
- Detailed statistics collection
- Per-token timestamp tracking

## Architecture

### Request State Flow

```
New Request
    ↓
WAITING (in worker queue)
    ↓
KVC Lookup
    ↓
Has Prefix Match?
    ├─ Yes → FETCH_KVC (retrieve KV cache)
    │           ↓
    │        READY
    └─ No  → READY
                ↓
        Scheduled in Batch
                ↓
        PREFILL/PREFILL_CHUNKED
                ↓
        Prefill Complete?
                ↓
            DECODE
                ↓
        All Tokens Generated?
                ↓
            COMPLETED
```

### Core Components

#### 1. **RequestPhase (Enum)**
Single source of truth for request states:
- `WAITING`: Request queued, not yet processed for KVC lookup
- `FETCH_KVC`: KVC retrieve in progress, moved to back of waiting queue
- `READY`: KVC complete or no match, ready to be scheduled
- `PREFILL`: Processing prompt tokens (non-chunked mode)
- `PREFILL_CHUNKED`: Processing prompt in chunks (chunked mode)
- `DECODE`: Generating output tokens one by one
- `COMPLETED`: Request finished

#### 2. **VLLMSchedulerConfig**
Configuration parameters for scheduler:
- `max_num_seqs`: Maximum sequences in a batch (default 256)
- `max_num_batched_tokens`: Maximum tokens per batch (default 2048)
- `chunked_prefill`: Enable chunked prefill (default True)
- `max_model_len`: Maximum sequence length
- `gpu_memory_kvcache_bytes`: Available GPU memory for KV cache

#### 3. **VLLMSchedulerRequest**
Extends `SchedulerRequest` to wrap `LLMRequest`:
- Maintains reference to original request for stats tracking
- Includes `hash_ids` for KVC lookup
- Tracks prefix matching results
- Tracks allocated GPU memory blocks

#### 4. **LLMWorkerVLLMScheduler**
Main worker class with key methods:

- `_vllm_scheduling_loop()`: Continuous scheduling loop
- `_scheduler_step()`: Execute one scheduling iteration
- `_check_new_requests()`: Poll for new requests from router
- `_process_kvc_transitions()`: Handle KVC lookup and fetch
- `_create_batch()`: Form batch with vLLM logic and block allocation
- `_calculate_batch_time()`: Compute batch execution time
- `_update_request_states_and_stats()`: Advance states and update metrics
- `_move_newly_started_requests()`: Move requests from waiting to running queue
- `_move_completed_requests()`: Move requests from running to completed queue
- `_handle_completed_requests()`: Store KVC, release blocks, and output results

## Configuration

### Required Configuration Parameters

Add to your `sim_config/*.json`:

```json
{
  "worker": {
    "worker_params": {
      "worker_local_queue_capacity": 100,
      "periodic_infra_update_time": 1.0,
      "kvcevent_coalesce_time": 0.1
    },
    "vllm_params": {
      "max_num_seqs": 8,
      "max_num_batched_tokens": 2048,
      "chunked_prefill": true,
      "block_size": 16
    },
    "hw": {
      "gpu": "H100",
      "memory_gb": 80,
      "tflops": 989,
      "mem_bw_TBps": 3.35,
      "tp": 1
    }
  }
}
```

### Configuration Parameters Explained

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|--------|
| `max_num_seqs` | Max sequences in a batch | 256 | Batch size limit |
| `max_num_batched_tokens` | Max tokens per batch | 2048 | Token throughput & chunk size |
| `chunked_prefill` | Enable chunked prefill | true | Large prompt handling |
| `block_size` | Tokens per GPU memory block | 16 | Memory granularity |
| `memory_gb` | GPU memory in GB | 16 | Total GPU memory |
| `tp` | Tensor parallelism degree | 1 | Model sharding |

## Usage Example

### Basic Usage

```python
from opal.vllm_worker import LLMWorkerVLLMScheduler
from opal.environment import OpalSimulatorEnvironment
from opal.opal_config import OpalConfig

# Load configuration
config = OpalConfig("path/to/config.json")
env = OpalSimulatorEnvironment(config)

# Create worker with vLLM scheduler
worker = LLMWorkerVLLMScheduler(
    opal_env=env,
    opal_config=config,
    output_req_queue=output_queue,
    infra_update_queue=infra_queue
)

# Worker automatically starts scheduling loop
# Requests are processed as they arrive in worker.worker_local_queue
```

### Integration with Router

Use `LLMWorkerVLLMScheduler` in your router configuration:

```python
# In router.py
from opal.vllm_worker import LLMWorkerVLLMScheduler

worker = LLMWorkerVLLMScheduler(
    opal_env=env,
    opal_config=config,
    output_req_queue=output_queue,
    infra_update_queue=infra_queue
)
```

## Scheduling Algorithm

### Batch Creation Logic

The scheduler follows continuous batching with FIFO ordering:

1. **Phase 1: Running Requests (FIFO Order)**
   - Add all running requests (both prefill and decode)
   - Decode requests generate 1 token per step
   - Prefill requests process up to chunk size tokens
   - Respect token budget and memory block constraints

2. **Phase 2: New READY Requests (FIFO Order)**
   - Add new requests ready for prefill
   - Start with first chunk
   - Allocate GPU memory blocks as needed
   - Stop when constraints are violated

### Resource Constraints

Batch creation respects three constraints:

1. **Max Sequences**: `max_num_seqs` limit
2. **Max Tokens**: `max_num_batched_tokens` limit
3. **GPU Memory Blocks**: Available free blocks

### GPU Memory Block Management

```python
# Calculate blocks needed for new request
if request.phase == READY:
    total_tokens = prefix_match_tokens + tokens_to_add
    total_blocks = (total_tokens + block_size - 1) // block_size
    blocks_needed = total_blocks - allocated_blocks

# Calculate blocks needed for running request
else:
    current_blocks = (kv_cache_tokens + block_size - 1) // block_size
    new_blocks = (kv_cache_tokens + tokens_to_add + block_size - 1) // block_size
    blocks_needed = new_blocks - current_blocks

# Allocate blocks
if blocks_needed <= free_gpu_blocks:
    request.allocated_blocks += blocks_needed
    free_gpu_blocks -= blocks_needed
```

### Timing Calculation

For each batch:

```python
# Calculate prefill times
for each prefill_request:
    if first_chunk:
        # Can use prefix match benefit
        effective_prefix = min(prefix_match_tokens, tokens_to_process)
        prefill_time = gpu_model.get_prefill_latency(
            tokens_to_process, effective_prefix
        )
    else:
        # Subsequent chunks - no prefix match benefit
        prefill_time = gpu_model.get_prefill_latency(
            tokens_to_process, 0
        )
    max_prefill_time = max(max_prefill_time, prefill_time)

# Calculate decode times
for each decode_request:
    current_length = prompt_tokens + decode_tokens_generated
    decode_time = gpu_model.get_decode_latency(current_length, 1)
    max_decode_time = max(max_decode_time, decode_time)

# Batch time (parallel execution)
batch_time = max(max_prefill_time, max_decode_time)
```

## Statistics Collected

### Per-Request Statistics

Tracked in `request.stats`:

- `creation_time`: Request creation timestamp
- `arrival_time`: Arrival at worker
- `start_processing_time`: Start of processing
- `end_kvc_time`: KVC operations complete
- `start_gpu_time`: GPU processing starts
- `completion_time`: Request fully complete
- `token_timestamps`: List of per-token/chunk times
- `kvc_prefix_reuse`: Number of prefix-matched tokens

### Worker-Level Metrics

- `gpu_busy_time`: Total GPU active time
- Queue lengths: `waiting_requests`, `running_requests`, `completed_requests`
- Batch statistics: size, tokens, memory usage
- GPU memory: `total_gpu_blocks`, `free_gpu_blocks`

## Comparison with worker_single_stage.py

| Feature | worker_single_stage.py | vllm_worker.py |
|---------|------------------------|----------------|
| Batching | No batching (1 req at a time) | Continuous batching |
| Scheduling | Simple FIFO | FIFO with resource constraints |
| Prefill | All-at-once | Chunked support |
| KVC Lookup | Optional, commented out | Integrated with state machine |
| Timing | Per-request sequential | Per-batch parallel |
| State Tracking | Basic | Full vLLM state machine |
| Memory Management | Simple | GPU block management |
| Realism | Low | High (matches vLLM v1) |

## Performance Considerations

### Advantages

1. **More Realistic**: Matches actual vLLM v1 behavior
2. **Better Throughput**: Continuous batching improves GPU utilization
3. **Chunked Prefill**: Handles large prompts efficiently
4. **KVC Integration**: Proper prefix matching simulation
5. **Memory Management**: Realistic GPU memory block tracking

### Trade-offs

1. **Complexity**: More complex than simple sequential processing
2. **Memory**: Tracks more state per request
3. **Computation**: Batch formation has overhead

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Key Log Messages

- `"New request X added to waiting queue (phase: WAITING)"`
- `"KVC lookup for request X"`
- `"Request X: prefix match = Y/Z tokens"`
- `"Scheduler step_N: batch=[...], tokens=T/MAX, free_blocks=F/TOTAL"`
- `"Added READY request X: tokens=T, blocks_needed=B"`
- `"Batch execution time: T seconds"`
- `"Request X: prefill complete, transitioning to DECODE"`
- `"Request X: released B blocks after KVC store"`
- `"Request X completed: prefill=P tokens, decode=D tokens, prefix_match=M tokens"`

## Testing

### Unit Tests

```python
# Test batch creation
def test_batch_creation():
    # Create worker and requests
    # Verify batch respects constraints
    # Check FIFO ordering

# Test KVC integration
def test_kvc_lookup():
    # Create request with hash_ids
    # Verify lookup called correctly
    # Check state transitions

# Test timing calculation
def test_batch_timing():
    # Create batch with mixed requests
    # Verify timing = max(prefill, decode)

# Test GPU memory blocks
def test_block_allocation():
    # Create requests
    # Verify blocks allocated/deallocated correctly
    # Check memory constraints enforced
```

### Integration Tests

Run with sample workload:

```bash
cd opal
python -m pytest tests/test_worker2_example.py -v
```

## Module Structure

The `vllm_worker.py` module contains:

1. **Scheduler Data Structures** (lines 50-151)
   - `RequestPhase`: Request state enum
   - `VLLMSchedulerConfig`: Scheduler configuration
   - `SchedulerRequest`: Base request class
   - `BatchMetadata`: Batch information
   - `VLLMSchedulerRequest`: Extended request with LLM integration

2. **Scheduler Helper Functions** (lines 162-178)
   - `schedule_prefill_chunk()`: Calculate tokens to process

3. **Worker Implementation** (lines 186-744)
   - `LLMWorkerVLLMScheduler`: Main worker class
   - All scheduling, batching, and execution logic

## Future Enhancements

Potential improvements:

1. **Preemption**: Support for request preemption
2. **Priority Scheduling**: Request priority levels
3. **Speculative Decoding**: Multi-token generation
4. **Memory Swapping**: CPU offloading simulation
5. **Multi-GPU**: Tensor parallelism support
6. **Advanced Scheduling**: More sophisticated policies

## References

- [vLLM v1 Architecture](https://docs.vllm.ai/en/latest/)
- [Continuous Batching Paper](https://arxiv.org/abs/2309.06180)
- [LMCache Integration](https://github.com/LMCache/LMCache)
- `opal/vllm_worker.py`: Complete implementation
- `opal/worker_single_stage.py`: Original worker implementation

## Support

For questions or issues:
1. Check debug logs
2. Review configuration parameters
3. Compare with worker_single_stage.py behavior
4. Open GitHub issue with reproduction steps

---

**Author**: Opal Simulator Team  
**Date**: 2026-03-11  
**Version**: 2.0.0  
**Branch**: vllm_sched