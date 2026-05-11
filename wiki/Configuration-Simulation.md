# Configuring Opal 

TODO: add the default parameters for all, and auto generate documentation. 

### simulation
  * `simulation_time` : run the simulation for the given seconds. If given as -1, then run the simulation until the workload finishes which typically is either (i) total amount of requests have been generated; or (ii) all trace events have been replayed.  
  * `seed`: python random seed 
  * `num_worker`: initial number of llm workers 
    * todo: move it to worker 

### model
  * `name` : the model name 
  * `config_dir` : model config file if you have it locally in some directory. 

 TODO: use the hf tokens, and model loading logic and to make it clean. 

 ### router
  * `enable_scaling`: do dynamic worker scaling up. Deletion is not yet implemented 
  * `max_queue_threshold`: at what max queue size at any worker, we add another worker. 
  * `scale_latency`: how much time it takes to start a new worker 
  * `max_workers`: how many max workers we should scale up to

### workload 
  * `type`: what workload to run, there are three supported type: `UniformReqRate` (uniform sampling), `ExponentialReqRate` (poisson arrival) and `Moonshot` (trace).
  * `request_rate`: mean requests generated per second. 
  * `total_request`: total number of requests to generate. If this is -1, then the simulation will run for simulation_time (virtual) seconds or all entries from a trace are replayed. If this is > 0 with trace replaying, then only first total_request entries are replayed from the trace. 
  * `prompt_size_min` and `prompt_size_max`: min and max prompt sizes. We uniformly sample the prompt size from this range.    
  * `default_prefix_length`: how much of a prefix to sample from the previous request so that there is a hit rate (not yet implemented). 
  * `chunk_size`: in case of traces, what is the chunk_size used in the hashes 
  * `jitter`: when doing ExponentialReqRate sampling how much to deviate from the mean value. 0 = not at all, close to the uniform sampling, 1.0 = as much as allowed, maximum. 
  * `trace_file`: when the workload type is Moonshot, this trace_file is replayed. 

### worker 
 * `single_stage`: we have two worker types, single or dual stage. Single stage worker has kvc fetching and GPU processing as a single stage. Hence every request goes with two stages in a single go. The number of concurrent with the single stage worker is min(`max_kvc_inflight`, `max_gpu_inflight`). When this is false, dual stage worker is used where kvc fetching and gpu processing can be scaled independently. 
 * `max_kvc_inflight`: maximum number of requests that can have kvc fetching request in flight per worker. 
 * `max_gpu_inflight`: maximum number of requests that can have GPU computation going on per worker. 
 * `worker_local_queue_capacity`: each worker's queue capacity on which the router queues the request. 
 * `fixed_gpu_latency`: When I added this parameter only I and God knew what I meant. Now only God knows. 

 in the `hw` section: 
  * `gpu`: Which GPU (name)
  * `tflops`: teraflops capacity of the GPU

in the `inference_params` section: 
 * `model`: which prefill analytical model to use to predict the TTFT times. Supported models are: `moonshot` or `exponential`. The former uses `a` and `b` regression parameter. No need to change them at the moment. The latter uniformly samples the latency between [0, `mean_latency_secs`]. 
     * TODO: to be fixed so that atleast it takes input token sizes as a parameter.
 * `mean_latency_secs`: mean TTFT latency when using the `exponential` analytical model for TTFT latency predictions. 
 * `a` and `b`: regression constants in the `moonshot` analytical model. 


Moonshot analytical model for TTFT:  $L . (a . N . D^2 + b . N^2 .D)$ where L = number of layer, D = model dimensions, N = input prompt length, and (a, b) are environment and infrastructure specific regression parameters. 

### kvc 
This is mean to cover tiering and policies for kvc management. Currently it is work in progress. 
 * `kvc_tiers` : which tiers are enabled for kvc. 1 = GPU, 2 = CPU DRAM, 3 = local NVMe, 4 = distributed, shared storage. 

### storage 
Performance characteristics of storage tier locations. 

  * `backend`: which backend to use? `DFSBackend` or `FixedLatBackend`. DFS is meant to emulate a distributed shared file system. While `FixedLatBackend` has infinite bandwidth and fixed latency for all operations (used in debugging). You can define your own local backend that will be dynamically loaded from `./storage_backend/` folder. 
  * `max_bandwidth`: maximum bandwidth for the `DFSBackend` backend that is shared between all concurrent requests in-flight. 
  * `min_latency`: what is the latency (network delay) in seconds. 
  * `fixed_latency_useconds`: fixed latency in microseconds. 