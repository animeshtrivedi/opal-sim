# Opal Discreete Event-Based simulation environment for KVCache reuse workloads

## High-level idea
The Opal simulation environment can generate LLM request workloads or replay LLM traces. It simulates all KVCache performance-critical events and estimates request-level TTFT and TPS, as well as the platform-level TTFT latency percentiles and token throughput vs. time.

## Architecture
The current inference platform components are simulated:
  - Workload generators:
    - Poisson distribution with a configurable arrival rate (lambda).
  - Request routing gateway (simulates llm-d). The current routing policy is to load balance (send requests to the least loaded worker).
  - LLM worker (simulates vLLM). There can be multiple vLLM instances serving requests in parallel.
  - Distributed storage system (simulates a Scale deployment). Implements basic concepts as request queueing, fair BW sharing between requests, async()/sync() interfaces.

## Coding style
### Archiecture
 - The simulation is build on top of SimPy. SimPy is a discrete-event simulation Python library in Python used to model event passing and queueing effects of complex systems.
 - SimPy introduces the concepts of: virtual time, processes, events, and shared resources (e.g., queues, capacities, containers). Instead of tracking every simulated time moment, SimPy "time jumps" between scheduled events, making the simulation a lot more efficient.
 - Processes and concurency in general are based on Python generators. It is very helpful to understand the precise generator mechanics in Python 3.
### Components
 - The code is currently organized into separated units (LLM request routing, LLM worker, distributed storage system, workload generators, parsing/plotting, requests, etc.).
### Suggestions
 - To ease development, consider adding a simple test case / example in each file. Ideally, each Python file can be executed and tested individually. The tests also work as examples or documentation.
 - Each simulation should be in a separate file that initializes the simulation environment and configures each unit. In the future, we could add support for configuration files.


# Semantics to be documented in the code 

## Worker 
has any arbitrary number of I/O and GPU slots for processing. They can be in a single stage, or decoupled. 
For example you can have 1 kvc I/O and 1 GPU slot that means they will be pipelines. 
In case when they are single stage, then there is only "X" number of minimum request in processing. X is defined as the minimum of kvc I/O and GPU slots. 