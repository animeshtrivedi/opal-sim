# How to calculate a, b 

## Online mode 
In this case, the measured and predicted values are generated in-place, in a single run and all values are calculated. 


Example run: 
```shell
```

## Offline mode
In this case you may already have a run of values from another machine and you just want to solve the regression 

Example run (for IBM Research Blog values): 
```shell
python ./offline_calc_a_b.py -m meta-llama/Llama-3.1-70B -gpu H100 --measurements=./ibm-research-blog_data.json  -tp=4
```