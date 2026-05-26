# RAD: Retrieval High-quality Demonstrations to Enhance Decision-making

Code for the paper "RAD: Retrieval High-quality Demonstrations to Enhance Decision-making".



## Running the code
Train and evaluate RAD on Antmaze benchmark tasks
```bash
$ python pipelines/RAD_d4rl_antmaze.py task=antmaze-large-diverse-v2 mode=iql_training
$ python pipelines/RAD_d4rl_antmaze.py task=antmaze-large-diverse-v2 mode=training
$ python pipelines/RAD_d4rl_antmaze.py task=antmaze-large-diverse-v2 mode=inference k=500 minscore1=0.7 minscore2=0.7 
```

