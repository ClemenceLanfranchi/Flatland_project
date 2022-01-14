🚂 This code is based on the official starter kit - Flatland 3
---

The main folders of interest is ```reinforcement_learning``` where we implement our PPO agent and ```reinforcement_learning/env/observations``` our observations.

The Judge implementation was adapted from the code of the JBR team from the Flatland 2020 edition: https://github.com/mahkons/jbr-flatland

In order to run the code, you need to be in the ```reinforcement_learning``` folder. Then, as an example, if you want to train the PPO agent with deadlock priority on 10k steps on the environment configuration n°2 and test it on the same configuration, you can run the following command line:

```
python multi_agent_training_deadlock_priority.py --policy PPO -n 10000 --use_gpu True -t 2 -e 2
```


Main links
---

* [Flatland Challenge](https://www.aicrowd.com/challenges/flatland-3)