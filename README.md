# mouse_test_9

 DDQN

## Purpose

1. Try Double DQN and Dueling DQN. See if they help.

## Lessons from last experiment

1. Even with the exact same condition, results may vary.

2. Always think of falling into local minima.

## TODO in this experiment

1. Implement Double DQN

2. Implement Dueling DQN

## Plan

1. Try Double DQN first.

2. Add Dueling DQN to see if it makes any difference.

## Other changes

1. Changed some code so that the Player class is more generally usable.

2. Now both CartPoleTest and SanityCheck uses the same Player class from Agent.

## Tests

### Sanity check with Cartpole (Simple DQN)

![image](https://user-images.githubusercontent.com/45917844/92990374-03801500-f517-11ea-9a39-12bcae624410.png)

![image](https://user-images.githubusercontent.com/45917844/92990379-0bd85000-f517-11ea-9acc-5e315be55c21.png)

1. The model was unable to learn. The maxQ value was not fluctuating, and was converging to some point.

2. Seems like it suffers from maximization bias (only choses one action)

3. Try Double DQN

### Cartpole with Double DQN

![image](https://user-images.githubusercontent.com/45917844/92990930-08df5e80-f51b-11ea-8b4c-9ad4eb527843.png)

![image](https://user-images.githubusercontent.com/45917844/92990914-f9f8ac00-f51a-11ea-8e8e-89fc901594f9.png)

1. Same thing happens.

2. Looked at the codes when the model was able to learn cart-pole (Only once). Discovered that there was a learning rate bug that instead of exponential decaying, it was exponential growing. Fixed the bug and tested with a few learning rates.

### Cartpole with Double DQN and modified lr

![image](https://user-images.githubusercontent.com/45917844/92990997-87d49700-f51b-11ea-8e2f-268bbfdcfa4c.png)

![image](https://user-images.githubusercontent.com/45917844/92991001-8e630e80-f51b-11ea-818b-3c11200af394.png)

1. Working learning rate : 1e-5 start, 1e-10 end for 500k steps, and 1e-10 stay

>![image](https://user-images.githubusercontent.com/45917844/92991105-2e209c80-f51c-11ea-83de-d22047c2dae8.png)

2. High learning rate was the problem. Now try original test with low learning rate.

### Double DQN with low lr

![image](https://user-images.githubusercontent.com/45917844/92991131-79d34600-f51c-11ea-8b27-d80efacb3b21.png)

![image](https://user-images.githubusercontent.com/45917844/92991138-8eafd980-f51c-11ea-99a6-8d659fc66c86.png)

![image](https://user-images.githubusercontent.com/45917844/92991144-966f7e00-f51c-11ea-8284-07730f729c9f.png)

1. Learning rate : 1e-5 start, decay 1e-5 per 1M steps

2. epsilon : 1 start 0.1 end for (total_steps // 2) steps, 0.1 stay

3. Learns well.

## Discussion

1. Learning rate matters.

2. If maxQ is not fluctuating much, it's may due to high learning rate.

## TODO

1. Try some longer steps and see what is the converging reward. (To use as a baseline)

2. Move onto Actor-Critic algorithms