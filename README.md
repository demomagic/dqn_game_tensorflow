# dqn_game_tensorflow
playing Atari game with Deep Q Learning (DQN &amp; DDQN) in tensorflow
# Requirement main
	python3.6

	gym[atari]

	opencv-python

	tensorflow-1.10
# Usage
For DQN train:

	python game_main.py --episode=15000 --env_name=MsPacman-v0 --model_type=dqn --train=True --load_network=False

For DDQN train:

	--model_type=ddqn
	
For test model:
	
	--train=False --load_network=True
# Result
The saved_network and summary file is saved with 5 hours of training data

![game_test](https://github.com/demomagic/dqn_game_tensorflow/blob/master/img/img.gif)
# Summary
	tensorboard --logdir=./summary/MsPacman-v0/dqn
	tensorboard --logdir=./summary/MsPacman-v0/ddqn

For DQN summary:

![dqn_summary](https://github.com/demomagic/dqn_game_tensorflow/blob/master/img/dqn_summary.png)

For DDQN summary:

![ddqn_summary](https://github.com/demomagic/dqn_game_tensorflow/blob/master/img/ddqn_summary.png)
# Reference
[DQN in Keras + TensorFlow + OpenAI Gym](https://github.com/tokb23/dqn)
