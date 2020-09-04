import tensorflow as tf
from tensorflow import math as tm
from tensorflow import keras
from tensorflow.keras import layers
import agent_assets.A_hparameters as hp
from datetime import datetime
from os import path, makedirs
import random
import cv2
import numpy as np
from agent_assets.replaybuffer import ReplayBuffer
from agent_assets.mousemodel import QModel
import pickle
from tqdm import tqdm

#leave memory space for opencl
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

keras.backend.clear_session()

class Player():
    """A agent class which plays the game and learn.

    Algorithms
    ----------
    e-greedy
    Prioritized sampling
    Double DQN
    """
    def __init__(self, observation_space, action_space, model_f, tqdm, m_dir=None,
                 log_name=None, start_step=0, start_round=0,load_buffer=False):
        """
        Parameters
        ----------
        observation_space : gym.Space
            Observation space of the environment.
        action_space : gym.Space
            Action space of the environment. Current agent expects only
            a discrete action space.
        model_f
            A function to build the Q-model. 
            It should take obeservation space and action space as inputs.
            It should not compile the model.
        tqdm : tqdm.tqdm
            A tqdm object to update every step.
        m_dir : str
            A model directory to load the model if there's a model to load
        log_name : str
            A name for log
        start_step : int
            Total step starts from start_step
        start_round : int
            Total round starts from start_round
        load_buffer : bool
            Whether to load the buffer from the model directory
        """
        # model : The actual training model
        # t_model : Fixed target model
        print('Model directory : {}'.format(m_dir))
        print('Log name : {}'.format(log_name))
        print('Starting from step {}'.format(start_step))
        print('Starting from round {}'.format(start_round))
        print('Load buffer? {}'.format(load_buffer))
        self.tqdm = tqdm
        self.action_n = action_space.n
        self.observation_space = observation_space
        #Inputs
        if m_dir is None :
            self.model = model_f(observation_space, action_space)
            # compile models
            optimizer = keras.optimizers.Adam(learning_rate=self._lr)
            self.model.compile(optimizer=optimizer)
        else:
            self.model = keras.models.load_model(m_dir)
            print('model loaded')
        self.t_model = keras.models.clone_model(self.model)
        self.t_model.set_weights(self.model.get_weights())
        self.model.summary()

        # Buffers
        if load_buffer:
            print('loading buffers...')
            with open(path.join(m_dir,'buffer.bin'),'rb') as f :
                self.buffer = pickle.load(f)
            print('loaded : {} filled in buffer'.format(self.buffer.num_in_buffer))
            print('Current buffer index : {}'.format(self.buffer.next_idx))
        else :
            self.buffer = ReplayBuffer(hp.Buffer_size, self.observation_space)

        # File writer for tensorboard
        if log_name is None :
            self.log_name = datetime.now().strftime('%m_%d_%H_%M_%S')
        else:
            self.log_name = log_name
        self.file_writer = tf.summary.create_file_writer(path.join('log',
                                                         self.log_name))
        self.file_writer.set_as_default()
        print('Writing logs at '+ self.log_name)

        # Scalars
        self.start_training = False
        self.total_steps = start_step
        self.current_steps = 1
        # self.score = 0
        self.rounds = start_round
        self.cumreward = 0
        
        # Savefile folder directory
        if m_dir is None :
            self.save_dir = path.join('savefiles',
                            self.log_name)
            self.save_count = 0
        else:
            self.save_dir, self.save_count = path.split(m_dir)
            self.save_count = int(self.save_count)

    def _lr(self):
        # if self.total_steps > hp.lr_nsteps:
        #     return hp.lr_end
        # else:
        return hp.lr_start*\
            ((hp.lr_end/hp.lr_start)**(self.total_steps/hp.lr_nsteps))

    @property
    def epsilon(self):
        if self.total_steps > hp.epsilon_nstep :
            return hp.epsilon_min
        else:
            return hp.epsilon-(hp.epsilon-hp.epsilon_min)*\
                (self.total_steps/hp.epsilon_nstep)

    @tf.function
    def pre_processing(self, observation:dict):
        """
        Preprocess input data
        """
        processed_obs = {}
        for name, obs in observation.items():
            # If only one observation is given, reshape to [1,...]
            if len(observation[name].shape)==\
                len(self.observation_space[name].shape):
                processed_obs[name] = tf.cast(obs[np.newaxis,...],tf.float32)/255
            else :
                processed_obs[name] = tf.cast(obs, tf.float32)/255
        return processed_obs

    def choose_action(self, q):
        """
        Policy part; uses e-greedy
        """
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_n)
        else :
            m = np.max(q[0])
            indices = [i for i, x in enumerate(q[0]) if x==m]
            return random.choice(indices)

    def act(self, before_state, record=True):
        q = self._tf_q(before_state)
        action = self.choose_action(q.numpy())
        if record:
            tf.summary.scalar('maxQ', tf.math.reduce_max(q), self.total_steps)
        return action
        

    @tf.function
    def _tf_q(self, before_state):
        processed_state = self.pre_processing(before_state)
        q = self.model(processed_state, training=False)
        return q

    @tf.function
    def train_step(self, o, r, d, a, sp_batch, total_step, weights):
        # next Q values from t_model to evaluate
        target_q = self.t_model(sp_batch, training=False)
        # next Q values from model to select action (Double DQN)
        another_q = self.model(sp_batch, training=False)
        idx = tf.math.argmax(another_q, axis=-1)
        # Then retrieve the q value from target network
        selected_q = tf.gather(target_q, idx, batch_dims=1)

        q_samp = r + tf.cast(tm.logical_not(d), tf.float32) * \
                     hp.Q_discount * \
                     selected_q
        mask = tf.one_hot(a, self.action_n, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q = self.model(o, training=True)
            q_sa = tf.math.reduce_sum(q*mask, axis=1)
            unweighted_loss = tf.math.square(q_samp - q_sa)
            loss = tf.math.reduce_mean(weights * unweighted_loss)
            tf.summary.scalar('Loss', loss, total_step)

        priority = (tf.math.abs(q_samp - q_sa) + hp.Buf.epsilon)**hp.Buf.alpha
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return priority


    def step(self, before, action, reward, done, info):
        self.buffer.store_step(before, action, reward, done)
        self.tqdm.update()
        # Record here, so that it won't record when evaluating
        # if info['ate_apple']:
        #     self.score += 1
        self.cumreward += reward
        tf.summary.scalar('lr', self._lr(),self.total_steps)
        if done:
            # tf.summary.scalar('Score', self.score, self.rounds)
            tf.summary.scalar('Reward', self.cumreward, self.rounds)
            # tf.summary.scalar('Score_step', self.score, self.total_steps)
            tf.summary.scalar('Reward_step', self.cumreward, self.total_steps)
            info_dict = {
                'Round':self.rounds,
                'Steps':self.current_steps,
                # 'Score':self.score,
                'Reward':self.cumreward,
            }
            self.tqdm.set_postfix(info_dict)
            # self.score = 0
            self.current_steps = 0
            self.cumreward = 0
            self.rounds += 1

        if self.total_steps % hp.histogram == 0:
            for var in self.model.trainable_weights:
                tf.summary.histogram(var.name, var, step=self.total_steps)

        if self.buffer.num_in_buffer < hp.Learn_start :
            self.tqdm.set_description(
                f'filling buffer'
                f'{self.buffer.num_in_buffer}/{hp.Learn_start}'
            )

        else :
            if self.start_training == False:
                self.tqdm.set_description()
                self.start_training = True
            s_batch, a_batch, r_batch, d_batch, sp_batch, indices, weights = \
                                    self.buffer.sample(hp.Batch_size)
            s_batch = self.pre_processing(s_batch)
            sp_batch = self.pre_processing(sp_batch)
            tf_total_steps = tf.constant(self.total_steps, dtype=tf.int64)
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)

            data = (
                s_batch,
                r_batch, 
                d_batch, 
                a_batch, 
                sp_batch, 
                tf_total_steps,
                weights,
            )

            new_priors = self.train_step(*data).numpy()
            self.buffer.update_prior_batch(indices, new_priors)

            if not self.total_steps % hp.Target_update:
                self.t_model.set_weights(self.model.get_weights())

        self.total_steps += 1
        self.current_steps += 1

    def save_model(self):
        """
        Saves the model and return next save file number
        """
        self.save_count += 1
        if not path.exists(self.save_dir):
            makedirs(self.save_dir)
        self.model_dir = path.join(self.save_dir, str(self.save_count))
        self.model.save(self.model_dir)
        with open(path.join(self.model_dir,'buffer.bin'),'wb') as f :
            pickle.dump(self.buffer, f)

        return self.save_count

