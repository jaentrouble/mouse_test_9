import tensorflow as tf
from tensorflow import math as tm
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
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
# if len(gpus)>0:
#     policy = mixed_precision.Policy('mixed_float16')
#     print('policy = mixed_float16')
# else : 
#     policy = mixed_precision.Policy('float32')
# mixed_precision.set_policy(policy)

class Player():
    def __init__(self, observation_space, action_space, tqdm, m_dir=None,
                 log_name=None, start_step=0, start_round=0,load_buffer=False):
        """
        model : The actual training model
        t_model : Fixed target model
        """
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
            obs_input = keras.Input(observation_space['obs'].shape,
                                    name='obs')
            outputs = self.brain_layers(obs_input)
            outputs = layers.Activation('linear',dtype='float32')(outputs)
            # Build models
            self.model = keras.Model(inputs=[obs_input],
                                outputs=outputs)
            lr = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01, 
                decay_steps=1000,
                decay_rate=0.95
            )
            optimizer = keras.optimizers.Adam(learning_rate=lr)
            optimizer = mixed_precision.LossScaleOptimizer(optimizer,
                                                        loss_scale='dynamic')
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
        self.score = 0
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

    # def eye_model(self, input_shape, left_or_right):
    #     """
    #     Return an eye model
    #     """
    #     inputs = layers.Input(input_shape)
    #     x = layers.Reshape((inputs.shape[1],
    #                         inputs.shape[2]*inputs.shape[3]))(inputs)
    #     x = layers.Conv1D(64, 7, strides=1, activation='relu')(x)
    #     x = layers.Conv1D(128, 5, strides=2, activation='relu')(x)
    #     x = layers.Conv1D(192, 3, strides=2, activation='relu')(x)
    #     outputs = layers.Conv1D(256, 3, strides=2, activation='relu')(x)
    #     return keras.Model(inputs=inputs, outputs=outputs, 
    #                 name=left_or_right+'_eye')

    def brain_layers(self, x):
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(self.action_n)(x)
        return outputs

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
        if len(observation['obs'].shape)==\
            len(self.observation_space['obs'].shape):
            for name, obs in observation.items():
                processed_obs[name] = tf.cast(obs[np.newaxis,...],tf.float32)
        else :
            for name, obs in observation.items():
                processed_obs[name] = tf.cast(obs, tf.float32)
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

    def act(self, before_state):
        q = self._tf_q(before_state)
        action = self.choose_action(q.numpy())
        tf.summary.scalar('maxQ', tf.math.reduce_max(q), self.total_steps)
        return action
        

    @tf.function
    def _tf_q(self, before_state):
        processed_state = self.pre_processing(before_state)
        q = self.model(processed_state, training=False)
        return q

    @tf.function
    def train_step(self, o, r, d, a, sp_batch, total_step, weights):
        target_q = self.t_model(sp_batch, training=False)
        q_samp = r + tf.cast(tm.logical_not(d), tf.float32) * \
                     hp.Q_discount * \
                     tm.reduce_max(target_q, axis=1)
        mask = tf.one_hot(a, self.action_n, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q = self.model(o, training=True)
            q_sa = tf.math.reduce_sum(q*mask, axis=1)
            # loss = keras.losses.MSE(q_samp, q_sa)
            unweighted_loss = tf.math.square(q_samp - q_sa)
            loss = tf.math.reduce_mean(weights * unweighted_loss)
            tf.summary.scalar('Loss', loss, total_step)
            scaled_loss = self.model.optimizer.get_scaled_loss(loss)

        priority = (tf.math.abs(q_samp - q_sa) + hp.Buf.epsilon)**hp.Buf.alpha
        trainable_vars = self.model.trainable_variables
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        gradients = self.model.optimizer.get_unscaled_gradients(scaled_gradients)
        self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return priority


    def step(self, before, action, reward, done, info):
        self.buffer.store_step(before, action, reward, done)
        self.tqdm.update()
        # Record here, so that it won't record when evaluating
        self.cumreward += reward
        if done:
            tf.summary.scalar('Reward', self.cumreward, self.rounds)
            tf.summary.scalar('Reward_step', self.cumreward, self.total_steps)
            info_dict = {
                'Round':self.rounds,
                'Steps':self.current_steps,
                'Reward':self.cumreward,
            }
            self.tqdm.set_postfix(info_dict)
            self.score = 0
            self.current_steps = 0
            self.cumreward = 0
            self.rounds += 1

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

    def evaluate(self, env, video_type):
        print('Evaluating...')
        done = False
        video_dir = path.join(self.model_dir, 'eval.{}'.format(video_type))
        eye_dir = path.join(self.model_dir, 'eval_eye.{}'.format(video_type))
        score_dir = path.join(self.model_dir, 'score.txt')
        if 'avi' in video_type :
            fcc = 'DIVX'
        elif 'mp4' in video_type:
            fcc = 'mp4v'
        else:
            raise TypeError('Wrong videotype')
        fourcc = cv2.VideoWriter_fourcc(*fcc)
        # Becareful : cv2 order of image size is (width, height)
        eye_out = cv2.VideoWriter(eye_dir, fourcc, 10, (205*5,50))
        out = cv2.VideoWriter(video_dir, fourcc, 10, env.image_size)
        eye_bar = np.ones((5,3),dtype=np.uint8)*np.array([255,255,0],dtype=np.uint8)
        o = env.reset()
        score = 0
        loop = 0
        while not done :
            loop += 1
            if not loop % 100:
                print('Eval : {}step passed'.format(loop))
            a = self.act(o)
            o,r,done,i = env.step(a)
            if i['ate_apple']:
                score += 1
            #eye recording
            rt_eye = np.flip(o['Right'][:,-1,:],axis=0)
            lt_eye = o['Left'][:,-1,:]
            eye_img = np.concatenate((lt_eye,eye_bar,rt_eye))
            eye_img = np.broadcast_to(eye_img.reshape((1,205,1,3)),(50,205,5,3))
            eye_img = eye_img.reshape(50,205*5,3)
            eye_out.write(np.flip(eye_img, axis=-1))
            # This will turn image 90 degrees, but it does not make any difference,
            # so keep it this way to save computations
            out.write(np.flip(env.render('rgb'), axis=-1))
        out.release()
        eye_out.release()
        with open(score_dir, 'w') as f:
            f.write(str(score))
        print('Eval finished')
        return score

