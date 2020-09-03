import tensorflow as tf
from tensorflow import keras
import tensorflow.math as tm
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

#leave memory space for opencl
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

keras.backend.clear_session()
if len(gpus) > 0 :
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

class Player():
    def __init__(self, observation_space, action_space, m_dir=None,
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
        self.action_n = action_space.n
        self.observation_space = observation_space
        #Inputs
        if m_dir is None :
            left_input = keras.Input(observation_space['Left'].shape,
                                    name='Left')
            right_input = keras.Input(observation_space['Right'].shape,
                                    name='Right')
            # Spare eye model for later use
            left_input_shape = observation_space['Left'].shape
            right_input_shape = observation_space['Right'].shape
            left_eye_model = self.eye_model(left_input_shape,'Left')
            right_eye_model = self.eye_model(right_input_shape,'Right')
            # Get outputs of the model
            left_encoded = left_eye_model(left_input)
            right_encoded = right_eye_model(right_input)
            # Concatenate both eye's inputs
            concat = layers.Concatenate()([left_encoded,right_encoded])
            # concat = layers.Concatenate()([left_input,right_input])
            outputs = self.brain_layers(concat)
            # x = layers.Flatten()(concat)
            # x = layers.Dense(self.action_n)(x)
            # outputs = layers.Activation('linear',dtype='float32')(x)
            # Build models
            self.model = keras.Model(inputs=[left_input, right_input],
                                outputs=outputs)
            self.optimizer = keras.optimizers.Adam()
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer,
                                                        loss_scale='dynamic')
        else:
            self.model = keras.models.load_model(m_dir)
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
                            datetime.now().strftime('%m_%d_%H_%M_%S'))
            self.save_count = 0
        else:
            self.save_dir, self.save_count = path.split(m_dir)
            self.save_count = int(self.save_count)

    def eye_model(self, input_shape, left_or_right):
        """
        Return an eye model
        """
        inputs = layers.Input(input_shape)
        x = layers.Reshape((inputs.shape[1],
                            inputs.shape[2]*inputs.shape[3]))(inputs)
        x = layers.Conv1D(64, 7, strides=2, activation='relu')(x)
        x = layers.Conv1D(32, 5, strides=2, activation='relu')(x)
        outputs = layers.Conv1D(16, 3, strides=2, activation='relu')(x)
        return keras.Model(inputs=inputs, outputs=outputs, 
                    name=left_or_right+'_eye')

    def brain_layers(self, x):
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(self.action_n)(x)
        outputs = layers.Activation('linear',dtype='float32')(x)
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
        if len(observation['Right'].shape)==\
            len(self.observation_space['Right'].shape):
            for name, obs in observation.items():
                processed_obs[name] = tf.cast(obs[np.newaxis,:,:,:],tf.float32)/255
        else :
            for name, obs in observation.items():
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

    def act(self, before_state, training:bool):
        if training :
            self.buf_idx = self.buffer.store_obs(before_state)
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
    def train_step(self, o, r, d, a, sp_batch, total_step):
        target_q = self.t_model(sp_batch, training=False)
        q_samp = r + tf.cast(tm.logical_not(d), tf.float32) * \
                     hp.Q_discount * \
                     tm.reduce_max(target_q, axis=1)
        mask = tf.one_hot(a, self.action_n, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q = self.model(o, training=True)
            q_sa = tf.math.reduce_sum(q*mask, axis=1)
            loss = keras.losses.MSE(q_samp, q_sa)
            tf.summary.scalar('Loss', loss, total_step)
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        trainable_vars = self.model.trainable_variables
        scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    def step(self, action, reward, done, info):
        self.buffer.store_effect(self.buf_idx, action, reward, done)
        # Record here, so that it won't record when evaluating
        if info['ate_apple']:
            self.score += 1
        self.cumreward += reward
        if done:
            tf.summary.scalar('Score', self.score, self.rounds)
            tf.summary.scalar('Reward', self.cumreward, self.rounds)
            tf.summary.scalar('Score_step', self.score, self.total_steps)
            tf.summary.scalar('Reward_step', self.cumreward, self.total_steps)
            print('\n{0} round({1} steps) || Score: {2} | Reward: {3:.1f}'.format(
                self.rounds, self.current_steps, self.score, self.cumreward
            ))
            self.score = 0
            self.current_steps = 0
            self.cumreward = 0
            self.rounds += 1

        if self.buffer.num_in_buffer < hp.Learn_start :
            if self.buffer.num_in_buffer % 100 == 0:
                print('filling buffer {0}/{1}'.format(
                        self.buffer.num_in_buffer, hp.Learn_start))

        else :
            s_batch, a_batch, r_batch, d_batch, sp_batch = self.buffer.sample(
                                                                hp.Batch_size)
            s_batch = self.pre_processing(s_batch)
            sp_batch = self.pre_processing(sp_batch)
            tf_total_steps = tf.constant(self.total_steps, dtype=tf.int64)
            data = (s_batch, r_batch, d_batch, a_batch, sp_batch, tf_total_steps)
            self.train_step(*data)

            if not self.total_steps % hp.Target_update:
                self.t_model.set_weights(self.model.get_weights())

        self.total_steps += 1
        self.current_steps += 1

    def save_model(self):
        """
        Return next save file number
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
            a = self.act(o, training=False)
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

