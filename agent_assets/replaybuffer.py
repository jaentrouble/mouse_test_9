import numpy as np
import random
import agent_assets.A_hparameters as hp

class ReplayBuffer():
    """A replay buffer with Prioritized sampling.
    
    """

    def __init__(self, buffer_size, observation_space:dict):
        """
        Stores observation space in uint8 dtype
        """
        self.size = buffer_size
        self.obs_buffer = {}
        for name, space in observation_space.spaces.items():
            shape = space.shape
            self.obs_buffer[name] = np.zeros([buffer_size]+list(shape),
                                            dtype=space.dtype)
        self.action_buffer = np.zeros(buffer_size, dtype=np.int32)
        self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.done_buffer = np.zeros(buffer_size, dtype=np.bool)
        self.prior_tree = np.zeros(2*self.size - 1)
        self.max_prior = 1
        self.next_idx = 0
        self.num_in_buffer = 0

    def store_step(self, observation:dict, action, reward, done) :
        """Give the original observation in uint8

        Parameters
        ----------
        observation : dict
            s_t
        action
            a_t
        reward
            r_t
        done
            d_t
        """
        for name, obs in observation.items() :
            self.obs_buffer[name][self.next_idx] = obs
        self.action_buffer[self.next_idx] = action
        self.reward_buffer[self.next_idx] = reward
        self.done_buffer[self.next_idx] = done
        self._update_prior(self.next_idx, self.max_prior)
        self.next_idx = (self.next_idx +1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer +1)

    def _to_tree_idx(self, idx):
        """Convert Data idx into tree idx"""
        return idx + self.size - 1

    def _to_data_idx(self, idx):
        """Convert Tree idx into data idx"""
        return idx - self.size + 1

    def _update_prior(self, idx, new_priority):
        """Update priority and propagate to parent nodes
        
        Parameters
        ----------
        idx : int
            DATA index
        new_priority
            New priority to add
        """
        if new_priority > self.max_prior:
            self.max_prior = new_priority
        tree_idx = self._to_tree_idx(idx)
        delta = new_priority - self.prior_tree[tree_idx]

        parent_idx = tree_idx
        while parent_idx >= 0 :
            self.prior_tree[parent_idx] += delta

            parent_idx = (parent_idx -1) //2

    def _get_idx_from_s(self, s):
        """Get tree index from sampled s value"""
        idx = 0
        left = 1
        right = 2
        while left < len(self.prior_tree):
            if s <= self.prior_tree[left]:
                idx = left
            else:
                idx = right
                s -= self.prior_tree[left]
            left = 2 * idx + 1
            right = left + 1

        return idx

    def _sample_indices(self, batch_size):
        total_sum = self.prior_tree[0]
        step_size = total_sum / batch_size
        sampled_s = np.linspace(0, total_sum, batch_size, endpoint=False) + \
                    np.random.random_sample(batch_size)*step_size
        sampled_i = np.array([self._get_idx_from_s(s) for s in sampled_s])
        return sampled_i

    def sample(self, batch_size):
        indices = self._to_data_idx(self._sample_indices(batch_size))
        next_indices = (indices + 1) % self.size
        obs_sample = {}
        next_obs_sample = {}
        for name, buf in self.obs_buffer.items():
            obs_sample[name] = buf[indices]
            next_obs_sample[name] = buf[next_indices]
        action_sample = self.action_buffer[indices]
        reward_sample = self.reward_buffer[indices]
        done_sample = self.done_buffer[indices]

        # alpha is already implemented in Agent's train step
        IS_weights = (self.num_in_buffer*\
                    self.prior_tree[self._to_tree_idx(indices)]/\
                    self.prior_tree[0])**(-hp.Buf.beta)
        IS_weights = IS_weights / np.max(IS_weights)

        return (obs_sample, 
                action_sample, 
                reward_sample, 
                done_sample,
                next_obs_sample,
                indices,
                IS_weights,
                )

    def update_prior_batch(self, indices, new_priors):
        """Update batch of priorities"""
        for i, p in zip(indices, new_priors):
            self._update_prior(i, p)