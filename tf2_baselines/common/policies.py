import tensorflow as tf
from tf2_baselines.common import tf_util
from tf2_baselines.a2c.utils import fc, flatten
from tf2_baselines.common.distributions import make_pdtype
from tf2_baselines.common.input import observation_placeholder, encode_observation
from tf2_baselines.common.tf_util import adjust_shape
from tf2_baselines.common.mpi_running_mean_std import RunningMeanStd
from tf2_baselines.common.models import get_network_builder

import gym


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """
    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None, **tensors):  # sess=None,
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder comes from keras.layers.Input in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent
        print('vf_latent.shape = ', vf_latent)
        print('latent.shape = ', latent)

        # vf_latent = flatten(vf_latent)
        # print('vf_latent.shape = ', vf_latent)
        # latent = flatten(latent)
        # print('latent.shape = ', latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        # self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]

    # @tf.function
    def _evaluate(self, variables, observation, **extra_feed):
        # sess = self.sess
        # feed_dict = {self.X: adjust_shape(self.X, observation)}

        # x = adjust_shape(self.X, observation)
        print('self.X  = ', self.X)
        self.X = adjust_shape(self.X, observation)
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    # inpt_name = tf.keras.layers.Input(name=inpt_name, tensor=adjust_shape(inpt, data))
                    # feed_dict[inpt] = adjust_shape(inpt, data)
                    inpt_name = tf.keras.layers.Input(name=inpt_name,  shape=data.shape)
                    print(inpt_name)


        # return sess.run(variables, feed_dict)
        return variables

    # @tf.function
    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        # if tf.size(state) == 0:
        #     state = None
        return a, v, state, neglogp

    # @tf.function
    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=None)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=None)


def build_policy(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, observ_placeholder=None):  #  sess=None,
        ob_space = env.observation_space
        print('ob_space = ', ob_space)
        print('observ_placeholder = ', observ_placeholder)

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)
        # X = tf.keras.layers.Input(shape=(batch_size,) + ob_space.shape, dtype=dtype, name=name)
        print('X.shape', X.shape)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        print('encoded_x.shape', encoded_x.shape)

        encoded_x = encode_observation(ob_space, encoded_x)

        print('encoded_x.shape', encoded_x.shape)

        # with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
        policy_latent = policy_network(encoded_x)
        print('policy_latent= ', policy_latent)
        if isinstance(policy_latent, tuple):
            policy_latent, recurrent_tensors = policy_latent

            if recurrent_tensors is not None:
                # recurrent architecture, need a few more steps
                nenv = nbatch // nsteps
                assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            # with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
            # TODO recurrent architectures are not supported with value_network=copy yet
            vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,  # it's a placeholder returned from keras.layers.Input
            latent=policy_latent,
            vf_latent=vf_latent,
            # sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

