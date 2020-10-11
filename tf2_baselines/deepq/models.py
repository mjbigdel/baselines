import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations

def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from tf2_baselines.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
        # with tf.variable_scope(scope, reuse=reuse):
        latent = network(input_placeholder)
        if isinstance(latent, tuple):
            if latent[1] is not None:
                raise NotImplementedError("DQN is not compatible with recurrent policies yet")
            latent = latent[0]

        latent = layers.Flatten(latent)

        # with tf.variable_scope("action_value"):
        action_out = latent
        for hidden in hiddens:
            action_out = layers.Dense(action_out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                action_out = layers.BatchNormalization(action_out, center=True, scale=True)
            action_out = activations.relu(action_out)
        action_scores = layers.Dense(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            # with tf.variable_scope("state_value"):
            state_out = latent
            for hidden in hiddens:
                state_out = layers.Dense(state_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    state_out = layers.BatchNormalization(state_out, center=True, scale=True)
                state_out = activations.relu(state_out)
            state_score = layers.Dense(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)

            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out

    return q_func_builder
