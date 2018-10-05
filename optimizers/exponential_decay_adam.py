from models import *
from tools.lazy_decorator import *


class ExponentialDecayAdam:
    def __init__(self, model, params: dict):
        # Variable to increment by one after the variables have been updated by the optimizers
        # this helps to keep track of the progress of the training (e.g. adapting learning rate and decay)
        self.global_step = tf.Variable(0, name='global_step_counter')
        self._model = model
        self._base_learning_rate = params['initial_lr']
        self._decay_step = params['decay_step']
        self._decay_rate = params['decay_rate']

        self._gradient_clipping = params.get('gradient_clipping')

    @lazy_property
    def learning_rate(self):
        learning_rate = tf.cond(self.global_step * self._model.batch_generator.batch_size < self._decay_step,
                                lambda: tf.constant(self._base_learning_rate),
                                lambda: tf.train.exponential_decay(
                                    self._base_learning_rate,  # Base learning rate.
                                    self.global_step * self._model.batch_generator.batch_size - self._decay_step,
                                    self._decay_step // 2,  # Decay step.
                                    self._decay_rate,  # Decay rate.
                                    staircase=False))
        return learning_rate

    @lazy_function
    def register_summary(self):
        tf.summary.scalar('global_step', self.global_step)
        tf.summary.scalar('learning_rate', self.learning_rate)

    @lazy_property
    def optimize(self):
        if self._gradient_clipping is not None:
            # Clipping gradients - check if gradients explode
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self._model.loss, tf.trainable_variables())
            trainables = tf.trainable_variables()
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self._gradient_clipping)

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                return optimizer.apply_gradients(zip(clipped_gradients, trainables), global_step=self.global_step)
        else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            return optimizer.minimize(self._model.loss, global_step=self.global_step)
