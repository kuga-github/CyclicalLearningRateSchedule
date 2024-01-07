import tensorflow as tf

# triangular2
class CyclicalLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, steps_per_epoch, stepsize_factor=8, base_lr=0.001, max_lr=0.006):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_epoch = tf.cast(self.steps_per_epoch, tf.float32)

        self.stepsize = stepsize_factor * self.steps_per_epoch

        self.base_lr = base_lr
        self.base_lr = tf.cast(self.base_lr, tf.float32)

        self.max_lr = max_lr
        self.max_lr = tf.cast(self.max_lr, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        cycle = tf.math.floor(1 + step / (2 * self.stepsize))
        lamb = 2. ** (1 - cycle)
        x = tf.math.abs(step / self.stepsize - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * tf.math.maximum(0., 1-x) * lamb
        return lr


# debug
if __name__ == "__main__":
    temp_learning_rate_schedule = CyclicalLRSchedule(100)
    plt.plot(temp_learning_rate_schedule(tf.range(10000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
