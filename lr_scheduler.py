import tensorflow as tf
from matplotlib import pyplot as plt


class CyclicalLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, step_per_epoch, stepsize_factor=8, base_lr=0.001, max_lr=0.006, policy="triangular", t2_gamma=0.5, er_gamma=0.99994):
        super().__init__()

        self.stepsize = stepsize_factor * step_per_epoch
        self.stepsize = tf.cast(self.stepsize, tf.float32)

        self.base_lr = base_lr
        self.base_lr = tf.cast(self.base_lr, tf.float32)

        self.max_lr = max_lr
        self.max_lr = tf.cast(self.max_lr, tf.float32)

        if policy not in ["triangular", "triangular2", "exp_range"]:
            raise ValueError(f'The `policy` argument must be one of "triangular", "triangular2" or "exp_range". Received: {policy}')
        self.policy = policy

        self.t2_gamma = t2_gamma
        self.t2_gamma = tf.cast(self.t2_gamma, tf.float32)

        self.er_gamma = er_gamma
        self.er_gamma = tf.cast(self.er_gamma, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        cycle = tf.math.floor(1 + step / (2 * self.stepsize))
        x = tf.math.abs(step / self.stepsize - 2 * cycle + 1)

        if self.policy=="triangular":
            decay_rate = 1
        elif self.policy=="triangular2":
            decay_rate = self.t2_gamma**(cycle-1)
        else:
            decay_rate = self.er_gamma**step

        return self.base_lr + (self.max_lr - self.base_lr) * tf.math.maximum(0., 1-x) * decay_rate


if __name__ == "__main__":
    policies = ["triangular", "triangular2", "exp_range"]
    colors = ["#006FBC", "#04AF7A", "#EB6120"]
    linestyles = ["-","--", "-."]
    for policy, color, linestyle in zip(policies, colors, linestyles):
        temp_learning_rate_schedule = CyclicalLearningRateSchedule(100, policy=policy)
        plt.plot(
            temp_learning_rate_schedule(tf.range(10000, dtype=tf.float32)),
            color = color,
            linestyle = linestyle,
            label=policy
        )
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.savefig("demo.png", bbox_inches="tight")
