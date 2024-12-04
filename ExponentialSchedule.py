import math

class ExponentialSchedule:
    def __init__(self, value_from, value_to, num_steps):
        """
        $value(t) = a \exp (b t)$
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        self.a = self.value_from
        self.b = math.log(self.value_to / self.value_from) / (self.num_steps - 1)

    def value(self, step) -> float:
        if step <= 0:
            return self.value_from
        elif step >= self.num_steps - 1:
            return self.value_to
        else:
            return self.a * math.exp(self.b * step)