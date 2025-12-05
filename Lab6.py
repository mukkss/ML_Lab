import random


def uniform_random():
    return random.random()


def f(x):
    return x**2


N = 10
samples = [uniform_random() for _ in range(N)]
values = [f(x) for x in samples]
estimate = sum(values) / N


print("random sample:", samples)
print("function values:", values)
print("estimate:", estimate)
