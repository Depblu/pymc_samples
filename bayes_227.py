from pymc3 import Model, Normal, HalfNormal, Uniform, Bernoulli, find_MAP, NUTS, sample, Slice, Deterministic
from scipy import optimize
import pymc3 as pm
N = 100

basic_model = Model()
with basic_model:
    p = Uniform("freq_cheating", 0, 1)
    true_answers = Bernoulli("truths", p)
    first_coin_flips = Bernoulli("first_flips", 0.5)
    second_coin_flips = Bernoulli("second_flips", 0.5)

    determin_val1 = Deterministic('determin_val1', first_coin_flips * true_answers + (1 - first_coin_flips)*second_coin_flips)
    determin_val = determin_val1.sum()/float(N)

    start = find_MAP(fmin=optimize.fmin_powell)

    # instantiate sampler
    step = Slice(vars=[true_answers])
    # draw 5000 posterior samples
    trace = sample(100, step=step, start=start)

    step = Slice(vars=[first_coin_flips])
    # draw 5000 posterior samples
    trace = sample(100, step=step, start=start)

    step = Slice(vars=[second_coin_flips])
    # draw 5000 posterior samples
    trace = sample(100, step=step, start=start)

    #print(first_coin_flips.getattr_value())
    print(determin_val)

map_estimate = find_MAP(model=basic_model)

print(map_estimate.values())