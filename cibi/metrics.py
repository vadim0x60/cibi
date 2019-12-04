from itertools import islice

# https://docs.python.org/release/2.3.5/lib/itertools-example.html
def sliding_window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def best_reward_window(rewards, window_size=100):
    return max(sum(period) for period in sliding_window(rewards, n=window_size))  