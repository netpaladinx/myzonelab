def collect_options(opts, prefix, share_remaining=False):
    assert isinstance(opts, dict)
    if isinstance(prefix, str):
        prefix = [prefix]
    assert isinstance(prefix, (list, tuple))

    collected = [{} for _ in prefix]
    for key, val in opts.items():
        for i, s in enumerate(prefix):
            if key.startswith(s):
                k = key[len(s):]
                collected[i][k] = val
                break
        else:
            if share_remaining:
                for col in collected:
                    col[key] = val
    return collected
