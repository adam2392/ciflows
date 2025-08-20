from ciflows.datasets.multidistr import StratifiedSampler


def test_stratified_sampler():
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    batch_size = 4
    sampler = StratifiedSampler(labels, batch_size)

    for epoch in range(3):
        this_idx = list(iter(sampler))
        print(f"Epoch {epoch}: {this_idx[:batch_size], this_idx[batch_size:]}")
        # for i, batch in enumerate(sampler):
        # print(f"Epoch {epoch}: {batch}")
        # print(f"Epoch {epoch}: {list(iter(sampler))}")
    assert False
