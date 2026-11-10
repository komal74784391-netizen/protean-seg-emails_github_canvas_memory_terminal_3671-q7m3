import random


def sample_episode(dataset, n_way=1, k_shot=5):
    classes = random.sample(list(dataset.classes), n_way)
    support = [dataset.sample(c, k_shot) for c in classes]
    query = [dataset.sample(c, 10) for c in classes]
    return support, query
