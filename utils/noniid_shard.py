def cifar_noniid_shard(dataset, num_users, shards_per_client, rs=np.random.RandomState(SEED)):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset: The CIFAR-10 dataset
    :param num_users: Number of clients/users
    :param shards_per_client: Number of shards per client
    :param rs: Random state for reproducibility
    :return: Dictionary mapping each user to their corresponding data indices
    """
    num_shards = shards_per_client * num_users
    num_imgs = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # Sort indices based on labels to create non-IID shards
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide and assign shards to users
    for i in range(num_users):
        rand_set = set(rs.choice(idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        rs.shuffle(dict_users[i])
    
    return dict_users
