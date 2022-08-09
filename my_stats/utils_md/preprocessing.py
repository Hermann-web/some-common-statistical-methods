from numpy import array, isnan, std


def clear_list(L):
    L = array(L).flatten()
    L = L[~isnan(L)]
    return L


def clear_list_pair(L1, L2):
    L1 = array(L1).flatten()
    L2 = array(L2).flatten()
    if len(L1) != len(L2):
        raise Exception("lists must be of the same size")

    mask = ~(isnan(L1) | isnan(L2))
    L1 = L1[mask]
    L2 = L2[mask]

    if len(L1) != len(L2):
        raise Exception("internal pb")

    return L1, L2
