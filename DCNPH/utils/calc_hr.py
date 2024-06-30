import numpy as np
import pickle

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_nwmap(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    nwmap = 0.
    wmap = 0.
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        sim = (np.dot(queryL[iter, :], retrievalL.transpose())).astype(np.float32)
        pos = (sim > 0).astype(np.float32)
        alln = pos.sum()
        if alln < 0.5:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        max_ind = np.argsort(-sim)
        max_sim = sim[max_ind] * 1.
        max_pos = (max_sim > 0).astype(np.float32)

        sim = sim[ind] * 1.
        sim_pos = (sim > 0).astype(np.float32)
        for k in range(1, retrievalL.shape[0]):
            max_sim[k] = max_sim[k - 1] + max_sim[k]
            sim[k] = sim[k - 1] + sim[k]
        tsum = retrievalL.shape[0]
        count = np.linspace(1, tsum, tsum)
        now = ((sim_pos * sim / alln) / count).sum()
        maxw = ((max_pos * max_sim / alln) / count).sum()
        wmap += now
        nwmap += now / maxw
    wmap = wmap / float(num_query)
    nwmap /= float(num_query)
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return nwmap, wmap
def calc_map(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = int(np.sum(gnd))
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map

def calc_topMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap

if __name__=='__main__':
    # B_L = {'Qi': qi, 'Qt': qt,
    #        'Di': ri, 'Dt': rt,
    #        'retrieval_L': database_labels.numpy(), 'query_L': test_labels.numpy()}
    with open('/s2_md0/leiji/v-rtu/2dtan/ccmh/ccmh_vgg11coco1_32.pkl', 'rb') as f:
        B_L = pickle.load(f)
    qi = B_L['Qi']
    qt = B_L['Qt']
    ri = B_L['Di']
    rt = B_L['Dt']
    test_labels = B_L['query_L']
    database_labels = B_L['retrieval_L']
    map_ti = calc_map(qt, ri, test_labels, database_labels)
    print('txt_i_map:', map_ti)
    map_it = calc_map(qi, rt, test_labels, database_labels)
    print('i_t_map:', map_it)