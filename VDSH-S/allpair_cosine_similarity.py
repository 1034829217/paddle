import os
import numpy as np
from tqdm import *
from scipy import sparse
from datasets import *
import argparse

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--gpunum")
parser.add_argument("--dataset")
parser.add_argument("--usetrain", dest='usetrain', action='store_true')
parser.add_argument("--usetest", dest='usetrain', action='store_false')
parser.set_defaults(usetrain=False)
parser.add_argument("--docBatchSize", default=100, type=int)
parser.add_argument("--queryBatchSize", default=500, type=int)
parser.add_argument("--num_neighbors", default=100, type=int)

args = parser.parse_args()
if args.gpunum:
    print("Use GPU #:{}".format(args.gpunum))
    gpunum = args.gpunum
else:
    print("Use GPU #0 as a default gpu")
    gpunum = "0"
    
os.environ["CUDA_VISIBLE_DEVICES"]=gpunum

if args.dataset:
    print("load {} dataset".format(args.dataset))
    dataset = args.dataset
else:
    parser.error("Need to provide the dataset.")
    
dataset = args.dataset
docBatchSize = args.docBatchSize
queryBatchSize = args.queryBatchSize
gpunum = args.gpunum

TopK = args.num_neighbors + 1 # need to plus one because for 'use_train' mode, the nearest node is itself.
usetrain = args.usetrain

##################################################################################################

dataset, data_fmt = dataset.split('.')
if dataset in ['reuters', 'tmc', 'rcv1']:
    single_label = False
else:
    single_label = True

#########################################################################################################

if single_label:
    train_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
    if usetrain:
        query_set = train_set
    else:
        query_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
else:
    train_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
    if usetrain:
        query_set = train_set
    else:
        query_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
        
##################################################################################################

print('num query: {} num candidate: {}'.format(len(query_set), len(train_set)))

os.environ["CUDA_VISIBLE_DEVICES"]=gpunum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##################################################################################################

queries = query_set.df
documents = train_set.df
n_docs = len(documents)
n_queries = len(queries)

##################################################################################################
nearest_neighbors = []
for q_idx in tqdm(range(0, n_queries, queryBatchSize), desc='Query', ncols=0):
    query_batch_s_idx = q_idx
    query_batch_e_idx = min(query_batch_s_idx + queryBatchSize, n_queries)
    queryMats = sparse.vstack(list(queries.bow.iloc[query_batch_s_idx: query_batch_e_idx])).toarray()
    queryMats = torch.from_numpy(queryMats).to(device)
    queryNorm2 = torch.norm(queryMats, 2, dim=1)
    queryNorm2.unsqueeze_(1)
    queryMats.unsqueeze_(2)

    scoreList = []
    indicesList = []
    for idx in tqdm(range(0, n_docs, docBatchSize), desc='Doc', leave=False, ncols=0):
    #for idx in range(0, n_docs, docBatchSize):
        batch_s_idx = idx
        batch_e_idx = min(batch_s_idx + docBatchSize, n_docs)
        n_doc_in_batch = batch_e_idx - batch_s_idx

        candidateMats = sparse.vstack(list(documents.bow.iloc[batch_s_idx: batch_e_idx])).toarray()
        candidateMats = torch.from_numpy(candidateMats).to(device)

        candidateNorm2 = torch.norm(candidateMats, 2, dim=1)
        candidateNorm2.unsqueeze_(0)

        candidateMats.unsqueeze_(2)
        candidateMats = candidateMats.permute(2, 1, 0)

        # compute cosine similarity
        queryMatsExpand = queryMats.expand(queryMats.size(0), queryMats.size(1), candidateMats.size(2))
        candidateMats = candidateMats.expand_as(queryMatsExpand)

        cos_sim_scores = torch.sum(queryMatsExpand * candidateMats, dim=1) / (queryNorm2 * candidateNorm2)

        K = min(TopK, n_doc_in_batch)
        scores, indices = torch.topk(cos_sim_scores, K, dim=1, largest=True)

        del cos_sim_scores
        del queryMatsExpand
        del candidateMats
        del candidateNorm2

        scoreList.append(scores)
        indicesList.append(indices + batch_s_idx)
        
    all_scores = torch.cat(scoreList, dim=1)
    all_indices = torch.cat(indicesList, dim=1)
    top_scores, top_indices = torch.topk(all_scores, TopK, dim=1, largest=True)
    topK_indices = torch.gather(all_indices, 1, top_indices) # convert index to the document index
    
    del queryMats
    del queryNorm2
    del scoreList
    del indicesList
    
    if usetrain:
        topK_indices = topK_indices[:, 1:] # ignore the first entry because it is itself
        top_scores = top_scores[:, 1:]
    else:
        topK_indices = topK_indices[:, :-1] # ignore the last one
        top_scores = top_scores[:, :-1]
        
    for i in range(query_batch_s_idx, query_batch_e_idx):
        nn_doc_index = topK_indices[i - query_batch_s_idx]
        nn_doc_ids = list(documents.iloc[nn_doc_index].index)
        nn_data = {'doc_id': queries.iloc[i].name, 
                   'bow': queries.iloc[i].bow,
                   'label': queries.iloc[i].label,
                   'top_nn': nn_doc_ids, 
                   'scores': list(top_scores[i - query_batch_s_idx].cpu().numpy())}
        nearest_neighbors.append(nn_data)
    
    torch.cuda.empty_cache()
 
###################################################################################################
df = pd.DataFrame.from_dict(nearest_neighbors)

save_dir = 'dataset/{}'.format(dataset)
print('saving the results to {} ...'.format(save_dir))

if usetrain:
    save_fn = os.path.join(save_dir, 'train.NN.pkl')
else:
    save_fn = os.path.join(save_dir, 'test.NN.pkl')
    
df.set_index('doc_id', inplace=True)
df.to_pickle(save_fn)