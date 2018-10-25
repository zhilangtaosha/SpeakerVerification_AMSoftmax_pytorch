import numpy as np

def cosine_distance_numpy(vector1 , vector2):
    vector1 = vector1.reshape([-1])
    vector2 = vector2.reshape([-1])
    cosV12 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return cosV12

def caculate_roc(embeddings1, embeddings2, actual_issame):
    l = len(embeddings1)
    score = [(cosine_distance_numpy(np.asarray(embeddings1[i]),np.asarray(embeddings2[i])), actual_issame[i]) for i in range(l)]
    score = sorted(score, key=lambda b : b[0], reverse=True)
    actual_issame = np.asarray(actual_issame)
    tp_fn = np.sum(actual_issame)
    tn_fp = l - tp_fn
    roc = []
    acc = []
    tp = 0.0
    fp = 0.0
    tn = 0.0
    auc = 0.0
    for i in range(l):
        if score[i][1] > 0:
            tp += 1
        else:
            fp += 1
        tn = tn_fp - fp
        tpr = tp / tp_fn
        fpr = fp / tn_fp
        acc.append([score[i][0], (tp + tn) / l])
        roc.append([fpr, tpr])
    for i in range(1, l):
        auc += (roc[i][0] - roc[i - 1][0]) * roc[i - 1][1]
    # print('auc : ', auc)
    max_acc = 0.0
    for i in acc:
        if i[1] > max_acc:
            max_acc = i[1]
    print("acc : ", max_acc)
    roc = np.asarray(roc)
    acc = np.asarray(acc)
    return roc, acc