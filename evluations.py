import numpy as np
from sklearn import metrics
from sklearn.metrics import auc
from statsmodels.tsa.stattools import adfuller


def eval_fun(res_dic, accident_dic,threshold=0.3):
    '''
    Calculate evaluation indicators
    by cxy
    Args:
        res_dic: Prediction result dictionary
        accident_dic: Accident sample keyframe node
        threshold: ADF p_value threshold

    Returns:

    '''
    length = len(res_dic.keys())
    score_dic = {}
    preds = np.zeros(length)
    targets = np.zeros(length)  # No accident 0 Accident 1
    arr_index = 0
    for video_index in res_dic.keys():
        if video_index >= 400:
            targets[arr_index] = 0
        else:
            targets[arr_index] = 1
        frame_list = res_dic[video_index][0] # frame sequence
        score_list = res_dic[video_index][1] # Corresponding cos distance sequence
        # Reorder results by time
        new_score = [val for (_, val) in sorted(zip(frame_list, score_list), key=lambda x: x[0])]
        new_frame = sorted(frame_list)

        result = adfuller(new_score)
        p_value = result[1] # ADF p value

        if p_value >threshold:
            preds[arr_index] = 1
            if video_index < 400:
                max_score_index = np.argmax(new_score)
                mean_score = np.mean(new_score)
                std_score = np.std(new_score)
                outpoints = np.where(new_score > (mean_score + std_score * 3))[0]  # Get the outlier list
                if len(outpoints)>1:
                    temp = np.min(outpoints)
                    if temp<max_score_index:
                        max_score_index = temp
                have_accident_frame_list = []
                accident_frame = new_frame[max_score_index]
                have_accident_frame_list.append(accident_frame)
                score_dic[video_index] = np.array(have_accident_frame_list)
        else:
            preds[arr_index] = 0
        arr_index += 1

    FPR, TPR, thresholds = metrics.roc_curve(targets, preds)
    F1 = metrics.f1_score(targets, preds)
    AUC = auc(FPR, TPR)
    # Calculate nrmse
    tp_num = 0
    maximum_frame = 25 * 10  # 10s
    sum_dis = 0
    for video_index in res_dic.keys():
        dis = 0
        if video_index < 400:
            tp_num += 1
            key_frame = accident_dic[video_index]
            if video_index in score_dic.keys():
                dis = min(abs(score_dic[video_index] - key_frame))
            else:
                dis = maximum_frame
            dis = np.square(dis)
            sum_dis += dis
    nrmse = (min(np.sqrt(sum_dis / tp_num), maximum_frame)) / maximum_frame
    return nrmse, F1, AUC