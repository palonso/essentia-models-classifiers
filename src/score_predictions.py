import argparse
import json
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

import shared

Score = namedtuple('Score', ['mean', 'std'])


def score_predictions(args):
    config = json.load(open(Path(args.config_file), 'r'))

    results_file = Path(config['exp_dir'], 'results_whole')
    predictions_file = Path(config['exp_dir'], 'predictions')

    ids, folds, predictions, groundtruth = [], [], dict(), dict()

    n_folds = config['config_train']['n_folds']
    grondtruth_folder = Path(config['config_train']['gt_test']).parent
    dataset = config['dataset']

    # get fold-wise predictions
    for i in range(n_folds):
        groundtruth_file = grondtruth_folder / f"gt_test_{i}.csv"

        with open('{}_{}.json'.format(predictions_file, i), 'r') as f:
            fold = json.load(f)
            folds.append(fold)
            predictions.update(fold)

        ids_fold, gt_fold = shared.load_id2gt(groundtruth_file)

        ids += ids_fold
        groundtruth.update(gt_fold)

    groundtruth_ids = set(ids)
    predictions_ids = set(predictions.keys())

    # check if there are missing predictions and update ids
    missing_ids = groundtruth_ids.symmetric_difference(predictions_ids)
    if missing_ids:
        print('ids without predictions or groundtruth: {}'.format(missing_ids))
        ids = list(predictions_ids - missing_ids)

    y_true, y_pred = zip(*[(groundtruth[i], predictions[i]) for i in ids])

    fold_gt, fold_pred = [], []
    for i, fold in enumerate(folds):
        keys = [i for i in fold.keys() if i in groundtruth.keys()]
        fold_pred.append([predictions[k] for k in keys])
        fold_gt.append([groundtruth[k] for k in keys])

    task_type = config['config_train']['task_type']
    scores, report = get_metrics(y_true, y_pred, fold_gt, fold_pred, n_folds, task_type)

    store_results(
        results_file,
        scores,
        report,
        dataset,
        task_type,
    )


def get_metrics(y_true, y_pred, fold_gt, fold_pred, n_folds, task_type):

    if task_type == "regression":

        # compute micro metrics
        micro_metrics = {}
        micro_metrics["p_corr"] = shared.compute_pearson_correlation(y_true, y_pred)
        micro_metrics["ccc"] = shared.compute_ccc(y_true, y_pred)
        micro_metrics["r2"] = shared.compute_r2_score(y_true, y_pred)
        micro_metrics["adjusted_r2"] = shared.compute_adjusted_r2_score(y_true, y_pred, np.shape(y_true)[1])
        micro_metrics["rmse"] = shared.compute_root_mean_squared_error(y_true, y_pred)

        p_corrs = []
        cccs = []
        r2s = []
        adjusted_r2s = []
        rmses = []
        for i in range(n_folds):
            y_true_fold = fold_gt[i]
            y_pred_fold = fold_pred[i]
            p_corrs.append(shared.compute_pearson_correlation(y_true_fold, y_pred_fold))
            cccs.append(shared.compute_ccc(y_true_fold, y_pred_fold))
            r2s.append(shared.compute_r2_score(y_true_fold, y_pred_fold))
            adjusted_r2s.append(shared.compute_adjusted_r2_score(y_true_fold, y_pred_fold, np.shape(y_true)[1]))
            rmses. append(shared.compute_root_mean_squared_error(y_true_fold, y_pred_fold))

        # compute pearson correlation
        macro_p_corr = np.mean(p_corrs, axis=0)
        print("Macro Pearson Corr:", macro_p_corr)
        print("Micro Pearson Corr:", micro_metrics["p_corr"])

        # compute ccc
        macro_ccc = np.mean(cccs, axis=0)
        print("Macro CCC:", macro_ccc)
        print("Micro CCC:", micro_metrics["ccc"])

        # compute r2 score
        macro_r2 = np.mean(r2s, axis=0)
        print("Macro R2:", macro_r2)
        print("Micro R2:", micro_metrics["r2"])

        # compute adjusted r2 score
        macro_adjusted_r2 = np.mean(adjusted_r2s, axis=0)
        print("Macro Adjusted R2:", macro_adjusted_r2)
        print("Micro Adjusted R2:", micro_metrics["adjusted_r2"])

        # compute RMSE
        macro_rmse = np.mean(rmses, axis=0)
        print("Macro RMSE:", macro_rmse)
        print("Micro RMSE:", micro_metrics["rmse"])

        p_corr_std, ccc_std = np.std(p_corrs, axis=0), np.std(cccs, axis=0)
        r2_std, adjusted_r2_std = np.std(r2s, axis=0), np.std(adjusted_r2s, axis=0)
        rmse_std = np.std(rmses, axis=0)

        #! classification report is only available for classification tasks

        p_corr_score = Score(macro_p_corr, p_corr_std)
        ccc_score = Score(macro_ccc, ccc_std)
        r2_score = Score(macro_r2, r2_std)
        adjusted_r2_score = Score(macro_adjusted_r2, adjusted_r2_std)
        rmse_score = Score(macro_rmse, rmse_std)
        scores = (p_corr_score, ccc_score, r2_score, adjusted_r2_score, rmse_score, micro_metrics)
        report = None
    else:
        # for classification tasks
        micro_acc = shared.compute_accuracy(y_true, y_pred)
        accs = []
        roc_aucs, pr_aucs = [], []
        for i in range(n_folds):
            y_true_fold = fold_gt[i]
            y_pred_fold = fold_pred[i]
            accs.append(shared.compute_accuracy(y_true_fold, y_pred_fold))
            roc_auc_i, pr_auc_i = shared.compute_auc(y_true_fold, y_pred_fold)
            roc_aucs.append(roc_auc_i)
            pr_aucs.append(pr_auc_i)

        macro_acc = np.mean(accs)
        print("Macro Acc:", macro_acc)
        print("Micro Acc:", micro_acc)
        acc_std = np.std(accs)
        roc_auc, pr_auc = np.mean(roc_aucs), np.mean(pr_aucs)
        roc_auc_std, pr_auc_std = np.std(roc_aucs), np.std(pr_aucs)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if shared.type_of_groundtruth(y_true) == "multilabel-indicator":
            y_pred_indicator = np.round(y_pred)
            report = classification_report(y_true, y_pred_indicator)
        else:
            y_true_argmax = np.argmax(y_true, axis=1)
            y_pred_argmax = np.argmax(y_pred, axis=1)
            report = classification_report(y_true_argmax, y_pred_argmax)

        roc_auc_score = Score(roc_auc, roc_auc_std)
        pr_auc_score = Score(pr_auc, pr_auc_std)
        macro_acc_score = Score(macro_acc, acc_std)

        scores = (roc_auc_score, pr_auc_score, macro_acc_score, micro_acc)
    return scores, report


def store_results(
    output_file,
    scores,
    report,
    dataset,
    task_type,
):
    if task_type == "regression":
        p_corr_score, ccc_score, r2_score, adjusted_r2_score, rmse_score, micro_metrics = scores

        # print experimental results
        print('Pearson Correlation: ' + str(p_corr_score.mean))
        print('Pearson Correlation STD: ' + str(p_corr_score.std))
        print('CCC: ' + str(ccc_score.mean))
        print('CCC STD: ' + str(ccc_score.std))
        print('R2 Score Micro: ' + str(micro_metrics["r2"]))
        print('R2 Score Macro: ' + str(r2_score.mean))
        print('R2 Score STD: ' + str(r2_score.std))
        print('Adjusted R2 Score Micro: ' + str(micro_metrics["adjusted_r2"]))
        print('Adjusted R2 Score Macro: ' + str(adjusted_r2_score.mean))
        print('Adjusted R2 Score STD: ' + str(adjusted_r2_score.std))
        print('RMSE Micro: ' + str(micro_metrics["rmse"]))
        print('RMSE Macro: ' + str(rmse_score.mean))
        print('RMSE STD: ' + str(rmse_score.std))
        print('latext format:')
        for i in range(len(micro_metrics["r2"])):
            print(f'{i}: {micro_metrics["r2"][i]:.2f}\\pm{r2_score.std[i]:.2f}')
        print('-' * 20)

        # store experimental results
        with open(output_file, 'w') as to:
            to.write('\nPearson Correlation: ' + str(p_corr_score.mean))
            to.write('\nStD: ' + str(p_corr_score.std))
            to.write('\nCCC: ' + str(ccc_score.mean))
            to.write('\nStD: ' + str(ccc_score.std))
            to.write('\nR2 Score Micro: ' + str(micro_metrics["r2"]))
            to.write('\nR2 Score Macro: ' + str(r2_score.mean))
            to.write('\nStD: ' + str(r2_score.std))
            to.write('\nAdjusted R2 Score Micro: ' + str(micro_metrics["adjusted_r2"]))
            to.write('\nAdjusted R2 Score Macro: ' + str(adjusted_r2_score.mean))
            to.write('\nStD: ' + str(adjusted_r2_score.std))
            to.write('\nRMSE Micro: ' + str(micro_metrics["rmse"]))
            to.write('\nRMSE Macro: ' + str(rmse_score.mean))
            to.write('\nStD: ' + str(rmse_score.std))
            to.write('\n')
    else:
        roc_auc_score, pr_auc_score, macro_acc_score, micro_metrics = scores

        # print experimental results
        print('ROC-AUC: ' + str(roc_auc_score.mean))
        print('PR-AUC: ' + str(pr_auc_score.mean))
        print('Balanced Micro Acc: ' + str(micro_metrics))
        print('Balanced Macro Acc: ' + str(macro_acc_score.mean))
        print('Balanced Acc STD: ' + str(macro_acc_score.std))
        print('latext format:')
        print('{:.2f}\\pm{:.2f}'.format(micro_metrics, macro_acc_score.std))
        print('-' * 20)

        # store experimental results
        with open(output_file, 'w') as to:
            to.write('\nROC AUC: ' + str(roc_auc_score.mean))
            to.write('\nStD: ' + str(roc_auc_score.std))
            to.write('\nPR AUC: ' + str(pr_auc_score.mean))
            to.write('\nStD: ' + str(pr_auc_score.std))
            to.write('\nAcc Micro: ' + str(micro_metrics))
            to.write('\nAcc Macro: ' + str(macro_acc_score.mean))
            to.write('\nStD: ' + str(macro_acc_score.std))
            to.write('\n')
            to.write('Report:\n')
            to.write('{}\n'.format(report))

    output_summary = output_file.parent.parent / 'results.json'

    try:
        with open(output_summary, 'r') as fp:
            data = json.load(fp)
    except:
        data = dict()

    with open(output_summary, 'w+') as fp:
        data[dataset] = defaultdict(dict)
        if task_type == "regression":
            data[dataset]['R2 Micro']['mean'] = micro_metrics["r2"]
            data[dataset]['R2 Macro']['mean'] = r2_score.mean
            data[dataset]['R2 Macro']['std'] = r2_score.std
            data[dataset]['Adjusted R2 Micro']['mean'] = micro_metrics["adjusted_r2"]
            data[dataset]['Adjusted R2 Macro']['mean'] = adjusted_r2_score.mean
            data[dataset]['Adjusted R2 Macro']['std'] = adjusted_r2_score.std
            data[dataset]['Pearson Correlation']['mean'] = p_corr_score.mean
            data[dataset]['Pearson Correlation']['std'] = p_corr_score.std
            data[dataset]['CCC']['mean'] = ccc_score.mean
            data[dataset]['CCC']['std'] = ccc_score.std
            data[dataset]['RMSE']['mean'] = rmse_score.mean
            data[dataset]['RMSE']['std'] = rmse_score.std
        else:
            data[dataset]['Accuracy Micro']['mean'] = micro_metrics
            data[dataset]['Accuracy Macro']['mean'] = macro_acc_score.mean
            data[dataset]['Accuracy']['std'] = macro_acc_score.std
            data[dataset]['ROC AUC']['mean'] = roc_auc_score.mean
            data[dataset]['ROC AUC']['std'] = roc_auc_score.std
            data[dataset]['PR AUC']['mean'] = pr_auc_score.mean
            data[dataset]['PR AUC']['std'] = pr_auc_score.std

        json.dump(data, fp, indent=4, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='configuration file')
    args = parser.parse_args()

    score_predictions(args)
