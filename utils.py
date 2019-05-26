import os
import time
import json
import numpy as np


def PRF(instances):
    # acc
    correct_id = []
    correct_count = 0
    all_count = len(instances)
    for qa_id, yp1, yp2, y1, y2 in instances:
        if yp1 == y1 and yp2 == y2:
            correct_count += 1
            correct_id.append(qa_id)

    acc = correct_count / all_count * 100

    return {"acc": acc}


def print_metrics(metrics, metrics_type, save_dir=None):
    matrix = metrics['matrix']
    acc = metrics['acc']
    each_prf = [[v * 100 for v in prf] for prf in zip(*metrics['each_prf'])]
    macro_prf = [v * 100 for v in metrics['macro_prf']]
    # loss = metrics['loss']
    # epoch = metrics['epoch']
    lines = ['\n\n**********************************************************************************',
             '*                                                                                *',
             '*                           {}                                  *'.format(
                 time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
             '*                                                                                *',
             '**********************************************************************************\n',
             'Confusion matrix:',
             '{0:>6}|{1:>6}|{2:>6}|<-- classified as'.format(' ', 'Good', 'Bad'),
             '------|--------------------|{0:>6}'.format('-SUM-'),
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('Good', *matrix[0].tolist()),
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('Bad', *matrix[1].tolist()),
             '------|-------------|------',
             '{0:>6}|{1:>6}|{2:>6}|{3:>6}'.format('-SUM-', *matrix[2].tolist()),
             '\nAccuracy = {0:6.2f}%\n'.format(acc * 100),
             'Results for the individual labels:',
             '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Good', *each_prf[0]),
             '\t{0:>6}: P ={1:>6.2f}%, R ={2:>6.2f}%, F ={3:>6.2f}%'.format('Bad', *each_prf[1]),
             '\n<<Official Score>>Macro-averaged result:',
             'P ={0:>6.2f}%, R ={1:>6.2f}%, F ={2:>6.2f}%'.format(*macro_prf),
             '--------------------------------------------------\n']

    [print(line) for line in lines]

    if save_dir is not None:
        with open(save_dir + ".json", 'a') as fw:
            log_output = {'global_step': '{:<6}'.format(metrics['global_step']),
                          'MAP': '{:<6.2f}'.format(metrics['MAP'] * 100),
                          'AvgRec': '{:<6.2f}'.format(metrics['AvgRec'] * 100),
                          'MRR': '{:<6.2f}'.format(metrics['MRR']),
                          'ACC': '{:<6.2f}'.format(metrics['acc'] * 100),
                          'F1(True)': '{:<6.2f}'.format(each_prf[0][2]),
                          'F1(macro)': '{:<6.2f}'.format(macro_prf[2])}
            json.dump(log_output, fw)
            fw.write('\n')


def transferring(matrix: np.ndarray, categories_num=3):
    conf_matrix = {'true': {'true': {}, 'false': {}}, 'false': {'true': {}, 'false': {}}}
    if categories_num == 3:
        conf_matrix['true']['true'] = matrix[0, 0]
        conf_matrix['true']['false'] = matrix[0, 1] + matrix[0, 2]
        conf_matrix['false']['true'] = matrix[1, 0] + matrix[2, 0]
        conf_matrix['false']['false'] = matrix[1, 1] + matrix[1, 2] + matrix[2, 1] + matrix[2, 2]
    else:
        conf_matrix['true']['true'] = matrix[0, 0]
        conf_matrix['true']['false'] = matrix[0, 1]
        conf_matrix['false']['true'] = matrix[1, 0]
        conf_matrix['false']['false'] = matrix[1, 1]
    return conf_matrix

