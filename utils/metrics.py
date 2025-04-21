import numpy as np
import pandas as pd

def save_result(file_path, result_name, task_num=11, eval_num=11, save_path='./results/', is_print=False):
    lines = []
    file = open(file_path, "r")
    for line in file:
        lines.append(line)

    accs = [float(line[:-1].split("Top-1 accuracy: ")[1]) \
        for line in lines if line.startswith("Top-1 accuracy: ")]
    
    if is_print:
        print(np.reshape(accs,(task_num,eval_num)))

    acc_mat = np.array(accs)
    df = pd.DataFrame(acc_mat.reshape((task_num,eval_num)))
    df.to_csv(save_path+f'{result_name}.csv', index=False)

def get_metrics(data):
    data = np.array(data)
    avg = np.mean(data)
    last = np.mean(data[-1,:])

    transfer = 0
    for col in range(1,data.shape[1]):
        transfer += np.mean(data[0:col,col])
    transfer = transfer / (data.shape[1]-1)

    forget = 0
    for task in range(data.shape[1]):
        forget += (data[-1][task]-data[task][task])
    forget = forget / (data.shape[1]-1)

    metrics = {}
    metrics['Avg.']=avg
    metrics['Last']=last
    metrics['Transfer']=transfer
    metrics['Forget']=forget

    metrics = {key: "{:.2f}".format(value) for key, value in metrics.items()}
    return metrics