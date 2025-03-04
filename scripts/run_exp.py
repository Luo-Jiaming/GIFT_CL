import argparse
import subprocess
import json

with open('project_config.json', 'r') as f:
    project_config = json.load(f)

SYN_DATA_LOCATION = project_config['SYN_DATA_LOCATION']
CL_DATA_LOCATION = project_config['CL_DATA_LOCATION']
MTIL_DATA_LOCATION = project_config['MTIL_DATA_LOCATION']
CKPT_LOCATION = project_config['CKPT_LOCATION']

def set_device(config,cmd):
     # set devices
    cmd.append("--devices")
    for device in config["devices"]:
        cmd.append(str(device))
    return cmd

# add extra settings here
def extra_settings(config,cmd):

    if 'we' in config and config['we']==True:
        cmd.append('--we')
        cmd.append("--avg_freq")
        cmd.append(config['avg_freq'])

        if 'pwe' in config and config['pwe']==True:
            cmd.append('--pwe')
    
    if 'awc' in config and config['awc']==True:
        cmd.append('--awc')

    if 'dump' in config and config['dump']:
        cmd.append('--dump')

    if "ref_model" in config:
        cmd.append('--ref-model')
        cmd.append(config["ref_model"])
    if "ref_dataset" in config:
        cmd.append("--ref-dataset")
        cmd.append(config["ref_dataset"])
    
    if 'image_nums' in config:
        cmd.append('--image-nums')
        cmd.append(str(config['image_nums']))

    if "ablate_image_only" in config and config["ablate_image_only"]:
        cmd.append("--image_only")
    elif "ablate_text_only" in config and config["ablate_text_only"]:
        cmd.append("--text_only")

    if "feature_mse" in config and config["feature_mse"]:
        cmd.append("--feature_mse")

    if "kl_div" in config and config["kl_div"]:
        cmd.append("--kl_div")

    if "static_awc" in config and config["static_awc"]!=0:
        cmd.append("--static_awc")
        cmd.append(str(config["static_awc"]))

    return cmd

def run_experiment(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    
    exp_no = config["exp_no"]
    dataset = config["train_dataset"]
    lr = config["learning_rate"]
    iterations = config["iterations"]
    label_smoothing = config["label_smoothing"]
    L2 = config["L2"]
    distill = config["distill"]

    if 'ita' not in config:
        ita = "0"
    else:
        ita = config['ita']

    if 'train_loss' not in config:
        train_loss = 'cross_entropy'
    else:
        train_loss = config['train_loss']

    eval_dataset = ""
    for d in config["eval_dataset"]:
        eval_dataset += d + ","
    eval_dataset = eval_dataset[:-1]

    if 'save_path' in config:
        save_path = config['save_path']
    else:
        save_path = CKPT_LOCATION

    if config["train"]:
        cmd = [
            "python", "-m", "src.main",
            "--train-mode=whole",
            "--train-dataset=" + dataset[0],
            "--lr=" + lr[0],
            "--ls", label_smoothing,
            "--iterations", iterations,
            "--l2", L2,
            "--ita", ita,
            "--distill", distill,
            "--method", "GIFT",
            "--train_loss", train_loss,
            # "--select_method", "gradient_importance",
            "--save", save_path + "exp_" + exp_no,
        ]
        cmd = set_device(config,cmd)
        cmd = extra_settings(config,cmd)
        subprocess.run(cmd)
        
        # not useful now
        stop_task = len(dataset)  
        if "exclude_sun397" in config and config["exclude_sun397"]:
            stop_task-=1
         
        for i in range(1, stop_task):
            dataset_cur = dataset[i]
            dataset_pre = dataset[i - 1]
            # continue training
            cmd = [
                "python", "-m", "src.main",
                "--train-mode=whole",
                "--train-dataset=" + dataset_cur,
                "--lr=" + lr[i],
                "--ls", label_smoothing,
                "--ita", ita,
                "--l2", L2,
                "--distill", distill,
                "--method", "GIFT",
                "--train_loss", train_loss,
                # "--select_method", "gradient_importance",
                "--iterations", iterations,
                "--save", save_path + "exp_" + exp_no,
                "--load", save_path + "exp_" + exp_no + "/" + dataset_pre + ".pth",
            ]
            cmd = set_device(config,cmd)
            cmd = extra_settings(config,cmd)
            subprocess.run(cmd)
    
    # zero-shot
    if config['zero-shot']:
        cmd = [
            "python", "-m", "src.main",
            "--eval-only",
            "--train-mode=whole",
            "--eval-datasets=" + eval_dataset,
            "--devices", str(config["devices"][0])
        ]
        if 'text_ensemble_test' in config and config['text_ensemble_test']:
            cmd.append("--text_ensemble_test")
        subprocess.run(cmd)

    # evaluate
    if config['evaluate']:

        # not useful now
        stop_task = len(dataset)  
        if "exclude_sun397" in config and config["exclude_sun397"]:
            stop_task-=1

        for i in range(0, stop_task):
            dataset_cur = dataset[i]
            cmd = [
                "python", "-m", "src.main",
                "--eval-only",
                "--train-mode=whole",
                "--eval-datasets=" + eval_dataset,
                "--load", save_path + "exp_" + exp_no + "/" + dataset_cur + ".pth",
                "--devices", str(config["devices"][0])
            ]
            if 'text_ensemble_test' in config and config['text_ensemble_test']:
                cmd.append("--text_ensemble_test")
            subprocess.run(cmd)
    
    print("Experiment %s finished!"%(exp_no))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment with config')
    parser.add_argument('--config_path', type=str, help='Path to the experiment config file')
    args = parser.parse_args()
    run_experiment(args.config_path)    