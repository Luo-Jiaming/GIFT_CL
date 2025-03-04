import subprocess
import json

with open('project_config.json', 'r') as f:
    project_config = json.load(f)

SYN_DATA_LOCATION = project_config['SYN_DATA_LOCATION']
CL_DATA_LOCATION = project_config['CL_DATA_LOCATION']
MTIL_DATA_LOCATION = project_config['MTIL_DATA_LOCATION']
CKPT_LOCATION = project_config['CKPT_LOCATION']

def set_device(devices,cmd):
     # set devices
    cmd.append("--devices")
    for device in devices:
        cmd.append(str(device))
    return cmd

if __name__ == "__main__":
    exp_no = "zscl_mtil_I"
    dataset = ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]
    lr = ["5e-5", "1e-5", "1e-5", "1e-5", "1e-5", "1e-5", "1e-5", "5e-5", "1e-5", "1e-5", "1e-5", "1e-5"]
    
    ref_dataset = "ImageNetSUB"
    num = str(100000)
    batch_size = str(64)
    devices = [3,4,5]

    is_train = True
    is_eval = True
    is_zero_shot = False

    if is_train:
        # first dataset
        cmd = [
            "python", "-m", "src.main",
            "--train-mode=whole",
            "--train-dataset=" + dataset[0],
            "--lr=" + lr[0],
            "--ls","0.2",
            "--iterations", "1000",
            "--method", "ZSCL",
            "--image_loss",
            "--text_loss",
            "--we",
            "--avg_freq", "100",
            "--l2", "1",
            "--ref-dataset", ref_dataset,
            "--num", num,
            "--batch-size", batch_size,
            # "--ref-sentences", "conceptual_captions",
            "--save", CKPT_LOCATION+"exp_" + exp_no
        ]
        cmd = set_device(devices,cmd)
        subprocess.run(cmd)

        for i in range(1, len(dataset)):
            dataset_cur = dataset[i]
            dataset_pre = dataset[i - 1]
            print("Task %d:"%(i))
            # continue training
            cmd = [
                "python", "-m", "src.main",
                "--train-mode=whole",
                "--train-dataset=" + dataset_cur,
                "--lr=" + lr[i],
                "--ls", "0.2",
                "--method", "ZSCL",
                "--image_loss",
                "--text_loss",
                "--we",
                "--avg_freq", "100",
                "--l2", "1",
                "--ref-dataset", ref_dataset,
                "--num", num,
                "--batch-size", batch_size,
                # "--ref-sentences", "conceptual_captions",
                "--iterations", "1000",
                "--save", CKPT_LOCATION+"exp_" + exp_no,
                "--load", CKPT_LOCATION+"exp_" + exp_no + "/" + dataset_pre + ".pth"
            ]
            cmd = set_device(devices,cmd)
            subprocess.run(cmd)

    # zero-shot
    if is_zero_shot:
        cmd = [
                "python", "-m", "src.main",
                "--eval-only",
                "--train-mode=whole",
                "--eval-datasets="+"ImageNet,"+",".join(dataset),
                "--devices", str(devices[0])
            ]
        subprocess.run(cmd)

    # for evaluation
    if is_eval:
        for i in range(0, len(dataset)):
            dataset_cur = dataset[i]
            print("Task %d:"%(i))
            # continue training
            cmd = [
                "python", "-m", "src.main",
                "--eval-only",
                "--train-mode=whole",
                "--eval-datasets="+"ImageNet,"+",".join(dataset),
                "--load", CKPT_LOCATION+"exp_" + exp_no + "/" + dataset_cur + ".pth",
                "--devices", str(devices[0])
            ]
            subprocess.run(cmd)

    print("Experiment %s finished!"%(exp_no))