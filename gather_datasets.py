from src import datasets
import json

with open('project_config.json', 'r') as f:
    project_config = json.load(f)

SYN_DATA_LOCATION = project_config['SYN_DATA_LOCATION']
CL_DATA_LOCATION = project_config['CL_DATA_LOCATION']
MTIL_DATA_LOCATION = project_config['MTIL_DATA_LOCATION']
CKPT_LOCATION = project_config['CKPT_LOCATION']

dataset_names = ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]

for dataset_name in dataset_names:
    try:
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            None,
            location=MTIL_DATA_LOCATION,
            batch_size=64,
            batch_size_eval=128,
        )
        print(f"{dataset_name} train:{len(dataset.train_dataset)} test:{len(dataset.test_dataset)}")
    except:
        print(f"{dataset_name} download failed..")