import argparse
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

import torch
from PIL.Image import Image
from src import datasets
import random
import csv
import numpy as np
import json

import os
import sys
import time
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, filename='log.txt', mode='a', timestamp=False, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    save_file = os.path.join(output_dir, filename)
    if timestamp:
        basename, extname = os.path.splitext(save_file)
        save_file = basename + time.strftime("-%Y-%m-%d-%H:%M:%S", time.localtime()) + extname
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    file_handler = logging.FileHandler(save_file, mode=mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def get_logger(name):
    return logging.getLogger(name)

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

with open('project_config.json', 'r') as f:
    project_config = json.load(f)

SYN_DATA_LOCATION = project_config['SYN_DATA_LOCATION']
CL_DATA_LOCATION = project_config['CL_DATA_LOCATION']
MTIL_DATA_LOCATION = project_config['MTIL_DATA_LOCATION']

class ImgGenerator:
    def __init__(self, device, model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5", scheduler=False):
        self.device = device
        self.pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        
        if scheduler:
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        
        self.pipeline.to(f"cuda:{device}")

    def generate_img(self, prompt, img_name, steps = 50, seed = 42):
        generator = torch.Generator(f"cuda:{self.device}").manual_seed(seed) 
        output_image: Image = self.pipeline(prompt, generator=generator, num_inference_steps=steps).images[0]
        # Save the image to a local file
        with open(img_name, "w") as f:
            output_image.save(f, format="JPEG")

def generate_mtil(args, datasetname='synthetic_data_a'):
    logger = get_logger(name=f'txt2img_logger_{args.device_id}')

    if args.setting == "MTIL_I":
        dataset_names = ["ImageNet", "Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]
    elif args.setting == "MTIL_II":
        dataset_names = ["ImageNet", "StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397", "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]
    else:
        logger.info(f"Setting {args.setting} not supported.")
        raise ValueError(f"Setting {args.setting} not supported.")

    logger.info(f"[setting]: {args.setting}")

    if args.start_task < 0 or args.start_task >= (len(dataset_names)-1):
        logger.info(f"Invalid start_task {args.start_task}")
        raise ValueError(f"Invalid start_task {args.start_task}")
    else:
        logger.info(f"[tasks]: {dataset_names[args.start_task+1:args.end_task+2]}")
    logger.info(f"[image num per task]: {args.image_num}")
    logger.info(f"[starting from]: {args.break_place}")
    torch.cuda.set_device(args.device_id)
    logger.info(f"[using device]: {args.device_id}")

    save_path = os.path.join(SYN_DATA_LOCATION, datasetname)
    logger.info(f"[save path]: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    generator = ImgGenerator(device = args.device_id, scheduler=(args.steps < 50))

    # prepare prompts and write into csv first
    prompts_pool = []
    for i in range(len(dataset_names) - 1):
        dataset_name = dataset_names[i]
        dataset_current = dataset_names[i+1]

        if args.sample_prompts:
            dataset_class = getattr(datasets, dataset_name)
            if dataset_name == 'ImageNet':
                data_path = CL_DATA_LOCATION
            else:
                data_path = MTIL_DATA_LOCATION
            dataset = dataset_class(
                None,
                location= data_path,
                batch_size=64,
                batch_size_eval=64,
            )
            template = dataset.template
            prompts_pool.extend([template(x) for x in dataset.classnames])

        if not os.path.exists(os.path.join(save_path, dataset_current + '_Syn')):
            os.makedirs(os.path.join(save_path, dataset_current + '_Syn'))
        
        if not os.path.exists(os.path.join(save_path, dataset_current + '_Syn', 'image_text.csv')):
            logger.info(f"Pick prompts for [{dataset_current}] ...")
            print(os.path.join(save_path, dataset_current + '_Syn'))
            with open(os.path.join(save_path, dataset_current + '_Syn', 'image_text.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image','text'])

                for j in range(args.image_num):
                    prompt = random.choice(prompts_pool)
                    img_name = os.path.join(save_path, dataset_current + '_Syn', f'{j}.jpeg')
                    writer.writerow([img_name, prompt])
    
    for i in range(len(dataset_names) - 1):

        dataset_name = dataset_names[i]
        dataset_current = dataset_names[i+1]

        if i < args.start_task:
            continue
        elif i > args.end_task:
            break

        logger.info(f"Start generating images for [{dataset_current}] ...")

        if i == args.start_task and args.break_place > 0:
            start_place = args.break_place
        else:
            start_place = 0

        with open(os.path.join(save_path, dataset_current + '_Syn', 'image_text.csv'), 'r', newline='') as f:
            
            reader = csv.reader(f)
            reader = list(reader)
            rows = [row for row in reader[1:]] # skip the head
            prompt_dict = {row[0]: row[1] for row in rows}

            for j in range(start_place, args.image_num): # image_num = 10000
                img_name = os.path.join(save_path, dataset_current + '_Syn', f'{j}.jpeg')
                prompt = prompt_dict[img_name]
                logger.info(f'Prompt [{j}]: {prompt}')
                generator.generate_img(prompt, img_name, steps = args.steps, seed = j + 100000 * (dataset_names.index(dataset_name)-1) + 10000000 * args.seed)
                logger.info(f'Photo [{j}] generated at [{img_name}] from seed [{j + 100000 * (dataset_names.index(dataset_name)-1) + 10000000 * args.seed}]')
        
        logger.info(f"Finish generating images for [{dataset_current}] ...")
    logger.info('All done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='args for txt2img.py')
    parser.add_argument('--setting', type=str, default="MTIL_alphabet", help='Generate imgae for which setting')
    parser.add_argument('--device_id', type=int, default=0, help='Device_id')
    parser.add_argument('--steps', type=int, default=50, help='Denoising steps')
    parser.add_argument('--start_task', type=int, default=0, help='Path to the experiment config file')
    parser.add_argument('--end_task', type=int, default=10, help='Path to the experiment config file')
    parser.add_argument('--break_place', type=int, default=0, help='Continue from where break')
    parser.add_argument('--image_num', type=int, default=1000, help='Image num per task')
    parser.add_argument('--sample_prompts', action='store_true', help='Whether to sample prompts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--datasetname', type=str, default='synthetic_data_a', help='Dataset name')
    args = parser.parse_args()

    seed_all(args.seed)
    logger = create_logger('.', filename=f'txt2img_out_{args.device_id}.log', name=f'txt2img_logger_{args.device_id}')
    
    if args.setting == "MTIL_I" or args.setting == "MTIL_II":
        generate_mtil(args, args.datasetname)
    else:
        logger.info(f"Setting {args.setting} not supported.")
        raise ValueError(f"Setting {args.setting} not supported.")