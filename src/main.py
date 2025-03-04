import copy
import os
import json
from random import random
import time
from datetime import datetime
import clip
import torch

from . import utils
from .args import parse_arguments
from .models import evaluate, evaluate_fc, evaluate_wise_ft, finetune, finetune_fc, finetune_icarl,  finetune_gift
from .models.modeling import create_image_classifier


def merge(model_0, model_1, alpha=0.95):
    key_name = [k for k, v in model_0.named_parameters()]
    for i, (param_q, param_k) in enumerate(zip(model_0.parameters(), model_1.parameters())):
        param_k.data = param_k.data * alpha + param_q.data * (1 - alpha)
    return model_1

def mtil_main(args):
    # start_time = time.time()
    # print(args)
    print("MTIL setting")
    utils.seed_all(args.seed)

    if "fc" in args.train_mode:
        assert args.train_mode in ["image-fc", "image-fc-fixed"]
        if args.eval_only:
            model = create_image_classifier(
                args, initialize=args.fc_init, setnone=args.fc_setnone
            )
            if args.load:
                utils.torch_load(model, args.load)
            elif args.save:
                checkpoint_pth = os.path.join(
                    args.save, f"zeroshot_{args.train_dataset}.pth"
                )
                utils.torch_load(model, checkpoint_pth)
            evaluate_fc(model, args)
        else:
            model = finetune_fc(args)
    else:
        assert args.train_mode in ["whole", "text", "image",]

        if args.eval_only:
            torch.cuda.set_device(args.devices[0]) 
            model, tra_preprocess, val_preprocess = clip.load(args.model, jit=False)
            if args.load:
                if args.wise_ft:
                    print("Use wise-ft.")
                    model_0 = copy.deepcopy(model)
                utils.torch_load(model, args.load, device = args.devices[0])
                if args.wise_ft:
                    model = merge(model_0, model, alpha=args.alpha)
            elif args.save:
                checkpoint_pth = os.path.join(
                    args.save, f"clip_zeroshot_{args.train_dataset}.pth"
                )
                utils.torch_save(checkpoint_pth, model)
            evaluate(model, args, val_preprocess, None)
            # evaluate(model, args, tra_preprocess)

        elif args.method in ["icarl"]:
            model = finetune_icarl(args)
        elif args.method in ["GIFT"]:
            model = finetune_gift(args)
        else:
            model = finetune(args)
    
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Start at {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}.")
    # print(f"End at {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}.")
    # print(f"Executed in {execution_time} seconds.")

if __name__ == "__main__":
    args = parse_arguments()
    mtil_main(args)
