import copy
import os
import pickle
import random
import clip.clip as clip

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .. import datasets, templates, utils
from .evaluation import evaluate, zeroshot_classifier
from .helpers import merge_we, wise_we, moving_avg, l2_loss, distillation, prepare_ref_dataset, ewc_loss, kl_divergence
import torch
import numpy as np

def finetune_gift(args):

    devices = args.devices
    torch.cuda.set_device(devices[0])

    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)

    # load model
    if args.load is not None:
        utils.torch_load(model, args.load, device=devices[0])
        # require grads (not that useful)
        for name, param in model.named_parameters():
            if name != 'logit_scale':
                param.requires_grad = True
    else:
        pass
    
    # prepare model for ensemble (not used now)
    if args.we_wise or (args.wise_merge and args.wise_ft_model != "zeroshot"):
        print("Using WiSE-FT with Loaded Model")
        model_fix, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
        if args.load is not None:
            utils.torch_load(model_fix, args.load)
    if args.we or args.moving_avg or args.we_wise:
        print("Averaging training")
        if args.moving_avg and args.mv_avg_model == "zeroshot": # mv+zeroshot
            we_model, _, _ =  clip.load(args.model, jit=False)
            we_model.cuda()
            we_n = 0
        else: #we; mv+m; mv+t; we_wise
            we_model = copy.deepcopy(model)
            we_n = 0
            we_model.cuda()
            
    if args.l2 > 0:
        print("L2 norm")
        l2_model = copy.deepcopy(model) # L2 model is the initial model
        l2_model.cuda()
        
    # prepare training dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=8,
    )
    
    # prepare template for training dataset
    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()        

    # number of iterations
    num_batches = len(dataset.train_loader)
    if args.epochs is not None:
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations
    if args.eval_every_epoch:
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval
    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    # params to tune
    frozen_params = ['logit_scale']
    params = [ v for k, v in model.named_parameters() if k not in frozen_params ]
    
    # optimizer
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    # move model to device
    model = model.cuda()
    logit_scale = model.logit_scale
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    # prepare ref dataset for distillation or ita
    if args.distill > 0 or args.ita > 0:

        print("Distillation")
        ref_model, _, test_preprocess = clip.load(args.model, jit=False)
        
        if args.ref_model == "load" and args.load is not None:
            utils.torch_load(ref_model, args.load, device=devices[0])
        
        elif "merge" in args.ref_model:
            # e.g. merge:0.2
            merge_ratio = float(args.ref_model.split(":")[1])
            print (f"Merge Teacher Model: {merge_ratio}")
            for param_q, param_k in zip(ref_model.parameters(), model.module.parameters()):
                param_q.data = param_q.data * merge_ratio + param_k.data * (1- merge_ratio)

        else:
            print(f"Teacher Model: [Initial CLIP]")

        ref_model = ref_model.cuda()
        ref_model = torch.nn.DataParallel(ref_model, device_ids=devices)
        ref_model.eval()

        if args.ita > 0:
            print("ita")

        # prepare ref dataset
        ref_dataset, ref_texts = prepare_ref_dataset(args, test_preprocess)
        ref_iter = iter(ref_dataset.train_loader)

    print(f"Start training on {args.train_dataset}")
    progress_bar = tqdm(range(total_iterations + 1))
    for iteration in progress_bar:

        if eval_iterations is not None and iteration % eval_iterations == 0:
            evaluate(model.module, args, val_preprocess)
        # training
        if iteration % num_batches == 0:
            data_iter = iter(dataset.train_loader)

        # prepare model
        model.train()
        scheduler(iteration)
        # prepare data iter
        if args.train_dataset == 'ImageNet':
            try:
                train_batch = next(data_iter)
            except:
                data_iter = iter(dataset.train_loader)
                train_batch = next(data_iter)
            images, labels = train_batch["images"], train_batch["labels"]
        else:
            try:
                images, labels = next(data_iter)
            except:
                data_iter = iter(dataset.train_loader)
                images, labels = next(data_iter)

        images, labels = images.cuda(), labels.cuda()

        # -- get image embedding --
        out = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)

        loss_dict = {}  # record all parts of losses 


        # -- get text embedding --
        embeddings = model(None, texts)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        logits_per_image = logit_scale.exp() * out @ embeddings.t()

        ce_loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)
        loss = ce_loss
        loss_dict['ce_loss'] = ce_loss.item()

        batch_acc = (logits_per_image.argmax(dim=-1) == labels).float().mean().item()
        ref_batch_acc = None

        # -- distillation and ita --
        if args.distill > 0 or args.ita > 0:
            try:
                ref_batch = next(ref_iter)
            except:
                ref_iter = iter(ref_dataset.train_loader)
                ref_batch = next(ref_iter)
                
            ref_images, ref_labels = ref_batch["images"].cuda(), ref_batch["labels"].cuda()

            batch_ref_texts = [ref_texts[x] for x in ref_labels]
            batch_ref_texts = clip.tokenize(batch_ref_texts).cuda()

            with torch.no_grad():
                ref_embeddings = ref_model(None, batch_ref_texts)
                ref_embeddings = ref_embeddings / ref_embeddings.norm(
                    dim=-1, keepdim=True
                )
                # -- get ref image embedding --
                ref_out = ref_model(ref_images, None)
                ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)
                logits_ref = logit_scale.exp() * ref_out @ ref_embeddings.t() 
            
            ref_out_current = model(ref_images, None)
            ref_out_current = ref_out_current / ref_out_current.norm(
                dim=-1, keepdim=True
            )

            ref_embeddings_current = model(None, batch_ref_texts)
            ref_embeddings_current = ref_embeddings_current / ref_embeddings_current.norm(
                dim=-1, keepdim=True
            )

            if args.image_only:
                logits_current = logit_scale.exp() * ref_out_current @ ref_embeddings.t()
            elif args.text_only:
                logits_current = logit_scale.exp() * ref_out @ ref_embeddings_current.t()
            else:
                logits_current = logit_scale.exp() * ref_out_current @ ref_embeddings_current.t()
            
            # accuracy on ref batch
            batch_ref_labels = torch.arange(ref_images.shape[0]).cuda()
            ref_batch_acc = (logits_current.argmax(dim=-1) == batch_ref_labels).float().mean().item()

            ref_labels_repeated = ref_labels.view(1, -1).repeat(ref_images.shape[0], 1)
            ref_equal_labels = (ref_labels_repeated == ref_labels.view(-1, 1)).type(torch.float)
            batch_ref_labels = ref_equal_labels / torch.sum(ref_equal_labels, dim=1).view(-1, 1).cuda()

            if args.distill > 0:
                distill_loss = 0
                # if args.image_loss:
                if args.feature_mse: # another form of KD under mild assumption
                    # image_distill_loss = args.distill * F.mse_loss(logits_ref, logits_current)
                    image_distill_loss = args.distill * F.mse_loss(ref_out, ref_out_current)
                elif args.kl_div:
                    image_distill_loss = args.distill * kl_divergence(logits_ref, logits_current, T=args.T)
                else:
                    image_distill_loss = args.distill * distillation(logits_ref, logits_current, T=args.T)
                loss_dict['image_distill_loss'] = image_distill_loss.item()
                distill_loss += image_distill_loss
                # if args.text_loss:
                if args.feature_mse:
                    # text_distill_loss = args.distill * F.mse_loss(logits_ref.t(), logits_current.t())
                    text_distill_loss = args.distill * F.mse_loss(ref_embeddings, ref_embeddings_current)
                elif args.kl_div:
                    text_distill_loss = args.distill * kl_divergence(logits_ref.t(), logits_current.t(), T=args.T)
                else:
                    text_distill_loss = args.distill * distillation(logits_ref.t(), logits_current.t(), T=args.T)
                loss_dict['text_distill_loss'] = text_distill_loss.item()
                distill_loss += text_distill_loss
                loss += distill_loss

            if args.ita > 0:
                ita_loss = 0
                # if args.image_loss:
                image_ita_loss = args.ita * F.cross_entropy(logits_current, batch_ref_labels, label_smoothing=args.ls)
                loss_dict['image_ita_loss'] = image_ita_loss.item()
                ita_loss += image_ita_loss
                # if args.text_loss:
                text_ita_loss = args.ita * F.cross_entropy(logits_current.t(), batch_ref_labels, label_smoothing=args.ls) 
                loss_dict['text_ita_loss'] = text_ita_loss.item()
                ita_loss += text_ita_loss
                loss += ita_loss
                
            # elastic weight consolidation
            if args.awc and (args.static_awc == 0 or iteration < args.static_awc):
                optimizer.zero_grad()

                if args.ita > 0 and args.distill > 0:
                    (distill_loss+ita_loss).backward(retain_graph=True)
                elif args.ita == 0:
                    distill_loss.backward(retain_graph=True)
                elif args.distill == 0:
                    ita_loss.backward(retain_graph=True)

                active_params = {n: p for n, p in model.named_parameters()}

                if args.static_awc != 0 and iteration > 0:
                    t = iteration
                    for n, p in active_params.items():
                        if p.grad is not None and 'logit_scale' not in n:
                            fisher_temp = p.grad.clone() ** 2
                            fisher_temp /= fisher_temp.mean()
                        else:
                            fisher_temp = torch.zeros_like(p) + 1e-6
                        _precision_matrices[n] = _precision_matrices[n].cuda()
                        _precision_matrices[n] = t / (t+1) * _precision_matrices[n] + 1 / (t+1) * fisher_temp
                        _precision_matrices[n] = _precision_matrices[n].detach()
                else:
                    _precision_matrices = {}
                    for n, p in active_params.items():
                        if p.grad is not None and 'logit_scale' not in n:
                            _precision_matrices[n] = p.grad.clone() ** 2
                            _precision_matrices[n] /= _precision_matrices[n].mean()
                        else:
                            _precision_matrices[n] = torch.zeros_like(p) + 1e-6
                    
                        _precision_matrices[n] = _precision_matrices[n].detach()
                
        if args.l2 > 0:
            if args.awc:
                wc_loss = args.l2 * ewc_loss(model, l2_model, _precision_matrices)
            else:
                wc_loss = l2_loss(model, l2_model)
                loss_dict['l2_dis'] = wc_loss.item()
                wc_loss = args.l2 * wc_loss
            loss +=  wc_loss
            loss_dict['wc_loss'] = wc_loss.item()

        formatted_loss_dict = {key: f'{value:.4f}' for key, value in loss_dict.items()}
        progress_bar.set_postfix(loss=loss.item(), batch_acc=batch_acc, ref_batch_acc=ref_batch_acc, loss_dict=formatted_loss_dict)

        # update
        if args.awc:
            (wc_loss+ce_loss).backward()
        else:
            optimizer.zero_grad()
            loss.backward()
        optimizer.step()

        # we
        if (args.we or args.moving_avg or args.we_wise) and iteration % args.avg_freq == 0:
            we_n += 1
            if args.moving_avg:
                if args.mv_avg_model == "t":
                    next_we_model = copy.deepcopy(model.module)
                    moving_avg(model.module, we_model, args.mv_avg_decay)
                    we_model = next_we_model.cuda()
                else: ### args.moving_avg_model == "n" or "zeroshot"
                    moving_avg(model.module, we_model, args.mv_avg_decay)
            elif args.we:
                merge_we(model.module, we_model, we_n)
            else:
                wise_we(model.module, we_model, we_n, model_fix, args.we_wise_alpha)

        elif args.dump and iteration % args.avg_freq == 0:
            dump_id = iteration // args.avg_freq
            to_save_model = model.module
            path = os.path.join(args.save, f"{args.train_dataset}_{str(dump_id)}.pth")
            utils.torch_save(to_save_model, path)

    # Saving model
    if args.save is not None:
        if args.we or args.we_wise:
            to_save_model = we_model
        else:
            to_save_model = model.module
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(to_save_model, path)