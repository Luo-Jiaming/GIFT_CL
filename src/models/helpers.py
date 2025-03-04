import json
import os
import clip.clip as clip
import torch
import torch.nn.functional as F

from .. import datasets

with open('project_config.json', 'r') as f:
    project_config = json.load(f)

SYN_DATA_LOCATION = project_config['SYN_DATA_LOCATION']
CL_DATA_LOCATION = project_config['CL_DATA_LOCATION']
MTIL_DATA_LOCATION = project_config['MTIL_DATA_LOCATION']

def batch(iterable, n=64):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_datasets_text(ds, args):
    texts = []
    for d in ds:
        ref_sentences_cls = getattr(datasets, d)
        ref_sentences = ref_sentences_cls(
            None,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        ref_template = ref_sentences.template
        ref_texts = [ref_template(x) for x in ref_sentences.classnames]
        texts.extend(ref_texts)
    ret = clip.tokenize(texts).cuda()
    return ret

def merge_we(model_0, model_1, sma_count):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_k.data = (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
    return model_1

def wise_we(model_0, model_1, sma_count, model_n, alpha=0.95):
    for param_q, param_k, param_n in zip(model_0.parameters(), model_1.parameters(), model_n.parameters()):
        param_k.data = (
                        (param_k.data * sma_count + param_q.data) / (1.0 + sma_count)
                    ) * alpha + param_n.data * (1-alpha)
    return model_1

def moving_avg(model_0, model_1, alpha=0.999):
    for param_q, param_k in zip(model_0.parameters(), model_1.parameters()):
        param_q.data = param_q.data * alpha + param_k.data * (1 - alpha)

def l2_loss(model, model_ref):
    loss = 0.0
    for param_q, param_k in zip(model.parameters(), model_ref.parameters()):
        loss += F.mse_loss(param_q, param_k.detach(), reduction="sum")
    return loss

def ewc_loss(model, model_ref, precision_matrices):
    loss = 0.0
    for (name_q, param_q), (_, param_k) in zip(model.named_parameters(), model_ref.named_parameters()):
        _loss = precision_matrices[name_q] * (param_q - param_k.detach()) ** 2
        loss += _loss.sum()
    return loss

def virtual_vocab(length=10, n_class=1000):
    voc_len = len(clip._tokenizer.encoder)
    texts = torch.randint(0, voc_len, (n_class, length))
    start = torch.full((n_class, 1), clip._tokenizer.encoder["<start_of_text>"])
    end = torch.full((n_class, 1), clip._tokenizer.encoder["<end_of_text>"])
    zeros = torch.zeros((n_class, 75 - length), dtype=torch.long)

    texts = torch.cat([start, texts, end, zeros], dim=1)
    return texts

def distillation(t, s, T=2, reduction="mean"):
    p = F.softmax(t / T, dim=1)
    loss = F.cross_entropy(s / T, p, reduction=reduction) * (T ** 2)
    return loss

def kl_divergence(t, s, T=2, reduction="batchmean"):
    p = F.log_softmax(t / T, dim=1)
    q = F.softmax(s / T, dim=1)
    loss = F.kl_div(p, q, reduction=reduction) * (T ** 2)
    return loss

def paired_loss_new(old_pred, old_true):
    T = 2
    pred_soft = F.softmax(old_pred[:, : old_true.shape[0]] / T, dim=1)
    true_soft = F.softmax(old_true[:, : old_true.shape[0]] / T, dim=1)
    loss_old = true_soft.mul(-1 * torch.log(pred_soft))
    loss_old = loss_old.sum(1)
    loss_old = loss_old.mean() * T * T
    return loss_old

def prepare_ref_dataset(args, test_preprocess):
    if args.ref_dataset == "SyntheticDataset_A":
        ref_dataset_cls = getattr(datasets, "SyntheticDataset")
        ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=os.path.join(SYN_DATA_LOCATION, 'synthetic_data_a/' + args.train_dataset +'_Syn/'),
                batch_size=args.batch_size,
                num_workers=8,
                image_nums=args.image_nums,
            )
        
        ref_texts = ref_dataset.all_prompts
    elif args.ref_dataset == "SyntheticDataset_B":
        ref_dataset_cls = getattr(datasets, "SyntheticDataset")
        ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=SYN_DATA_LOCATION+'synthetic_data_b/' + args.train_dataset +'_Syn/',
                batch_size=args.batch_size,
                num_workers=8,
            )
        ref_texts = ref_dataset.all_prompts
    elif args.ref_dataset == "ImageNetSUB":
        dataset_names =  ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]
        seed_offset = dataset_names.index(args.train_dataset)
        ref_dataset_cls = getattr(datasets, "ImageNetSUB")
        ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=CL_DATA_LOCATION,
                batch_size=args.batch_size,
                num=args.image_nums,
                random_seed=42+seed_offset,
                num_workers=8,
            )
        ref_template = ref_dataset.template
        ref_texts = [ref_template(x) for x in ref_dataset.classnames]
    elif args.ref_dataset == "ImageNet":
        # Use ImageNet as reference 
        ref_dataset_cls = getattr(datasets, "ImageNet")
        ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=CL_DATA_LOCATION,
                batch_size=args.batch_size,
                num_workers=8,
            )
        
        ref_template = ref_dataset.template
        ref_texts = [ref_template(x) for x in ref_dataset.classnames]

    return ref_dataset, ref_texts
    
    