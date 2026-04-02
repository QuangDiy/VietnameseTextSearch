from typing import Callable
from torch import Tensor
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from .loss import cosine_similarity, Losses
from datasets import load_dataset
from .model import SimilarityLoss
from .dataset import MergeContextDataset, GeneralCollator

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def gte_wrapper(model, input_ids, attention_mask, **kwargs):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    embeddings = average_pool(outputs.last_hidden_state, attention_mask)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def get_batch_similar_info(similar: Tensor) -> Tensor:
    batch_size = similar.shape[0]
    data_pair_similar = similar.diagonal()
    diff_pair_similar = similar.sum() - data_pair_similar.sum()

    return (float(data_pair_similar.mean()), float(diff_pair_similar / (batch_size*batch_size-batch_size)), )

def check_model(model, eval_dataset, collator, eval_batch=64):
    import torch
    from tqdm import tqdm
    
    eval_result = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_dataset), eval_batch)):
            model_inputs = collator([eval_dataset[k] for k in range(i, min(i+eval_batch, len(eval_dataset)))])
            loss, query, pos, neg = model(**model_inputs, logger=True)

            result = (float(loss),)
            pos_query_sim = cosine_similarity(query, pos)
            result += get_batch_similar_info(pos_query_sim)

            if neg is not None:
                neg_query_sim = cosine_similarity(query, neg)
                result += get_batch_similar_info(neg_query_sim)

            eval_result.append(result)

    eval_result = torch.tensor(eval_result).mean(dim=0).tolist()
    return {
        'description': f'loss, pair_pos, diff_pos, pair_neg, diff_neg: {eval_result}',
        'loss': eval_result[0],
    }

def training(model,
             tokenizer,
             collator,
             train_dataset,
             eval_dataset=None,
             hub_model_id=None,
             token=None,
             output_dir='./rmtest',
             call_every_train:Callable=None,
             num_repeat=100,
             **training_kwargs,
             ):
    import random

    seed = random.randint(0, 100)
    
    training_arg = TrainingArguments(
        output_dir=output_dir,
        hub_token=token,
        do_train=True,
        seed=seed,
        **training_kwargs,
    )

    trainer = Trainer(
        args=training_arg,
        tokenizer=tokenizer,
        model = model,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    best_result = 999999999999
    per_device_eval_batch_size = training_kwargs.get('per_device_eval_batch_size', 32)

    for _ in range(num_repeat):
        if eval_dataset is not None:
            output_dict = check_model(model, eval_dataset, collator, eval_batch=per_device_eval_batch_size)
            
            print(output_dict['description'])
            if output_dict['loss'] < best_result:
                best_result = output_dict['loss']
                print('find better model, push to hub')
                model.push_to_hub(hub_model_id, token=token, commit_message=output_dict['description'])
        else:
            model.push_to_hub(hub_model_id, token=token)

        if call_every_train is not None:
            call_every_train()

        trainer.train()

    if eval_dataset is not None:
        output_dict = check_model(model, eval_dataset, collator, eval_batch=per_device_eval_batch_size)
        
        print(output_dict['description'])
        if output_dict['loss'] < best_result:
            best_result = output_dict['loss']
            print('find better model, push to hub')
            model.push_to_hub(hub_model_id, token=token, commit_message=output_dict['description'])
    else:
        model.push_to_hub(hub_model_id, token=token)

    if call_every_train is not None:
        call_every_train()

def JustTrainIt(
    dataset_repo={},
    model_name_or_path = 'Alibaba-NLP/gte-base-en-v1.5',
    model_wrapper = gte_wrapper,
    tokenizer_name_or_path = 'vilm/vinallama-7b-chat',
    model_repo_id = 'npvinHnivqn/example_repo_of_just_train_it',
    dropout_list = (0.1, 0.15),
    token=None,
    loss_fn=Losses.ELOSS,
    call_every_train:Callable=None,
    logging_steps=1000,
    learning_rate=5e-5,
    batch_size=32,
    num_repeat=100,
    drop_columns = ['query', 'pos', 'neg'],
    **kwargs,
):
    model, tokenizer = SimilarityLoss.create_from_pretrained(model_name_or_path,
                                                        tokenizer_name_or_path=tokenizer_name_or_path,
                                                        token=token,
                                                        loss_fn=loss_fn,
                                                        model_wrapper=model_wrapper,
                                                        push_to_hub_id=model_repo_id,
                                                        dropout_list=dropout_list)

    train_dataset = MergeContextDataset([
        load_dataset(dataset_repo_id, token=token, **dkwargs['train']) for dataset_repo_id, dkwargs in dataset_repo.items()
    ], remove_columns=drop_columns)

    eval_dataset = MergeContextDataset([
        load_dataset(dataset_repo_id, token=token, **dkwargs['evaluate']) for dataset_repo_id, dkwargs in dataset_repo.items()
    ], remove_columns=drop_columns)

    collator = GeneralCollator()

    training(
        model,
        tokenizer=tokenizer,
        collator=collator,
        train_dataset=train_dataset.workingdataset,
        eval_dataset=eval_dataset.workingdataset,
        hub_model_id=model_repo_id,
        token=token,
        num_repeat=num_repeat,
        call_every_train=call_every_train,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        **kwargs
    )
        