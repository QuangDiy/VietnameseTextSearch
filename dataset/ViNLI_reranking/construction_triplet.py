from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd


def spliter(samples):
    result = {
        'anchor': [],
        'pos': [],
        'neg': [],
    }
        
    for index in range(len(samples['anchor'])):
        for pos_sentence in samples['pos'][index]:
            for neg_sentence in samples['neg'][index]:
                result['anchor'].append(samples['anchor'][index])
                result['pos'].append(pos_sentence)
                result['neg'].append(neg_sentence)
                
    return result

def create(data):
    data = data.map(spliter, batched=True)
    return data

def construct():
    dataset_name = 'ContextSearchLM/ViNLI_reranking'

    dataset = load_dataset(dataset_name, token=None)

    for sub_dataset in dataset:
        dataset[sub_dataset] = create(dataset[sub_dataset])

    return dataset

if __name__ == '__main__':
    dataset = construct()
    print(dataset)
    dataset.push_to_hub('ContextSearchLM/ViNLI_triplet', token=None)