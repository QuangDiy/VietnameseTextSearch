from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd


def create(data):
    data = data.to_pandas().groupby('sentence1').agg(lambda x: list(x)).reset_index()
    
    full_data = {
        'entailment': [],
        'contradiction': [],
        'other': [],
        'neutral': [],
        'anchor': []
    }
    for item in data.iterrows():
        spliter = {}
        for label, sentence in zip(item[1]['gold_label'],item[1]['sentence2']):
            if label not in spliter:
                spliter[label] = [sentence]
            else:
                spliter[label].append(sentence)
        spliter['anchor'] = item[1]['sentence1']
        
        if full_data.keys() == spliter.keys():
            for key in full_data:
                full_data[key].append(spliter[key])
        
    dataframe = pd.DataFrame(full_data)
    return Dataset.from_dict(dataframe.to_dict(orient='list'))

def construct():
    dataset_name = 'presencesw/vinli_4_label'

    dataset = load_dataset(dataset_name)

    for sub_dataset in dataset:
        dataset[sub_dataset] = create(dataset[sub_dataset])

    return dataset

if __name__ == '__main__':
    dataset = construct()
    print(dataset)
    dataset.push_to_hub('ContextSearchLM/ViNLI_remake', token=None)