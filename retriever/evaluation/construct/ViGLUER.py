from datasets import load_dataset, concatenate_datasets

def create(
        token=None,
        dataset_name_or_path='tmnam20/ViGLUE',
        dataset_repo_id='ContextSearchLM/ViGLUE-R',
        subset=['mnli', 'qnli'],
):
    for name in subset:
        dataset = load_dataset(dataset_name_or_path, name=name, split='test', token=token)
    pass