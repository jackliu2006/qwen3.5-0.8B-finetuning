from datasets import get_dataset_config_names, load_dataset_builder

dataset_name = "jackliu2006/car_knowledge"
ds_builder = load_dataset_builder(dataset_name)


from datasets import load_dataset

dataset = load_dataset(dataset_name, split="train")
print(dataset["model"])