from utils.helpers_functions import get_configuration, collect_data_confs, load_split, get_random_elements, collect_datastats
from datasets import load_dataset, get_dataset_split_names

print("\n\n === RUNNING EXERCISE 1.1: Loading datasets and split == \n\n")
conf = get_configuration("config.yaml")
print("> Configuration retrieved in config file: {}".format(conf))
print("\n")

DATA_SLUG = collect_data_confs(conf)
print("> Caching HuggingFace Dataset with slug: {}...".format(DATA_SLUG))
splits = get_dataset_split_names(DATA_SLUG)
print("\n")
print("> Current Dataset is composed of: {} splits".format(splits))

print("\n")
print("> Loading different splits...")
train_data = load_split(DATA_SLUG, 'train')
print("Training data: {}".format(train_data))

validation_data = load_split(DATA_SLUG, 'validation')
print("Validation data: {}".format(validation_data))

test_data = load_split(DATA_SLUG, 'test')
print("Test data: {}".format(test_data))

get_random_elements(2, train_data)
print("\n")
get_random_elements(2, validation_data)
print("\n")
get_random_elements(2, test_data)
print("\n")

print("> Collecting some datastats: ")
collect_datastats(train_data, "training_set")
collect_datastats(validation_data, "validation_set")
collect_datastats(test_data, "test_set")

print("\n")



