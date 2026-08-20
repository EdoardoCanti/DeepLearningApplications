
import yaml
from pathlib import Path
from datasets import load_dataset, get_dataset_split_names
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import torch

def set_seed(seed=99):
    # https://docs.pytorch.org/docs/2.13/notes/randomness.html
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return seed

### Helpers related to data
# All config files (but probably yhere will be one) will be saved in configs dir
# This function will take only the name of the config file of interest and will load it 
# from the configs dir
def get_configuration(config_file_name: str) -> dict:
    CONFIGS_DIRNAME = "configs"
    # wrt to THIS location (__file__) move "up" by 2
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # filenames will be like <config_file_name>.yaml
    # if the passed string doesn't have the extension add it
    name, extension = os.path.splitext(config_file_name)
    if not extension:
        extension = ".yaml"
        config_file_name = name+".yaml"
    config_file_path = os.path.join(project_root, CONFIGS_DIRNAME ,config_file_name)
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# given a configuration file
# return only those values related to data
def collect_data_confs(yaml_as_dict:  dict):
    dataset_slug = yaml_as_dict["DATA"]["slug"]
    return dataset_slug

# create a dictionary of type {<name_split>:<Dataset itself>}
def load_split(dataset_slug: str, in_split: str) -> dict:
    data_split = load_dataset(dataset_slug, split = in_split)
    return data_split

# given a configuration file
# return only those values related to the model
def collect_model_confs(yaml_as_dict:  dict):
    model_slug = yaml_as_dict["MODEL"]["slug"]
    return model_slug

# Function for familiarize with HuggingFace Datasets
# https://huggingface.co/docs/datasets/access
def get_random_elements(number_of_elements: int, data):
    # using numpy randint
    # random.randint(low, high=None, size=None, dtype=int)
    print("> Printing {} random items from data...".format(number_of_elements))
    random_indices = np.random.randint(low=0, high=len(data), size=number_of_elements)
    for i in range(len(random_indices)):
        print("{}".format(data[i]))

# Given a huggingface dataset and a path for saving a json data
# collects simple stats about texts lenghts and labels count in that dataset
def collect_datastats(data, split_name: str, statistics_dirname: str = "data_stats") -> str:
    stats_dict = dict()
    texts_lengths = list()
    # as shown in https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
    # this dataset only have two labels so
    labels_counter = dict()
    positive_lbl_count = 0
    negative_lbl_count = 0
    for sample in data:
        text = sample['text']
        txt_len = len(text)
        texts_lengths.append(txt_len)
        if sample['label'] == 1:
            positive_lbl_count += 1
        else:
            negative_lbl_count += 1
    texts_min_len = np.min(texts_lengths)
    texts_max_len = np.max(texts_lengths)
    text_mean_len = np.mean(texts_lengths)
    labels_counter['positives'] = positive_lbl_count
    labels_counter['negatives'] = negative_lbl_count 
    stats_dict['minimum text length'] = int(texts_min_len)
    stats_dict['maximum text length'] = int(texts_max_len)
    stats_dict['mean text lenght'] = float(text_mean_len)
    stats_dict['label counts'] = labels_counter
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, statistics_dirname)
    os.makedirs(full_path, exist_ok=True)
    with open(os.path.join(full_path, "{}.json".format(split_name)), "w") as file:
        json.dump(stats_dict, file, indent=4)

    # It would be interesting to know if data are collected wrt the text lengths
    # if yes this could be somehow to take into account during the batching (or at least considered)
    # Doing this using a matplotlib barplot x := sample index, y := sample text len
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(texts_lengths)), texts_lengths)
    plt.title("Texts lengths in {}".format(split_name))
    plt.xlabel("index")
    plt.ylabel("text length")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(full_path, "{}_texts_lengths.png".format(split_name)))
    plt.close()
    print("Statistics for data saved at: {}".format(os.path.join(full_path, "{}.json".format(split_name))))
    return os.path.join(full_path, "{}.json".format(split_name))


### Helpers related to model

# Given a HuggingFace slug collect the model from AutoModel
def collect_model_confs(yaml_as_dict:  dict):
    model_slug = yaml_as_dict["MODEL"]["slug"]
    return model_slug

def get_model_from_slug(hf_slug: str):
    model = AutoModel.from_pretrained(hf_slug)
    if model:
        print("Model with slug: {} correctly retrieved.".format(hf_slug))
    return model

# Given a HuggingFace slug collect the tokenizer from AutoTokenizer
def get_tokenizer_from_slug(hf_slug: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_slug)
    if tokenizer:
        print("Tokenizer with slug: {} correctly retrieved.".format(hf_slug))
    return tokenizer

# Loading both model and tokenizer from the same slug
def load_model_tok(hf_slug: str):
    model = get_model_from_slug(hf_slug)
    tokenizer = get_tokenizer_from_slug(hf_slug)
    return model, tokenizer

from utils.CustomTokenizer import CustomTokenizer
# Now it would be interesting to collect statistics (on each splits)
# about tokens; so applying a "tokens-oriented" version of the previous defined collect_datastats
def collect_tokenstats(custom_tokenizer: CustomTokenizer, data, split_name: str, statistics_dirname: str = "data_stats") -> str:
    stats_dict = dict()
    tokens_per_sample = list()
    for sample in data:
        text = sample['text']
        # The follow will return a list of tokens id (depending on the vocabulary used by the AUTOTOKENIZER)
        #current_sample_tokens = tokenizer.encode(text) 
        current_sample_tokens = custom_tokenizer.count_tokens(text)
        tokens_per_sample.append(current_sample_tokens)
    tokens_min_len = np.min(tokens_per_sample)
    tokens_max_len = np.max(tokens_per_sample)
    tokens_mean_len = np.mean(tokens_per_sample)
    stats_dict['minimum number of tokens'] = int(tokens_min_len)
    stats_dict['maximum number of tokens'] = int(tokens_max_len)
    stats_dict['mean number of tokens'] = float(tokens_mean_len)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(project_root, statistics_dirname)
    os.makedirs(full_path, exist_ok=True)
    with open(os.path.join(full_path, "{}_TOKENS.json".format(split_name)), "w") as file:
        json.dump(stats_dict, file, indent=4)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(tokens_per_sample)), tokens_per_sample)
    plt.title("Texts lengths in {}".format(split_name))
    plt.xlabel("index")
    plt.ylabel("tokens numbers")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(full_path, "{}_tokens_lengths.png".format(split_name)))
    plt.close()
    print("Statistics for tokens saved at: {}".format(os.path.join(full_path, "{}_TOKENS.json".format(split_name))))
    return os.path.join(full_path, "{}_TOKENS.json".format(split_name))

def encode_random_text(custom_tokenizer: CustomTokenizer, data, to_tensor: bool = False):
    random_index = np.random.randint(low=0, high=len(data))
    random_sample = data[random_index]
    random_text = random_sample['text']
    text, tokens = custom_tokenizer.encode_string(random_text, as_tensor = to_tensor)
    return text, tokens

def count_unk_collisions(custom_tokenizer: CustomTokenizer, data) -> int:
    UNK_ID = 100
    unk_counter = 0
    for sample in data:
        text = sample["text"]
        _, tokens_from_text = custom_tokenizer.encode_string(text)
        for t in tokens_from_text:
            if t == UNK_ID:
                unk_counter += 1
    return unk_counter

# LLM USED HERE
# asjed Gemini how to modify an existing json file by adding some new infos
# left exactly as Gemini wrote it, just added the project_root path stuff
def update_json(file_path, new_data):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(project_root, file_path)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                # Handle empty or malformed files
                data = {}
    else:
        data = {}

    data.update(new_data)

    
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# Helpers for LAB2_exercise1_3.py
from transformers import pipeline
# Here are written several helper functions to start the feature extraction pipeline

def instanciate_extractor(model, tokenizer, device):
    extractor = pipeline("feature-extraction", model = model, tokenizer = tokenizer, device=device)
    if extractor:
        return extractor
    else:
        raise ValueError("[FATAL] Unable to instanciate pipeline extractor!")
    
# This returns a list of features
# The list contains as many elements as CARDINALITY(data)
# each element is a tensor of batch size = 1 (a singleton for each unit in data)
# every singleton has T tokens (where T is the cardinality of tokens extracted from the text)
# embedded in a D vector space (in the case of distil-BERT uncased is 768, as Bert-Base)
# By remembdering the fact that BERT uses CLS(pre-attached) and SEP(post-attached) special tokens
# it is possible to state the following:
# (example on trivial case 1 token == 1 word)
# "This is a sample extracted from data"
# features[i].shape is([1, 9, 768]), 9 because 7 words + 2 special tokens and 768 embedding dimension 
def get_features(extractor, data):
    features = extractor(list(data), return_tensors = 'pt')
    if features:
        return features
    else:
        print("[WARNING] Features not extracted")
        return None
    
# Given what stated in the previous comment consider that
# in order to access to the CLS token of each unit in data we should do
# features[i][0][0]
def collect_cls_tokens(features_list: list):
    # For tensor in features_list:
    # Let's consider its unique singleton/batch_element (first 0)
    # of that singleton consider the first generated token (which is [CLS] by construction) (second 0)
    # consider all 768 dimensions of that [CLS] token! (:)
    cls_embeddings = [f[0, 0, :] for f in features_list]
    as_tensor = torch.stack(cls_embeddings)
    return as_tensor

# This function merges last two function declared here
def get_cls_embeddings(extractor, data, save_filename: str = None):
    features = get_features(extractor, data)
    cls_embeddings = collect_cls_tokens(features)
    if save_filename:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        artifacts_folder = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_folder, exist_ok=True)
        saving_path = os.path.join(artifacts_folder, save_filename)
        torch.save(cls_embeddings.cpu(), saving_path)
        print("[CLS] embeddings saved at: {}".format(saving_path))
    return cls_embeddings

# Utils for exercise2
def preprocessing_function(dataset, tokenizer):
    return tokenizer(dataset['text'], truncation = True)

# Functions for CLIP (Exercise 3)
import random
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
from transformers import AutoProcessor
import wandb

def zero_shot_random_image(dataset, model, labels_names):
    random_index = random.randint(0, dataset['train'].num_rows-1)
    image = dataset['train']['image'][random_index]
    real_class = dataset['train']['label'][random_index]
    preds = model(image, candidate_labels = labels_names)
    return preds, real_class, image

# This function is used in order to run inference using CLIP
# will be used for zero-shot, text-encoder only fine tuned, vision-encoder only fine tuned and both encoders fine tuned
# by default the description is setted to the zero shot 
def evaluate_clip(dataset_split, model, labels_names, prompts, desc = "CLIP zero shot on tobacco3843"):
    y_true_names = [] # a list for real classes names (gold standards as names, not as integers)
    y_pred_names = [] # the same list but for predicted classes by CLIP (we need them at the end for metrics eval)
    probs_to_true_names = [] # (sorry for the name) this will contain the probability that CLIP assigned to the REAL class
    progress_bar = tqdm(dataset_split, desc=desc) #CLIP zer
    results = [] # This will contain for each i in the datase the y_true_name, y_pred_name_ probs_to_true, in order to serialize results!

    for i, item in enumerate(progress_bar):
        # the image in the dataset
        image = item['image']
        # the associated label (the real one)
        # AS SAW IN EDA THIS IS AN INTEGER
        real_class_idx = item['label']
        # attaching the class label to the current image label index
        y_true_name = labels_names[real_class_idx]
        # for each label there is a prompt, recover it
        label_related_prompt = prompts[real_class_idx]

        # The core is here: Running CLIP zero-shot inference on the current image
        # by asking it to assign scores on candidate labels (those defined before)
        preds = model(image, candidate_labels=prompts)
        
        # As saw in previous tests (and in previous cell)
        # CLIP predictions are returned as a list of dicts
        # each dictionary is composed of {<label>:<score>}, where label is clearly from candidate_labels argument
        # IT IS IMPORTANT TO REMEMBER THAT THOSE PREDICTIONS ARE RETURNED IN DECREASING ORDER OF PROBABILITY
        # This means that the first dictionary in the list is the one with highest probability on the attached label
        best_predicted_prompt = preds[0]['label'] 
        # LLM used here: asked gemini a fast way to retrieve the index given the label
        # for instance let says that the best predicted prompt is 'A scanned image representing a Letter document'
        pred_class_idx = prompts.index(best_predicted_prompt)  # The related class index is 3
        pred_class_name = labels_names[pred_class_idx] # From this I'mg going to retrieve the label itself
        
        # I'd like also to return the probability given by CLIP to the REAL
        # this could be interesting in order to understand "HOW MUCH CLIP WAS WRONG?"
        true_class_score = 0.0
        for p in preds: # for each dictionary in the predicitions
            # consider the one with the prompt attached to the REAL LABEL
            if p['label'] == label_related_prompt: 
                true_class_score = p['score'] # This is is the probability that CLIP assigns to the prompt of the REAL CLASS
                break
                
        y_true_names.append(y_true_name)
        y_pred_names.append(pred_class_name)
        probs_to_true_names.append(true_class_score)

        # At this point we have:
        # Three different lists of the same length
        # for the same index i exists:
        #   1. THE TRUE CLASS OF THE IMAGE
        #   2. THE CLIP PREDICTED CLASS OF THE IMAGE
        #   3. THE PROBABILITY THAT CLIP ASSIGNED TO THE REAL CLASS DURING THE ZERO SHOT
        #
        # It would be nice to find a way to save these list with also a preview of the image
        # like several images dataset do in HuggingFace but I think it will take time
        # and probably I would not get the expected result.
        # Maybe what could be done is to randomly select a sample of images and show predicted and real
        # class (as done over LAB1 GTSRB)
        #

        # Once in a while tell me you are alive
        if i % 250 == 0 and i != 0:
            print("> 250 images passed\n") 

        results.append({"i": i, "real_class": y_true_name, "predicted_class": pred_class_name, "clip_prob_to_real_class": round(true_class_score, 2)})
    
    return y_true_names, y_pred_names, probs_to_true_names, results

# Why this function?
# The starting point of this was to have a custom Dataset Class that would have preprocess data on the fly
# it was a huge bottleneck... The new way was to preprocess tobacco data before and then pass them to the model
# kernel crashed several times due to mem issues
def preprocess_in_batches(images, texts, processor, batch_size=100):
    pixel_values_list = []
    input_ids_list = []
    # preprocess data in blocks of batch_size items at time
    for i in range(0, len(images), batch_size):
        batch_imgs = list(images[i:i+batch_size]) # cosnder images in batch
        batch_txts = texts[i:i+batch_size] # with attached texts
        inputs = processor(images=batch_imgs, text=batch_txts, return_tensors="pt", padding="max_length", truncation=True)
        pixel_values_list.append(inputs["pixel_values"])
        input_ids_list.append(inputs["input_ids"])
    # the return tensor concatenation was produced with Gemini
    return {"pixel_values": torch.cat(pixel_values_list, dim=0), "input_ids": torch.cat(input_ids_list, dim=0)}

def build_lora_model(model_slug, device, model_component="all"):
    if model_component not in ["text_model", "vision_model", "all"]:
        raise ValueError("Error: model_component must be: 'text_model', or 'vision_model', or 'all'")
    # After several attempts I think we should always retrieve the starting model
    clip_model = AutoModel.from_pretrained(model_slug)
    clip_processor = AutoProcessor.from_pretrained(model_slug)
    # This is the list of modules that will be passed to LoRA
    target_modules = []
    for name in clip_model.named_modules():
        if model_component == "all":
            if 'k_proj' in name[0] or 'q_proj' in name[0]:
                target_modules.append(name[0])
        else:
            if name[0].startswith(model_component) and ('k_proj' in name[0] or 'q_proj' in name[0]):
                target_modules.append(name[0])
    encoder_lora_config = LoraConfig(r=16, target_modules=target_modules, lora_alpha=32, lora_dropout=0.05)
    # Citing the official documentation at https://huggingface.co/docs/peft/guides/peft_model_config: 
    # "Use the get_peft_model() function to create a PeftModel from the base facebook/opt-350m model and the lora_config you created earlier."
    # this means that we can use the already instanced model to get a PEFT MODEL
    peft_encoder = get_peft_model(clip_model, encoder_lora_config)
    peft_encoder.print_trainable_parameters()
    peft_encoder.to(device)
    return peft_encoder, clip_processor, target_modules

def train_peft_model(model, train_dataloader, test_dataloader, project_name, run_name, lr, epochs, batch_size, device):
    # Consider that now the model is the PEFT MODEL:
    # The optimizer must optimize the PEFT MODEL
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if wandb.run is not None:
        wandb.finish()

    wandb.init(reinit=True, project=project_name, name=run_name,  
        config={"learning_rate": lr, "epochs": epochs, "batch_size": batch_size},
        settings=wandb.Settings(start_method="thread") # suggested by gemini
    )

    for epoch in range(epochs):
        model.train()
        current_training_loss = 0.0
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            # return loss is necessary because by default CLIP didn't return it
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_training_loss = current_training_loss + loss.item()
        avg_train_loss = current_training_loss / len(train_dataloader)

        # At the end of the epoch run over the test
        model.eval()
        cur_val_loss = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True)
                cur_val_loss = cur_val_loss + outputs.loss.item()
        avg_val_loss = cur_val_loss / len(test_dataloader)
        print("Epoch {}: Train Loss: {}, Val Loss: {};".format(epoch, avg_train_loss, avg_val_loss))
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch + 1})
    wandb.finish()
    return model