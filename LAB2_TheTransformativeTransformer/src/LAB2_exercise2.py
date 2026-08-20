
from utils.helpers_functions import get_configuration, collect_data_confs, get_dataset_split_names, load_split, collect_model_confs, load_model_tok, preprocessing_function
from transformers import pipeline
import torch
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os
import numpy as np

from transformers import set_seed
set_seed(99)

# Devices
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Running device mode: {}".format(device))

conf = get_configuration("config.yaml")
DATA_SLUG = collect_data_confs(conf)
splits = get_dataset_split_names(DATA_SLUG)
train_data = load_split(DATA_SLUG, 'train')
validation_data = load_split(DATA_SLUG, 'validation')
test_data = load_split(DATA_SLUG, 'test')
if train_data:
    print("Training data loaded")
if validation_data:
    print("Validation data loaded")
if test_data:
    print("Test data loaded")

MODEL_SLUG = collect_model_confs(conf)
model, tokenizer = load_model_tok(MODEL_SLUG)
if model:
    print("Model loaded")
if tokenizer:
    print("Tokenizer loaded")

#### EXERCISE 2.1 ####

EXECUTE_TOKENIZATION = conf['TOKENIZED_DATA']['exec_tokenization']
SAVE_PATH = conf['TOKENIZED_DATA']['save_path']

if EXECUTE_TOKENIZATION:
    print("> TOKENIZING TRAINING SET...")
    tokenized_train = train_data.map(preprocessing_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_train.set_format('pt', columns=['input_ids'], output_all_columns=True)

    print("> TOKENIZING VALIDATION SET...")
    tokenized_validation = validation_data.map(preprocessing_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_validation.set_format('pt', columns=['input_ids'], output_all_columns=True)

    print("> TOKENIZING TEST SET...")
    tokenized_test = test_data.map(preprocessing_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_test.set_format('pt', columns=['input_ids'], output_all_columns=True)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print(f"> Saving tokenized datasets to {SAVE_PATH}...")
    torch.save({'train': tokenized_train, 'validation': tokenized_validation, 'test': tokenized_test}, SAVE_PATH)

else:
    print(f"> Loading tokenized datasets from {SAVE_PATH}...")
    data = torch.load(SAVE_PATH)
    tokenized_train = data['train']
    tokenized_validation = data['validation']
    tokenized_test = data['test']
    print("Tokenized Training Set successfully loaded!")
    print("Tokenized Validation Set successfully loaded!")
    print("Tokenized Test Set successfully loaded!")
    print(tokenized_train)

print(model)

#### EXERCISE 2.2 ####
from transformers import AutoModelForSequenceClassification

#https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification

cls_model = AutoModelForSequenceClassification.from_pretrained(MODEL_SLUG, num_labels=2)
print(cls_model)

#### EXERCISE 2.3 ####
"""
For this exercise I think it would be nice to explore some of the training arguments
I'd like to make HPO over: Learning Rate, Batch Size and a couple of optimizers(?)

By looking at:
https://huggingface.co/docs/transformers/v4.51.1/en/main_classes/trainer#transformers.TrainingArguments

I notice that interesting args could be:
- output_dir: The output directory where the model predictions and checkpoints will be written.
- do_train (bool, optional, defaults to False) — Whether to run training or not. This argument is not directly used by Trainer, 
            it’s intended to be used by your training/evaluation scripts instead. See the example scripts for more details.
- do_eval: Whether to run evaluation on the validation set or not. Will be set to True if eval_strategy 
           is different from "no". This argument is not directly used by Trainer, it’s intended to be used 
           by your training/evaluation scripts instead. See the example scripts for more details.
- eval_strategy (str or IntervalStrategy, optional, defaults to "no") — The evaluation strategy to adopt during training. Possible values are:
    "no": No evaluation is done during training.
    "steps": Evaluation is done (and logged) every eval_steps.
    "epoch": Evaluation is done at the end of each epoch. <-- I want to use epoch in order to make a validation run ath the end 
    of every epoch
- learning_rate (float, optional, defaults to 5e-5) — The initial learning rate for AdamW optimizer. I want to HPO this
- optim (str or training_args.OptimizerNames, optional, defaults to "adamw_torch") — 
The optimizer to use, such as “adamw_torch”, “adamw_torch_fused”, “adamw_apex_fused”, “adamw_anyprecision”, “adafactor”. 
See OptimizerNames in training_args.py for a full list of optimizers. <-- lets try AdamW default and SGD
SGD = "sgd"; ADAMW_TORCH = "adamw_torch"
"""

HPO_LRS = conf['FINE_TUNING_PHASE']['learning_rate']
HPO_BS = conf['FINE_TUNING_PHASE']['batch_size']
HPO_OPTS = conf['FINE_TUNING_PHASE']['optimizers']
RUN_HPO = conf['FINE_TUNING_PHASE']['run_hpo']
BEST_MODEL = conf['FINE_TUNING_PHASE']['best_model_path']

from transformers import DataCollatorWithPadding, Trainer
from utils.HyperparamOptim import HyperparamSearcher

# The collator is used in order to prepare data in the correct "padded" format
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

# Now get an instance of the HP Searcher defined in utils 
# This will do the same process of Flipped Lecture, but on multiple configs in order
# to run an HPO

if RUN_HPO:
    searcher = HyperparamSearcher(model_slug=MODEL_SLUG, tokenizer=tokenizer, data_collator=data_collator, 
                                  lr_list=HPO_LRS, bs_list=HPO_BS, optims_list=HPO_OPTS, output_dir="hpo_results")
    searcher.connect_datasets(training_data=tokenized_train, validation_data=tokenized_validation)
    hpo_results, best_model_path = searcher.run()

    """Config: {'learning_rate': 2e-05, 'batch_size': 16, 'optimizer': 'adamw_torch'} | Val Loss: 0.480242520570755
    Config: {'learning_rate': 2e-05, 'batch_size': 16, 'optimizer': 'sgd'} | Val Loss: 0.6943866610527039
    BEST Config: {'learning_rate': 2e-05, 'batch_size': 64, 'optimizer': 'adamw_torch'} | Val Loss: 0.351376473903656
    Config: {'learning_rate': 2e-05, 'batch_size': 64, 'optimizer': 'sgd'} | Val Loss: 0.6947872042655945
    Config: {'learning_rate': 2e-06, 'batch_size': 16, 'optimizer': 'adamw_torch'} | Val Loss: 0.40191367268562317
    Config: {'learning_rate': 2e-06, 'batch_size': 16, 'optimizer': 'sgd'} | Val Loss: 0.6949373483657837
    Config: {'learning_rate': 2e-06, 'batch_size': 64, 'optimizer': 'adamw_torch'} | Val Loss: 0.48633065819740295
    Config: {'learning_rate': 2e-06, 'batch_size': 64, 'optimizer': 'sgd'} | Val Loss: 0.6949838399887085"""
else:
    best_model_path = BEST_MODEL
    print(f"\n[EVAL] Skipping HPO. Loading pre-configured best model from: {best_model_path}")

# This is performer OR after the HPO, or instead of the HPO recovering the best model path from the config file yaml
print(f"\n> Evaluating best model ({best_model_path}) on Test Set...")
model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
trainer = Trainer(model=model, data_collator=data_collator)
output = trainer.predict(tokenized_test)
y_pred = np.argmax(output.predictions, axis=-1)
y_true = output.label_ids
accuracy = accuracy_score(y_true, y_pred)
print("Test Loss:", output.metrics["test_loss"])
print("Test Accuracy:", accuracy)
db_ft_classification_report = classification_report(y_true, y_pred)
print("\nClassification Report on Test Set:\n")
print(db_ft_classification_report)
exp_folder = "experiments"
os.makedirs(exp_folder, exist_ok=True)
report_path = os.path.join(exp_folder, "distilbert_classification_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(db_ft_classification_report)

print(f"> Report correctly saved to {report_path}")