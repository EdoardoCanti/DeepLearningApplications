import os
import torch
import wandb
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from utils.helpers_functions import get_configuration
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# I want to define this object over the lists of possible hyperparams
# in our case: learning rates list, batch sizes list and optimizers list
class HyperparamSearcher:
    def __init__(self, model_slug: str, tokenizer, data_collator, 
                 lr_list: list, bs_list: list, optims_list: list, output_dir: str = "hpo"):
        self.model_slug = model_slug
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.learning_rates = lr_list
        self.batch_sizes = bs_list
        self.optimizers = optims_list
        print("HPO space: \n learning_rates: {} \n batch sizes: {} \n optimizers: {}".format(self.learning_rates, self.batch_sizes, self.optimizers))
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def connect_datasets(self, training_data, validation_data):
        self.training_data = training_data
        self.validation_data = validation_data
        if self.training_data:
            print("[HyperparamSearcher] Training dataset connected")
        if self.validation_data:
            print("[HyperparamSearcher] Validation dataset connected")

    def create_hp_space(self):
        self.confs_list = list()
        for lr in self.learning_rates:
            for bs in self.batch_sizes:
                for opt in self.optimizers:
                    conf_dict = { 'learning_rate': lr, 'batch_size': bs, 'optimizer': opt }
                    self.confs_list.append(conf_dict)
        return self.confs_list

    # HEAVILY BASED on HF video tutorial: 
    # https://www.youtube.com/watch?v=nvBXf7s7vTI
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="macro")
        recall = recall_score(labels, preds, average="macro")
        f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # for each configuration in hpo space I want to do the same we did i flipped lectrure
    def run(self):
        confs = self.create_hp_space()
        total_runs = len(confs)
        print("\n[HyperparamSearcher] Starting HPO on {} configurations".format(total_runs))
        results = []
        for conf in confs:
            lr = conf['learning_rate']
            bs = conf['batch_size']
            opt = conf['optimizer']
            run_name = "run_lr_{}_bs_{}_opt_{}".format(lr, bs, opt)
            run_output_dir = os.path.join(self.output_dir, run_name)
            print(">> Running: LR={}, BS={}, OPT={}".format(lr, bs, opt))
            wandb.init(project="rotten-tomatoes-hpo-final-version", name=run_name, reinit=True)

            # Considering the model that I want to train
            # https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/models/distilbert/modeling_distilbert.py#L767
            # At the previous link we can see labels and crossEntropyLoss
            model = AutoModelForSequenceClassification.from_pretrained(self.model_slug, num_labels=2)

            # Preparing the arguments, this is parametrized in order to run HPO
            current_training_args = TrainingArguments(
                output_dir=run_output_dir, # current cofnig output dir
                eval_strategy="epoch", # eval at the end of each epoch
                save_strategy="epoch", # seaving at each epoch
                learning_rate=lr, # current lr
                per_device_train_batch_size=bs, # current bs (same for traing ed eval)
                per_device_eval_batch_size=bs,
                optim=opt, # the current optimizer
                num_train_epochs=3,
                report_to="wandb",
                run_name=run_name,
                logging_strategy="steps",
                logging_steps=1, # logging each step as flipped lecture
                load_best_model_at_end=True,
                metric_for_best_model="accuracy"
            )
            
            # Get a trainer over the current model, with previous args to be validate over validation data 
            # using the collator in order to pad inputs
            trainer = Trainer(model=model, args=current_training_args, train_dataset=self.training_data,
                eval_dataset=self.validation_data, tokenizer=self.tokenizer, data_collator=self.data_collator, compute_metrics=self.compute_metrics)
            
            trainer.train()

            # This validation part is heavily based on the original HF Tutorial:
            # https://www.youtube.com/watch?v=nvBXf7s7vTI
            # Predictions over ValSet
            # calling predict on trainer, will call compute metrics by its own
            raw_predictions = trainer.predict(self.validation_data)
            metrics = raw_predictions.metrics
            print("Metrics {}".format(metrics))
            val_loss = metrics["test_loss"]
            val_acc = metrics["test_accuracy"]
            val_prec = metrics["test_precision"]
            val_rec = metrics["test_recall"]
            val_f1 = metrics["test_f1"]

            best_check = trainer.state.best_model_checkpoint # Asked Gemini how to get the best checkpoint if using eval dataset
            if best_check is None:
                best_check = run_output_dir
            results.append({'config': conf, 'val_loss': val_loss, 'val_accuracy': val_acc, 'val_precision': val_prec,
                'val_recall': val_rec,'val_f1': val_f1, 'best_checkpoint': best_check})
            
            wandb.finish()

        print("> HPO COMPLETED")
        print("> Summary of all runs:")
        for item in results:
            print("Config: {} | Val Loss: {}".format(item['config'], item['val_loss']))

        # Asked Gemini how to retrieve the best model in order to return the path
        best_run = max(results, key=lambda x: x['val_accuracy'])
        best_checkpoint_path = best_run['best_checkpoint']
        
        return results, best_checkpoint_path