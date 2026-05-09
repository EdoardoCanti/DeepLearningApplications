import torch
from torch import nn
from collections import OrderedDict
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import math
import wandb

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

"""
The following class has been written in order to encapsulate the responsabilities
of handling an experiment in a single object.

General idea:
Given the training data and the test data, an experiment is identified using a title.
A good idea is to write the reason explaining what is the interest in the experiment itself.
The object will create a base directory for all the experiments (if it doesn't exists yet) and,
inside of that, a subdirectory with the experiment title name will be created.

We can manually specify whether a validation set is required and, if so, the percentage of data to be split from the training set.
Optimizer, learning rate, batch dimensions and number of epochs can be passed as argument.
Regarding this, note that a config.yaml file have been defined and that an automatic grid search will be applied.

In general, when possible, some "warnings/errors mechanisms" has been implemented in order to avoid settings that
are in contrast one with the otherm for instance, if you provide the need of a validation set, but do not provide
a valid proportion there will be an error.
Further explainations can be found in the code.

"""
class Experiment:
    def __init__(self, training_data, testing_data,
                       title: str, motivation: str, 
                       directory_name: str, base_dir: str = "experiments", 
                       need_valset: bool = False, valset_proportion: float = None,
                       learning_rate=None, optim=None, batch_size=None, num_epochs=None,
                       use_wandb: bool = False, wandb_project: str = None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.title = title
        self.motivation = motivation
        self.base_dir = base_dir
        self.directory_name = directory_name
        self.experiment_path = os.path.join(self.base_dir, self.directory_name)
        self.training_data = training_data 
        self.testing_data = testing_data   
        self.valset_proportion = valset_proportion
        self.need_valset = need_valset
        self.learning_rate = learning_rate
        self.optim = optim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_wandb = use_wandb
        
        if self.use_wandb:
            wandb.init(project=wandb_project, name=self.title, config={"learning_rate": self.learning_rate, "batch_size": self.batch_size, "num_epochs": self.num_epochs, "optimizer": self.optim})
        
        # if an experiment is created on the assumption that it requires a validation set
        if self.need_valset:
            # but the valset proportion is not provided or invalid
            if self.valset_proportion is None or self.valset_proportion <= 0 or self.valset_proportion >= 1:
                # Error
                raise ValueError("[ERROR] valset_proportion argument must be in (0,1)")

            # otherwise we consider the total length of the training data
            total_train_size = len(self.training_data)
            # compute the size of the validation set given the argument proportion in the ctor
            val_size = int(total_train_size * self.valset_proportion)

            # now we are going to split using SKLearn - Stratified Sampling
            # over the indices
            original_training_indices = []
            original_training_classes = []

            # for each index and related sample in training data
            for i, tup in enumerate(self.training_data._samples):
                # populate the two lists
                original_training_indices.append((i))
                original_training_classes.append((tup[1]))
            
            # Apply a stratified sampling over those indices, stratifying on the related classes 
            # (this maintains proporotions between classes) 
            TRAIN_INDICES, VAL_INDICES = train_test_split(original_training_indices,
                                                              stratify=original_training_classes, 
                                                              test_size=val_size)
            # creating a deep copy of the original data
            self.validation_set = copy.deepcopy(self.training_data)
            # maintaining in this copy ONLY THOSE WHOSE INDEX is in the validation indices list
            self.validation_set._samples = [self.training_data._samples[i] for i in VAL_INDICES]
            # same thing but for training data
            self.training_data._samples = [self.training_data._samples[i] for i in TRAIN_INDICES]

        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path, exist_ok=True)

    # Connecting the current experiment to the model we want to use
    def connect_model(self, model):
        #self.model = model
        self.model = model.to(self.device)

    # preparing dataloaders
    def allocate_dataloaders(self, in_batch_size: int):
        self.train_loader = DataLoader(self.training_data, batch_size=in_batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testing_data, batch_size=in_batch_size, shuffle=False)
        if self.need_valset:
            self.validation_loader = DataLoader(self.validation_set, batch_size=in_batch_size, shuffle=False)

    def train_one_epoch(self, dataloader, optimizer, verbose=False):
        self.model.train()
        batch_losses = []
        num_batches = len(dataloader)
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            if verbose:
                print(">> Training on batch: {}/{}".format(i+1, num_batches))
            optimizer.zero_grad()
            logits = self.model(x)
            batch_loss = F.cross_entropy(logits, y)
            batch_loss.backward()
            optimizer.step()
            batch_losses.append(batch_loss.item())
        return np.mean(batch_losses)
    
    def evaluate(self, dataloader, verbose=False):
        self.model.eval()
        ys_pred = []
        ys_true = []
        num_batches = len(dataloader)
        batch_losses = []
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                if verbose:
                    print("> Evaluating batch: {}/{}".format(i, num_batches))
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                batch_losses.append(loss.item())
                
                predictions = logits.argmax(dim=1)
                ys_true.append(y.cpu().numpy())
                ys_pred.append(predictions.cpu().numpy())

        ys_true = np.hstack(ys_true)
        ys_pred = np.hstack(ys_pred)
        epoch_loss = np.mean(batch_losses)
        return (accuracy_score(ys_true, ys_pred),classification_report(ys_true, ys_pred, zero_division=0, digits=3),epoch_loss)

    def train(self, num_epochs: int = None, learning_rate=None, optim = None, 
              early_stopping: bool = False, patience: int = np.inf, max_iterations: int = 2000, verbose: int = 1):
        
        if num_epochs is None:
            num_epochs = self.num_epochs
        if learning_rate is None:
            learning_rate = self.learning_rate
        if optim is None:
            optim = self.optim

        if early_stopping and patience == np.inf:
            raise ValueError("[ERROR] Cannot define early stopping without providing a valid patience argument.")

        if optim is None:
            print("[WARNING] No optimizer provided, setting Adam default")
            opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            opt = optim

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.opt_to_str = str(type(opt))
        
        self.epoch_losses = [] 
        if self.need_valset:
            self.validation_losses = [] 
            lowest_validation_loss = np.inf
            no_better_validation_count = 0

        print("Training started on {} total epochs with learning_rate {}".format(num_epochs, learning_rate))

        for epoch_idx in range(num_epochs):
            if epoch_idx < max_iterations:
                if verbose == 2 or verbose == 1:
                    print("---"*10)
                    print("> Training epoch {}/{}".format(epoch_idx+1, num_epochs))
                if verbose == 2:
                    epoch_loss = self.train_one_epoch(self.train_loader, opt, verbose=True)
                else:
                    epoch_loss = self.train_one_epoch(self.train_loader, opt)
                
                self.epoch_losses.append(epoch_loss)
                if self.use_wandb:
                    log_dict = {"train_loss": epoch_loss, "epoch": epoch_idx + 1}
                print("> Epoch: {}/{} Loss: {}".format(epoch_idx+1, num_epochs, epoch_loss))
                if self.need_valset:
                    if verbose == 2:
                        _, _, val_epoch_loss = self.evaluate(self.validation_loader, verbose=True)
                    else:
                        _, _, val_epoch_loss = self.evaluate(self.validation_loader)
                    self.validation_losses.append(val_epoch_loss)
                    if self.use_wandb:
                        log_dict["val_loss"] = val_epoch_loss

                    if self.use_wandb:
                        wandb.log(log_dict)
                    print("> Epoch: {}/{} Validation Loss: {}".format(epoch_idx+1, num_epochs, val_epoch_loss))
                    if early_stopping:
                        if val_epoch_loss < lowest_validation_loss:
                            lowest_validation_loss = val_epoch_loss
                            no_better_validation_count = 0
                            model_path = os.path.join(self.experiment_path, "{}_model.pth".format(self.title))
                            torch.save(self.model.state_dict(), model_path)
                        else:
                            no_better_validation_count += 1
                            if no_better_validation_count >= patience:
                                print("[EARLY STOPPING] No validation loss decreasing after {} iterations.\n>>Training stopped.".format(patience))
                                break
            else:
                print("[WARNING] Training interrupted for reaching max_iterations")
                break


    def save_results(self, test_metrics: dict):
        model_path = os.path.join(self.experiment_path, "{}_last_model.pth".format(self.title))
        torch.save(self.model.state_dict(), model_path)
        results = {
            "title": self.title,
            "motivation": self.motivation,
            "hyperparams":{
                "learning_rate": self.learning_rate,
                "optimizer": self.opt_to_str,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size
            },
            "test_metrics": {
                "accuracy": float(test_metrics["accuracy"]),
                "loss": float(test_metrics["loss"])
            },
            "training_losses": [float(x) for x in self.epoch_losses],
            "validation_losses": [float(x) for x in self.validation_losses] if self.need_valset else []
        }
        
        with open(os.path.join(self.experiment_path, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_losses, label='Training Loss', color='blue')
        if self.need_valset:
            plt.plot(self.validation_losses, label='Validation Loss', color='orange')
        plt.title("Training Loss - {}".format(self.title))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.experiment_path, "loss_chart.png"))
        plt.close()
        
        with open(os.path.join(self.experiment_path, "classification_report.txt"), "w") as f:
            f.write(test_metrics["report"])
        
        print("Results saved at: {}".format(self.experiment_path))
        if self.use_wandb:
            wandb.log({
                "test_accuracy": test_metrics["accuracy"],
                "test_loss": test_metrics["loss"]
            })
            wandb.finish()

    def start_experiment(self, model, num_epochs=None, in_batch_size=None, learning_rate=None, 
                         optim=None, early_stopping=False, patience=10, max_iterations=2000, verbose=1):
        
        if in_batch_size is None:
            in_batch_size = self.batch_size
        if num_epochs is None:
            num_epochs = self.num_epochs
        if learning_rate is None:
            learning_rate = self.learning_rate
        if optim is None:
            optim = self.optim

        self.batch_size = in_batch_size
        self.connect_model(model)
        self.allocate_dataloaders(in_batch_size)
        
        self.train(num_epochs=num_epochs, learning_rate=learning_rate, optim=optim, 
                   early_stopping=early_stopping, patience=patience, 
                   max_iterations=max_iterations, verbose=verbose)
        
        model_path = os.path.join(self.experiment_path, "{}_model.pth".format(self.title))
        if early_stopping and os.path.exists(model_path):
            print("> Retrieving best model params")
            self.model.load_state_dict(torch.load(model_path))

        acc, report, loss = self.evaluate(self.test_loader)
        test_results = {"accuracy": acc, "report": report, "loss": loss}
        self.save_results(test_results)
        print(">>> Experiment Ended")
        

# Simple implementation of MultiLayerPreceptron (FFN) with only one hidden layer
# this will be able to handle three different weights initializations:
# 0) completely random
# 1) he uniform
# 2) he normal
# also we are going to pass as argument the activation function

class MLP(nn.Module):
    def __init__(self, input_layer_dim: int, hidden_layer_dim: int, output_layer_dim: int, weight_init = None, activation_function: str = 'relu'):
        super().__init__()
        self.input_layer_dim = input_layer_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.output_layer_dim = output_layer_dim
        self.weight_init = weight_init
        self.activation_function = activation_function
        self.W0 = nn.Parameter(torch.empty(self.input_layer_dim, self.hidden_layer_dim))
        self.b0 = nn.Parameter(torch.zeros(self.hidden_layer_dim))
        self.W1 = nn.Parameter(torch.empty(self.hidden_layer_dim, self.output_layer_dim))
        self.b1 = nn.Parameter(torch.zeros(self.output_layer_dim))
        # If no init method requested do it randomily
        if self.weight_init is None:
            with torch.no_grad():
                self.W0.normal_(0, 1)
                self.W1.normal_(0, 1)
        elif self.weight_init == "he_uniform":
            # input layer weights matrix
            limit0 = math.sqrt(6 / self.input_layer_dim)
            with torch.no_grad():
                self.W0.uniform_(-limit0, limit0)
            # Hidden layer weights matrix
            limit1 = math.sqrt(6 / self.hidden_layer_dim)
            with torch.no_grad():
                self.W1.uniform_(-limit1, limit1)
        elif self.weight_init == "he_normal":
            stadev_input_layer = math.sqrt(2/self.input_layer_dim)
            stadev_hidden_layer = math.sqrt(2/self.hidden_layer_dim)
            with torch.no_grad():
                self.W0.normal_(0, stadev_input_layer)
                self.W1.normal_(0, stadev_hidden_layer)
        else:
            raise ValueError("[Error] Provided weight initialization method: {} is not allowed. Choose 'he_uniform' or 'he_normal'.".format(self.weight_init))

    def forward(self, x):
        x = x @ self.W0 + self.b0
        if self.activation_function == "relu":
            x = F.relu(x)
        elif self.activation_function == "gelu":
            x = F.gelu(x)
        elif self.activation_function == "lrelu":
            x = F.leaky_relu(x, negative_slope=0.01)
        x = x @ self.W1 + self.b1
        return x

