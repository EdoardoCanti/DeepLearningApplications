# Exercize 1.3: A Stable Baseline
from utils.helpers_functions import get_configuration, collect_data_confs, get_dataset_split_names, load_split, collect_model_confs, load_model_tok
from transformers import pipeline
import torch
from sklearn.svm import LinearSVC
import optuna
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

# Devices
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Running device mode: {}".format(device))


# 1) Loading Data 
conf = get_configuration("config.yaml")
DATA_SLUG = collect_data_confs(conf)
splits = get_dataset_split_names(DATA_SLUG)
train_data = load_split(DATA_SLUG, 'train')
validation_data = load_split(DATA_SLUG, 'validation')
test_data = load_split(DATA_SLUG, 'test')
if train_data:
    print("Training data loaded")
    train_data_txt = train_data["text"]
    train_data_lbl = train_data["label"]
if validation_data:
    val_data_txt = validation_data["text"]
    val_data_lbl = validation_data["label"]
    print("Validation data loaded")
if test_data:
    test_data_txt = test_data["text"]
    test_data_lbl = test_data["label"]
    print("Test data loaded")

# 2) Loading model and tokenizer
MODEL_SLUG = collect_model_confs(conf)
model, tokenizer = load_model_tok(MODEL_SLUG)
if model:
    print("Model loaded")
if tokenizer:
    print("Tokenizer loaded")

# 3) preparing the pipeline
# https://huggingface.co/tasks/feature-extraction
# https://huggingface.co/docs/transformers/v5.14.0/en/main_classes/pipelines#transformers.FeatureExtractionPipeline

RECOMPUTE_EMBEDDINGS = conf["EXEC"]["recompute_embeddings"]
if RECOMPUTE_EMBEDDINGS:
    print("Computing embeddings")
    from utils.helpers_functions import instanciate_extractor, get_cls_embeddings
    extractor = instanciate_extractor(model, tokenizer, device = device)
    print("\n>> Extracting and saving features for training set")
    training_cls_embeddings = get_cls_embeddings(extractor, list(train_data_txt), "training_cls_embeddings.pt")
    print("\n>> Extracting and saving features for validation set")
    validation_cls_embeddings = get_cls_embeddings(extractor, list(val_data_txt), "val_cls_embeddings.pt")
    print("\n>> Extracting and saving features for test set")
    test_cls_embeddings = get_cls_embeddings(extractor, list(test_data_txt), "test_cls_embeddings.pt")
else:
    print("Retrieving existing embeddings")
    training_cls_embeddings = torch.load(conf["EXEC"]["training_cls_embeddings_path"])
    validation_cls_embeddings = torch.load(conf["EXEC"]["vali_cls_embeddings_path"])
    test_cls_embeddings = torch.load(conf["EXEC"]["test_cls_embeddings_path"])

# Since we are working using a Distil-BERT I'd like
# to search a good baseline (which will be of course limited wrt to Distil-BERT but not too trivial)
# I think this is a good moment to use Optuna (https://optuna.readthedocs.io/en/stable/index.html)
# (another reason is that I used it for random tests few times and this could be a good moment to refresh how it works)
print("="*20+" BASELINE "+"="*20)

# Optuna's objective function is used to define the... objective function
# Here we are going to tell Optuna which hyperparams we are interested to optimize
# what is returned by this function will be the objective itself.
# Since we already splitted into Train, Validation and Test set, our goal will be to train over the training set
# and choose hypeparams over the validation set.
# Hyperparams to be optimized will be:
# 1) C (regularization param) 
# 2) kernel {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}
# 
# and were selected from here: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#
# In Optuna we are allowed to define the type of the hyperparam we want to optimize
# for us we have only: suggest_categorical and sugget_float 
# docs at (https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial)

def objective(trial):
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    # The C regularizer is what in Optimization Methods is said to "model" the hardness of the misclassification
    # an HARD CONSTRAINT (high c) means that we are penalizing more the classificaton error (larger margin)
    # sources: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html#sphx-glr-auto-examples-svm-plot-svm-scale-c-py
    # and LLM used here: asked further explaination of C hyperparam
    c_lower = float(conf["SVC"]["c_regularizer_lowerbound"])
    c_upper = float(conf["SVC"]["c_regularizer_upperbound"])
    c_regularizer = trial.suggest_float("c_reg", c_lower, c_upper, log=True)
    # Defining the model family on which we want to find the optimal hyparams
    svc = SVC(kernel = kernel, C = c_regularizer, random_state=conf["SVC"]["random_state"], max_iter=5000)
    # Fitting over training set with current hyperparams
    svc.fit(training_cls_embeddings, train_data_lbl) 
    # Predicting the validation set, using the model trained on current train and CURRENT hyperparams
    y_val_hat = svc.predict(validation_cls_embeddings)
    acc_score = accuracy_score(y_val_hat, val_data_lbl)
    # Returning the accuracy score because we want to optimize it (MAXIMIZE IT!!)
    return acc_score

print("Launching Optuna study...")
# LLM used here, didn't remember how to use the Optuna dashboard for cool visualization in real time (sqlite etc...)
study = optuna.create_study(direction = "maximize", study_name="SVC_baseline_HPO", storage="sqlite:///db.sqlite3", load_if_exists=True)
study.optimize(objective, n_trials=30)
print("Optuna found as best hyperparams: {} with acc: {}".format(study.best_params, study.best_value))

# Baseline evaluation on test set (using Optuna's best model found )
print("Using Optuna's best params to evaluate the test set...")
test_svc = SVC(kernel = study.best_params["kernel"], C=study.best_params["c_reg"], random_state=conf["SVC"]["random_state"])
test_svc.fit(training_cls_embeddings, train_data_lbl)
y_test = test_svc.predict(test_cls_embeddings)
#baseline_classification_report = classification_report(y_test, test_data_lbl)
baseline_classification_report = classification_report(test_data_lbl, y_test)
print("Test Set results on best model found by Optuna:")
print(baseline_classification_report)
# We want to save the report itself
exp_folder = "experiments"
os.makedirs(exp_folder, exist_ok=True)
report_path = os.path.join(exp_folder, "svc_baseline_classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(baseline_classification_report)
print("SVC Baseline classification report saved at: {}".format(report_path))