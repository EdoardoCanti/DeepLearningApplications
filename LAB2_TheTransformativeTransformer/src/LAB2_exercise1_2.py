from datasets import load_dataset, get_dataset_split_names
from utils.helpers_functions import get_configuration, collect_data_confs, load_split
from utils.helpers_functions import collect_model_confs, get_model_from_slug, encode_random_text, collect_tokenstats, count_unk_collisions, update_json, load_model_tok
from utils.CustomTokenizer import CustomTokenizer
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BatchEncoding #[ISSUE-1]

# For the first part we are going to recall some basic operations 
# previously applied in LAB2_excercise1_1.py

print("\n\n === RUNNING EXERCISE 1.2: A Pre-trained BERT and Tokenizer === \n")
conf = get_configuration("config.yaml")
DATA_SLUG = collect_data_confs(conf)
splits = get_dataset_split_names(DATA_SLUG)
train_data = load_split(DATA_SLUG, 'train')
validation_data = load_split(DATA_SLUG, 'validation')
test_data = load_split(DATA_SLUG, 'test')
print("\n")

# Second part:

MODEL_SLUG = collect_model_confs(conf)
print(" --- Using the CustomTokenizer class in order to collect tokens statistics accross splits and save the tokenizer vocabulary ---")
# Using the CustomTokenizer object in order to take a look at the tokenizer behavior
custom_tokenizer = CustomTokenizer(MODEL_SLUG)
print(">> Collecting statistics about tokens cardinalities in each split...")
collect_tokenstats(custom_tokenizer, train_data, "training_set")
collect_tokenstats(custom_tokenizer, validation_data, "validation_set")
collect_tokenstats(custom_tokenizer, test_data, "test_set")
vocab_path = custom_tokenizer.save_vocabulary()
print(">> Vocabulary saved at: {}\n".format(vocab_path))

# Difference between tokenization and encoding
# Tokenization is the process from original text to Pieces of WordPiece (see next long comment)
# Encoding is the process in which each piece is mapped into an id
# a vocabulary is a dictionary Token/Piece -> ID 
print("\n\n --- Using the CustomTokenizer class to encode random samples accross splits ---")
random_training_sample, generated_training_encodings = encode_random_text(custom_tokenizer, train_data, to_tensor=True)
tokenized_training_sample = custom_tokenizer.tokenize_string(random_training_sample)
print(">Original text: {}\nTokenized as: {}\nEncoded as: {}\n".format(random_training_sample, tokenized_training_sample ,generated_training_encodings))
print("> Detokenized tokens: {}\n\n".format(custom_tokenizer.decode_tokens(generated_training_encodings)))

random_validation_sample, generated_validation_encodings = encode_random_text(custom_tokenizer, validation_data)
tokenized_validation_sample = custom_tokenizer.tokenize_string(random_validation_sample)
print(">Original text: {}\nTokenized as: {}\nencoded as: {}\n".format(random_validation_sample,tokenized_validation_sample,generated_validation_encodings))
print("> Detokenized tokens: {}\n\n".format(custom_tokenizer.decode_tokens(generated_validation_encodings)))

random_test_sample, generated_test_encodings = encode_random_text(custom_tokenizer, test_data)
tokenized_test_sample = custom_tokenizer.tokenize_string(random_test_sample)
print(">Original text: {}\nTokenized as: {}\nEncoded as: {}\n".format(random_test_sample, tokenized_test_sample, generated_test_encodings))
print("> Detokenized tokens: {}\n".format(custom_tokenizer.decode_tokens(generated_test_encodings)))

# Now the saved vocabulary shows the presence of the token "[UNK]": 100
# In BERT Paper (Devlin et al. 2019), the authors mention that their model uses the WordPiece algorithm
# WordPiece is a subword tokenizer, that is more complex of the easier Byte Pair Encoding (at least in my opinion),
# and by taking a look at its original paper, https://arxiv.org/pdf/1609.08144, (to be fair: I didn't read it, I only "cherry-picked"
# infos by searching for ["UNK"])
# the "UNK" token is used in order to handle the Out Of Vocabulary words (and actually subwords).
# I think that it would be interesting to check how many collisions there are between our DATA and the [UNK] token.
# In order to do this a function named count_unk_collisions can be found in utils/helper_functions.py

# I'm going to update the json statistics files generated during the execution of LAB2_exercise1_1.py
# LLM used for writing function update_json(file_path, new_data) in utils/helpers_function
print("\n\n --- Using the CustomTokenizer class to evaluate the presence of unknown tokens accross splits ---")
print("> Evaluating UNKNOWS PRESENCE IN EACH SPLIT: ")
path_to_training_set_stats_json = "data_stats/training_set.json"
unknows_in_training = count_unk_collisions(custom_tokenizer, train_data)
unk_train_info = {"Number of unknows": unknows_in_training}
update_json(path_to_training_set_stats_json, unk_train_info)
print(">> UNKs in training set: {}".format(unknows_in_training))

path_to_validation_set_stats_json = "data_stats/validation_set.json"
unknows_in_validation = count_unk_collisions(custom_tokenizer, validation_data)
unk_validation_info = {"Number of unknows": unknows_in_validation}
update_json(path_to_validation_set_stats_json, unk_validation_info)
print(">> UNKs in validation set: {}".format(unknows_in_validation))

path_to_test_set_stats_json = "data_stats/test_set.json"
unknows_in_test = count_unk_collisions(custom_tokenizer, test_data)
unk_test_info = {"Number of unknows": unknows_in_test}
update_json(path_to_test_set_stats_json, unk_test_info)
print(">> UNKs in test set: {}".format(unknows_in_test))
print("\n =================== \n")

# From now the CustomTokenizer will not be used anymore
# starting to use official HuggingFace AutoTokenizer and AutoModel
# del custom_tokenizer

# !!!!!! UPDATE: After some tests I was struggling a little bit with the AutoTokenizer
# class so I decided to modify the CustomTokenizer class in a way that is defined in comments
# of the method CustomTokenizer::encode_string. 
# (This means I will keep using my CustomTokenizer class and eventually modify it when
# needed)
# Loading the model only, the tokenizer is still CustomTokenizer

model = get_model_from_slug(MODEL_SLUG)
EXAMPLE_SENTENCE = "This notebook is written in order acquire familiarity with AutoModel and AutoTokenizer object."
print(" >> Original sentence: {}".format(EXAMPLE_SENTENCE))


print("==="*10)
# Trying the CustomTokenizer encode_string with extended_mode and return tensors
# extended mode will call the wrapped tokenizer as a functor (not with the encode method)
# Further details in CustomTokenizer::encode_string [ISSUE-1]
_, encoding = custom_tokenizer.encode_string(EXAMPLE_SENTENCE, as_tensor=True, extended_mode=True)
print("Encoding in extended_mode and as_tensor: {}".format(encoding))
assert(isinstance(encoding, BatchEncoding)) # is encoding a dictionary as defined CustomTokenizer::encode_string?
assert('input_ids' in list(encoding.keys())) # if yes has that dictionary a key called 'input_ids'?
assert(isinstance(encoding['input_ids'], torch.Tensor)) # if yes is that a Tensor?
decoding = custom_tokenizer.decode_tokens(encoding)
print(decoding)

print(" $$$$$ ")
print("\n not tensor but extended_mode")
_, encoding = custom_tokenizer.encode_string(EXAMPLE_SENTENCE, extended_mode=True)
print(type(encoding))
print("Encoding in extended_mode and NOT as_tensor: {}".format(encoding))
assert(isinstance(encoding, BatchEncoding)) # is encoding a dictionary as defined CustomTokenizer::encode_string?
assert('input_ids' in list(encoding.keys())) # if yes has that dictionary a key called 'input_ids'?
assert(isinstance(encoding['input_ids'], list)) # if yes is that a list?
decoding = custom_tokenizer.decode_tokens(encoding)
print(decoding)

print(" $$$$$ ")
print("\n  tensor but not extended_mode")
_, encoding = custom_tokenizer.encode_string(EXAMPLE_SENTENCE, as_tensor=True)
print(type(encoding))
print("Encoding as_tensor not extended mode: {}".format(encoding))
assert(isinstance(encoding, torch.Tensor)) # is encoding a dictionary as defined CustomTokenizer::encode_string?
decoding = custom_tokenizer.decode_tokens(encoding)
print(decoding)

print(" $$$$$ ")
print("\n  not tensor and not extended_mode")
_, encoding = custom_tokenizer.encode_string(EXAMPLE_SENTENCE, as_tensor=False, extended_mode=False)
print(type(encoding))
print("Encoding neither as_tensor and neither as extended mode: {}".format(encoding))
decoding = custom_tokenizer.decode_tokens(encoding)
print(decoding)

# Final part: actually requested by exercise
print("\n\n")
print("=== FINAL PART (ACTUALLY REQUESTED) ===")
model, tokenizer = load_model_tok(MODEL_SLUG)
EXAMPLE_SENTENCE = "This notebook is written in order to acquire familiarity with AutoModel and AutoTokenizer object."
print(" >> Original sentence: {}".format(EXAMPLE_SENTENCE))
tokens = tokenizer(EXAMPLE_SENTENCE, return_tensors="pt")
print(">> Tokenized inputs keys:", list(tokens.keys()))
print(">> Input IDs shape:", tokens['input_ids'].shape)
outputs = model(**tokens)
print("====")
print(">> Model output type:", type(outputs))
print(">> Model output :", outputs)
print(">> Last hidden state shape:", outputs.last_hidden_state.shape)