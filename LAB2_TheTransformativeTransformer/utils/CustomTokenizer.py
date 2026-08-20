# Defining here a custom wrapper for the AutoTokenizer model
from transformers import AutoTokenizer, BatchEncoding
import json
import os
import torch


"A simple Wrapper class that encapsulates methods for handling the HuggingFace AutoTokenizer object"
class CustomTokenizer:
    
    def __init__(self, hf_slug: str):
        self.tokenizer = AutoTokenizer.from_pretrained(hf_slug)
        self.vocabulary = self.tokenizer.get_vocab()
        self.tokens = set(list(self.vocabulary.keys()))
        if self.tokenizer:
            print("[CustomTokenizer] AutoTokenizer correctly loaded from slug: {}".format(hf_slug))

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_vocabulary(self) -> dict:
        return self.vocabulary
        
    def save_vocabulary(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.vocabulary = self.get_vocabulary()
        with open(os.path.join(project_root, "tokenizer_vocabulary.json"), "w") as file:
            json.dump(self.vocabulary, file, indent=4)
        return os.path.join(project_root, "tokenizer_vocabulary.json")
    
    def tokenize_string(self, in_string: str):
        return self.tokenizer.tokenize(in_string)

    # [ISSUE-1]
    # After some "experiments" like:
    #    tokens = tokenizer('this is a sample sentece.')
    #    tokens = tokenizer.encode('this is a sample sentece.')
    # it turns out that:
    #   if you tokenize using the encode method you get the input_ids only (as tensors or not depends)
    #   but if you tokenize calling the tokenizer itself, as a functor, it returns both the input_ids and the attention_mask
    #   as a transformers.BatchEncoding object (https://huggingface.co/transformers/v4.0.1/main_classes/tokenizer.html#transformers.BatchEncoding).
    # so the followinf method is going to handle it in this way:
    # I will call "the functor way" as extended_mode: boolean = False by default
    def encode_string(self, in_string: str, as_tensor: bool = False, extended_mode: bool = False):
        encodings = None
        if extended_mode:
            if as_tensor:
                encodings = self.tokenizer(in_string, return_tensors="pt")
            else:
                encodings = self.tokenizer(in_string)
        else:
            if as_tensor:
                encodings = self.tokenizer.encode(in_string, return_tensors = "pt")
            else:
                encodings = self.tokenizer.encode(in_string)
        return in_string, encodings

    def count_tokens(self, in_string: str):
        _, encodings = self.encode_string(in_string)
        return len(encodings)
    
    # [ISSUE-1]
    # If applying only decode(tokens) when tokens is a torch Tensor:
    #   TypeError: argument 'ids': 'list' object cannot be interpreted as an integer
    # So I'm going to extract the first element of the tensor (which is the list to ids itself)
    def decode_tokens(self, tokens):
        # if it is a BatchEncoding (i.e: the original string was encoded in extended_mode)
        # I need to extract the input_ids
        if type(tokens) == BatchEncoding:
            input_ids = tokens['input_ids']
            # Now if the original string was also encoded as tensor, I need to extract them
            # So if it is a single string
            if type(input_ids) == torch.Tensor:
                if input_ids.shape[0] == 1: # if it is a single string
                    return self.tokenizer.decode(input_ids[0])
                if input_ids.shape[0] > 1: # if it is a batch of tokens lists
                    tokens_decoded = [] # prepare a list for every item
                    for i in range(input_ids.shape[0]):
                        tokens_decoded.append(self.tokenizer.decode(input_ids[i])) # append here avery token list
                    return tokens_decoded
        else:
            if type(tokens) == torch.Tensor: # EXACTLY LIKE BEFORE
                if tokens.shape[0] == 1: # if it is a single string
                    return self.tokenizer.decode(tokens[0])
                if tokens.shape[0] > 1: # if it is a batch of tokens lists
                    tokens_decoded = [] # prepare a list for every item
                    for i in range(tokens.shape[0]):
                        tokens_decoded.append(self.tokenizer.decode(tokens[i])) # append here avery token list
                    return tokens_decoded
            else: 
                return self.tokenizer.decode(tokens)
    
    # if arg is list, for each string in arg return dict {w: True/False}
    # if arg is string: if contains spaces (i.e: if more than one word= raise error, else return only True/False 
    def is_in_vocab(self, arg) -> bool:
        # taking all the vocabulary keys as list, and transforming into set
        if type(arg) == list:
            presence_dict = dict()
            for w in arg:
                if str(w) in self.tokens:
                    presence_dict[str(w)] = True
                else:
                    presence_dict[str(w)] = False
            return presence_dict
        elif type(arg) == str:
            # if exists an object returned by splitting on space raise value error
            if " " in arg.strip(): # LLM USED HERE: original (wrong) condition was: if arg.split(" ")
                raise ValueError("[CustomTokenizer] Error: argument must be a list of strings, or a single string.")
            else:
                if str(arg) in self.tokens:
                    return True
                else:
                    return False
                
    
    

