# Referenced https://huggingface.co/docs/transformers/v5.14.0/en/main_classes/tokenizer

import math

gen_AI_model = ""  # The name of the model used for the answer generation stage. Defaults if left blank.

"""Preprocessing"""
from datasets import load_from_disk
import re
custom_corpus = load_from_disk("custom_corpus")
article_fulltexts = []
for article in custom_corpus:
  article_fulltexts.append(re.sub(r'(?<=\D)\.(\d+)', '.', article["article"]))  # Remove most citation numbers, because they confuse the sentence splitter

"""Fixed Size Chunking
Splits on a set number of characters or tokens"""
from transformers import AutoTokenizer  # Using HuggingFace's method that picks a tokenizer based on the given generative AI model

# num_tokens is the number of tokens per chunk, and overlap is the number of tokens of overlap expressed as a percentage of num_tokens.
# Uses a default overlap of 15% of num_tokens, a number verified by the literature.
def fixed_size_token(document, num_tokens, overlap=0.15):
  # Assigns the default overlap if invalid
  overlap = validate_parameters(document, num_tokens, overlap, True)
  
  if gen_AI_model:  # Possibilities include "meta-llama/Llama-2-7b-chat-hf", "google/gemma-2-2b", a bert or gpt2 tokenizer, etc.
    tokenizer = AutoTokenizer.from_pretrained(gen_AI_model)
  else:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.model_max_length = int(1e30)  # Used to suppress a warning that isn't necessary for the tokenization stage

  num_tokens_overlap = round(num_tokens * overlap)
  tokens = tokenizer.encode(document, add_special_tokens=False)
  chunks = []
  for i in range(0, len(tokens), num_tokens - num_tokens_overlap):
    chunk = tokenizer.decode(tokens[i:i + num_tokens])
    chunks.append(chunk)
  return chunks


"""Recursive Chunking
Repeatedly splits chunks until they reach a target chunk size. Overlaps are also approximate."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

# num_tokens is the target maximum size of each chunk, and overlap is the target overlap, expressed as a percentage of num_tokens.
# Arguments should ideally be the same as fixed_size_token to compare the methods properly.
def recursive(document, num_tokens, overlap=0.15):
  # Assigns the default overlap if invalid
  overlap = validate_parameters(document, num_tokens, overlap, True)
  
  if gen_AI_model:  # Possibilities include "meta-llama/Llama-2-7b-chat-hf", "google/gemma-2-2b", a bert or gpt2 tokenizer, etc.
    tokenizer = AutoTokenizer.from_pretrained(gen_AI_model)
  else:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

  def token_len(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

  num_tokens_overlap = round(num_tokens * overlap)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = num_tokens, chunk_overlap = num_tokens_overlap,
                                                 length_function = token_len)
  return text_splitter.split_text(document)


"""Sentence/paragraph level chunking"""
# Looks for breaks that make more sense, like between sentences or paragraphs
import nltk
from nltk.tokenize import sent_tokenize

# Here overlap is number of sentences, not a percentage
def fixed_size_sentence(document, num_sentences=1, overlap=0):
  validate_parameters(document, num_sentences, overlap, False)

  sentences = sent_tokenize(document)
  chunks = []
  for i in range(0, len(sentences), num_sentences - overlap):
    chunks.append(" ".join(sentences[i:i + num_sentences]))
  return chunks

"""Not using since paragraph breaks from the original paper don't always make it into the downloaded papers"""
# Inspired by https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089
# Makes each paragaph a chunk, except for paragraphs below min_chunk_size which it combines together, and above max_chunk_size which it splits up
# delimiter is the characters that indicate the end of a paragraph. min_chunk_size is in characters. Overlap is the percent overlap that should be used
# when paragraphs that are too large are split.
# From a brief look at the data, the average word (separated by spaces) length is 6 to 6.5 characters. For limiting paragraphs to 128 to 384 tokens,
# I will use 750 for min characters and 1500 for max characters

# def split_paragraphs(document, delimiter="\n\n", min_chunk_size=750, max_chunk_size=1500, overlap_percent=0.15):
#   paragraphs = document.split(delimiter)
#   chunks = []
#   current_chunk = ""
#   for para in paragraphs:
#     if not para.strip():
#       continue
#     current_chunk += para
#     print("len of current chunk:", len(current_chunk))
#     if len(current_chunk) > max_chunk_size:
#       size_without_overlap = max_chunk_size / (1 + overlap_percent)
#       print("target size of new chunks in characters is", size_without_overlap)
#       num_chunks_to_make = math.ceil(len(current_chunk) / size_without_overlap)
#       print("num_chunks_to_make is", num_chunks_to_make)
#       len_of_each_chunk = len(current_chunk) / num_chunks_to_make
#       words_without_overlap = round(len_of_each_chunk / 6)  # Assuming that 6 is the average word length
#       print("words_without_overlap is", words_without_overlap)
#       overlap = round(words_without_overlap * overlap_percent)
#       print("number of words of overlap is", overlap)
#       current_chunk_words = current_chunk.split()
#       for i in range(0, len(current_chunk_words), words_without_overlap):
#         if i + words_without_overlap + overlap > len(current_chunk_words):
#           chunks.append(current_chunk_words[i:])
#           print("Added to chunks:", current_chunk_words[i:])
#         else:
#           chunks.append(current_chunk_words[i : i + words_without_overlap + overlap])
#           print("Added to chunks:", current_chunk_words[i : i + words_without_overlap + overlap])
#       current_chunk = ""
#     elif len(current_chunk) >= min_chunk_size:
#       print("Current chunk was not bigger than max_chunk_size. Adding to chunks and resetting current_chunk.")
#       chunks.append(current_chunk)
#       current_chunk = ""
#   if len(current_chunk) < min_chunk_size:
#     chunks.append(current_chunk)
#   return chunks


"""Semantic Chunking
Calculates the embedding vector for each sentences, calculates cosine similarities and starts a new chunk whenever the similarity is too low"""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

def semantic(document, target_num_chunks=60):
  embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
  text_splitter = SemanticChunker(embeddings, number_of_chunks=target_num_chunks)
  return text_splitter.split_text(document)


"""Helper functions"""
# Returns overlap if valid, otherwise returns 15%
def validate_parameters(document, size, overlap, percentage):
  if not isinstance(document, str) or len(document) == 0:
    raise TypeError("Document is empty or invalid.")
  if not size > 0:
    raise ValueError("The number of tokens or sentences per chunk must be greater than 0.")
  if percentage:
    if overlap < 0 or overlap >= 1:
      print("Invalid overlap given, using the default overlap of 15%.")
      return 0.15
  else:
    if overlap < 0 or overlap >= size:
      raise ValueError("Overlap must be less than num_sentences.")
  return overlap

import statistics
# Returns a dictionary with keys "num_chunks", "mean_token_count", "median_token_count", "min_token_count", "max_token_count", and "stdev"
def compute_chunk_stats(chunks):
  if gen_AI_model:  # Possibilities include "meta-llama/Llama-2-7b-chat-hf", "google/gemma-2-2b", a bert or gpt2 tokenizer, etc.
    tokenizer = AutoTokenizer.from_pretrained(gen_AI_model)
  else:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.model_max_length = int(1e30)  # Used to suppress a warning that isn't necessary for the tokenization stage

  results = {"num_chunks": len(chunks)}
  chunk_sizes = [len(tokenizer.encode(chunk, add_special_tokens=False)) for chunk in chunks]
  results["mean_token_count"] = sum(chunk_sizes) / len(chunks)
  results["median_token_count"] = statistics.median(chunk_sizes)
  results["min_token_count"] = min(chunk_sizes)
  results["max_token_count"] = max(chunk_sizes)
  results["stdev"] = statistics.stdev(chunk_sizes)
  return results


"""Testing"""
show_sample_chunks = False
article1 = article_fulltexts[1]
# print(article1["article"][0:1000] + "\n")  # Head of first paper

# Test chunk sizes from 125 to 250 to 500
# Using an overlap greater than 15% since academic text requires more overlap in general
chunks = fixed_size_token(article1, 125, 0.25)
print("Fixed Number of Tokens")
if show_sample_chunks:
  for chunk in chunks[0:7]:
    print(chunk + "\n")
  print()
print(compute_chunk_stats(chunks))
print()
num_chunks_estimate = compute_chunk_stats(chunks)["num_chunks"]

chunks = recursive(article1, 125, 0.25)
print("Recursive Splitting")
if show_sample_chunks:
  for chunk in chunks[0:7]:
    print(chunk + "\n")
  print()
print(compute_chunk_stats(chunks))
print()

# Test with 2 and 3
chunks = fixed_size_sentence(article1, 2)
print("Fixed Number of Sentences")
if show_sample_chunks:
  for chunk in chunks[0:7]:
    print(chunk + "\n")
  print()
print(compute_chunk_stats(chunks))
print()

chunks = semantic(article1, num_chunks_estimate)
print("Semantic/Embeddings-based Chunking")
if show_sample_chunks:
  for chunk in chunks[0:7]:
    print(chunk + "\n")
  print()
print(compute_chunk_stats(chunks))
print()


"""Notes"""
# Fixed-size chunking divides documents into equal-sized token windows with a fixed overlap, whereas recursive chunking attempts to
# preserve document structure by splitting on increasingly finer separators while respecting an approximate maximum chunk size.
# Fixed token chunking produces uniform chunks with exact overlap. Recursive chunking uses the same target values but allows chunk sizes and overlap
# to vary in order to split at natural boundaries. The question we are interested in is, does preserving document structure improve retrieval enough
# to outweigh the less regular chunking?
# The semantic chunker was controlled for number of chunks produced, as the default settings could result in way larger chunks than all the other methods.
# Article one can be found here: https://www.tandfonline.com/doi/full/10.2147/NDT.S50763#d1e104
# Article two can be found here: https://pmc.ncbi.nlm.nih.gov/articles/PMC3400365/