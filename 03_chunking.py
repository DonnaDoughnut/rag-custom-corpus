# Developed based on https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089
# Referenced https://huggingface.co/docs/transformers/v5.14.0/en/main_classes/tokenizer

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


# def fixed_size_paragraph(document, num_paragraphs, overlap):
"""Semantic Chunking"""

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

"""Testing"""
article1 = article_fulltexts[0]
# print(article1["article"][0:1000] + "\n")  # Head of first paper

# Test chunk sizes from 125 to 250 to 500
# Using an overlap greater than 15% since academic text requires more overlap in general

chunks = fixed_size_token(article1, 125, 0.2)
print("Fixed Number of Tokens")
for chunk in chunks[0:7]:
  print(chunk + "\n")

# I'd test performance with 0.21-0.25 too, as it seems to be missing the beginnings of sentences

chunks = recursive(article1, 125, 0.2)
print("Recursive Splitting")
for chunk in chunks[0:7]:
  print(chunk + "\n")

chunks = fixed_size_sentence(article1, 2, 1)
print("Fixed Number of Sentences")
for chunk in chunks[0:7]:
  print(chunk + "\n")


"""Notes"""
# Fixed-size chunking divides documents into equal-sized token windows with a fixed overlap, whereas recursive chunking attempts to
# preserve document structure by splitting on increasingly finer separators while respecting an approximate maximum chunk size.
# Fixed token chunking produces uniform chunks with exact overlap. Recursive chunking uses the same target values but allows chunk sizes and overlap
# to vary in order to split at natural boundaries. The question we are interested in is, does preserving document structure improve retrieval enough
# to outweigh the less regular chunking?