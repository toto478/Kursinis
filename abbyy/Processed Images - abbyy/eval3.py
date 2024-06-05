import pandas as pd
from difflib import SequenceMatcher
from collections import Counter

def read_txt_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text.strip()

def read_xlsx_file(path):
    df = pd.read_excel(path, engine='openpyxl', header=None)
    text = ' '.join(df.fillna('').astype(str).values.flatten())
    return text.strip()

def compute_accuracies(gt_text, ext_text):
    # Normalize texts by removing extra spaces and converting to lower case
    gt_text = ' '.join(gt_text.lower().split())
    ext_text = ' '.join(ext_text.lower().split())

    # Compute symbol accuracy using SequenceMatcher (direct match, character by character)
    direct_symbol_matcher = SequenceMatcher(None, ext_text, gt_text)
    direct_symbol_matches = sum(triple.size for triple in direct_symbol_matcher.get_matching_blocks())
    direct_symbol_accuracy = direct_symbol_matches / len(ext_text) if ext_text else 0

    # Compute longest common subsequence (LCS) for symbols
    lcs_symbol_matcher = SequenceMatcher(None, ext_text, gt_text)
    lcs_symbol_size = sum(block.size for block in lcs_symbol_matcher.get_matching_blocks())
    sequential_symbol_accuracy = lcs_symbol_size / len(ext_text) if ext_text else 0

    # Compute sequential word accuracy using SequenceMatcher for word sequences
    gt_words = gt_text.split()
    ext_words = ext_text.split()
    word_matcher = SequenceMatcher(None, ext_words, gt_words)
    lcs_word_size = sum(block.size for block in word_matcher.get_matching_blocks())
    sequential_word_accuracy = lcs_word_size / len(ext_words) if ext_words else 0

    # Compute word accuracy using Counter for word frequencies (Non-sequential)
    gt_word_counts = Counter(gt_words)
    ext_word_counts = Counter(ext_words)
    correct_words = sum(min(ext_word_counts[word], gt_word_counts[word]) for word in ext_word_counts)
    frequency_aware_word_accuracy = correct_words / sum(ext_word_counts.values()) if ext_words else 0

    print("Detailed Calculations:")
    print(f"Total Characters in Extracted Text: {len(ext_text)}")
    print(f"Total Matching Characters: {direct_symbol_matches}")
    print(f"Symbol Accuracy: {direct_symbol_accuracy:.4f} ({direct_symbol_accuracy*100:.2f}%)")
    print(f"Sequential symbol Accuracy: {sequential_symbol_accuracy:.4f} ({sequential_symbol_accuracy*100:.2f}%)")
    print(f"Total Words in Extracted Text: {len(ext_words)}")
    print(f"Longest Common Subsequence (Word Level): {lcs_word_size}")
    print(f"Sequential Word Accuracy: {sequential_word_accuracy:.4f} ({sequential_word_accuracy*100:.2f}%)")
    print(f"Total Correct Words (Frequency Aware): {correct_words}")

    return {
        'direct_symbol_accuracy': direct_symbol_accuracy,
        'sequential_symbol_accuracy': sequential_symbol_accuracy,
        'sequential_word_accuracy': sequential_word_accuracy,
        'frequency_aware_word_accuracy': frequency_aware_word_accuracy
    }





# Paths to your files
txt_path = '232_AKROPOLIS GROUP FS EN 2021.06.30.txt'
xlsx_path = '232_AKROPOLIS GROUP FS EN 2021.06.30.xlsx'

#txt_path = 'test.txt'
#xlsx_path = 'test.xlsx'

# Reading text from files
ground_truth_text = read_txt_file(txt_path)
extracted_text = read_xlsx_file(xlsx_path)

# Computing accuracies
accuracies = compute_accuracies(ground_truth_text, extracted_text)

# Output results
print("Accuracy Results:")
print(f"Symbol Accuracy: {accuracies['direct_symbol_accuracy']*100:.2f}%")
print(f"Sequential Symbol Accuracy: {accuracies['sequential_symbol_accuracy']*100:.2f}%")
print(f"Sequential Word Accuracy: {accuracies['sequential_word_accuracy']*100:.2f}%")
