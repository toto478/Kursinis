import os
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
    gt_text = ' '.join(gt_text.lower().split())
    ext_text = ' '.join(ext_text.lower().split())

    symbol_matcher = SequenceMatcher(None, ext_text, gt_text)
    symbol_matches = sum(triple.size for triple in symbol_matcher.get_matching_blocks())
    symbol_accuracy = symbol_matches / len(ext_text) if ext_text else 0

    symbol_matcher_autojunk_off = SequenceMatcher(None, ext_text, gt_text, autojunk=False)
    symbol_matches_autojunk_off = sum(triple.size for triple in symbol_matcher_autojunk_off.get_matching_blocks())
    symbol_accuracy_autojunk_off = symbol_matches_autojunk_off / len(ext_text) if ext_text else 0

    gt_words = gt_text.split()
    ext_words = ext_text.split()
    word_matcher = SequenceMatcher(None, ext_words, gt_words)
    lcs_size = sum(block.size for block in word_matcher.get_matching_blocks())
    sequential_word_accuracy = lcs_size / len(ext_words) if ext_words else 0

    print(f"Total Symbol Accuracy: {symbol_accuracy*100:.2f}% ({symbol_matches}/{len(ext_text)})")
    print(f"Total Symbol Accuracy (Autojunk off): {symbol_accuracy_autojunk_off*100:.2f}% ({symbol_matches_autojunk_off}/{len(ext_text)})")
    print(f"Total Word Accuracy: {sequential_word_accuracy*100:.2f}% ({lcs_size}/{len(ext_words)})")
    
    return symbol_matches, symbol_matches_autojunk_off, len(ext_text), lcs_size, len(ext_words) 

def process_folder(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    
    total_symbol_matches = 0
    total_symbol_matches_autojunk_off = 0
    total_ext_symbols = 0
    total_lcs_size = 0
    total_ext_words = 0
    i=1
    for txt_file in txt_files:
        base_name = txt_file.rsplit('.', 1)[0]
        corresponding_xlsx = base_name + '.xlsx'
        if corresponding_xlsx in xlsx_files:
            print(i)
            i+=1
            gt_text = read_txt_file(os.path.join(folder_path, txt_file))
            ext_text = read_xlsx_file(os.path.join(folder_path, corresponding_xlsx))
            symbol_matches, symbol_matches_autojunk_off, ext_symbols, lcs_size, ext_words = compute_accuracies(gt_text, ext_text)
            total_symbol_matches += symbol_matches
            total_symbol_matches_autojunk_off += symbol_matches_autojunk_off
            total_ext_symbols += ext_symbols
            total_lcs_size += lcs_size
            total_ext_words += ext_words
            print(f"Processed: {txt_file} and {corresponding_xlsx}")

    symbol_accuracy = total_symbol_matches / total_ext_symbols if total_ext_symbols else 0
    symbol_accuracy_autojunk_off = total_symbol_matches_autojunk_off / total_ext_symbols if total_ext_symbols else 0
    word_accuracy = total_lcs_size / total_ext_words if total_ext_words else 0

    print("\nCumulative Accuracy Results:")
    print(f"Total Symbol Accuracy: {symbol_accuracy*100:.2f}% ({total_symbol_matches}/{total_ext_symbols})")
    print(f"Total Symbol Accuracy (Autojunk off): {symbol_accuracy_autojunk_off*100:.2f}% ({total_symbol_matches_autojunk_off}/{total_ext_symbols})")
    print(f"Total Word Accuracy: {word_accuracy*100:.2f}% ({total_lcs_size}/{total_ext_words})")

# Specify the path to the folder containing the TXT and XLSX files
folder_path = r'C:\Users\Simas\Desktop\Ataskaitos\kursinis\GOOD abbyy\Processed Images - abbyy'
process_folder(folder_path)
