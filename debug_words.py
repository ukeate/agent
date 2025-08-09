#!/usr/bin/env python3
import sys
import re
import os
sys.path.append('/Users/runout/awork/code/my_git/agent/apps/api/src')

content = "重复重复重复重复重复"
print(f'Content: {content}')

words = re.findall(r'\b\w+\b', content.lower())
print(f'Words found: {words}')

unique_words = set(words)
print(f'Unique words: {unique_words}')

if len(words) > 0:
    vocabulary_diversity = len(unique_words) / len(words)
    print(f'Diversity: {vocabulary_diversity}')
    
    repetition_ratio = 1 - vocabulary_diversity
    print(f'Repetition ratio: {repetition_ratio}')
else:
    print("No words found")