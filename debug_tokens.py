#!/usr/bin/env python3
import sys
import os
sys.path.append('/Users/runout/awork/code/my_git/agent/apps/api/src')

from ai.agentic_rag.context_composer import ContextComposer

c = ContextComposer()
test_cases = [
    ('Hello world', 2),
    ('你好世界', 4),
    ('Hello 世界 test', 5),
    ('```python\nprint("hello")\n```', 8)
]

for text, expected_min in test_cases:
    actual = c._estimate_tokens(text)
    print(f'Text: {repr(text)}')
    print(f'Expected >= {expected_min}, Actual: {actual}')
    print(f'Pass: {actual >= expected_min}')
    print('---')