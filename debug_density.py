#!/usr/bin/env python3
import sys
import os
sys.path.append('/Users/runout/awork/code/my_git/agent/apps/api/src')

from ai.agentic_rag.context_composer import ContextComposer

c = ContextComposer()
test_cases = [
    ("短文本", 0.2, 0.8),  # 简单文本，密度较低
    ("这是一个包含多种结构的文档：\n\n# 标题\n\n- 列表项1\n- 列表项2\n\n```code```", 0.4, 1.0),  # 结构化文档，密度较高
    ("重复重复重复重复重复", 0.0, 0.3),  # 重复内容，密度很低
]

for content, min_density, max_density in test_cases:
    density = c._calculate_information_density(content)
    print(f'Content: {repr(content)}')
    print(f'Expected: {min_density} <= density <= {max_density}, Actual: {density}')
    print(f'Pass: {min_density <= density <= max_density}')
    print('---')