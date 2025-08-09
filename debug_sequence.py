#!/usr/bin/env python3
import sys
import os
sys.path.append('/Users/runout/awork/code/my_git/agent/apps/api/src')

from ai.agentic_rag.context_composer import ContextComposer, KnowledgeFragment, FragmentType

c = ContextComposer()

step1_frag = KnowledgeFragment(
    id="step1",
    content="第一步：准备数据和环境",
    source="test.md",
    fragment_type=FragmentType.PROCEDURE,
    relevance_score=0.8,
    quality_score=0.8,
    information_density=0.7,
    tokens=30
)

step2_frag = KnowledgeFragment(
    id="step2",
    content="第二步：训练机器学习模型",
    source="test.md",
    fragment_type=FragmentType.PROCEDURE,
    relevance_score=0.8,
    quality_score=0.8,
    information_density=0.7,
    tokens=40
)

sequence = c._detect_sequence(step1_frag, step2_frag)
print(f'Sequence detection result: {sequence}')
print(f'Expected > 0.5, Pass: {sequence > 0.5}')