# Skill Evaluation Reference

This document provides guidance for evaluating biomedical skills.

## Evaluation Directory Structure

```
skill-workspace/
└── iteration-N/
    ├── benchmark.json           # Aggregate results
    ├── eval-1-name/
    │   ├── eval_metadata.json   # Test case definition
    │   ├── with_skill/
    │   │   ├── outputs/         # Generated files
    │   │   ├── grading.json     # Assertion results
    │   │   └── timing.json      # Duration and tokens
    │   └── without_skill/
    │       └── ... (same structure)
    ├── eval-2-name/
    │   └── ...
    └── eval-3-name/
        └── ...
```

## Creating Test Cases

### eval_metadata.json

```json
{
  "eval_id": 1,
  "eval_name": "descriptive-name",
  "prompt": "The user's task prompt",
  "assertions": [
    {
      "name": "assertion_name",
      "text": "Human-readable description",
      "check": "What to verify"
    }
  ]
}
```

### Assertion Types

**Output Content:**
- `includes_metric` - Output contains specific metric (QED, SA, LogP)
- `includes_entity` - Output mentions specific entity (molecule name, protein)
- `structured_format` - Output uses tables, sections, clear structure

**Workflow Execution:**
- `uses_tool` - Correct tool was invoked
- `handles_input_type` - Properly handles SMILES/name/sequence input
- `error_handling` - Gracefully handles errors

**Domain-Specific:**
- `correct_interpretation` - Scores interpreted correctly
- `valid_recommendation` - Recommendation follows from data
- `complete_analysis` - All required analyses performed

### Example Assertions

```json
{
  "assertions": [
    {
      "name": "includes_qed_score",
      "text": "Report includes QED score",
      "check": "Output contains QED score with numerical value"
    },
    {
      "name": "correct_qed_interpretation",
      "text": "QED score is interpreted correctly",
      "check": "QED > 0.7 should be 'Excellent', 0.5-0.7 'Good', < 0.5 'Poor'"
    },
    {
      "name": "uses_open_biomed_tools",
      "text": "Analysis uses OpenBioMed tools",
      "check": "Methodology section lists TOOLS dictionary usage"
    },
    {
      "name": "provides_recommendation",
      "text": "Provides actionable recommendation",
      "check": "Output includes clear recommendation for next steps"
    }
  ]
}
```

## Grading Process

### Manual Grading

Read the output and check each assertion:

```json
{
  "eval_id": 1,
  "eval_name": "test-case",
  "run_id": "with_skill",
  "assertions": [
    {
      "name": "includes_qed_score",
      "text": "Report includes QED score",
      "passed": true,
      "evidence": "QED Score: 0.69 found in Drug-likeness Scores table"
    },
    {
      "name": "uses_open_biomed_tools",
      "text": "Analysis uses OpenBioMed tools",
      "passed": false,
      "evidence": "Report states 'Analysis performed without using skill tools'"
    }
  ],
  "pass_rate": 0.5
}
```

### Automated Grading

For objective checks, use scripts:

```python
import json
import re

def grade_qed_score(output_path):
    """Check if QED score is present and valid."""
    with open(output_path) as f:
        content = f.read()

    # Check for QED mention
    qed_match = re.search(r'QED[^:]*:\s*([\d.]+)', content, re.IGNORECASE)
    if not qed_match:
        return {"passed": False, "evidence": "No QED score found"}

    qed_value = float(qed_match.group(1))
    if not 0 <= qed_value <= 1:
        return {"passed": False, "evidence": f"Invalid QED value: {qed_value}"}

    return {
        "passed": True,
        "evidence": f"QED Score: {qed_value} found and valid"
    }
```

## Benchmark Aggregation

### benchmark.json Format

```json
{
  "skill_name": "skill-name",
  "iteration": 1,
  "timestamp": "2026-03-14",
  "results": [
    {
      "eval_id": 1,
      "eval_name": "test-case-1",
      "with_skill": {
        "pass_rate": 1.0,
        "assertions_passed": 6,
        "assertions_total": 6
      },
      "without_skill": {
        "pass_rate": 0.833,
        "assertions_passed": 5,
        "assertions_total": 6
      }
    }
  ],
  "summary": {
    "with_skill": {
      "mean_pass_rate": 0.95,
      "total_assertions_passed": 17,
      "total_assertions_total": 18
    },
    "without_skill": {
      "mean_pass_rate": 0.85,
      "total_assertions_passed": 15,
      "total_assertions_total": 18
    },
    "delta": {
      "pass_rate_improvement": 0.10,
      "additional_assertions_passed": 2
    }
  }
}
```

## Comparator (Blind A/B Testing)

Use when comparing two versions of a skill.

### Process

1. Give two outputs to independent agent
2. Agent doesn't know which is which (A vs B)
3. Agent judges quality on specific criteria
4. Reveal which version won
5. Analyze why

### Comparator Prompt

```
You are judging two outputs for a biomedical analysis task.

Task: [The original prompt]

Output A:
[First output]

Output B:
[Second output]

Evaluate on these criteria:
1. Completeness: Did it address all aspects of the prompt?
2. Accuracy: Are the scientific interpretations correct?
3. Clarity: Is the output well-structured and easy to understand?
4. Actionability: Does it provide useful recommendations?

Which output is better? Explain your reasoning.
```

## Analyzer

Analyze benchmark results for patterns.

### Key Patterns to Identify

**Non-Discriminating Assertions:**
- Always pass in both configurations
- Don't help distinguish skill quality
- Consider removing or making harder

**High-Variance Assertions:**
- Pass/fail inconsistently
- May indicate flaky tests
- Investigate root cause

**Time/Quality Tradeoffs:**
- Does better quality take longer?
- Is the tradeoff worth it?

**Unexpected Failures:**
- Where did the skill fall short?
- What assumptions were violated?

### Analyzer Output Template

```markdown
## Benchmark Analysis

### Summary Statistics
- With Skill: X% pass rate, Ys average duration
- Without Skill: Z% pass rate, Ws average duration
- Delta: +D% improvement

### Assertion Analysis

#### Non-Discriminating (always pass)
- assertion_name: Passes in all configurations

#### Discriminating (differentiates skill)
- assertion_name: With-skill X%, Without-skill Y%

#### High-Variance (inconsistent)
- assertion_name: Inconsistent results across runs

### Recommendations
1. [Specific improvement based on analysis]
2. [Another improvement]
```

## Evaluation Best Practices

### Test Case Design

1. **Realistic prompts** - What users would actually type
2. **Edge cases** - Invalid inputs, missing data, API failures
3. **Diverse inputs** - Different molecules, proteins, text queries
4. **Objective criteria** - Verifiable outcomes, not subjective quality

### Assertion Design

1. **Specific** - Check one thing per assertion
2. **Verifiable** - Can be checked programmatically or by inspection
3. **Meaningful** - Relates to skill quality, not trivia
4. **Descriptive names** - Clear what's being checked

### Timing Considerations

- Record duration for each run
- Compare time between configurations
- Consider if slower is acceptable for better quality

## Iteration Workflow

1. **Run evaluation** - All test cases, with and without skill
2. **Grade outputs** - Check assertions
3. **Aggregate results** - Create benchmark.json
4. **Analyze patterns** - What worked, what didn't
5. **Improve skill** - Address identified issues
6. **Re-run** - New iteration with improved skill
7. **Compare** - Previous iteration as baseline

Continue until:
- User satisfied
- All feedback empty
- No meaningful progress
