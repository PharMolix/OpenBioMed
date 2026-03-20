# SKILL.md Template

Full template for creating a biomedical skill.

## YAML Frontmatter

```yaml
---
name: skill-name
description: |
  [When to trigger + what it does + trigger phrases]
  Use this skill when [specific scenario]. Triggers on phrases like
  "phrase 1", "phrase 2", "phrase 3".
---
```

## Complete Structure

```markdown
---
name: skill-name
description: |
  [When to trigger + what it does + trigger phrases]
---

# Skill Title

Brief one-line description of the skill.

## When to Use

- User asks about [specific scenario]
- User provides [specific input type]
- User wants to [specific goal]

## Workflow

### Step 1: [Action Name]

Brief description of what this step does.

```python
# Keep code snippets under 20 lines
from open_biomed.tools.tool_registry import TOOLS

tool = TOOLS["tool_name"]
result, message = tool.run(parameter=value)
entity = result.get("key")
```

### Step 2: [Action Name]

```python
# Next step code
another_tool = TOOLS["another_tool"]
output, msg = another_tool.run(entity=entity)
```

### Step 3: [Action Name]

Description and code...

## Expected Outputs

| Step | Output | Description |
|------|--------|-------------|
| Step 1 | Entity object | Retrieved data |
| Step 2 | Result | Processed output |

## Error Handling

### [Common Error Type]

**Symptom**: [What user sees]

**Solution**: [How to fix]

```python
# Fallback code example
```

## Interpretation Guide

### [Metric Name]

| Value | Quality | Meaning |
|-------|---------|---------|
| > 0.7 | Excellent | [Interpretation] |
| 0.5-0.7 | Good | [Interpretation] |
| < 0.5 | Poor | [Interpretation] |

## Example

```
Input: [example input]

Step 1: [step result]
Step 2: [step result]

Output: [final output]
```

## See Also

- `examples/basic_example.py` - Full runnable example
- `references/troubleshooting.md` - Detailed error handling
```

## Section Guidelines

### When to Use
- List 3-5 specific scenarios
- Include input types user might provide

### Workflow
- Number steps clearly
- Keep each code block under 20 lines
- For longer code, link to `examples/` file

### Expected Outputs
- Use table format
- Be specific about output types

### Error Handling
- List common errors
- Provide concrete solutions
- Include fallback approaches

### Interpretation Guide
- Explain what scores/metrics mean
- Provide ranges and their significance

### Example
- Show realistic input/output
- Keep it concise (5-10 lines)

## Long Code Handling

When workflow code exceeds 20 lines:

**In SKILL.md:**
```python
# Main workflow
result = analyze_mutation(uniprot_id, mutation)
# See examples/basic_analysis.py for full implementation
```

**In examples/basic_analysis.py:**
```python
def analyze_mutation(uniprot_id: str, mutation: str):
    """Full implementation with all steps."""
    # Complete code here
    ...
```
