# Contributing to PyTorch Course

Thank you for your interest in contributing! This guide explains how to help improve the course.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report errors, typos, unclear explanations, or broken code examples
- Include the module number and section
- Provide context: what was confusing, what you expected

### Improving Content
1. **Fork** the repository
2. **Create a branch**: `git checkout -b improve/module-XX-topic`
3. **Make changes** to the relevant `.md` file
4. **Test code examples** locally if you modify them
5. **Commit** with clear messages: `git commit -m "Module 05: Fix CNN stride calculation example"`
6. **Push** and **open a Pull Request**

### Adding New Content
- Follow the existing module structure:
  - Learning objectives
  - Mathematical derivations (where relevant)
  - Conceptual explanations with diagrams
  - Full annotated PyTorch code snippets
  - Real-world use cases
  - Best practices table
  - Exercises with solutions
  - Module summary
  - 10-question quiz with answers
- Use consistent Markdown formatting
- Include runnable Python code examples
- Test all code on PyTorch 2.0+

### Fixing Code Examples
- Test all code snippets locally before submitting
- Ensure compatibility with PyTorch 2.0+ and Python 3.8+
- Add comments explaining non-obvious logic
- Include expected output or behavior in docstrings
- Verify GPU/CPU compatibility where relevant

## Pull Request Process

1. Update relevant sections if you modify module content
2. Ensure all code examples run without errors
3. Add a clear description of your changes
4. Reference any related issues: `Fixes #123`
5. Wait for review and address feedback

## Style Guide

- **Markdown**: Use standard GitHub Flavored Markdown
- **Code**: Follow PEP 8 and PyTorch conventions
- **Comments**: Explain the "why", not the "what"
- **Examples**: Keep them concise and educational
- **Math**: Use LaTeX for equations (e.g., `$\nabla L$`)

## Testing Code Examples

Before submitting:
```bash
# Verify Python syntax
python -m py_compile your_code.py

# Run examples if they're standalone
python your_example.py

# Check for common PyTorch issues
# - Device compatibility (CPU/GPU)
# - Gradient tracking (requires_grad)
# - In-place operations
```

## Questions?

Open a GitHub Discussion or Issue. We're here to help!

---

**Thank you for making this course better for everyone!**
