.PHONY: help validate clean

help:
	@echo "PyTorch Course — Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help           Show this help message"
	@echo "  validate       Validate markdown and code examples"
	@echo "  clean          Remove cache and build artifacts"

validate:
	@echo "Validating markdown syntax..."
	@python -c "import glob, re; \
	for file in glob.glob('*.md'): \
		with open(file) as f: \
			content = f.read(); \
			if content.count('\`\`\`') % 2 != 0: \
				print(f'ERROR: Unclosed code block in {file}'); exit(1)"
	@echo "✓ All markdown files valid"
	@echo "Checking for required files..."
	@test -f README.md || (echo "ERROR: README.md not found"; exit 1)
	@test -f PROJECTS_AND_SOLUTIONS.md || (echo "ERROR: PROJECTS_AND_SOLUTIONS.md not found"; exit 1)
	@test -f QUIZ_BANK.md || (echo "ERROR: QUIZ_BANK.md not found"; exit 1)
	@echo "✓ All required files present"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned up cache and build artifacts"
