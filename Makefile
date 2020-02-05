all:
	jupyter nbconvert --to markdown --output-dir=docs BGWpy_Tutorial.ipynb
	mkdocs build

clean:
	rm -f docs/BGWpy_Tutorial.md
	rm -rf Runs
