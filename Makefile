all:
	jupyter nbconvert --to markdown --output-dir=docs BGWpy_Tutorial.ipynb
	mkdocs build

clean:
	rm -rf site
	rm -rf Runs
