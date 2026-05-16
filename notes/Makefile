MAIN := main
OUT := build

.PHONY: all clean watch view

all: $(OUT)/$(MAIN).pdf

$(OUT)/$(MAIN).pdf: $(MAIN).tex preamble.tex refs.bib $(wildcard content/*.tex) $(wildcard figures/*)
	mkdir -p $(OUT)
	pdflatex -interaction=nonstopmode -output-directory=$(OUT) $(MAIN).tex
	BIBINPUTS=.: bibtex $(OUT)/$(MAIN) || true
	pdflatex -interaction=nonstopmode -output-directory=$(OUT) $(MAIN).tex
	pdflatex -interaction=nonstopmode -output-directory=$(OUT) $(MAIN).tex

view: $(OUT)/$(MAIN).pdf
	open $(OUT)/$(MAIN).pdf

watch:
	# Requires `latexmk`. Auto-rebuild on file changes.
	latexmk -pdf -pvc -interaction=nonstopmode -output-directory=$(OUT) $(MAIN).tex

clean:
	rm -rf $(OUT)
