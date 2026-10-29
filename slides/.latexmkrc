# Always build these slides with XeLaTeX (fontspec + xeCJK need it).
$pdf_mode = 5;                      # plain `latexmk` -> xelatex
# Editors that call `latexmk -pdf` (e.g. VS Code LaTeX Workshop's default recipe) force pdflatex mode;
# pointing the pdflatex command at xelatex makes that path work too.
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$xelatex  = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -no-pdf %O %S';
# After every successful build of main.tex, refresh the deliverable under its required file name.
$success_cmd = 'if [ "%R" = "main" ]; then mkdir -p ../submission && cp %D ../submission/r14725022_report.pdf; fi';
