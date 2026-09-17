Main file: manuscript.tex
Compile from this directory.
Set TEXINPUTS and BSTINPUTS to include ./vendor/elsarticle//.
On Windows PowerShell:
$env:TEXINPUTS = ".;./vendor/elsarticle//;"
$env:BSTINPUTS = ".;./vendor/elsarticle//;"
$env:BIBINPUTS = ".;"
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
The supplied manuscript.bbl is the compiled bibliography.
