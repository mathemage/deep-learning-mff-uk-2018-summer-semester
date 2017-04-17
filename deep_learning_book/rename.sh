from_format="pdf"
to_format="html"

for f in *.$from_format
do
  mv "$f" "`basename $f .$from_format`.$to_format"
done
