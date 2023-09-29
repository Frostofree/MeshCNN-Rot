array=
for n in {0..9} ;
do
    for m in {test,train} ;
    do
        mkdir -p $n/$m ;
    done
done