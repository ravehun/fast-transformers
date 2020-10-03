rm -f test_arb
gcc -v test_arb.cc -o test_arb -I/usr/include/ -I/usr/include/flint  -L/usr/lib/x86_64-linux-gnu/ -L/usr/lib/ -lflint-arb -lflint -lmpfr  -lm -lpthread -lgmp  -g -rdynamic
./test_arb