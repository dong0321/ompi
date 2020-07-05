#!/bin/bash

echo "ompi version with AVX512 -- Usage: arg1: count of elements, args2: 'i'|'u'|'f'|'d' : datatype: signed, unsigned, float, double. args3 size of type. args4 operation"
echo "make sure to use -u -t -s !"
echo "mpirun -np 1 reduce_local -u 1024 -t i -s 16  sum -v"

Orange="\033[0;33m"
Blue="\033[0;34m"
Purple="\033[0;35m"

NC="\033[0m"

echo "=========Signed Integer type all operations & all sizes========"
echo ""
for op in max min sum mul band bor bxor; do
    echo -e "\n===Operation  $op test==="
    for type_size in 8 16 32 64; do
        for size in 0 1 7 15 31 63 127 130; do
            foo=$((1024 * 1024 + $size))
            echo -e "Test \e[1;33m __mm512 instruction for loop \e[m Total_num_bits = $foo * $type_size "
            echo "mpirun -np 1 reduce_local -u $foo -t i -s $type_size $op -v"
            mpirun -np 1 reduce_local -u $foo -t i -s $type_size $op -v
        done
        echo -e "\n\n"
    done
    echo -e "\n\n"
done
echo "=========Signed Integer type all operations & all sizes========"
echo -e "\n\n"

echo "=========Unsigned Integer type all operations & all sizes========"
echo ""
for op in max min sum mul band bor bxor; do
    echo -e "\n===Operation  $op test==="
    for type_size in 8 16 32 64; do
        for size in 0 1 7 15 31 63 127 130; do
            foo=$((1024 * 1024 + $size))
            echo -e "Test \e[1;33m __mm512 instruction for loop \e[m Total_num_bits = $foo * $type_size"
            echo "mpirun -np 1 reduce_local -u $foo -t u -s $type_size $op -v"
            mpirun -np 1 reduce_local -u $foo -t u -s $type_size $op -v
        done
    done
done
echo "=========Unsigned Integer type all operations & all sizes========"
echo -e "\n\n"

echo "=======Float type all operations========"
echo ""
for op in max min sum mul; do
    for size in 1024 127 130; do
        foo=$((1024 * 1024 + $size))
        echo -e "Test \e[1;33m __mm512 instruction for loop \e[m Total_num_bits = $foo * 32"
        echo "mpirun -np 1 reduce_local -u $foo -t f -s 32 $op -v"
        mpirun -np 1 reduce_local -u $foo -t f -s 32 $op -v
    done
done

echo "========Double type all operations========="
echo ""
for op in max min sum mul; do
    for size in 1024 127 130; do
        foo=$((1024 * 1024 + $size))
        echo -e "Test \e[1;33m __mm512 instruction for loop \e[m Total_num_bits = $foo * 64"
        echo "mpirun -np 1 reduce_local -u $foo -t d -s 64 $op -v"
        mpirun -np 1 reduce_local -u $foo -t d -s 64 $op -v
    done
done

