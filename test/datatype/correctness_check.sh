echo "ompi version with AVX512 -- Usage: arg1: count of elements, args2: 'i'|'f'|'d' : datatype: integer, float, double. args3 size of type. args4 operation"
echo "/home/zhongdong/opt/git/avx512_reduction/bin/mpirun  -mca   op_sve_hardware_available 0  -mca   op_avx_hardware_available 0   -np 1 /home/zhongdong/Downloads/git/ARM/arm_sve_reduction/test/datatype/Reduce_local_float 1048576  i 8 max"

echo "=========Integer type all operations & all sizes========"
echo ""
for op in max min sum mul band bor bxor
do
    echo ""
    echo "===Operation  $op test==="
    for size in 8 16 32 64
    do
        /home/zhongdong/opt/git/avx512_reduction/bin/mpirun -np 1 /home/zhongdong/Downloads/git/ARM/arm_sve_reduction/test/datatype/Reduce_local_float 1024 i $size $op 
    done
done

echo "=======Float type all operations========"
echo ""
for op in max min sum mul
do
    /home/zhongdong/opt/git/avx512_reduction/bin/mpirun -np 1 /home/zhongdong/Downloads/git/ARM/arm_sve_reduction/test/datatype/Reduce_local_float 1024 f 32 $op 
done

echo "========Double type all operations========="
echo ""
for op in max min sum mul
do
    /home/zhongdong/opt/git/avx512_reduction/bin/mpirun -np 1 /home/zhongdong/Downloads/git/ARM/arm_sve_reduction/test/datatype/Reduce_local_float 1024 d 32 $op 
done
