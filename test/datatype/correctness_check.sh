echo "ompi version with AVX512 -- Usage: arg1: count of elements, args2: 'i'|'f'|'d' : datatype: integer, float, double. args3 size of type. args4 operation"
echo "/home/zhongdong/opt/git/avx512_reduction/bin/mpirun  -mca   op_sve_hardware_available 0  -mca   op_avx_hardware_available 0   -np 1 /home/zhongdong/Downloads/git/ARM/arm_sve_reduction/test/datatype/Reduce_local_float 1048576  i 8 max"

Orange= "\033[0;33m"
Blue= '\033[0;34m'
Purple= '\033[0;35m'

NC='\033[0m'

echo "=========Integer type all operations & all sizes========"
echo ""
for op in max min sum mul band bor bxor
do
    echo ""
    echo "===Operation  $op test==="
    for size in 8 16 32 64
    do
        echo -e "Test \e[1;33m __mm512 instruction for loop \e[m Total_num_bits = 512*N "
      /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1 -mca pml ob1  -np 1  armie -msve-vector-bits=256 --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_local_float  1024 i $size $op 
        echo -e "Test \e[1;34m __mm256 instruction for loop$ \e[m (512*(N-1) + 256) <  Total_num_bits < 512*N "
        echo -e "Test \e[1;35m  duff device code \e[m  512*N <  Total_num_bits < 512*N + 256  "
    done
done

echo "=======Float type all operations========"
echo ""
echo -e "Test \e[1;33m __mm512 instruction for loop \e[m Total_num_bits = 512*N "
for op in max min sum mul
do
   /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1  -mca pml ob1  -np 1  armie -msve-vector-bits=256 --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_local_float 1024 f 32 $op 
done

echo -e "Test \e[1;34m __mm256 instruction for loop$ \e[m (512*(N-1) + 256) <  Total_num_bits < 512*N "
## 28= 16+8 + 4

echo -e "Test \e[1;35m  duff device code \e[m  512*N <  Total_num_bits < 512*N + 256  "
##40=16+4

echo "========Double type all operations========="
echo ""

echo -e "Test \e[1;33m __mm512 instruction for loop \e[m Total_num_bits = 512*N "
for op in max min sum mul
do
   /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1 -mca pml ob1  -np 1  armie -msve-vector-bits=256 --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_local_float 1024 d 64 $op
done

echo -e "Test \e[1;34m __mm256 instruction for loop$ \e[m (512*(N-1) + 256) <  Total_num_bits < 512*N "
## 20= 8 +8 + 3

echo -e "Test \e[1;35m  duff device code \e[m  512*N <  Total_num_bits < 512*N + 256  "
##12=8+2
