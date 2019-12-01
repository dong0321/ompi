/*
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * Copyright (c) 2019      Arm Ltd.  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#include "opal/util/output.h"

#include "ompi/op/op.h"
#include "ompi/mca/op/op.h"
#include "ompi/mca/op/base/base.h"
#include "ompi/mca/op/arm_sve_op/op_sve.h"
#include "ompi/mca/op/arm_sve_op/op_sve_functions.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

/*
 * Since all the functions in this file are essentially identical, we
 * use a macro to substitute in names and types.  The core operation
 * in all functions that use this macro is the same.
 *
 * This macro is for (out op in).
 *
 */
#define OP_SVE_FUNC(name, type_name, type_size, type, op) \
    static void ompi_op_sve_2buff_##name##_##type(void *_in, void *_out, int *count, \
            struct ompi_datatype_t **dtype, \
            struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int types_per_step = svcnt##type_name();                           \
    int left_over = *count; \
    type* in = (type*)_in; \
    type* out = (type*)_out; \
    svbool_t Pg = svptrue_b##type_size(); \
    for (; left_over >= types_per_step; left_over -= types_per_step) { \
        sv##type  vsrc = svld1(Pg, in);                                \
        sv##type  vdst = svld1(Pg, out);                               \
        in += types_per_step;                                          \
        vdst=sv##op##_z(Pg,vdst,vsrc);                                 \
        svst1(Pg, out,vdst);                                           \
        out += types_per_step; \
    }                                                                  \
    \
    if (left_over !=0){                                                \
       switch(left_over) {                         \
           case 256: out[255] = current_func(out[255],in[255]) ;                        \
           case 255: out[254] = current_func(out[254],in[254]) ;                        \
           case 254: out[253] = current_func(out[253],in[253]) ;                        \
           case 253: out[252] = current_func(out[252],in[252]) ;                        \
           case 252: out[251] = current_func(out[251],in[251]) ;                        \
           case 251: out[250] = current_func(out[250],in[250]) ;                        \
           case 250: out[249] = current_func(out[249],in[249]) ;                        \
           case 249: out[248] = current_func(out[248],in[248]) ;                        \
           case 248: out[247] = current_func(out[247],in[247]) ;                        \
           case 247: out[246] = current_func(out[246],in[246]) ;                        \
           case 246: out[245] = current_func(out[245],in[245]) ;                        \
           case 245: out[244] = current_func(out[244],in[244]) ;                        \
           case 244: out[243] = current_func(out[243],in[243]) ;                        \
           case 243: out[242] = current_func(out[242],in[242]) ;                        \
           case 242: out[241] = current_func(out[241],in[241]) ;                        \
           case 241: out[240] = current_func(out[240],in[240]) ;                        \
           case 240: out[239] = current_func(out[239],in[239]) ;                        \
           case 239: out[238] = current_func(out[238],in[238]) ;                        \
           case 238: out[237] = current_func(out[237],in[237]) ;                        \
           case 237: out[236] = current_func(out[236],in[236]) ;                        \
           case 236: out[235] = current_func(out[235],in[235]) ;                        \
           case 235: out[234] = current_func(out[234],in[234]) ;                        \
           case 234: out[233] = current_func(out[233],in[233]) ;                        \
           case 233: out[232] = current_func(out[232],in[232]) ;                        \
           case 232: out[231] = current_func(out[231],in[231]) ;                        \
           case 231: out[230] = current_func(out[230],in[230]) ;                        \
           case 230: out[229] = current_func(out[229],in[229]) ;                        \
           case 229: out[228] = current_func(out[228],in[228]) ;                        \
           case 228: out[227] = current_func(out[227],in[227]) ;                        \
           case 227: out[226] = current_func(out[226],in[226]) ;                        \
           case 226: out[225] = current_func(out[225],in[225]) ;                        \
           case 225: out[224] = current_func(out[224],in[224]) ;                        \
           case 224: out[223] = current_func(out[223],in[223]) ;                        \
           case 223: out[222] = current_func(out[222],in[222]) ;                        \
           case 222: out[221] = current_func(out[221],in[221]) ;                        \
           case 221: out[220] = current_func(out[220],in[220]) ;                        \
           case 220: out[219] = current_func(out[219],in[219]) ;                        \
           case 219: out[218] = current_func(out[218],in[218]) ;                        \
           case 218: out[217] = current_func(out[217],in[217]) ;                        \
           case 217: out[216] = current_func(out[216],in[216]) ;                        \
           case 216: out[215] = current_func(out[215],in[215]) ;                        \
           case 215: out[214] = current_func(out[214],in[214]) ;                        \
           case 214: out[213] = current_func(out[213],in[213]) ;                        \
           case 213: out[212] = current_func(out[212],in[212]) ;                        \
           case 212: out[211] = current_func(out[211],in[211]) ;                        \
           case 211: out[210] = current_func(out[210],in[210]) ;                        \
           case 210: out[209] = current_func(out[209],in[209]) ;                        \
           case 209: out[208] = current_func(out[208],in[208]) ;                        \
           case 208: out[207] = current_func(out[207],in[207]) ;                        \
           case 207: out[206] = current_func(out[206],in[206]) ;                        \
           case 206: out[205] = current_func(out[205],in[205]) ;                        \
           case 205: out[204] = current_func(out[204],in[204]) ;                        \
           case 204: out[203] = current_func(out[203],in[203]) ;                        \
           case 203: out[202] = current_func(out[202],in[202]) ;                        \
           case 202: out[201] = current_func(out[201],in[201]) ;                        \
           case 201: out[200] = current_func(out[200],in[200]) ;                        \
           case 200: out[199] = current_func(out[199],in[199]) ;                        \
           case 199: out[198] = current_func(out[198],in[198]) ;                        \
           case 198: out[197] = current_func(out[197],in[197]) ;                        \
           case 197: out[196] = current_func(out[196],in[196]) ;                        \
           case 196: out[195] = current_func(out[195],in[195]) ;                        \
           case 195: out[194] = current_func(out[194],in[194]) ;                        \
           case 194: out[193] = current_func(out[193],in[193]) ;                        \
           case 193: out[192] = current_func(out[192],in[192]) ;                        \
           case 192: out[191] = current_func(out[191],in[191]) ;                        \
           case 191: out[190] = current_func(out[190],in[190]) ;                        \
           case 190: out[189] = current_func(out[189],in[189]) ;                        \
           case 189: out[188] = current_func(out[188],in[188]) ;                        \
           case 188: out[187] = current_func(out[187],in[187]) ;                        \
           case 187: out[186] = current_func(out[186],in[186]) ;                        \
           case 186: out[185] = current_func(out[185],in[185]) ;                        \
           case 185: out[184] = current_func(out[184],in[184]) ;                        \
           case 184: out[183] = current_func(out[183],in[183]) ;                        \
           case 183: out[182] = current_func(out[182],in[182]) ;                        \
           case 182: out[181] = current_func(out[181],in[181]) ;                        \
           case 181: out[180] = current_func(out[180],in[180]) ;                        \
           case 180: out[179] = current_func(out[179],in[179]) ;                        \
           case 179: out[178] = current_func(out[178],in[178]) ;                        \
           case 178: out[177] = current_func(out[177],in[177]) ;                        \
           case 177: out[176] = current_func(out[176],in[176]) ;                        \
           case 176: out[175] = current_func(out[175],in[175]) ;                        \
           case 175: out[174] = current_func(out[174],in[174]) ;                        \
           case 174: out[173] = current_func(out[173],in[173]) ;                        \
           case 173: out[172] = current_func(out[172],in[172]) ;                        \
           case 172: out[171] = current_func(out[171],in[171]) ;                        \
           case 171: out[170] = current_func(out[170],in[170]) ;                        \
           case 170: out[169] = current_func(out[169],in[169]) ;                        \
           case 169: out[168] = current_func(out[168],in[168]) ;                        \
           case 168: out[167] = current_func(out[167],in[167]) ;                        \
           case 167: out[166] = current_func(out[166],in[166]) ;                        \
           case 166: out[165] = current_func(out[165],in[165]) ;                        \
           case 165: out[164] = current_func(out[164],in[164]) ;                        \
           case 164: out[163] = current_func(out[163],in[163]) ;                        \
           case 163: out[162] = current_func(out[162],in[162]) ;                        \
           case 162: out[161] = current_func(out[161],in[161]) ;                        \
           case 161: out[160] = current_func(out[160],in[160]) ;                        \
           case 160: out[159] = current_func(out[159],in[159]) ;                        \
           case 159: out[158] = current_func(out[158],in[158]) ;                        \
           case 158: out[157] = current_func(out[157],in[157]) ;                        \
           case 157: out[156] = current_func(out[156],in[156]) ;                        \
           case 156: out[155] = current_func(out[155],in[155]) ;                        \
           case 155: out[154] = current_func(out[154],in[154]) ;                        \
           case 154: out[153] = current_func(out[153],in[153]) ;                        \
           case 153: out[152] = current_func(out[152],in[152]) ;                        \
           case 152: out[151] = current_func(out[151],in[151]) ;                        \
           case 151: out[150] = current_func(out[150],in[150]) ;                        \
           case 150: out[149] = current_func(out[149],in[149]) ;                        \
           case 149: out[148] = current_func(out[148],in[148]) ;                        \
           case 148: out[147] = current_func(out[147],in[147]) ;                        \
           case 147: out[146] = current_func(out[146],in[146]) ;                        \
           case 146: out[145] = current_func(out[145],in[145]) ;                        \
           case 145: out[144] = current_func(out[144],in[144]) ;                        \
           case 144: out[143] = current_func(out[143],in[143]) ;                        \
           case 143: out[142] = current_func(out[142],in[142]) ;                        \
           case 142: out[141] = current_func(out[141],in[141]) ;                        \
           case 141: out[140] = current_func(out[140],in[140]) ;                        \
           case 140: out[139] = current_func(out[139],in[139]) ;                        \
           case 139: out[138] = current_func(out[138],in[138]) ;                        \
           case 138: out[137] = current_func(out[137],in[137]) ;                        \
           case 137: out[136] = current_func(out[136],in[136]) ;                        \
           case 136: out[135] = current_func(out[135],in[135]) ;                        \
           case 135: out[134] = current_func(out[134],in[134]) ;                        \
           case 134: out[133] = current_func(out[133],in[133]) ;                        \
           case 133: out[132] = current_func(out[132],in[132]) ;                        \
           case 132: out[131] = current_func(out[131],in[131]) ;                        \
           case 131: out[130] = current_func(out[130],in[130]) ;                        \
           case 130: out[129] = current_func(out[129],in[129]) ;                        \
           case 129: out[128] = current_func(out[128],in[128]) ;                        \
           case 128: out[127] = current_func(out[127],in[127]) ;                        \
           case 127: out[126] = current_func(out[126],in[126]) ;                        \
           case 126: out[125] = current_func(out[125],in[125]) ;                        \
           case 125: out[124] = current_func(out[124],in[124]) ;                        \
           case 124: out[123] = current_func(out[123],in[123]) ;                        \
           case 123: out[122] = current_func(out[122],in[122]) ;                        \
           case 122: out[121] = current_func(out[121],in[121]) ;                        \
           case 121: out[120] = current_func(out[120],in[120]) ;                        \
           case 120: out[119] = current_func(out[119],in[119]) ;                        \
           case 119: out[118] = current_func(out[118],in[118]) ;                        \
           case 118: out[117] = current_func(out[117],in[117]) ;                        \
           case 117: out[116] = current_func(out[116],in[116]) ;                        \
           case 116: out[115] = current_func(out[115],in[115]) ;                        \
           case 115: out[114] = current_func(out[114],in[114]) ;                        \
           case 114: out[113] = current_func(out[113],in[113]) ;                        \
           case 113: out[112] = current_func(out[112],in[112]) ;                        \
           case 112: out[111] = current_func(out[111],in[111]) ;                        \
           case 111: out[110] = current_func(out[110],in[110]) ;                        \
           case 110: out[109] = current_func(out[109],in[109]) ;                        \
           case 109: out[108] = current_func(out[108],in[108]) ;                        \
           case 108: out[107] = current_func(out[107],in[107]) ;                        \
           case 107: out[106] = current_func(out[106],in[106]) ;                        \
           case 106: out[105] = current_func(out[105],in[105]) ;                        \
           case 105: out[104] = current_func(out[104],in[104]) ;                        \
           case 104: out[103] = current_func(out[103],in[103]) ;                        \
           case 103: out[102] = current_func(out[102],in[102]) ;                        \
           case 102: out[101] = current_func(out[101],in[101]) ;                        \
           case 101: out[100] = current_func(out[100],in[100]) ;                        \
           case 100: out[99] = current_func(out[99],in[99]) ;                        \
           case 99: out[98] = current_func(out[98],in[98]) ;                        \
           case 98: out[97] = current_func(out[97],in[97]) ;                        \
           case 97: out[96] = current_func(out[96],in[96]) ;                        \
           case 96: out[95] = current_func(out[95],in[95]) ;                        \
           case 95: out[94] = current_func(out[94],in[94]) ;                        \
           case 94: out[93] = current_func(out[93],in[93]) ;                        \
           case 93: out[92] = current_func(out[92],in[92]) ;                        \
           case 92: out[91] = current_func(out[91],in[91]) ;                        \
           case 91: out[90] = current_func(out[90],in[90]) ;                        \
           case 90: out[89] = current_func(out[89],in[89]) ;                        \
           case 89: out[88] = current_func(out[88],in[88]) ;                        \
           case 88: out[87] = current_func(out[87],in[87]) ;                        \
           case 87: out[86] = current_func(out[86],in[86]) ;                        \
           case 86: out[85] = current_func(out[85],in[85]) ;                        \
           case 85: out[84] = current_func(out[84],in[84]) ;                        \
           case 84: out[83] = current_func(out[83],in[83]) ;                        \
           case 83: out[82] = current_func(out[82],in[82]) ;                        \
           case 82: out[81] = current_func(out[81],in[81]) ;                        \
           case 81: out[80] = current_func(out[80],in[80]) ;                        \
           case 80: out[79] = current_func(out[79],in[79]) ;                        \
           case 79: out[78] = current_func(out[78],in[78]) ;                        \
           case 78: out[77] = current_func(out[77],in[77]) ;                        \
           case 77: out[76] = current_func(out[76],in[76]) ;                        \
           case 76: out[75] = current_func(out[75],in[75]) ;                        \
           case 75: out[74] = current_func(out[74],in[74]) ;                        \
           case 74: out[73] = current_func(out[73],in[73]) ;                        \
           case 73: out[72] = current_func(out[72],in[72]) ;                        \
           case 72: out[71] = current_func(out[71],in[71]) ;                        \
           case 71: out[70] = current_func(out[70],in[70]) ;                        \
           case 70: out[69] = current_func(out[69],in[69]) ;                        \
           case 69: out[68] = current_func(out[68],in[68]) ;                        \
           case 68: out[67] = current_func(out[67],in[67]) ;                        \
           case 67: out[66] = current_func(out[66],in[66]) ;                        \
           case 66: out[65] = current_func(out[65],in[65]) ;                        \
           case 65: out[64] = current_func(out[64],in[64]) ;                        \
           case 64: out[63] = current_func(out[63],in[63]) ;                        \
           case 63: out[62] = current_func(out[62],in[62]) ;                        \
           case 62: out[61] = current_func(out[61],in[61]) ;                        \
           case 61: out[60] = current_func(out[60],in[60]) ;                        \
           case 60: out[59] = current_func(out[59],in[59]) ;                        \
           case 59: out[58] = current_func(out[58],in[58]) ;                        \
           case 58: out[57] = current_func(out[57],in[57]) ;                        \
           case 57: out[56] = current_func(out[56],in[56]) ;                        \
           case 56: out[55] = current_func(out[55],in[55]) ;                        \
           case 55: out[54] = current_func(out[54],in[54]) ;                        \
           case 54: out[53] = current_func(out[53],in[53]) ;                        \
           case 53: out[52] = current_func(out[52],in[52]) ;                        \
           case 52: out[51] = current_func(out[51],in[51]) ;                        \
           case 51: out[50] = current_func(out[50],in[50]) ;                        \
           case 50: out[49] = current_func(out[49],in[49]) ;                        \
           case 49: out[48] = current_func(out[48],in[48]) ;                        \
           case 48: out[47] = current_func(out[47],in[47]) ;                        \
           case 47: out[46] = current_func(out[46],in[46]) ;                        \
           case 46: out[45] = current_func(out[45],in[45]) ;                        \
           case 45: out[44] = current_func(out[44],in[44]) ;                        \
           case 44: out[43] = current_func(out[43],in[43]) ;                        \
           case 43: out[42] = current_func(out[42],in[42]) ;                        \
           case 42: out[41] = current_func(out[41],in[41]) ;                        \
           case 41: out[40] = current_func(out[40],in[40]) ;                        \
           case 40: out[39] = current_func(out[39],in[39]) ;                        \
           case 39: out[38] = current_func(out[38],in[38]) ;                        \
           case 38: out[37] = current_func(out[37],in[37]) ;                        \
           case 37: out[36] = current_func(out[36],in[36]) ;                        \
           case 36: out[35] = current_func(out[35],in[35]) ;                        \
           case 35: out[34] = current_func(out[34],in[34]) ;                        \
           case 34: out[33] = current_func(out[33],in[33]) ;                        \
           case 33: out[32] = current_func(out[32],in[32]) ;                        \
           case 32: out[31] = current_func(out[31],in[31]) ;                        \
           case 31: out[30] = current_func(out[30],in[30]) ;                        \
           case 30: out[29] = current_func(out[29],in[29]) ;                        \
           case 29: out[28] = current_func(out[28],in[28]) ;                        \
           case 28: out[27] = current_func(out[27],in[27]) ;                        \
           case 27: out[26] = current_func(out[26],in[26]) ;                        \
           case 26: out[25] = current_func(out[25],in[25]) ;                        \
           case 25: out[24] = current_func(out[24],in[24]) ;                        \
           case 24: out[23] = current_func(out[23],in[23]) ;                        \
           case 23: out[22] = current_func(out[22],in[22]) ;                        \
           case 22: out[21] = current_func(out[21],in[21]) ;                        \
           case 21: out[20] = current_func(out[20],in[20]) ;                        \
           case 20: out[19] = current_func(out[19],in[19]) ;                        \
           case 19: out[18] = current_func(out[18],in[18]) ;                        \
           case 18: out[17] = current_func(out[17],in[17]) ;                        \
           case 17: out[16] = current_func(out[16],in[16]) ;                        \
           case 16: out[15] = current_func(out[15],in[15]) ;                        \
           case 15: out[14] = current_func(out[14],in[14]) ;                        \
           case 14: out[13] = current_func(out[13],in[13]) ;                        \
           case 13: out[12] = current_func(out[12],in[12]) ;                        \
           case 12: out[11] = current_func(out[11],in[11]) ;                        \
           case 11: out[10] = current_func(out[10],in[10]) ;                        \
           case 10: out[9] = current_func(out[9],in[9]) ;                        \
           case 9: out[8] = current_func(out[8],in[8]) ;                        \
           case 8: out[7] = current_func(out[7],in[7]) ;                        \
           case 7: out[6] = current_func(out[6],in[6]) ;                        \
           case 6: out[5] = current_func(out[5],in[5]) ;                        \
           case 5: out[4] = current_func(out[4],in[4]) ;                        \
           case 4: out[3] = current_func(out[3],in[3]) ;                        \
           case 3: out[2] = current_func(out[2],in[2]) ;                        \
           case 2: out[1] = current_func(out[1],in[1]) ;                        \
           case 1: out[0] = current_func(out[0],in[0]) ;                        \
       }\
    } \
}

/*************************************************************************
 * Max
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) > (b) ? (a) : (b))
    OP_SVE_FUNC(max, b,  8,   int8_t, max)
    OP_SVE_FUNC(max, b,   8,  uint8_t, max)
    OP_SVE_FUNC(max, h,  16,  int16_t, max)
    OP_SVE_FUNC(max, h, 16, uint16_t, max)
    OP_SVE_FUNC(max, w,  32,  int32_t, max)
    OP_SVE_FUNC(max, w, 32, uint32_t, max)
    OP_SVE_FUNC(max, d,  64,  int64_t, max)
    OP_SVE_FUNC(max, d, 64, uint64_t, max)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC(max, h, 16, float16_t, max)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC(max, h, 16, float16_t, max)
#endif
    OP_SVE_FUNC(max, w, 32, float32_t, max)
    OP_SVE_FUNC(max, d, 64, float64_t, max)

/*************************************************************************
 * Min
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) < (b) ? (a) : (b))
    OP_SVE_FUNC(min, b,  8,   int8_t, min)
    OP_SVE_FUNC(min, b,   8,  uint8_t, min)
    OP_SVE_FUNC(min, h,  16,  int16_t, min)
    OP_SVE_FUNC(min, h, 16, uint16_t, min)
    OP_SVE_FUNC(min, w,  32,  int32_t, min)
    OP_SVE_FUNC(min, w, 32, uint32_t, min)
    OP_SVE_FUNC(min, d,  64,  int64_t, min)
    OP_SVE_FUNC(min, d, 64, uint64_t, min)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC(min, h, 16, float16_t, min)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC(min, h, 16, float16_t, min)
#endif
    OP_SVE_FUNC(min, w, 32, float32_t, min)
    OP_SVE_FUNC(min, d, 64, float64_t, min)

 /*************************************************************************
 * Sum
 ************************************************************************/
#undef current_func
#define current_func(a, b) ((a) + (b))
    OP_SVE_FUNC(sum, b,  8,   int8_t, add)
    OP_SVE_FUNC(sum, b,   8,  uint8_t, add)
    OP_SVE_FUNC(sum, h,  16,  int16_t, add)
    OP_SVE_FUNC(sum, h, 16, uint16_t, add)
    OP_SVE_FUNC(sum, w,  32,  int32_t, add)
    OP_SVE_FUNC(sum, w, 32, uint32_t, add)
    OP_SVE_FUNC(sum, d,  64,  int64_t, add)
    OP_SVE_FUNC(sum, d, 64, uint64_t, add)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC(sum, h, 16, float16_t, add)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC(sum, h, 16, float16_t, add)
#endif
    OP_SVE_FUNC(sum, w, 32, float32_t, add)
    OP_SVE_FUNC(sum, d, 64, float64_t, add)

/*************************************************************************
 * Product
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) * (b))
    OP_SVE_FUNC(prod, b,  8,   int8_t, mul)
    OP_SVE_FUNC(prod, b,   8,  uint8_t, mul)
    OP_SVE_FUNC(prod, h,  16,  int16_t, mul)
    OP_SVE_FUNC(prod, h, 16, uint16_t, mul)
    OP_SVE_FUNC(prod, w,  32,  int32_t, mul)
    OP_SVE_FUNC(prod, w, 32, uint32_t, mul)
    OP_SVE_FUNC(prod, d,  64,  int64_t, mul)
    OP_SVE_FUNC(prod, d, 64, uint64_t, mul)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
OP_SVE_FUNC(prod, h, 16, float16_t, mul)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
OP_SVE_FUNC(prod, h, 16, float16_t, mul)
#endif
    OP_SVE_FUNC(prod, w, 32, float32_t, mul)
OP_SVE_FUNC(prod, d, 64, float64_t, mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) & (b))
    OP_SVE_FUNC(band, b,  8,   int8_t, and)
    OP_SVE_FUNC(band, b,   8,  uint8_t, and)
    OP_SVE_FUNC(band, h,  16,  int16_t, and)
    OP_SVE_FUNC(band, h, 16, uint16_t, and)
    OP_SVE_FUNC(band, w,  32,  int32_t, and)
    OP_SVE_FUNC(band, w, 32, uint32_t, and)
    OP_SVE_FUNC(band, d,  64,  int64_t, and)
OP_SVE_FUNC(band, d, 64, uint64_t, and)

 /*************************************************************************
 * Bitwise OR
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) | (b))
    OP_SVE_FUNC(bor, b,  8,   int8_t, orr)
    OP_SVE_FUNC(bor, b,   8,  uint8_t, orr)
    OP_SVE_FUNC(bor, h,  16,  int16_t, orr)
    OP_SVE_FUNC(bor, h, 16, uint16_t, orr)
    OP_SVE_FUNC(bor, w,  32,  int32_t, orr)
    OP_SVE_FUNC(bor, w, 32, uint32_t, orr)
    OP_SVE_FUNC(bor, d,  64,  int64_t, orr)
OP_SVE_FUNC(bor, d, 64, uint64_t, orr)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
#undef current_func
#define current_func(a, b) ((a) ^ (b))
    OP_SVE_FUNC(bxor, b,  8,   int8_t, eor)
    OP_SVE_FUNC(bxor, b,   8,  uint8_t, eor)
    OP_SVE_FUNC(bxor, h,  16,  int16_t, eor)
    OP_SVE_FUNC(bxor, h, 16, uint16_t, eor)
    OP_SVE_FUNC(bxor, w,  32,  int32_t, eor)
    OP_SVE_FUNC(bxor, w, 32, uint32_t, eor)
    OP_SVE_FUNC(bxor, d,  64,  int64_t, eor)
OP_SVE_FUNC(bxor, d, 64, uint64_t, eor)

/*
 *  This is a three buffer (2 input and 1 output) version of the reduction
 *  routines, needed for some optimizations.
 */
#define OP_SVE_FUNC_3BUFF(name, type_name, type_size, type, op) \
        static void ompi_op_sve_3buff_##name##_##type(void * restrict _in1,   \
                void * restrict _in2, void * restrict _out, int *count, \
                struct ompi_datatype_t **dtype, \
                struct ompi_op_base_module_1_0_0_t *module) \
{                                                                      \
    int types_per_step = svcnt##type_name();                           \
    int left_over = *count; \
    type* in1 = (type*)_in1; \
    type* in2 = (type*)_in2; \
    type* out = (type*)_out; \
    svbool_t Pg = svptrue_b##type_size(); \
    for (; left_over >= types_per_step; left_over -= types_per_step) { \
        sv##type  vsrc = svld1(Pg, in1);                               \
        sv##type  vdst = svld1(Pg, in2);                               \
        in1 += types_per_step; \
        in2 += types_per_step; \
        vdst=sv##op##_z(Pg,vdst,vsrc);                                 \
        svst1(Pg, out,vdst);                                           \
        out += types_per_step; \
    }                                                                  \
    if (left_over !=0){                                                \
        Pg = svwhilelt_b##type_size##_u64(0, left_over);               \
        sv##type  vsrc = svld1(Pg, in1);                               \
        sv##type  vdst = svld1(Pg, in2);                               \
        vdst=sv##op##_z(Pg,vdst,vsrc);                                 \
        svst1(Pg, out,vdst);                                           \
    } \
}

/*************************************************************************
 * Max
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(max, b,  8,   int8_t, max)
    OP_SVE_FUNC_3BUFF(max, b,   8,  uint8_t, max)
    OP_SVE_FUNC_3BUFF(max, h,  16,  int16_t, max)
    OP_SVE_FUNC_3BUFF(max, h, 16, uint16_t, max)
    OP_SVE_FUNC_3BUFF(max, w,  32,  int32_t, max)
    OP_SVE_FUNC_3BUFF(max, w, 32, uint32_t, max)
    OP_SVE_FUNC_3BUFF(max, d,  64,  int64_t, max)
    OP_SVE_FUNC_3BUFF(max, d, 64, uint64_t, max)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC_3BUFF(max, h, 16, float16_t, max)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC_3BUFF(max, h, 16, float16_t, max)
#endif
    OP_SVE_FUNC_3BUFF(max, w, 32, float32_t, max)
    OP_SVE_FUNC_3BUFF(max, d, 64, float64_t, max)

/*************************************************************************
 * Min
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(min, b,  8,   int8_t, min)
    OP_SVE_FUNC_3BUFF(min, b,   8,  uint8_t, min)
    OP_SVE_FUNC_3BUFF(min, h,  16,  int16_t, min)
    OP_SVE_FUNC_3BUFF(min, h, 16, uint16_t, min)
    OP_SVE_FUNC_3BUFF(min, w,  32,  int32_t, min)
    OP_SVE_FUNC_3BUFF(min, w, 32, uint32_t, min)
    OP_SVE_FUNC_3BUFF(min, d,  64,  int64_t, min)
    OP_SVE_FUNC_3BUFF(min, d, 64, uint64_t, min)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC_3BUFF(min, h, 16, float16_t, min)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC_3BUFF(min, h, 16, float16_t, min)
#endif
    OP_SVE_FUNC_3BUFF(min, w, 32, float32_t, min)
    OP_SVE_FUNC_3BUFF(min, d, 64, float64_t, min)

 /*************************************************************************
 * Sum
 ************************************************************************/
    OP_SVE_FUNC_3BUFF(sum, b,  8,   int8_t, add)
    OP_SVE_FUNC_3BUFF(sum, b,   8,  uint8_t, add)
    OP_SVE_FUNC_3BUFF(sum, h,  16,  int16_t, add)
    OP_SVE_FUNC_3BUFF(sum, h, 16, uint16_t, add)
    OP_SVE_FUNC_3BUFF(sum, w,  32,  int32_t, add)
    OP_SVE_FUNC_3BUFF(sum, w, 32, uint32_t, add)
    OP_SVE_FUNC_3BUFF(sum, d,  64,  int64_t, add)
    OP_SVE_FUNC_3BUFF(sum, d, 64, uint64_t, add)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
    OP_SVE_FUNC_3BUFF(sum, h, 16, float16_t, add)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
    OP_SVE_FUNC_3BUFF(sum, h, 16, float16_t, add)
#endif
    OP_SVE_FUNC_3BUFF(sum, w, 32, float32_t, add)
    OP_SVE_FUNC_3BUFF(sum, d, 64, float64_t, add)

/*************************************************************************
 * Product
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(prod, b,  8,   int8_t, mul)
    OP_SVE_FUNC_3BUFF(prod, b,   8,  uint8_t, mul)
    OP_SVE_FUNC_3BUFF(prod, h,  16,  int16_t, mul)
    OP_SVE_FUNC_3BUFF(prod, h, 16, uint16_t, mul)
    OP_SVE_FUNC_3BUFF(prod, w,  32,  int32_t, mul)
    OP_SVE_FUNC_3BUFF(prod, w, 32, uint32_t, mul)
    OP_SVE_FUNC_3BUFF(prod, d,  64,  int64_t, mul)
OP_SVE_FUNC_3BUFF(prod, d, 64, uint64_t, mul)

    /* Floating point */
#if defined(HAVE_SHORT_FLOAT)
OP_SVE_FUNC_3BUFF(prod, h, 16, float16_t, mul)
#elif defined(HAVE_OPAL_SHORT_FLOAT_T)
OP_SVE_FUNC_3BUFF(prod, h, 16, float16_t, mul)
#endif
    OP_SVE_FUNC_3BUFF(prod, w, 32, float32_t, mul)
OP_SVE_FUNC_3BUFF(prod, d, 64, float64_t, mul)

/*************************************************************************
 * Bitwise AND
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(band, b,  8,   int8_t, and)
    OP_SVE_FUNC_3BUFF(band, b,   8,  uint8_t, and)
    OP_SVE_FUNC_3BUFF(band, h,  16,  int16_t, and)
    OP_SVE_FUNC_3BUFF(band, h, 16, uint16_t, and)
    OP_SVE_FUNC_3BUFF(band, w,  32,  int32_t, and)
    OP_SVE_FUNC_3BUFF(band, w, 32, uint32_t, and)
    OP_SVE_FUNC_3BUFF(band, d,  64,  int64_t, and)
OP_SVE_FUNC_3BUFF(band, d, 64, uint64_t, and)

 /*************************************************************************
 * Bitwise OR
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(bor, b,  8,   int8_t, orr)
    OP_SVE_FUNC_3BUFF(bor, b,   8,  uint8_t, orr)
    OP_SVE_FUNC_3BUFF(bor, h,  16,  int16_t, orr)
    OP_SVE_FUNC_3BUFF(bor, h, 16, uint16_t, orr)
    OP_SVE_FUNC_3BUFF(bor, w,  32,  int32_t, orr)
    OP_SVE_FUNC_3BUFF(bor, w, 32, uint32_t, orr)
    OP_SVE_FUNC_3BUFF(bor, d,  64,  int64_t, orr)
OP_SVE_FUNC_3BUFF(bor, d, 64, uint64_t, orr)

/*************************************************************************
 * Bitwise XOR
 *************************************************************************/
    OP_SVE_FUNC_3BUFF(bxor, b,  8,   int8_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, b,   8,  uint8_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, h,  16,  int16_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, h, 16, uint16_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, w,  32,  int32_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, w, 32, uint32_t, eor)
    OP_SVE_FUNC_3BUFF(bxor, d,  64,  int64_t, eor)
OP_SVE_FUNC_3BUFF(bxor, d, 64, uint64_t, eor)

/** C integer ***********************************************************/
#define C_INTEGER(name, ftype)                                              \
    [OMPI_OP_BASE_TYPE_INT8_T] = ompi_op_sve_##ftype##_##name##_int8_t,     \
    [OMPI_OP_BASE_TYPE_UINT8_T] = ompi_op_sve_##ftype##_##name##_uint8_t,   \
    [OMPI_OP_BASE_TYPE_INT16_T] = ompi_op_sve_##ftype##_##name##_int16_t,   \
    [OMPI_OP_BASE_TYPE_UINT16_T] = ompi_op_sve_##ftype##_##name##_uint16_t, \
    [OMPI_OP_BASE_TYPE_INT32_T] = ompi_op_sve_##ftype##_##name##_int32_t,   \
    [OMPI_OP_BASE_TYPE_UINT32_T] = ompi_op_sve_##ftype##_##name##_uint32_t, \
    [OMPI_OP_BASE_TYPE_INT64_T] = ompi_op_sve_##ftype##_##name##_int64_t,   \
    [OMPI_OP_BASE_TYPE_UINT64_T] = ompi_op_sve_##ftype##_##name##_uint64_t


/** Floating point, including all the Fortran reals *********************/
#if defined(HAVE_SHORT_FLOAT) || defined(HAVE_OPAL_SHORT_FLOAT_T)
#define SHORT_FLOAT(name, ftype) ompi_op_sve_##ftype##_##name##_float16_t
#else
#define SHORT_FLOAT(name, ftype) NULL
#endif
#define FLOAT(name, ftype) ompi_op_sve_##ftype##_##name##_float32_t
#define DOUBLE(name, ftype) ompi_op_sve_##ftype##_##name##_float64_t

#define FLOATING_POINT(name, ftype)                                        \
    [OMPI_OP_BASE_TYPE_SHORT_FLOAT] = SHORT_FLOAT(name, ftype),            \
    [OMPI_OP_BASE_TYPE_FLOAT] = FLOAT(name, ftype),                        \
    [OMPI_OP_BASE_TYPE_DOUBLE] = DOUBLE(name, ftype)

/*
 * MPI_OP_NULL
 * All types
 */
#define FLAGS_NO_FLOAT \
        (OMPI_OP_FLAGS_INTRINSIC | OMPI_OP_FLAGS_ASSOC | OMPI_OP_FLAGS_COMMUTE)
#define FLAGS \
        (OMPI_OP_FLAGS_INTRINSIC | OMPI_OP_FLAGS_ASSOC | \
         OMPI_OP_FLAGS_FLOAT_ASSOC | OMPI_OP_FLAGS_COMMUTE)

ompi_op_base_handler_fn_t ompi_op_sve_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] =
{
    /* Corresponds to MPI_OP_NULL */
    [OMPI_OP_BASE_FORTRAN_NULL] = {
        /* Leaving this empty puts in NULL for all entries */
        NULL,
    },
    /* Corresponds to MPI_MAX */
    [OMPI_OP_BASE_FORTRAN_MAX] = {
        C_INTEGER(max, 2buff),
        FLOATING_POINT(max, 2buff),
    },
    /* Corresponds to MPI_MIN */
    [OMPI_OP_BASE_FORTRAN_MIN] = {
        C_INTEGER(min, 2buff),
        FLOATING_POINT(min, 2buff),
    },
    /* Corresponds to MPI_SUM */
    [OMPI_OP_BASE_FORTRAN_SUM] = {
        C_INTEGER(sum, 2buff),
        FLOATING_POINT(sum, 2buff),
    },
    /* Corresponds to MPI_PROD */
    [OMPI_OP_BASE_FORTRAN_PROD] = {
        C_INTEGER(prod, 2buff),
        FLOATING_POINT(prod, 2buff),
    },
    /* Corresponds to MPI_LAND */
    [OMPI_OP_BASE_FORTRAN_LAND] = {
        NULL,
    },
    /* Corresponds to MPI_BAND */
    [OMPI_OP_BASE_FORTRAN_BAND] = {
        C_INTEGER(band, 2buff),
    },
    /* Corresponds to MPI_LOR */
    [OMPI_OP_BASE_FORTRAN_LOR] = {
        NULL,
    },
    /* Corresponds to MPI_BOR */
    [OMPI_OP_BASE_FORTRAN_BOR] = {
        C_INTEGER(bor, 2buff),
    },
    /* Corresponds to MPI_LXOR */
    [OMPI_OP_BASE_FORTRAN_LXOR] = {
        NULL,
    },
    /* Corresponds to MPI_BXOR */
    [OMPI_OP_BASE_FORTRAN_BXOR] = {
        C_INTEGER(bxor, 2buff),
    },
    /* Corresponds to MPI_REPLACE */
    [OMPI_OP_BASE_FORTRAN_REPLACE] = {
        /* (MPI_ACCUMULATE is handled differently than the other
           reductions, so just zero out its function
           implementations here to ensure that users don't invoke
           MPI_REPLACE with any reduction operations other than
           ACCUMULATE) */
        NULL,
    },

};

ompi_op_base_3buff_handler_fn_t ompi_op_sve_3buff_functions[OMPI_OP_BASE_FORTRAN_OP_MAX][OMPI_OP_BASE_TYPE_MAX] =
{
    /* Corresponds to MPI_OP_NULL */
    [OMPI_OP_BASE_FORTRAN_NULL] = {
        /* Leaving this empty puts in NULL for all entries */
        NULL,
    },
    /* Corresponds to MPI_MAX */
    [OMPI_OP_BASE_FORTRAN_MAX] = {
        C_INTEGER(max, 3buff),
        FLOATING_POINT(max, 3buff),
    },
    /* Corresponds to MPI_MIN */
    [OMPI_OP_BASE_FORTRAN_MIN] = {
        C_INTEGER(min, 3buff),
        FLOATING_POINT(min, 3buff),
    },
    /* Corresponds to MPI_SUM */
    [OMPI_OP_BASE_FORTRAN_SUM] = {
        C_INTEGER(sum, 3buff),
        FLOATING_POINT(sum, 3buff),
    },
    /* Corresponds to MPI_PROD */
    [OMPI_OP_BASE_FORTRAN_PROD] = {
        C_INTEGER(prod, 3buff),
        FLOATING_POINT(prod, 3buff),
    },
    /* Corresponds to MPI_LAND */
    [OMPI_OP_BASE_FORTRAN_LAND] ={
        NULL,
    },
    /* Corresponds to MPI_BAND */
    [OMPI_OP_BASE_FORTRAN_BAND] = {
        C_INTEGER(band, 3buff),
    },
    /* Corresponds to MPI_LOR */
    [OMPI_OP_BASE_FORTRAN_LOR] = {
        NULL,
    },
    /* Corresponds to MPI_BOR */
    [OMPI_OP_BASE_FORTRAN_BOR] = {
        C_INTEGER(bor, 3buff),
    },
    /* Corresponds to MPI_LXOR */
    [OMPI_OP_BASE_FORTRAN_LXOR] = {
        NULL,
    },
    /* Corresponds to MPI_BXOR */
    [OMPI_OP_BASE_FORTRAN_BXOR] = {
        C_INTEGER(bxor, 3buff),
    },
    /* Corresponds to MPI_REPLACE */
    [OMPI_OP_BASE_FORTRAN_REPLACE] = {
        /* MPI_ACCUMULATE is handled differently than the other
           reductions, so just zero out its function
           implementations here to ensure that users don't invoke
           MPI_REPLACE with any reduction operations other than
           ACCUMULATE */
        NULL,
    },
};
