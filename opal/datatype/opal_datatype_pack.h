/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2011      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2017-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED

#include "opal_config.h"
#include <immintrin.h>

#define vector_length 512/8

#if !defined(CHECKSUM) && OPAL_CUDA_SUPPORT
/* Make use of existing macro to do CUDA style memcpy */
#undef MEMCPY_CSUM
#define MEMCPY_CSUM( DST, SRC, BLENGTH, CONVERTOR ) \
    CONVERTOR->cbmemcpy( (DST), (SRC), (BLENGTH), (CONVERTOR) )
#endif

static inline void
pack_predefined_data( opal_convertor_t* CONVERTOR,
                      const dt_elem_desc_t* ELEM,
                      size_t* COUNT,
                      unsigned char** memory,
                      unsigned char** packed,
                      size_t* SPACE )
{
    const ddt_elem_desc_t* _elem = &((ELEM)->elem);
    size_t total_count = _elem->count * _elem->blocklen;
    size_t cando_count = (*SPACE) / opal_datatype_basicDatatypes[_elem->common.type]->size;
    size_t do_now, do_now_bytes;
    unsigned char* _memory = (*memory) + _elem->disp;

    assert( *(COUNT) <= total_count); //_elem->count * _elem->blocklen);

    if( cando_count > *(COUNT) )
        cando_count = *(COUNT);

    /**
     * First check if we already did something on this element ?
     */
    do_now = (total_count - *(COUNT));  /* done elements */
    DO_DEBUG( opal_output( 0, "Begin pack do_now:--%d  total_count:--%d *(COUNT):--%d ",  do_now, total_count, *(COUNT)););
    if( 0 != do_now ) {
        do_now = do_now % _elem->blocklen;  /* partial blocklen? */

        if( 0 != do_now ) {
            size_t left_in_block = _elem->blocklen - do_now;  /* left in the current blocklen */
            do_now = (left_in_block > cando_count ) ? cando_count : left_in_block;
            do_now_bytes = do_now * opal_datatype_basicDatatypes[_elem->common.type]->size;

            OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, do_now_bytes, (CONVERTOR)->pBaseBuf,
                                            (CONVERTOR)->pDesc, (CONVERTOR)->count );
            DO_DEBUG( opal_output( 0, "pack 1. memcpy( %p, %p, %lu ) => space %lu [prolog]\n",
                                   (void*)*(packed), (void*)_memory, (unsigned long)do_now_bytes, (unsigned long)(*(SPACE)) ); );
            MEMCPY_CSUM( *(packed), _memory, do_now_bytes, (CONVERTOR) );
            _memory      = (*memory) + _elem->disp + (ptrdiff_t)do_now_bytes;
            /* compensate if we just completed a blocklen */
            if( do_now == left_in_block )
                _memory += _elem->extent - (_elem->blocklen * opal_datatype_basicDatatypes[_elem->common.type]->size);
            *(packed)   += do_now_bytes;
            *(SPACE)    -= do_now_bytes;
            *(COUNT)    -= do_now;
            cando_count -= do_now;
        }
    }

    /**
     * Compute how many full blocklen we need to do and do them.
     */
    do_now = cando_count / _elem->blocklen;
    if( 0 != do_now ) {
        do_now_bytes = _elem->blocklen * opal_datatype_basicDatatypes[_elem->common.type]->size;

        /* each block can full fill vector, we use 4x version, or just copy once */
        if(vector_length < do_now_bytes || do_now == 1) {
            DO_DEBUG( opal_output( 0, "block larger than VL or single copy(%d)(cando %d)(len %d)(extend %d)(do_now %d)",
                        do_now,cando_count,_elem->blocklen, _elem->extent,do_now_bytes); );
            for(size_t _i = 0; _i < do_now; _i++ ) {
                OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, do_now_bytes, (CONVERTOR)->pBaseBuf,
                        (CONVERTOR)->pDesc, (CONVERTOR)->count );
                DO_DEBUG( opal_output( 0, "pack 2. memcpy( %p, %p, %lu ) => space %lu\n",
                            (void*)*(packed), (void*)_memory, (unsigned long)do_now_bytes, (unsigned long)*(SPACE) ); );
                MEMCPY_CSUM( *(packed), _memory, do_now_bytes, (CONVERTOR) );
                *(packed)   += do_now_bytes;
                _memory     += _elem->extent;
                *(SPACE)    -= do_now_bytes;
                *(COUNT)    -= _elem->blocklen;
                cando_count -= _elem->blocklen;
            }
        }
        /* each vector can deal with multi blocks */
        else {
            DO_DEBUG( opal_output( 0, "block in bytes is smaller than VL in bytes, use gather_load version"); );
            /* how many full blocks can be processed in each vector */
            int blocks_in_VL = vector_length/do_now_bytes;

            /*  cannot fullfill a whole vector to copy
             *
             *  |__-|__-|
             *
             *  |__|__|__|__|
             *
             */
            if(blocks_in_VL>do_now) {
                blocks_in_VL = do_now;
            }

            DO_DEBUG( opal_output( 0, "blength %d extend %d block bytes %d blocks %d do_now %d",
                        _elem->blocklen, _elem->extent, do_now_bytes, blocks_in_VL, do_now); );
            /* max VL 512/8/4 = 16 offsets for MPI_INT  */
            uint32_t off_sets[16];
            int start = 0;

            /*  get offsets for block items
             *
             *  blocks  "__":useful data; "-":no-copy
             *  |__-|__-|__-|__-|
             *  |   / /
             *  |  / /
             *  |__|__|__|__|
             *
             *  get offsets for block items
             */
            for(int j=0; j<blocks_in_VL; j++){
                for(size_t i =0; i<do_now_bytes/opal_datatype_basicDatatypes[_elem->common.type]->size; i++)
                {
                    /* offset in bytes offset=offset in block + extend(bytes)*j */
                    off_sets[start] = (i+j*_elem->extent/opal_datatype_basicDatatypes[_elem->common.type]->size);
                    DO_DEBUG( opal_output( 0, "off_sets --%d",off_sets[start]););
                    start++;
                }
            }
            /* cannot totally fullfill the vector but almost full, best we can do */
            __mmask16 load_mask;
            __m512i temp_src;
            __m512i xt = _mm512_load_epi32((__m512*)off_sets);// _mm512_loadu_si512((__m512*)off_sets);
/*
            uint32_t offs[256];
            for(int i=0; i<256; i=i+1)
            {   
                offs[i]=i+1;
            //    DO_DEBUG( opal_output( 0, "--%d",offs[i]););
            } 
*/
            /* loop thru how many vector copy need to do
             *
             * blocks  "__":useful data; "-":no-copy
             * |__-|__-|__-|__-|__-|__-|__-|__-|__-|__-|
             * |-------VL------|-------VL------|--rem--|
             *
             */

            int num_of_copys = cando_count/ (_elem->blocklen*blocks_in_VL);
            load_mask = _cvtu32_mask16((1<<_elem->blocklen*blocks_in_VL)-1);
            for(int i=0; i < num_of_copys; i++)
            {
                DO_DEBUG( opal_output( 0, "pack full VL. memcpy( %p, %p, %lu ) => space %lu copy seq %d copy(%d)(cando %d)(len %d)(extend %d)(do_now%d)"
                            ,(void*)*(packed), (void*)_memory, (unsigned long)do_now_bytes*blocks_in_VL, (unsigned long)*(SPACE),
                            i, do_now,cando_count,_elem->blocklen, _elem->extent, do_now); );
                //load_mask = _cvtu32_mask16((1<<_elem->blocklen*blocks_in_VL)-1);
                //__m512i vsrc = _mm512_mask_i32gather_epi32 (temp_src, load_mask, xt,  (void*)offs, 4); //_memory
                __m512i vsrc = _mm512_mask_i32gather_epi32 (temp_src, load_mask, xt, (void*)_memory, 4); //_memory
                //_mm512_mask_store_epi32( *(packed), load_mask, vsrc);
                _mm512_store_epi32 (*(packed), vsrc);
                /*
                for(int i=0; i<_elem->blocklen*blocks_in_VL; i++)
                {
                    DO_DEBUG( opal_output( 0, "No.%d -- %p %d  %p %d ", i, *packed+i*4,*(*packed+i*4),_memory+i ,*(_memory+i)););
                }
                */
                *(packed)   += do_now_bytes*blocks_in_VL;
                _memory     += _elem->extent*blocks_in_VL;
                *(SPACE)    -= do_now_bytes*blocks_in_VL;
                *(COUNT)    -= _elem->blocklen*blocks_in_VL;
                cando_count -= _elem->blocklen*blocks_in_VL;
            }
            /* remaining blocks */
            blocks_in_VL = cando_count / _elem->blocklen;
            DO_DEBUG( opal_output( 0, "Total ramining cando_count -- %d", cando_count););
            if (blocks_in_VL != 0) {
                // Need to use mask OP for partial load & store
                __mmask16 load_mask = _cvtu32_mask16((1<<cando_count)-1);
                //__m512i vsrc = _mm512_mask_i32gather_epi32 (temp_src, load_mask, xt,  (void*)offs, 4);// (uint8_t*)_memory, 1);
                __m512i vsrc = _mm512_mask_i32gather_epi32 (temp_src, load_mask, xt,  (void*)_memory, 4);
                _mm512_mask_store_epi32( *(packed), load_mask, vsrc);

                
                for(int i=0; i<_elem->blocklen*blocks_in_VL; i++)
                {
                    DO_DEBUG( opal_output( 0, "ramining No.%d -- %p %d  %p %d ", i, *packed+i*4,*(*packed+i*4),_memory+i ,*(_memory+i)););
                }
                

                *(packed)   += do_now_bytes*blocks_in_VL;
                _memory     += _elem->extent*blocks_in_VL;
                *(SPACE)    -= do_now_bytes*blocks_in_VL;
                *(COUNT)    -= _elem->blocklen*blocks_in_VL;
                cando_count -= _elem->blocklen*blocks_in_VL;
            }
        }
    }
    /**
     * As an epilog do anything left from the last blocklen.
     */
    do_now = cando_count;
    if( 0 != do_now ) {
        do_now_bytes = do_now * opal_datatype_basicDatatypes[_elem->common.type]->size;
        OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, do_now_bytes, (CONVERTOR)->pBaseBuf,
                                        (CONVERTOR)->pDesc, (CONVERTOR)->count );
        DO_DEBUG( opal_output( 0, "pack 3. memcpy( %p, %p, %lu ) => space %lu [epilog]\n",
                               (void*)*(packed), (void*)_memory, (unsigned long)do_now_bytes, (unsigned long)(*(SPACE)) ); );
        MEMCPY_CSUM( *(packed), _memory, do_now_bytes, (CONVERTOR) );
        _memory   += do_now_bytes;
        *(packed) += do_now_bytes;
        *(SPACE)  -= do_now_bytes;
        *(COUNT)  -= do_now;
    }

    *(memory)  = _memory - _elem->disp;
}

static inline void pack_contiguous_loop( opal_convertor_t* CONVERTOR,
                                         const dt_elem_desc_t* ELEM,
                                         size_t* COUNT,
                                         unsigned char** memory,
                                         unsigned char** packed,
                                         size_t* SPACE )
{
    const ddt_loop_desc_t *_loop = (ddt_loop_desc_t*)(ELEM);
    const ddt_endloop_desc_t* _end_loop = (ddt_endloop_desc_t*)((ELEM) + _loop->items);
    unsigned char* _memory = (*memory) + _end_loop->first_elem_disp;
    size_t _copy_loops = *(COUNT);

    if( (_copy_loops * _end_loop->size) > *(SPACE) )
        _copy_loops = (*(SPACE) / _end_loop->size);
    for(size_t _i = 0; _i < _copy_loops; _i++ ) {
        OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, _end_loop->size, (CONVERTOR)->pBaseBuf,
                                    (CONVERTOR)->pDesc, (CONVERTOR)->count );
        DO_DEBUG( opal_output( 0, "pack 3. memcpy( %p, %p, %lu ) => space %lu\n",
                               (void*)*(packed), (void*)_memory, (unsigned long)_end_loop->size, (unsigned long)(*(SPACE) - _i * _end_loop->size) ); );
        MEMCPY_CSUM( *(packed), _memory, _end_loop->size, (CONVERTOR) );
        *(packed) += _end_loop->size;
        _memory   += _loop->extent;
    }
    *(memory) = _memory - _end_loop->first_elem_disp;
    *(SPACE) -= _copy_loops * _end_loop->size;
    *(COUNT) -= _copy_loops;
}

#define PACK_PREDEFINED_DATATYPE( CONVERTOR,    /* the convertor */                       \
                                  ELEM,         /* the basic element to be packed */      \
                                  COUNT,        /* the number of elements */              \
                                  MEMORY,       /* the source pointer (char*) */          \
                                  PACKED,       /* the destination pointer (char*) */     \
                                  SPACE )       /* the space in the destination buffer */ \
pack_predefined_data( (CONVERTOR), (ELEM), &(COUNT), &(MEMORY), &(PACKED), &(SPACE) )

#define PACK_CONTIGUOUS_LOOP( CONVERTOR, ELEM, COUNT, MEMORY, PACKED, SPACE ) \
    pack_contiguous_loop( (CONVERTOR), (ELEM), &(COUNT), &(MEMORY), &(PACKED), &(SPACE) )

#endif  /* OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED */
