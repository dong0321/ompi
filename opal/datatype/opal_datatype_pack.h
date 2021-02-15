/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2011      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2017-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2020      Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED

#include "opal_config.h"

#include "opal_config.h"
#include <immintrin.h>

#if !defined(CHECKSUM) && OPAL_CUDA_SUPPORT
/* Make use of existing macro to do CUDA style memcpy */
#undef MEMCPY_CSUM
#define MEMCPY_CSUM( DST, SRC, BLENGTH, CONVERTOR ) \
    CONVERTOR->cbmemcpy( (DST), (SRC), (BLENGTH), (CONVERTOR) )
#endif


#define opal_datatype_pack_threshold 16

#define vector_bits 512
#define vector_bytes 512/8

/**
 * This function deals only with partial elements. The COUNT points however to the whole leftover count,
 * but this function is only expected to operate on an amount less than blength, that would allow the rest
 * of the pack process to handle only entire blength blocks (plus the left over).
 *
 * Return 1 if we are now aligned on a block, 0 otherwise.
 */
static inline int
pack_partial_blocklen( opal_convertor_t* CONVERTOR,
                       const dt_elem_desc_t* ELEM,
                       size_t* COUNT,
                       unsigned char** memory,
                       unsigned char** packed,
                       size_t* SPACE )
{
    const ddt_elem_desc_t* _elem = &((ELEM)->elem);
    size_t do_now_bytes = opal_datatype_basicDatatypes[_elem->common.type]->size;
    size_t do_now = *(COUNT);
    unsigned char* _memory = (*memory) + _elem->disp;
    unsigned char* _packed = *packed;

    assert( *(COUNT) <= ((size_t)_elem->count * _elem->blocklen) );

    /**
     * First check if we already did something on this element ? The COUNT is the number
     * of remaining predefined types in the current elem, not how many predefined types
     * should be manipulated in the current call (this number is instead reflected on the
     * SPACE).
     */
    if( 0 == (do_now = (*COUNT) % _elem->blocklen) )
        return 1;

    size_t left_in_block = do_now;  /* left in the current blocklen */

    if( (do_now_bytes * do_now) > *(SPACE) )
        do_now = (*SPACE) / do_now_bytes;

    do_now_bytes *= do_now;

    OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, do_now_bytes, (CONVERTOR)->pBaseBuf,
                                     (CONVERTOR)->pDesc, (CONVERTOR)->count );
    DO_DEBUG( opal_output( 0, "pack memcpy( %p, %p, %lu ) => space %lu [partial]\n",
                           (void*) _packed, (void*)_memory, (unsigned long)do_now_bytes, (unsigned long)(*(SPACE)) ); );
    MEMCPY_CSUM( _packed, _memory, do_now_bytes, (CONVERTOR) );
    *(memory)     += (ptrdiff_t)do_now_bytes;
    if( do_now == left_in_block )  /* compensate if completed a blocklen */
        *(memory) += _elem->extent - (_elem->blocklen * opal_datatype_basicDatatypes[_elem->common.type]->size);

    *(COUNT)  -= do_now;
    *(SPACE)  -= do_now_bytes;
    *(packed) += do_now_bytes;
    return (do_now == left_in_block);
}

/**
 * Pack entire blocks, plus a possible remainder if SPACE is constrained to less than COUNT elements.
 */
static inline void
pack_predefined_data( opal_convertor_t* CONVERTOR,
                      const dt_elem_desc_t* ELEM,
                      size_t* COUNT,
                      unsigned char** memory,
                      unsigned char** packed,
                      size_t* SPACE )
{
    const ddt_elem_desc_t* _elem = &((ELEM)->elem);
    size_t blocklen_bytes = opal_datatype_basicDatatypes[_elem->common.type]->size;
    size_t cando_count = *(COUNT), do_now_bytes;
    unsigned char* _memory = (*memory) + _elem->disp;
    unsigned char* _packed = *packed;

    assert( 0 == (cando_count % _elem->blocklen) );  /* no partials here */
    assert( *(COUNT) <= ((size_t)_elem->count * _elem->blocklen) );

    if( (blocklen_bytes * cando_count) > *(SPACE) )
        cando_count = (*SPACE) / blocklen_bytes;

    /* premptively update the number of COUNT we will return. */
    *(COUNT) -= cando_count;

    /* buffers aligned and we manipulate, on each copy, less data than a defined threshold */
    if( (0 == (( (uintptr_t)_memory ^ (uintptr_t)_packed) & (blocklen_bytes - 1))) && (_elem->blocklen <= opal_datatype_pack_threshold) ) {
        if( 1 == _elem->blocklen ) { /* Do as many full blocklen as possible */
            for(; cando_count > 0; cando_count--) {
                OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, blocklen_bytes, (CONVERTOR)->pBaseBuf,
                                                 (CONVERTOR)->pDesc, (CONVERTOR)->count );
                DO_DEBUG( opal_output( 0, "pack memcpy( %p, %p, %lu ) => space %lu [blen = 1]\n",
                                       (void*)_packed, (void*)_memory, (unsigned long)blocklen_bytes, (unsigned long)(*(SPACE) - (_packed - *(packed))) ); );
                MEMCPY_CSUM( _packed, _memory, blocklen_bytes, (CONVERTOR) );
                _packed     += blocklen_bytes;
                _memory     += _elem->extent;
            }
            goto update_and_return;
        }

        if( (1 < _elem->count) && (_elem->blocklen <= cando_count) ) {
            blocklen_bytes *= _elem->blocklen;

            if (vector_bytes < blocklen_bytes*2) {
                do { /* Do as many full blocklen as possible */
                    OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, blocklen_bytes, (CONVERTOR)->pBaseBuf,
                            (CONVERTOR)->pDesc, (CONVERTOR)->count );
                    DO_DEBUG( opal_output( 0, "pack 2. memcpy( %p, %p, %lu ) => space %lu\n",
                                (void*)_packed, (void*)_memory, (unsigned long)blocklen_bytes, (unsigned long)(*(SPACE) - (_packed - *(packed))) ); );
                    MEMCPY_CSUM( _packed, _memory, blocklen_bytes, (CONVERTOR) );
                    _packed     += blocklen_bytes;
                    _memory     += _elem->extent;
                    cando_count -= _elem->blocklen;
                } while (_elem->blocklen <= cando_count);
            }
            else {
                // avx gather pack
                int blocks_in_vl = vector_bytes/blocklen_bytes;

                if (blocks_in_vl > cando_count){
                    blocks_in_vl =  cando_count;
                }
                /* max VL 512/8/4 = 16 offsets for MPI_INT  */
                uint32_t off_sets[16];
                int start = 0;
                for(int j=0; j<blocks_in_vl; j++){
                    for(size_t i =0; i<blocklen_bytes/opal_datatype_basicDatatypes[_elem->common.type]->size; i++)
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
                __m512i xt = _mm512_load_epi32((__m512*)off_sets);
                int num_of_copys = cando_count/ (_elem->blocklen*blocks_in_vl);
                load_mask = _cvtu32_mask16((1<<_elem->blocklen*blocks_in_vl)-1);
                for(int i=0; i < num_of_copys; i++)
                {
                    DO_DEBUG( opal_output( 0, "pack full VL. memcpy( %p, %p, %lu ) => space %lu copy seq %d (cando %d)(len %d)(extend %d)"
                                ,(void*)*(packed), (void*)_memory, (unsigned long)blocklen_bytes*blocks_in_vl, (unsigned long)*(SPACE),
                                i, cando_count, _elem->blocklen, _elem->extent); );
                    __m512i vsrc = _mm512_mask_i32gather_epi32 (temp_src, load_mask, xt, (void*)_memory, 4);
                    _mm512_store_epi32 (*(packed), vsrc);
                    _packed     += blocklen_bytes*blocks_in_vl;
                    _memory     += _elem->extent*blocks_in_vl;
                    cando_count -= _elem->blocklen*blocks_in_vl;
                }
                /* remaining blocks */
                blocks_in_vl = cando_count / _elem->blocklen;
                DO_DEBUG( opal_output( 0, "Total ramining cando_count -- %d", cando_count););
                if (blocks_in_vl != 0) {
                    // Need to use mask OP for partial load & store
                    __mmask16 load_mask = _cvtu32_mask16((1<<cando_count)-1);
                    __m512i vsrc = _mm512_mask_i32gather_epi32 (temp_src, load_mask, xt,  (void*)_memory, 4);
                    _mm512_mask_store_epi32( *(packed), load_mask, vsrc);

                    for(int i=0; i<_elem->blocklen*blocks_in_vl; i++)
                    {
                    //    DO_DEBUG( opal_output( 0, "ramining No.%d -- %p %d  %p %d ", i, _packed+i*4,*(_packed+i*4),_memory+i ,*(_memory+i)););
                    }

                    _packed     += blocklen_bytes*blocks_in_vl;
                    _memory     += _elem->extent*blocks_in_vl;
                    cando_count -= _elem->blocklen*blocks_in_vl;
                }
            }
        }
        /**
         * As an epilog do anything left from the last blocklen.
         */
        if( 0 != cando_count ) {
            assert( (cando_count < _elem->blocklen) ||
                    ((1 == _elem->count) && (cando_count <= _elem->blocklen)) );
            do_now_bytes = cando_count * opal_datatype_basicDatatypes[_elem->common.type]->size;
            OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, do_now_bytes, (CONVERTOR)->pBaseBuf,
                                             (CONVERTOR)->pDesc, (CONVERTOR)->count );
            DO_DEBUG( opal_output( 0, "pack 3. memcpy( %p, %p, %lu ) => space %lu [epilog]\n",
                                   (void*)_packed, (void*)_memory, (unsigned long)do_now_bytes, (unsigned long)(*(SPACE) - (_packed - *(packed))) ); );
            MEMCPY_CSUM( _packed, _memory, do_now_bytes, (CONVERTOR) );
            _memory   += do_now_bytes;
            _packed   += do_now_bytes;
        }
        goto update_and_return;
    }
    if( 1 == _elem->blocklen ) { /* Do as many full blocklen as possible */
        for(; cando_count > 0; cando_count--) {
            OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, blocklen_bytes, (CONVERTOR)->pBaseBuf,
                                             (CONVERTOR)->pDesc, (CONVERTOR)->count );
            DO_DEBUG( opal_output( 0, "pack memcpy( %p, %p, %lu ) => space %lu [blen = 1]\n",
                                   (void*)_packed, (void*)_memory, (unsigned long)blocklen_bytes, (unsigned long)(*(SPACE) - (_packed - *(packed))) ); );
            MEMCPY_CSUM( _packed, _memory, blocklen_bytes, (CONVERTOR) );
            _packed     += blocklen_bytes;
            _memory     += _elem->extent;
        }
        goto update_and_return;
    }

    if( (1 < _elem->count) && (_elem->blocklen <= cando_count) ) {
        blocklen_bytes *= _elem->blocklen;

        do { /* Do as many full blocklen as possible */
            OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, blocklen_bytes, (CONVERTOR)->pBaseBuf,
                                             (CONVERTOR)->pDesc, (CONVERTOR)->count );
            DO_DEBUG( opal_output( 0, "pack 2. memcpy( %p, %p, %lu ) => space %lu\n",
                                   (void*)_packed, (void*)_memory, (unsigned long)blocklen_bytes, (unsigned long)(*(SPACE) - (_packed - *(packed))) ); );
            MEMCPY_CSUM( _packed, _memory, blocklen_bytes, (CONVERTOR) );
            _packed     += blocklen_bytes;
            _memory     += _elem->extent;
            cando_count -= _elem->blocklen;
        } while (_elem->blocklen <= cando_count);
    }

    /**
     * As an epilog do anything left from the last blocklen.
     */
    if( 0 != cando_count ) {
        assert( (cando_count < _elem->blocklen) ||
                ((1 == _elem->count) && (cando_count <= _elem->blocklen)) );
        do_now_bytes = cando_count * opal_datatype_basicDatatypes[_elem->common.type]->size;
        OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, do_now_bytes, (CONVERTOR)->pBaseBuf,
                                         (CONVERTOR)->pDesc, (CONVERTOR)->count );
        DO_DEBUG( opal_output( 0, "pack 3. memcpy( %p, %p, %lu ) => space %lu [epilog]\n",
                               (void*)_packed, (void*)_memory, (unsigned long)do_now_bytes, (unsigned long)(*(SPACE) - (_packed - *(packed))) ); );
        MEMCPY_CSUM( _packed, _memory, do_now_bytes, (CONVERTOR) );
        _memory   += do_now_bytes;
        _packed   += do_now_bytes;
    }

 update_and_return:
    *(memory)  = _memory - _elem->disp;
    *(SPACE)  -= (_packed - *packed);
    *(packed)  = _packed;
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

#define PACK_PARTIAL_BLOCKLEN( CONVERTOR,    /* the convertor */                       \
                               ELEM,         /* the basic element to be packed */ \
                               COUNT,        /* the number of elements */ \
                               MEMORY,       /* the source pointer (char*) */ \
                               PACKED,       /* the destination pointer (char*) */ \
                               SPACE )       /* the space in the destination buffer */ \
pack_partial_blocklen( (CONVERTOR), (ELEM), &(COUNT), &(MEMORY), &(PACKED), &(SPACE) )

#define PACK_PREDEFINED_DATATYPE( CONVERTOR,    /* the convertor */     \
                                  ELEM,         /* the basic element to be packed */      \
                                  COUNT,        /* the number of elements */              \
                                  MEMORY,       /* the source pointer (char*) */          \
                                  PACKED,       /* the destination pointer (char*) */     \
                                  SPACE )       /* the space in the destination buffer */ \
pack_predefined_data( (CONVERTOR), (ELEM), &(COUNT), &(MEMORY), &(PACKED), &(SPACE) )

#define PACK_CONTIGUOUS_LOOP( CONVERTOR, ELEM, COUNT, MEMORY, PACKED, SPACE ) \
    pack_contiguous_loop( (CONVERTOR), (ELEM), &(COUNT), &(MEMORY), &(PACKED), &(SPACE) )

#endif  /* OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED */
