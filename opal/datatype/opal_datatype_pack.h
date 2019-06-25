/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2009      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2011      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2017-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (C) 2019      Arm Ltd.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED
#define OPAL_DATATYPE_PACK_H_HAS_BEEN_INCLUDED

#include "opal_config.h"

#if !defined(CHECKSUM) && OPAL_CUDA_SUPPORT
/* Make use of existing macro to do CUDA style memcpy */
#undef MEMCPY_CSUM
#define MEMCPY_CSUM( DST, SRC, BLENGTH, CONVERTOR ) \
    CONVERTOR->cbmemcpy( (DST), (SRC), (BLENGTH), (CONVERTOR) )
#endif

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

    assert( *(COUNT) <= _elem->count * _elem->blocklen);

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
                           _packed, (void*)_memory, (unsigned long)do_now_bytes, (unsigned long)(*(SPACE)) ); );
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
    assert( *(COUNT) <= _elem->count * _elem->blocklen);

    if( (blocklen_bytes * cando_count) > *(SPACE) )
        cando_count = (*SPACE) / blocklen_bytes;

    /* premptively update the number of COUNT we will return. */
    *(COUNT) -= cando_count;

    if( 1 == _elem->blocklen ) { /* Do as many full blocklen as possible */
        for(; cando_count > 0; cando_count--) {
            OPAL_DATATYPE_SAFEGUARD_POINTER( _memory, blocklen_bytes, (CONVERTOR)->pBaseBuf,
                                             (CONVERTOR)->pDesc, (CONVERTOR)->count );
            DO_DEBUG( opal_output( 0, "pack memcpy( %p, %p, %lu ) => space %lu [blen = 1]\n",
                                  (void*)_packed, (void*)_memory, (unsigned long)blocklen_bytes,
                                  (unsigned long)(*(SPACE) - (_packed - *(packed))) ); );
            MEMCPY_CSUM( _packed, _memory, blocklen_bytes, (CONVERTOR) );
            _packed     += blocklen_bytes;
            _memory     += _elem->extent;
        }
        goto update_and_return;
    }

    /**
     * Compute how many full blocklen we need to do and do them.
     */
    if( (1 < _elem->count) && (_elem->blocklen <= cando_count) ) {
        do_now_bytes = _elem->blocklen * opal_datatype_basicDatatypes[_elem->common.type]->size;

        /* each block can full fill vector, we use 4x version, or just copy once */
        if(svcntb() < do_now_bytes || do_now == 1) {
            DO_DEBUG( opal_output( 0, "block larger than VL or single copy(%d)(cando %d)(len %d)(extend %d)(do_now %d),
                        4x version",do_now,cando_count,_elem->blocklen, _elem->extent,do_now_bytes); );
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
            int blocks_in_VL = svcntb()/do_now_bytes/4;

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
            /* max VL 2048/8/4 = 64 offsets */
            uint32_t off_sets[256];
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
                for(size_t i =0; i<do_now_bytes; i++)
                {
                    /* offset in bytes offset=offset in block + extend(bytes)*j */
                    off_sets[start] = (i+j*_elem->extent);
                    start++;
                }
            }
            /* cannot totally fullfill the vector but almost full, best we can do */
            svbool_t Pg1 =svwhilelt_b8_u32(0, (int)do_now_bytes*blocks_in_VL*4);
            svuint32_t xt = svld1(Pg1, off_sets);
            /*
            uint32_t offs[256];
            svst1(Pg1,offs,xt);
            for(int i=0; i<256; i=i+1)
            {   
                DO_DEBUG( opal_output( 0, "--%d",offs[i]););
            } 
            */

            /* loop thru how many vector copy need to do
             *
             * blocks  "__":useful data; "-":no-copy
             * |__-|__-|__-|__-|__-|__-|__-|__-|__-|__-|
             * |-------VL------|-------VL------|--rem--|
             *
             */
            svbool_t Pg =svwhilelt_b8(0, (int)do_now_bytes*blocks_in_VL*4); 
            int num_of_copys = cando_count/ (_elem->blocklen*blocks_in_VL);
            for(int i=0; i < num_of_copys; i++)
            {
                DO_DEBUG( opal_output( 0, "pack full VL. memcpy( %p, %p, %lu ) => space %lu copy seq %d copy(%d)(cando %d)(len %d)(extend %d)(do_now%d)"
                            ,(void*)*(packed), (void*)_memory, (unsigned long)do_now_bytes*blocks_in_VL, (unsigned long)*(SPACE),
                            i, do_now,cando_count,_elem->blocklen, _elem->extent, do_now); );

                svuint32_t vsrc = svld1ub_gather_offset_u32(Pg, (uint8_t*)_memory, xt);
                /* need to store with 1b, because vsrc can only be 32_t */
                svst1b(Pg, *(packed), vsrc);

                /*
                for(int i=0; i<_elem->extent*blocks_in_VL; i=i+4)
                {
                     DO_DEBUG( opal_output( 0, "-- %p %d  %p %d ",*packed+i,*(*packed+i),_memory+i ,*(_memory+i)););
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
            if (blocks_in_VL != 0) {
                svbool_t Pg = svwhilelt_b8_u32(0, do_now_bytes*blocks_in_VL*4);
                svuint32_t vsrc = svld1ub_gather_offset_u32(Pg, (uint8_t*)_memory, xt);
                /* need to store with 1b, because vsrc can only be 32_t */
                svst1b(Pg, *(packed), vsrc);
                /*
                for(int i=0; i<_elem->extent*blocks_in_VL; i=i+4)
                {
                    DO_DEBUG( opal_output( 0, "ramining -- %p %d  %p %d ",*packed+i,*(*packed+i),_memory+i ,*(_memory+i)););
                } 
                */
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
