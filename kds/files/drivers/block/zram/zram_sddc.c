// SPDX-License-Identifier: GPL-2.0
/*
 * SDDC: slot deduplication with dictionary (delta) compression for zram.
 *
 * Slot kinds on top of the ordinary compressed slot:
 *
 *   REF    a slot whose compressed stream stays addressable so later pages can
 *          be compressed against it.  It owns its zsmalloc handle.
 *   ALIAS  a slot whose compressed stream is byte identical to a REF; it stores
 *          no data at all and just points at the REF.
 *   DELTA  a slot stored as a dictionary coded stream relative to a REF.
 *
 * Lifetime: a REF is pinned by its dependents (meta->dependents).  When the
 * slot index of a REF is freed or overwritten the metadata survives as an
 * orphan and keeps the zsmalloc handle alive until the last dependent goes
 * away, so a dependent can never read a recycled reference.  The reference
 * graph is one level deep: only REF slots are referenced and REF slots have
 * no reference of their own.
 *
 * Locking: ->bucket_lock protects the candidate buckets and the
 * orphan/index transitions only; every pointer handed out is pinned with
 * ->dependents while the lock is held.  zsmalloc mapping and codec calls
 * always run outside that lock, so no sleeping operation happens under a
 * spinlock.
 *
 * All of this is a pure cache: any failure falls back to an ordinary
 * compressed slot.
 */

#include <linux/highmem.h>
#include <linux/jhash.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/vmalloc.h>
#include <linux/zsmalloc.h>

#include "zcomp.h"
#include "zram_sddc.h"

#define SDDC_MAGIC		0x43444453	/* "SDDC" */
#define SDDC_VERSION		1
#define SDDC_HDR_BYTES		24
#define SDDC_INDEX_MIN_BYTES	256
#define SDDC_MAX_CANDIDATES	4
#define SDDC_BUCKET_BITS	14
#define SDDC_BUCKETS		(1U << SDDC_BUCKET_BITS)
#define SDDC_BUCKET_WAYS	6
#define SDDC_HASH_SEED		0x9e3779b9

/* LZ4KD needs >= 5 bytes of delta input and the wire must fit one page. */
#define SDDC_DELTA_LIMIT	(PAGE_SIZE - SDDC_HDR_BYTES)
#define SDDC_DELTA_MIN_BYTES	8

enum sddc_kind {
	SDDC_KIND_REF = 1,
	SDDC_KIND_ALIAS,
	SDDC_KIND_DELTA,
};

struct sddc_wire_hdr {
	u32 magic;
	u16 version;
	u16 kind;
	u32 ref_hash;
	u32 target_len;
	u32 payload_len;
	u32 payload_hash;
	u32 reserved;
} __packed;

struct sddc_meta {
	struct sddc_meta *ref;	/* reference slot, NULL for REF */
	unsigned long handle;	/* own stream (REF and DELTA only) */
	int dependents;
	int orphan;
	int kind;
	u32 stream_hash;	/* hash of own stream; for DELTA the ref's hash */
	u32 sample_hash;	/* head/tail page sample hash, bucket key */
	u16 wire_len;		/* bytes stored in zsmalloc */
	u16 target_len;
};

struct sddc_bucket {
	struct sddc_meta *slot[SDDC_BUCKET_WAYS];
};

struct zram_sddc {
	struct zs_pool *pool;
	struct zcomp *comp;
	unsigned long huge_class_size;

	struct sddc_bucket *buckets;
	spinlock_t bucket_lock;

	/* encode scratch, serialised by ->enc_lock */
	void *enc_pair;		/* 2 contiguous pages: ref | target */
	void *enc_out;		/* 1 page: stream / delta wire */
	void *enc_tmp;		/* 1 page: candidate stream copy */
	void *proof_pair;	/* 2 contiguous pages: round-trip proof */
	struct mutex enc_lock;

	/* decode scratch, serialised by ->dec_lock */
	void *dec_pair;		/* 2 contiguous pages: ref | target */
	void *dec_hdr;		/* 1 page: delta wire */
	void *dec_tmp;		/* 1 page: candidate stream copy */
	spinlock_t dec_lock;

	atomic64_t alias_hits;
	atomic64_t delta_hits;
	atomic64_t fallbacks;
	atomic64_t read_errors;
	atomic64_t proof_failures;
};

static u32 sddc_hash_bytes(const void *buf, unsigned int len)
{
	return jhash(buf, len, SDDC_HASH_SEED);
}

/*
 * Sample hash over the first and the last 16 bytes of the page.  Similar
 * pages share it, which is what the delta candidate search needs.
 */
static u32 sddc_sample_hash(const u8 *page)
{
	u32 head = jhash(page, 16, SDDC_HASH_SEED);
	u32 tail = jhash(page + PAGE_SIZE - 16, 16, SDDC_HASH_SEED);

	return head ^ rol32(tail, 13);
}

static struct sddc_bucket *sddc_bucket(struct zram_sddc *sddc, u32 hash)
{
	return &sddc->buckets[hash >> (32 - SDDC_BUCKET_BITS)];
}

static int sddc_zs_write(struct zram_sddc *sddc, const void *buf,
			 unsigned int len, unsigned long *handle)
{
	void *dst;

	*handle = zs_malloc(sddc->pool, len,
			    GFP_NOIO | __GFP_HIGHMEM | __GFP_MOVABLE);
	if (IS_ERR((void *)*handle))
		return -ENOMEM;

	dst = zs_map_object(sddc->pool, *handle, ZS_MM_WO);
	memcpy(dst, buf, len);
	zs_unmap_object(sddc->pool, *handle);
	return 0;
}

static void sddc_read_stream(struct zram_sddc *sddc, unsigned long handle,
			     unsigned int len, void *buf)
{
	void *src;

	if (!handle)
		return;
	src = zs_map_object(sddc->pool, handle, ZS_MM_RO);
	memcpy(buf, src, len);
	zs_unmap_object(sddc->pool, handle);
}

/* Drop a pin; the last pin of an orphaned metadata frees it. */
static void sddc_put(struct zram_sddc *sddc, struct sddc_meta *meta)
{
	bool last;

	if (!meta)
		return;

	spin_lock(&sddc->bucket_lock);
	last = (--meta->dependents == 0) && meta->orphan;
	spin_unlock(&sddc->bucket_lock);

	if (last) {
		if (meta->handle)
			zs_free(sddc->pool, meta->handle);
		kfree(meta);
	}
}

/* Pin a candidate while the bucket lock is held. */
static struct sddc_meta *sddc_get_locked(struct zram_sddc *sddc,
					 struct sddc_meta *meta)
{
	if (!meta || meta->kind != SDDC_KIND_REF || meta->orphan)
		return NULL;
	meta->dependents++;
	return meta;
}

static void sddc_unindex(struct zram_sddc *sddc, struct sddc_meta *meta)
{
	struct sddc_bucket *b = sddc_bucket(sddc, meta->sample_hash);
	int i;

	for (i = 0; i < SDDC_BUCKET_WAYS; i++) {
		if (b->slot[i] == meta) {
			b->slot[i] = NULL;
			return;
		}
	}
}

static void sddc_index(struct zram_sddc *sddc, struct sddc_meta *meta)
{
	struct sddc_bucket *b = sddc_bucket(sddc, meta->sample_hash);
	int i, victim;

	for (i = 0; i < SDDC_BUCKET_WAYS; i++) {
		if (b->slot[i] == meta)
			return;
	}
	for (i = 0; i < SDDC_BUCKET_WAYS; i++) {
		if (!b->slot[i]) {
			b->slot[i] = meta;
			return;
		}
	}
	/* Bucket full: replace a random-ish entry, the old REF stays valid. */
	victim = sddc_hash_bytes(meta, sizeof(meta)) % SDDC_BUCKET_WAYS;
	b->slot[victim] = meta;
}

/*
 * Store @page.  Returns 0 with @out filled in; -ENOMEM when the allocation
 * itself failed; -EAGAIN when the caller must use the compression it already
 * performed elsewhere.
 */
static int sddc_store(struct zram_sddc *sddc, struct page *page,
		      struct zram_sddc_store *out);

int zram_sddc_store(struct zram_sddc *sddc, struct page *page,
		    struct zram_sddc_store *out)
{
	int ret;

	mutex_lock(&sddc->enc_lock);
	ret = sddc_store(sddc, page, out);
	mutex_unlock(&sddc->enc_lock);
	return ret;
}

static int sddc_store(struct zram_sddc *sddc, struct page *page,
		      struct zram_sddc_store *out)
{
	struct zcomp_strm *zstrm;
	struct sddc_meta *cand[SDDC_MAX_CANDIDATES];
	struct sddc_meta *meta = NULL;
	struct sddc_bucket *bucket;
	const u8 *pagebuf;
	unsigned long handle = 0;
	unsigned int slen = 0, dlen = 0, wire_len;
	u32 sample, shash;
	int i, ncand = 0, ret = 0, sel = -1;

	pagebuf = kmap_atomic(page);
	memcpy(sddc->enc_pair + PAGE_SIZE, pagebuf, PAGE_SIZE);
	sample = sddc_sample_hash(pagebuf);
	kunmap_atomic((void *)pagebuf);

	/* 1. plain compression of the page */
	zstrm = zcomp_stream_get(sddc->comp);
	ret = zcomp_compress(zstrm, sddc->enc_pair + PAGE_SIZE, &slen);
	if (ret || slen == 0 || slen >= PAGE_SIZE ||
	    slen >= sddc->huge_class_size) {
		zcomp_stream_put(sddc->comp);
		/* Let the caller store the raw page, like the stock path. */
		return -EAGAIN;
	}
	memcpy(sddc->enc_out, zstrm->buffer, slen);
	zcomp_stream_put(sddc->comp);

	shash = sddc_hash_bytes(sddc->enc_out, slen);

	/* 2. collect candidates, pinning them while the bucket lock is held */
	bucket = sddc_bucket(sddc, sample);
	spin_lock(&sddc->bucket_lock);
	for (i = 0; i < SDDC_BUCKET_WAYS && ncand < SDDC_MAX_CANDIDATES; i++) {
		struct sddc_meta *c = sddc_get_locked(sddc, bucket->slot[i]);

		if (c)
			cand[ncand++] = c;
	}
	spin_unlock(&sddc->bucket_lock);

	/* 3. exact stream match becomes an ALIAS */
	for (i = 0; i < ncand; i++) {
		if (cand[i]->wire_len != slen || cand[i]->stream_hash != shash)
			continue;
		sddc_read_stream(sddc, cand[i]->handle, slen, sddc->enc_tmp);
		if (memcmp(sddc->enc_tmp, sddc->enc_out, slen))
			continue;
		meta = kzalloc(sizeof(*meta), GFP_NOIO);
		if (!meta) {
			ret = -ENOMEM;
			goto drop;
		}
		meta->ref = cand[i];
		meta->kind = SDDC_KIND_ALIAS;
		meta->wire_len = 0;
		meta->target_len = PAGE_SIZE;
		meta->stream_hash = shash;
		cand[i] = NULL;	/* the new slot takes over the pin */
		atomic64_inc(&sddc->alias_hits);
		out->meta = meta;
		out->handle = 0;
		out->size = 0;
		goto drop;
	}

	/* 4. dictionary compression against the first useful candidate */
	if (slen > PAGE_SIZE / 8 && zcomp_has_delta(sddc->comp)) {
		for (i = 0; i < ncand; i++) {
			struct sddc_meta *c = cand[i];
			void *payload = sddc->enc_out + SDDC_HDR_BYTES;
			bool proof_bad = false;

			if (!c)
				continue;

			sddc_read_stream(sddc, c->handle, c->wire_len,
					 sddc->enc_tmp);
			zstrm = zcomp_stream_get(sddc->comp);
			ret = zcomp_decompress(zstrm, sddc->enc_tmp,
					       c->wire_len, sddc->enc_pair);
			if (!ret)
				ret = zcomp_compress_delta(zstrm,
						sddc->enc_pair,
						sddc->enc_pair + PAGE_SIZE,
						&dlen, SDDC_DELTA_LIMIT);
			if (!ret && (dlen < SDDC_DELTA_MIN_BYTES ||
				     dlen > SDDC_DELTA_LIMIT))
				ret = -E2BIG;
			if (!ret) {
				memcpy(payload, zstrm->buffer, dlen);
				/*
				 * Round-trip proof, exactly like the reference
				 * implementation: never commit a delta that the
				 * codec cannot restore to the exact page.
				 */
				memcpy(sddc->proof_pair, sddc->enc_pair,
				       PAGE_SIZE);
				memcpy(sddc->proof_pair + PAGE_SIZE,
				       sddc->enc_pair + PAGE_SIZE, PAGE_SIZE);
				if (zcomp_decompress_delta(zstrm, payload, dlen,
						sddc->proof_pair,
						sddc->proof_pair + PAGE_SIZE) ||
				    memcmp(sddc->proof_pair + PAGE_SIZE,
					   sddc->enc_pair + PAGE_SIZE,
					   PAGE_SIZE))
					proof_bad = true;
			}
			zcomp_stream_put(sddc->comp);
			if (proof_bad) {
				atomic64_inc(&sddc->proof_failures);
				pr_info_ratelimited("lz4kds: delta proof failed (%u bytes)\n",
						    dlen);
				continue;
			}
			if (ret)
				continue;

			wire_len = SDDC_HDR_BYTES + dlen;
			if (wire_len + 8 >= slen)
				continue;
			if (zs_lookup_class_index(wire_len) >=
			    zs_lookup_class_index(slen))
				continue;

			sel = i;
			break;
		}

		if (sel >= 0) {
			struct sddc_wire_hdr *hdr = sddc->enc_out;
			struct sddc_meta *c = cand[sel];

			memset(hdr, 0, SDDC_HDR_BYTES);
			hdr->magic = SDDC_MAGIC;
			hdr->version = SDDC_VERSION;
			hdr->kind = SDDC_KIND_DELTA;
			hdr->ref_hash = c->stream_hash;
			hdr->target_len = PAGE_SIZE;
			hdr->payload_len = dlen;
			hdr->payload_hash = sddc_hash_bytes(sddc->enc_out +
					SDDC_HDR_BYTES, dlen);

			wire_len = SDDC_HDR_BYTES + dlen;
			if (sddc_zs_write(sddc, sddc->enc_out, wire_len,
					  &handle))
				goto plain;

			meta = kzalloc(sizeof(*meta), GFP_NOIO);
			if (!meta) {
				zs_free(sddc->pool, handle);
				goto plain;
			}
			meta->ref = c;
			meta->handle = handle;
			meta->kind = SDDC_KIND_DELTA;
			meta->wire_len = wire_len;
			meta->target_len = PAGE_SIZE;
			meta->stream_hash = c->stream_hash;
			cand[sel] = NULL;	/* pin transferred to the new meta */
			atomic64_inc(&sddc->delta_hits);
			out->meta = meta;
			out->handle = handle;
			out->size = wire_len;
			goto drop;
		}
	}
	(void)shash;

plain:
	if (sddc_zs_write(sddc, sddc->enc_out, slen, &handle)) {
		ret = -ENOMEM;
		goto drop;
	}

	/* 5. promote the new slot to a reference when it is worth indexing */
	if (slen >= SDDC_INDEX_MIN_BYTES && slen < sddc->huge_class_size) {
		meta = kzalloc(sizeof(*meta), GFP_NOIO);
		if (meta) {
			meta->kind = SDDC_KIND_REF;
			meta->handle = handle;
			meta->wire_len = slen;
			meta->target_len = PAGE_SIZE;
			meta->stream_hash = shash;
			meta->sample_hash = sample;
			spin_lock(&sddc->bucket_lock);
			sddc_index(sddc, meta);
			spin_unlock(&sddc->bucket_lock);
			out->meta = meta;
			out->handle = handle;
			out->size = slen;
			goto drop;
		}
		atomic64_inc(&sddc->fallbacks);
	}

	out->meta = NULL;
	out->handle = handle;
	out->size = slen;
	ret = 0;

drop:
	for (i = 0; i < ncand; i++)
		if (cand[i])
			sddc_put(sddc, cand[i]);
	return ret;
}

/* Rebuild a page out of an SDDC slot.  Runs under the zram slot spinlock. */
int zram_sddc_load(struct zram_sddc *sddc, void *meta_ptr, struct page *page)
{
	struct sddc_meta *meta = meta_ptr;
	struct sddc_meta *ref;
	struct zcomp_strm *zstrm;
	struct sddc_wire_hdr *hdr;
	void *pagebuf;
	int ret = -EIO;

	if (!meta)
		return -EIO;

	spin_lock(&sddc->dec_lock);

	if (meta->kind == SDDC_KIND_REF || meta->kind == SDDC_KIND_ALIAS) {
		ref = (meta->kind == SDDC_KIND_REF) ? meta : meta->ref;
		if (!ref)
			goto out;

		sddc_read_stream(sddc, ref->handle, ref->wire_len,
				 sddc->dec_tmp);
		pagebuf = kmap_atomic(page);
		zstrm = zcomp_stream_get(sddc->comp);
		ret = zcomp_decompress(zstrm, sddc->dec_tmp, ref->wire_len,
				       pagebuf);
		zcomp_stream_put(sddc->comp);
		kunmap_atomic(pagebuf);
		if (ret)
			atomic64_inc(&sddc->read_errors);
		goto out;
	}

	if (meta->kind != SDDC_KIND_DELTA)
		goto out;

	ref = meta->ref;
	if (!ref)
		goto out;

	sddc_read_stream(sddc, meta->handle, meta->wire_len, sddc->dec_hdr);
	hdr = sddc->dec_hdr;
	if (hdr->magic != SDDC_MAGIC || hdr->version != SDDC_VERSION ||
	    hdr->kind != SDDC_KIND_DELTA ||
	    hdr->payload_len + SDDC_HDR_BYTES != meta->wire_len ||
	    hdr->ref_hash != ref->stream_hash ||
	    hdr->payload_hash != sddc_hash_bytes(sddc->dec_hdr + SDDC_HDR_BYTES,
						 hdr->payload_len)) {
		atomic64_inc(&sddc->read_errors);
		goto out;
	}

	sddc_read_stream(sddc, ref->handle, ref->wire_len, sddc->dec_tmp);
	zstrm = zcomp_stream_get(sddc->comp);
	ret = zcomp_decompress(zstrm, sddc->dec_tmp, ref->wire_len,
			       sddc->dec_pair);
	if (!ret) {
		void *payload = sddc->dec_hdr + SDDC_HDR_BYTES;

		ret = zcomp_decompress_delta(zstrm, payload, hdr->payload_len,
					     sddc->dec_pair,
					     sddc->dec_pair + PAGE_SIZE);
		if (!ret) {
			pagebuf = kmap_atomic(page);
			memcpy(pagebuf, sddc->dec_pair + PAGE_SIZE, PAGE_SIZE);
			kunmap_atomic(pagebuf);
		} else {
			atomic64_inc(&sddc->read_errors);
		}
	} else {
		atomic64_inc(&sddc->read_errors);
	}
	zcomp_stream_put(sddc->comp);

out:
	spin_unlock(&sddc->dec_lock);
	return ret;
}

void zram_sddc_free_slot(struct zram_sddc *sddc, void *meta_ptr)
{
	struct sddc_meta *meta = meta_ptr;
	struct sddc_meta *ref = meta->ref;
	int deps;

	spin_lock(&sddc->bucket_lock);
	if (meta->kind == SDDC_KIND_REF)
		sddc_unindex(sddc, meta);
	meta->orphan = 1;
	deps = meta->dependents;
	spin_unlock(&sddc->bucket_lock);

	if (ref)
		sddc_put(sddc, ref);

	if (deps == 0)
		meta->dependents = 1;	/* route through the common release */
	if (meta->kind == SDDC_KIND_ALIAS || deps == 0)
		sddc_put(sddc, meta);
}

struct zram_sddc *zram_sddc_create(struct zs_pool *pool, struct zcomp *comp,
				   unsigned long huge_class_size)
{
	struct zram_sddc *sddc;

	sddc = kzalloc(sizeof(*sddc), GFP_KERNEL);
	if (!sddc)
		return NULL;

	sddc->pool = pool;
	sddc->comp = comp;
	sddc->huge_class_size = huge_class_size;
	spin_lock_init(&sddc->bucket_lock);
	mutex_init(&sddc->enc_lock);
	spin_lock_init(&sddc->dec_lock);

	sddc->buckets = vzalloc(array_size(SDDC_BUCKETS,
					   sizeof(struct sddc_bucket)));
	sddc->enc_pair = (void *)__get_free_pages(GFP_KERNEL | __GFP_ZERO, 1);
	sddc->enc_out = (void *)__get_free_page(GFP_KERNEL | __GFP_ZERO);
	sddc->enc_tmp = (void *)__get_free_page(GFP_KERNEL | __GFP_ZERO);
	sddc->dec_pair = (void *)__get_free_pages(GFP_KERNEL | __GFP_ZERO, 1);
	sddc->dec_hdr = (void *)__get_free_page(GFP_KERNEL | __GFP_ZERO);
	sddc->dec_tmp = (void *)__get_free_page(GFP_KERNEL | __GFP_ZERO);
	sddc->proof_pair = (void *)__get_free_pages(GFP_KERNEL | __GFP_ZERO, 1);

	if (!sddc->buckets || !sddc->enc_pair || !sddc->enc_out ||
	    !sddc->enc_tmp || !sddc->dec_pair || !sddc->dec_hdr ||
	    !sddc->dec_tmp || !sddc->proof_pair)
		goto err;

	return sddc;
err:
	zram_sddc_destroy(sddc);
	return NULL;
}

void zram_sddc_destroy(struct zram_sddc *sddc)
{
	if (!sddc)
		return;

	vfree(sddc->buckets);
	free_pages((unsigned long)sddc->enc_pair, 1);
	free_page((unsigned long)sddc->enc_out);
	free_page((unsigned long)sddc->enc_tmp);
	free_pages((unsigned long)sddc->dec_pair, 1);
	free_pages((unsigned long)sddc->proof_pair, 1);
	free_page((unsigned long)sddc->dec_hdr);
	free_page((unsigned long)sddc->dec_tmp);
	kfree(sddc);
}
