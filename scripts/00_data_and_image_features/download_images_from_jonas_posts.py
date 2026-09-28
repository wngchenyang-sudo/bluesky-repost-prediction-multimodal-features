"""Download all image-bearing posts in postsFinal.json; never modify the 24k cache.

Default is a read-only inventory. Add --download to copy/download files.
Restart the same command to resume. Failed URLs are retried on the next run.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from PIL import Image

ROOT = Path(r"E:\Blue\MEng-project-jonas-original\MEng-project-original")
PAIR_FIELDS = ['hashtag', 'post_uri', 'image_number', 'image_cid', 'image_url', 'post_text']
RESULT_FIELDS = ['image_url', 'status', 'local_path', 'bytes', 'sha256',
                 'width', 'height', 'format', 'http_status', 'attempts', 'error', 'seconds', 'origin']
POST_FIELDS = ['post_uri', 'hashtag', 'source_image_count', 'has_image_source', 'source_status']


def own_images(embed):
    if not isinstance(embed, dict):
        return []
    return list(embed.get('images') or []) + own_images(embed.get('media'))


def inventory(posts):
    pairs, post_rows, errors = [], [], []
    for uri, post in posts.items():
        count = 0
        try:
            if not isinstance(post, dict) or post.get('uri', uri) != uri:
                raise ValueError('Missing or mismatched URI')
            author = uri.split('/')[2]
            if post.get('author', {}).get('did') != author:
                raise ValueError('Author DID mismatch')
            record = post.get('record') or {}
            blobs = []
            for im in own_images(record.get('embed')):
                cid = im.get('image', {}).get('ref', {}).get('$link')
                if not cid:
                    raise ValueError('Image without source CID')
                blobs.append(cid)
            ordered = list(dict.fromkeys(blobs))
            count = len(ordered)
            by_cid = {}
            for im in own_images(post.get('embed')):
                url = im.get('fullsize') or im.get('thumb') or im.get('url')
                if not url:
                    raise ValueError('Image without URL')
                parsed = urlparse(url)
                parts = unquote(parsed.path).split('/')
                if parsed.scheme != 'https' or parsed.hostname != 'cdn.bsky.app':
                    raise ValueError('Unexpected image host')
                if len(parts) < 2 or parts[-2] != author:
                    raise ValueError('Image URL author mismatch')
                cid = parts[-1].split('@')[0]
                if cid in by_cid and by_cid[cid] != url:
                    raise ValueError('Conflicting URLs for CID')
                by_cid[cid] = url
            if set(ordered) != set(by_cid):
                raise ValueError('Record/view image CID sets differ')
            for number, cid in enumerate(ordered, 1):
                pairs.append(dict(hashtag=post.get('hashtag', ''), post_uri=uri,
                                  image_number=number, image_cid=cid,
                                  image_url=by_cid[cid], post_text=record.get('text', '')))
            status = 'ok'
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            status = 'source_error'
            errors.append({'post_uri': uri, 'error': str(exc)})
        post_rows.append(dict(post_uri=uri, hashtag=post.get('hashtag', '') if isinstance(post, dict) else '',
                              source_image_count=count, has_image_source=int(count > 0), source_status=status))
    return pairs, post_rows, errors


def write_csv(path, rows, fields):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def inspect(data):
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    width=im.width, height=im.height, format=im.format)


def verified_bytes(root, row):
    path = (root / row['local_path']).resolve()
    if not path.is_relative_to((root / 'images').resolve()):
        raise ValueError('Cache path outside images directory')
    data = path.read_bytes()
    if not row.get('sha256') or hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('Cache SHA256 mismatch')
    inspect(data)
    return data


def check_output_directory(out, config):
    """Allow our script and a matching completed 100-URL trial on first full run."""
    if not out.exists():
        return
    config_path = out / 'download_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Output configuration differs; use a new output directory')
        return
    for entry in out.iterdir():
        if entry.resolve() == Path(__file__).resolve():
            continue
        if entry.name == 'trial_100' and entry.is_dir() and 'limit_unique_urls' not in config:
            trial_config = entry / 'download_config.json'
            trial_summary = entry / 'download_summary.json'
            expected = {**config, 'limit_unique_urls': 100}
            if (trial_config.exists() and trial_summary.exists()
                    and not (entry / 'download.lock').exists()
                    and json.loads(trial_config.read_text(encoding='utf-8')) == expected):
                summary = json.loads(trial_summary.read_text(encoding='utf-8'))
                if summary.get('limit_unique_urls') == 100 and summary.get('unique_urls') == 100:
                    continue
        raise ValueError('Unexpected existing output entry: ' + str(entry))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--posts', type=Path, default=ROOT / 'data/raw/posts/postsFinal.json')
    parser.add_argument('--output', type=Path, default=Path(r'E:\Blue\image_downloads\full_posts'))
    parser.add_argument('--reuse', type=Path, default=Path(r'E:\Blue\image_downloads\small sample'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None, help='Trial: first N unique image URLs; omit for full download')
    parser.add_argument('--no-reuse', action='store_true', help='Do not copy images from the old 24k cache')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('workers must be 1..32')
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    out, reuse = args.output.resolve(), args.reuse.resolve()
    if out == reuse or out.is_relative_to(reuse) or reuse.is_relative_to(out):
        raise ValueError('Output and old cache must be separate directories')

    source_bytes = args.posts.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    posts = json.loads(source_bytes)
    del source_bytes
    if not isinstance(posts, dict):
        raise ValueError('Expected URI-keyed postsFinal.json dictionary')
    pairs, post_rows, errors = inventory(posts)
    del posts
    urls = list(dict.fromkeys(p['image_url'] for p in pairs))
    full_post_count = len(post_rows)
    full_url_count = len(urls)
    if args.limit is not None:
        urls = urls[:args.limit]
        selected_urls = set(urls)
        pairs = [p for p in pairs if p['image_url'] in selected_urls]
        selected_posts = {p['post_uri'] for p in pairs}
        post_rows = [p for p in post_rows if p['post_uri'] in selected_posts]

    old = {}
    cache_csv = reuse / 'downloads_by_url.csv'
    if not args.no_reuse and cache_csv.exists():
        with cache_csv.open(encoding='utf-8-sig', newline='') as f:
            old = {r['image_url']: r for r in csv.DictReader(f)
                   if r['status'] in ('downloaded', 'skipped_existing')}
    print(json.dumps(dict(posts=len(post_rows), image_pairs=len(pairs), unique_urls=len(urls),
                          source_error_posts=len(errors), reuse_candidates=sum(u in old for u in urls),
                          output=str(out), mode='download' if args.download else 'inventory_only'),
                     ensure_ascii=False), flush=True)
    if not args.download:
        print('Read-only inventory complete. Add --download to start. No files written.')
        return

    config = dict(version=1, posts_source=str(args.posts.resolve()), source_sha256=source_hash,
                  reuse_source=str(reuse))
    if args.no_reuse:
        config['reuse_source'] = None
    if args.limit is not None:
        config['limit_unique_urls'] = args.limit
    config_path = out / 'download_config.json'
    check_output_directory(out, config)
    out.mkdir(parents=True, exist_ok=True)
    # A stale lock after a forced termination must be inspected, not automatically removed.
    lock = out / 'download.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    try:
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        (out / 'images').mkdir(exist_ok=True)
        write_csv(out / 'source_posts.csv', post_rows, POST_FIELDS)
        write_csv(out / 'source_image_pairs.csv', pairs, PAIR_FIELDS)
        write_csv(out / 'source_errors.csv', errors, ['post_uri', 'error'])
        previous = {}
        journal_path = out / 'download_progress.jsonl'
        if journal_path.exists():
            with journal_path.open(encoding='utf-8') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        previous[r['image_url']] = r
                    except (ValueError, KeyError):
                        continue
        local = threading.local()

        def fetch(url):
            started = time.monotonic()
            base = dict(image_url=url, status='failed', origin='', local_path='', bytes=0,
                        sha256='', width=0, height=0, format='', http_status=None,
                        attempts=0, error='', seconds=0.0)
            try:
                prior = previous.get(url)
                if prior and prior['status'] in ('downloaded', 'skipped_existing'):
                    try:
                        verified_bytes(out, prior)
                        return {**base, **prior, 'status': 'skipped_existing', 'origin': 'resumed',
                                'http_status': None, 'attempts': 0, 'error': '',
                                'seconds': round(time.monotonic() - started, 3)}
                    except (OSError, ValueError):
                        pass
                data, origin = None, 'network'
                if url in old:
                    try:
                        data = verified_bytes(reuse, old[url])
                        origin = 'copied_24k'
                    except (OSError, ValueError):
                        pass
                if data is None:
                    if not hasattr(local, 'session'):
                        local.session = requests.Session()
                        local.session.headers['User-Agent'] = 'FullPostImageDownloader/1.0'
                    for attempt in range(3):
                        base['attempts'] = attempt + 1
                        base['http_status'] = None
                        try:
                            began = time.monotonic()
                            with local.session.get(url, stream=True, timeout=(10, 20), allow_redirects=False) as res:
                                base['http_status'] = res.status_code
                                res.raise_for_status()
                                if res.status_code != 200:
                                    raise ValueError('Unexpected HTTP status ' + str(res.status_code))
                                chunks, size = [], 0
                                for chunk in res.iter_content(65536):
                                    size += len(chunk)
                                    if size > 100 * 1024 * 1024 or time.monotonic() - began > 90:
                                        raise ValueError('Image size/time limit exceeded')
                                    chunks.append(chunk)
                            data = b''.join(chunks)
                            inspect(data)
                            break
                        except requests.RequestException as exc:
                            status = exc.response.status_code if exc.response is not None else None
                            if status in (400, 401, 403, 404, 410) or attempt == 2:
                                raise
                            time.sleep(5 * (attempt + 1) if status == 429 else attempt + 1)
                info = inspect(data)
                ext = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'GIF': '.gif'}.get(info['format'], '.img')
                relative = 'images/' + hashlib.sha256(url.encode()).hexdigest() + ext
                target = out / relative
                temp = target.with_suffix(target.suffix + '.part')
                temp.write_bytes(data)
                os.replace(temp, target)
                return {**base, **info, 'status': 'downloaded', 'origin': origin, 'local_path': relative,
                        'seconds': round(time.monotonic() - started, 3)}
            except Exception as exc:
                return {**base, 'error': type(exc).__name__ + ': ' + str(exc),
                        'seconds': round(time.monotonic() - started, 3)}

        results, stats = {}, Counter()
        began = time.monotonic()
        with journal_path.open('a', encoding='utf-8') as journal:
            journal.write('\n')  # separate a possibly interrupted final record
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                # Bounded batches avoid creating one Future per full-dataset image.
                for offset in range(0, len(urls), 128):
                    for r in pool.map(fetch, urls[offset:offset + 128]):
                        results[r['image_url']] = r
                        stats[r['origin'] or r['status']] += 1
                        journal.write(json.dumps(r, ensure_ascii=False) + '\n')
                        journal.flush()
                    print(f'{len(results)}/{len(urls)} | {dict(stats)} | {(time.monotonic()-began)/60:.1f} min', flush=True)
        manifest = ({**p, **{k: v for k, v in results[p['image_url']].items() if k != 'image_url'}} for p in pairs)
        write_csv(out / 'image_manifest.csv', manifest, PAIR_FIELDS + RESULT_FIELDS[1:])
        write_csv(out / 'downloads_by_url.csv', (results[u] for u in urls), RESULT_FIELDS)
        successful = {p['post_uri'] for p in pairs if results[p['image_url']]['status'] != 'failed'}
        summary = dict(posts_source=str(args.posts.resolve()), total_posts=len(post_rows),
                       limit_unique_urls=args.limit, full_source_posts=full_post_count,
                       full_source_unique_urls=full_url_count,
                       source_error_posts=len(errors), post_image_pairs=len(pairs), unique_urls=len(urls),
                       status_counts=dict(Counter(r['status'] for r in results.values())),
                       origin_counts=dict(stats), posts_with_downloaded_image=len(successful),
                       total_image_bytes=sum(r['bytes'] for r in results.values()),
                       elapsed_minutes=round((time.monotonic()-began)/60, 2), features_computed=False)
        (out / 'download_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
