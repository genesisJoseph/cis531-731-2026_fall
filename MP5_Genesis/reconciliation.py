"""Validate inputs and compare all six recovered sinks to an independent clean run.
This does NOT perform the interruption; run an actual kill/restart separately.
"""
import json
import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from contracts import transcript_schema, corpus_schema, check_events

spark = (SparkSession.builder.master('local[2]').appName('MP5Reconciliation')
         .config('spark.sql.session.timeZone', 'UTC').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')
transcripts = spark.read.schema(transcript_schema).option('multiLine', 'true').json('data/transcripts')
corpus = spark.read.schema(corpus_schema).option('multiLine', 'true').json('data/corpus')
valid_asr = check_events(transcripts.filter(F.col('track') == 'ASR'))
valid_vdb = check_events(corpus.filter(F.col('track') == 'VDB'))
events = valid_asr.unionByName(valid_vdb.drop('license_class'))
print('ASR total/valid:', transcripts.count(), valid_asr.count())
print('VDB total/valid:', corpus.count(), valid_vdb.count())
late = events.withColumn('lateness_sec', F.unix_timestamp('ingest_ts') - F.unix_timestamp('ts_end'))
print('Lateness p50/p95/p99:', late.approxQuantile('lateness_sec', [0.5, 0.95, 0.99], 0.01))

recovered_root = Path(os.getenv('MP5_OUTPUT', 'output_fixed'))
clean_root = Path(os.getenv('MP5_CLEAN_OUTPUT', 'output_clean'))
all_good = True
compared = 0
for spec in ('tumbling', 'sliding'):
    for suffix in ('', '_tokens', '_terms'):
        name = spec + suffix
        rpath, cpath = recovered_root / name, clean_root / name
        if not list(rpath.glob('*.parquet')) or not list(cpath.glob('*.parquet')):
            print(name, ': MISSING OUTPUT OR BASELINE')
            all_good = False
            continue
        r, c = spark.read.parquet(str(rpath)), spark.read.parquet(str(cpath))
        key = ['window', 'track'] + (['token'] if suffix == '_terms' else [])
        duplicates = r.groupBy(*key).count().filter(F.col('count') > 1).count()
        if sorted(r.columns) != sorted(c.columns):
            print(name, ': SCHEMA MISMATCH')
            all_good = False
            continue
        cols = sorted(c.columns)
        missing = c.select(cols).exceptAll(r.select(cols)).count()
        extra = r.select(cols).exceptAll(c.select(cols)).count()
        nr, nc = r.count(), c.count()
        print(f'{name}: clean={nc}, recovered={nr}, duplicates={duplicates}, missing={missing}, extra={extra}')
        if nr == 0 or nc == 0 or duplicates or missing or extra:
            all_good = False
        compared += 1
print('COMPARISON:', 'PASS' if all_good and compared == 6 else 'NOT VERIFIED')
print('A PASS only supports recovery if a real interrupted run was restarted from its existing checkpoints.')
spark.stop()
