"""MP5 Structured Streaming: ASR/VDB, event-time windows, token statistics."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from contracts import transcript_schema, corpus_schema, check_events

OUT = os.getenv('MP5_OUTPUT', 'output_fixed')
CKPT = os.getenv('MP5_CHECKPOINT', 'checkpoints_fixed')
METRICS = os.getenv('MP5_METRICS', 'metrics.jsonl')
WATERMARK = os.getenv('MP5_WATERMARK', '2 minutes')
for folder in ('data/transcripts', 'data/corpus', OUT, CKPT):
    Path(folder).mkdir(parents=True, exist_ok=True)

spark = (SparkSession.builder.master(os.getenv('SPARK_MASTER', 'local[2]'))
         .appName('MP5StructuredStreaming')
         .config('spark.sql.shuffle.partitions', '2')
         .config('spark.sql.session.timeZone', 'UTC').getOrCreate())
spark.sparkContext.setLogLevel('WARN')

asr = spark.readStream.schema(transcript_schema).option('multiLine', 'true').json('data/transcripts')
vdb = spark.readStream.schema(corpus_schema).option('multiLine', 'true').json('data/corpus')
asr = check_events(asr.filter(F.col('track') == 'ASR'))
vdb = check_events(vdb.filter(F.col('track') == 'VDB'))
events = asr.unionByName(vdb.drop('license_class'))
events = events.withColumn('ingest_latency_sec', F.unix_timestamp('ingest_ts') - F.unix_timestamp('ts_end'))
words = (events.select('record_id', 'track', 'ts_end',
                       F.explode(F.split(F.lower(F.regexp_replace('text', r'[^a-zA-Z0-9]+', ' ')), r'\s+')).alias('token'))
         .filter(F.col('token') != ''))

specs = {'tumbling': ('60 seconds', None), 'sliding': ('5 minutes', '1 minute')}

def win(spec):
    duration, slide = specs[spec]
    return F.window('ts_end', duration, slide) if slide else F.window('ts_end', duration)

def summary(spec):
    return (events.withWatermark('ts_end', WATERMARK)
            .groupBy(win(spec).alias('window'), 'track')
            .agg(F.count('*').alias('record_count'),
                 F.avg('ingest_latency_sec').alias('mean_ingest_latency_sec')))

def tokens(spec):
    return (words.withWatermark('ts_end', WATERMARK)
            .groupBy(win(spec).alias('window'), 'track')
            .agg(F.approx_count_distinct('token').alias('approx_distinct_tokens')))

def terms(spec):
    return (words.withWatermark('ts_end', WATERMARK)
            .groupBy(win(spec).alias('window'), 'track', 'token')
            .agg(F.count('*').alias('term_count')))

def start(df, name):
    return (df.writeStream.queryName(name).format('parquet').outputMode('append')
            .option('path', f'{OUT}/{name}')
            .option('checkpointLocation', f'{CKPT}/{name}')
            .trigger(availableNow=True).start())

queries = {}
try:
    for spec in specs:
        queries[spec] = start(summary(spec), spec)
        queries[f'{spec}_tokens'] = start(tokens(spec), f'{spec}_tokens')
        queries[f'{spec}_terms'] = start(terms(spec), f'{spec}_terms')
    for query in queries.values():
        query.awaitTermination()
finally:
    with open(METRICS, 'a', encoding='utf-8') as log:
        for name, query in queries.items():
            for p in query.recentProgress:
                state = p.get('stateOperators', [])
                log.write(json.dumps({
                    'recorded_at': datetime.now(timezone.utc).isoformat(),
                    'query': name, 'batch_id': p.get('batchId'),
                    'input_rows': p.get('numInputRows'),
                    'input_rows_per_second': p.get('inputRowsPerSecond'),
                    'processed_rows_per_second': p.get('processedRowsPerSecond'),
                    'batch_duration_ms': p.get('durationMs', {}).get('triggerExecution'),
                    'state_rows_total': sum(s.get('numRowsTotal', 0) for s in state),
                    'watermark': p.get('eventTime', {}).get('watermark')
                }) + '\n')
    spark.stop()
print('Streaming finished.')
