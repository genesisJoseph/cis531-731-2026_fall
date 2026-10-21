"""Inspect finalized window aggregates and rank top-10 terms per window."""
import os
from pathlib import Path
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = SparkSession.builder.master('local[2]').appName('MP5Inspect').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
root = Path(os.getenv('MP5_OUTPUT', 'output_fixed'))
for spec in ('tumbling', 'sliding'):
    for suffix in ('', '_tokens', '_terms'):
        name = spec + suffix
        path = root / name
        if not list(path.glob('*.parquet')):
            print(f'{name}: no finalized Parquet rows')
            continue
        df = spark.read.parquet(str(path))
        print(f'\n{name}: {df.count()} rows')
        if suffix == '_terms':
            rank = Window.partitionBy('window', 'track').orderBy(F.desc('term_count'), F.asc('token'))
            df = df.withColumn('rank', F.row_number().over(rank)).filter('rank <= 10')
        df.show(100, truncate=False)
spark.stop()
