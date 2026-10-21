from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType
)
from pyspark.sql.functions import (
    col, trim, unix_timestamp
)


ALLOWED_SOURCES = ["local_replay"]


ALLOWED_LICENSES = ["public_domain"]

transcript_schema = StructType([
    StructField("record_id", StringType(), False),
    StructField("track", StringType(), False),
    StructField("source_id", StringType(), False),
    StructField("ts_start", TimestampType(), False),
    StructField("ts_end", TimestampType(), False),
    StructField("text", StringType(), False),
    StructField("ingest_ts", TimestampType(), False),
])

corpus_schema = StructType([
    StructField("record_id", StringType(), False),
    StructField("track", StringType(), False),
    StructField("source_id", StringType(), False),
    StructField("ts_start", TimestampType(), False),
    StructField("ts_end", TimestampType(), False),
    StructField("text", StringType(), False),
    StructField("ingest_ts", TimestampType(), False),
    StructField("license_class", StringType(), False),
])


def check_events(df):
    duration = (
        unix_timestamp("ts_end") -
        unix_timestamp("ts_start")
    )

    valid = (
        col("record_id").isNotNull()
        & (trim(col("record_id")) != "")
        & col("track").isin("ASR", "VDB")
        & col("source_id").isin(ALLOWED_SOURCES)
        & col("ts_start").isNotNull()
        & col("ts_end").isNotNull()
        & col("ingest_ts").isNotNull()
        & col("text").isNotNull()
        & (trim(col("text")) != "")
        & (col("ts_start") < col("ts_end"))
        & (duration > 0)
        & (duration <= 30)
    )

    if "license_class" in df.columns:
        valid = (
            valid
            & col("license_class").isin(ALLOWED_LICENSES)
        )

    return df.filter(valid)