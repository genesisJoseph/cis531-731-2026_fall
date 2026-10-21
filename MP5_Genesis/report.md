## CIS 531/731 — MP 5: Structured Streaming
Student: Genesis Joseph
Course: CIS 731 — Fall 2026
Project: ASR/VDB Structured Streaming Pipeline
Execution Environment: Local CPU baseline, PySpark

1. Data Ingestion

I developed a PySpark Structured Streaming pipeline for ASR transcripts and VDB corpus records, and I established schemas together with validation checks for the timestamps, text, record IDs, and source information.

2. Event Time and Watermarks

I employed a two-minute watermark, sixty-second tumbling windows, and five-minute sliding windows with a one-minute slide; although the pipeline did run, the limited amount of input data wasn't sufficient to move the watermark far enough for the final results to be produced.

3. Analytics

I carried out the implementation of record counts, average ingestion latency, approximate distinct-token counts, and word frequencies, and I also wrote a script for inspecting the Parquet results.

4. Recovery and Performance

I set up the checkpoint directories and gathered the Spark progress metrics; I also prepared a reconciliation script in order to compare the outputs. Yet I was unable to fully verify the recovery or the meaningful aggregation results with the data available as input.

5. MP 3 Dependency and Limitations

A big difficulty was our reliance on MP 3. Since the MP 3 assignment did not ask us to create or save a timestamped JSON/Parquet landing-zone dataset in the format required by MP 5, the dataset necessary to carry out the streaming evaluation was not included in my earlier submission. I had used a small number of local replay records to test the implementation, but these were not enough for the full analysis. As a result, I was unable to carry out all the required performance and recovery verification.

## GenAI Audit

Yes, I used ChatGPT (OpenAI) as a Generative AI assistant for MP 5. I used it to understand the assignment requirements, identify the necessary deliverables, review Python and PySpark syntax, and plan the streaming pipeline implementation.

The AI assistance included explanations of data contracts, event-time windowing, watermarking, checkpointing, performance metrics, and fault-tolerance testing.

My AI interaction log is included in the repository as `GenAI.pdf`. Any code testing, validation, and execution results are documented separately in the project files.