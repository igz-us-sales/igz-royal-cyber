from kfp import dsl
import mlrun

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(
    name="spark-pipeline",
    description="Example of batch spark pipeline for iris dataset"
)
def pipeline(
    source_url: str,
    label_column: str
):
    # Get MLRun project
    project = mlrun.get_current_project()
    
    # Fetch data set
    ingest = mlrun.run_function(
        'get-data',
        handler='prep_data',
        inputs={'source_url': source_url},
        params={'label_column': label_column},
        outputs=["cleaned_data"]
    )
    
    # Read CSV with Spark
    spark_read_fn = project.get_function("spark-read-csv")
    spark_read_fn.with_spark_service(spark_service="spark")
    spark_read = mlrun.run_function(
        'spark-read-csv',
        handler='spark_read_csv',
        inputs={'dataset': ingest.outputs["cleaned_data"]},
        outputs=["df_sample"]
    )