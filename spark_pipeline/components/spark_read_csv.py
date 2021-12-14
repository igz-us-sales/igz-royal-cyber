import mlrun
from pyspark.sql import SparkSession

def spark_read_csv(dataset: mlrun.DataItem):
    
    # Create MLRun context
    with mlrun.get_or_create_ctx("spark") as context:

        # Build Spark session with config
        spark = SparkSession.builder.appName("Spark job")\
            .config("spark.cores.max","1")\
            .config("spark.executor.memory","1g")\
            .getOrCreate()
        
        # Get location of artifact - looks in v3io://projects directory
        location = dataset.url.split("projects/")[1]

        # Read CSV
        df = spark.read.load(location, format="csv", sep=",", header="true")

        # Sample for logging
        df_to_log = df.toPandas()

        # Log CSV via MLRun experiment tracking
        context.log_dataset("df_sample", 
                            df=df_to_log,
                            format="csv", index=False,
                            artifact_path=context.artifact_path)

        spark.stop()