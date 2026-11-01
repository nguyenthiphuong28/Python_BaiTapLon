from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def main():
    # 1. Khởi tạo Spark
    spark = SparkSession.builder \
        .appName("Music Popularity Prediction - RandomForest") \
        .getOrCreate()

    # 2. Đọc + xử lý dữ liệu
    df = spark.read.csv(
        "hdfs://master:9000/user/spotify/dataset.csv",
        header=True,
        inferSchema=True
    ).select(
        col("track_genre").alias("genre"),
        col("popularity").cast("int"),
        col("danceability").cast("double"),
        col("energy").cast("double"),
        col("loudness").cast("double"),
        col("speechiness").cast("double"),
        col("acousticness").cast("double"),
        col("instrumentalness").cast("double"),
        col("liveness").cast("double"),
        col("valence").cast("double"),
        col("tempo").cast("double"),
        col("duration_ms").cast("double")
    ).dropna() \
     .filter(col("popularity") >= 5)

    print("=== Sample Data After Filter ===")
    df.show(5)

    # 3. Feature list
    feature_cols = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness",
        "valence", "tempo", "duration_ms", "genre_index"
    ]

    # 4. Pipeline stages
    indexer = StringIndexer(
        inputCol="genre",
        outputCol="genre_index"
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="popularity",
        numTrees=100,
        maxDepth=10,
        maxBins=256,
        seed=42
    )

    pipeline = Pipeline(stages=[indexer, assembler, rf])

    # 5. Train / Test split
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    print("Train count:", train_data.count())
    print("Test count:", test_data.count())

    # 6. Train model
    model = pipeline.fit(train_data)

    # Feature Importance
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances

    print("=== Feature Importances ===")
    for f, v in zip(feature_cols, importances):
        print(f"{f}: {v:.4f}")

    # 7. Predict
    predictions = model.transform(test_data)

    print("=== Predictions ===")
    predictions.select("popularity", "prediction").show(10)

    # 8. Evaluate
    evaluator = RegressionEvaluator(
        labelCol="popularity",
        predictionCol="prediction",
        metricName="rmse"
    )

    rmse = evaluator.evaluate(predictions)
    print("RMSE =", rmse)

    # R2
    evaluator_r2 = RegressionEvaluator(
        labelCol="popularity",
        predictionCol="prediction",
        metricName="r2"
    )

    r2 = evaluator_r2.evaluate(predictions)
    print("R2 =", r2)

    # 9. Save model
    model.write().overwrite().save("rf_music_model")
    print("Model saved to rf_music_model")

    # 10. Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()
