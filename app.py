
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)
from werkzeug.utils import secure_filename


# Project folders
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx"}
MAX_FILE_SIZE = 16 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

logging.basicConfig(
    filename=str(BASE_DIR / "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def clean_chunk(df):
    """Clean a DataFrame by handling spaces and missing values."""
    df = df.copy()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove extra spaces from text values
    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:
        df[column] = df[column].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    # Replace infinite values with missing values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill missing values
    for column in df.columns:
        if df[column].isna().any():

            if pd.api.types.is_numeric_dtype(df[column]):
                median = df[column].median()

                if pd.isna(median):
                    median = 0

                df[column] = df[column].fillna(median)

            else:
                mode = df[column].mode(dropna=True)

                fill_value = (
                    mode.iloc[0]
                    if not mode.empty
                    else "Unknown"
                )

                df[column] = df[column].fillna(fill_value)

    return df


def process_dataset(df):
    """Clean the dataset and generate a processing report."""
    start_time = time.perf_counter()

    if df.empty or len(df.columns) == 0:
        raise ValueError("Dataset is empty.")

    original_rows = len(df)
    original_columns = len(df.columns)

    missing_before = int(df.isna().sum().sum())
    duplicates_before = int(df.duplicated().sum())

    cleaned_df = clean_chunk(df)

    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates().reset_index(
        drop=True
    )

    report = {
        "status": "success",
        "original_rows": int(original_rows),
        "cleaned_rows": int(len(cleaned_df)),
        "original_columns": int(original_columns),
        "final_columns": int(len(cleaned_df.columns)),
        "missing_values_before": missing_before,
        "missing_values_after": int(
            cleaned_df.isna().sum().sum()
        ),
        "duplicates_found": duplicates_before,
        "duplicates_removed": int(
            original_rows - len(cleaned_df)
        ),
        "processing_time_seconds": round(
            time.perf_counter() - start_time, 4
        )
    }

    return cleaned_df, report


@app.route("/", methods=["GET", "POST"])
def home():
    message = ""
    result = None

    if request.method == "POST":
        uploaded_file = request.files.get("file")

        if not uploaded_file or not uploaded_file.filename:
            message = "Please select a file."

        elif not allowed_file(uploaded_file.filename):
            message = "Please upload a CSV or XLSX file."

        else:
            input_path = None

            try:
                filename = secure_filename(
                    uploaded_file.filename
                )

                if not filename:
                    raise ValueError("Invalid filename.")

                # Generate a unique ID for this job
                job_id = uuid.uuid4().hex

                input_path = UPLOAD_DIR / f"{job_id}_{filename}"
                uploaded_file.save(input_path)

                # Read the uploaded dataset
                if input_path.suffix.lower() == ".csv":
                    df = pd.read_csv(input_path)
                else:
                    df = pd.read_excel(
                        input_path,
                        engine="openpyxl"
                    )

                # Clean dataset and generate report
                cleaned_df, report = process_dataset(df)

                # Output filenames
                output_file = f"cleaned_{job_id}.csv"
                report_file = f"report_{job_id}.json"

                # Save cleaned CSV
                cleaned_df.to_csv(
                    OUTPUT_DIR / output_file,
                    index=False
                )

                # Save JSON report
                with open(
                    OUTPUT_DIR / report_file,
                    "w",
                    encoding="utf-8"
                ) as report_handle:
                    json.dump(
                        report,
                        report_handle,
                        indent=4
                    )

                # Prepare result for the webpage
                result = {
                    "job_id": job_id,
                    "filename": output_file,
                    "report_file": report_file,
                    "report": report,
                    "columns": cleaned_df.columns.tolist(),
                    "preview": cleaned_df.head(5).to_dict(
                        orient="records"
                    )
                }

                message = "Dataset processed successfully!"

                logging.info(
                    "Dataset processed successfully: %s",
                    filename
                )

            except Exception:
                logging.exception("Dataset processing failed")
                message = (
                    "Processing failed. Please check your "
                    "file format and contents."
                )

            finally:
                # Remove temporary uploaded file
                if input_path and input_path.exists():
                    input_path.unlink()

    return render_template(
        "index.html",
        message=message,
        result=result
    )


@app.route("/download/<filename>")
def download_file(filename):
    # Allow only generated output files
    valid_cleaned_file = (
        filename.startswith("cleaned_")
        and filename.endswith(".csv")
    )

    valid_report_file = (
        filename.startswith("report_")
        and filename.endswith(".json")
    )

    if not (valid_cleaned_file or valid_report_file):
        return "Invalid filename.", 400

    return send_from_directory(
        str(OUTPUT_DIR),
        filename,
        as_attachment=True
    )


if __name__ == "__main__":
    print("Starting Flask server...", flush=True)
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
        