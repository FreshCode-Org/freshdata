"""examples/integrations/fastapi_clean_service.py — FastAPI Cleaning Service.

Problem:
--------
Production systems often need to expose data-cleaning workflows through HTTP
APIs. Running CPU-bound dataframe cleaning directly inside an async FastAPI
endpoint can block the event loop.

This example provides a lightweight FastAPI service that accepts either JSON
tabular records or CSV uploads and runs FreshData cleaning in a worker thread.

Installation:
-------------
pip install fastapi uvicorn python-multipart "freshdata-cleaner[dev]"
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import UploadFile

import freshdata as fd

app = FastAPI(
    title="FreshData Cleaning Service",
    description="Asynchronous FastAPI integration for FreshData.",
)


def parse_bool(value: object, default: bool = False) -> bool:
    """Parse common boolean representations from form or JSON input."""
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "on"}


async def clean_dataframe(
    df: pd.DataFrame,
    *,
    strategy: object = "balanced",
    impute: object | None = None,
    drop_duplicates: bool = False,
) -> tuple[pd.DataFrame, Any]:
    """Run FreshData cleaning without blocking the FastAPI event loop."""
    clean_options: dict[str, object] = {
        "strategy": strategy,
        "drop_duplicates": drop_duplicates,
        "return_report": True,
    }

    if impute is not None:
        clean_options["impute"] = impute

    cleaned_df, report = await asyncio.to_thread(
        fd.clean,
        df,
        **clean_options,
    )

    return cleaned_df, report


@app.post("/clean")
async def clean_data(request: Request) -> dict[str, object]:
    """Clean JSON tabular records or an uploaded CSV file."""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON body.",
            ) from exc

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="JSON body must be an object.",
            )

        records = payload.get("records")

        if not isinstance(records, list) or not records:
            raise HTTPException(
                status_code=400,
                detail="JSON body must include non-empty 'records'.",
            )

        try:
            df = pd.DataFrame(records)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Unable to convert records into tabular data.",
            ) from exc

        strategy = payload.get("strategy", "balanced")
        impute = payload.get("impute")
        drop_duplicates = parse_bool(
            payload.get("drop_duplicates"),
            default=False,
        )

    elif "multipart/form-data" in content_type:
        form = await request.form()

        file = form.get("file")

        if not isinstance(file, UploadFile):
            raise HTTPException(
                status_code=400,
                detail="CSV file upload is required.",
            )

        contents = await file.read()

        try:
            df = pd.read_csv(BytesIO(contents))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Unable to read CSV file.",
            ) from exc

        strategy = form.get("strategy", "balanced")
        impute = form.get("impute")

        drop_duplicates = parse_bool(
            form.get("drop_duplicates"),
            default=False,
        )

    else:
        raise HTTPException(
            status_code=415,
            detail="Use application/json or multipart/form-data.",
        )

    try:
        cleaned_df, report = await clean_dataframe(
            df,
            strategy=strategy,
            impute=impute,
            drop_duplicates=drop_duplicates,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"FreshData cleaning failed: {exc}",
        ) from exc

    return {
        "data": cleaned_df.to_dict(orient="records"),
        "report": report.to_dict(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
