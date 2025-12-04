import json
import os
from typing import Any, Dict

import pandas as pd
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Suppress only the insecure request warning for self-signed HTTPS
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


def _request_sap(url: str, use_auth: bool = True) -> requests.Response:
    """
    Internal helper that performs a GET request to the given URL.

    If use_auth is True, basic auth with fixed credentials is used.
    If use_auth is False, no auth is sent.

    This function does NOT parse JSON; it just returns the Response object.
    """
    # Direct credentials (adapt as needed)
    SAP_USERNAME = "DEVELOPER"
    SAP_PASSWORD = "ABAPtr2023#00"

    headers = {
        "Accept": "application/json",
        "Connection": "close",  # sometimes helps with servers that dislike keep-alive
    }

    kwargs: Dict[str, Any] = {
        "headers": headers,
        "timeout": 10,
        # Since you "accept risk" in the browser (self-signed / untrusted cert),
        # we mirror that here by disabling certificate verification.
        # For production, you should instead point 'verify' to a proper CA bundle.
        "verify": False,
    }
    if use_auth:
        kwargs["auth"] = (SAP_USERNAME, SAP_PASSWORD)

    response = requests.get(url, **kwargs)
    response.raise_for_status()
    return response


def parse_json() -> Dict[str, Any]:
    """
    Fetch stock data from local SAP OData service.

    First, try with basic auth. If the server aborts the connection
    (RemoteDisconnected / ConnectionError), retry once without auth.

    Credentials are defined directly in this function (no environment variables).
    Adjust SAP_USERNAME and SAP_PASSWORD as needed.

    Raises:
    Raises:
        RuntimeError: if the request fails or JSON cannot be decoded.
    """
    # IMPORTANT: ensure this URL (scheme host, port, path) matches EXACTLY
    # what your browser shows after you "accept risk".
    # Note the change to 'https://' here.
    url = (
        "https://localhost:50001/"
        "sap/opu/odata4/sap/test_binding/srvd_a2x/sap/sd_stock_api/0001/"
        "view_stock_act?sap-client=001&$format=json"
    )

    # 1) Try with auth
    try:
        response = _request_sap(url, use_auth=True)
    except requests.exceptions.ConnectionError as exc:
        # Server aborted connection, possibly due to auth or protocol expectations.
        print(
            "Warning: connection aborted when using basic auth. "
            "Retrying once WITHOUT auth to see if the endpoint is public "
            "or requires a different authentication method..."
        )
        try:
            response = _request_sap(url, use_auth=False)
        except requests.exceptions.RequestException as exc2:
            raise RuntimeError(
                "SAP OData call failed both with and without basic auth. "
                "This strongly indicates a server/proxy configuration issue "
                "or a mismatch between the URL used in the browser and in this script. "
                f"Underlying error (no-auth): {exc2}"
            ) from exc2
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Error while calling SAP OData service with basic auth: "
            f"{exc}\nPlease verify that the URL, port, and protocol "
            "exactly match what you use in the browser."
        ) from exc

    # At this point, we have a successful HTTP response (status 2xx).
    # Try to decode JSON.
    try:
        data = response.json()
    except ValueError as exc:
        snippet = response.text[:500]
        raise RuntimeError(
            "Failed to decode JSON from SAP response. "
            "Ensure the service returns JSON for this URL.\n"
            f"Response snippet:\n{snippet}"
        ) from exc

    # Basic structural validation
    if "value" not in data or not isinstance(data["value"], list):
        raise RuntimeError(
            "Unexpected SAP OData response format: 'value' key with a list is required.\n"
            f"Top-level keys found: {list(data.keys())}"
        )

    return data


def get_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert SAP OData 'view_stock' response into a pandas DataFrame.

    Expected structure:
        {
            "@odata.context": "...",
            "@odata.metadataEtag": "...",
            "value": [
                {"product_id": "PRODUCT001", "jahr": "2023", "monat": "1", "req_qty": 480},
                ...
            ]
        }

    Returns:
        DataFrame indexed by a datetime (first day of each month),
        with a single column 'req_qty' as float.
    """
    records = data["value"]
    df = pd.DataFrame(records)

    required_columns = {"jahr", "monat", "act_qty"}
    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing expected columns in SAP data: {', '.join(sorted(missing))}"
        )

    # Normalize types
    df["jahr"] = df["jahr"].astype(int)
    df["monat"] = df["monat"].astype(int)
    df["act_qty"] = df["act_qty"].astype(float)

    # Create a datetime index: first day of each month
    df["date"] = pd.to_datetime(
        df["jahr"].astype(str) + "-" + df["monat"].astype(str) + "-01"
    )
    df = df.set_index("date").sort_index()

    # Keep only the target series for time-series analysis
    df = df[["act_qty"]]

    return df


def create_json() -> None:
    """
    Fetch SAP OData data and store the raw JSON into 'data/output_sap.json'.
    """
    data = parse_json()
    os.makedirs("data", exist_ok=True)
    file_name = "output_sap.json"
    file_path = os.path.join("data", file_name)
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Fetch from SAP, convert to DataFrame, and print.
    sap_data = parse_json()
    df = get_df(sap_data)
    print(df)
    # Optionally also write JSON to file:
    # create_json()
