import requests

# licor dataChannel, region_id
DATA_CHANNELS = {
    ("0942b00d-ca65-460d-9b34-8ca6f3912883", 25),  # Devereux Slough
    ("d7218efc-fea9-40f3-8789-ca39e78c0f1d", 51),  # Pajaro Lagoon
    ("f2ed8c17-707b-40ee-8002-1f44498a19c1", 2138),  # Pescadero Lagoon
    ("1908e69d-3637-41ff-a2b0-2a6fb679b8de", 13057),  # San Gregorio Creek
    ("d0cd3768-c0ea-45c8-b144-bb552fb67db6", 20),  # Santa Clara River
    ("cba5a28f-af65-4413-8a44-cc82963170ba", 21),  # Ventura River
    ("4aa09afb-a9b6-4327-a95f-d6c4bcaeeced", 53),  # Younger Lagoon
}

url = "https://www.licor.cloud/api/dashboard/public/query"

# Headers from the original request, sensitive ones are generalized or removed.
headers = {
    "content-type": "application/json",
}

payload = {
    "id": "cd5b1933-9022-44e4-a3c8-f4cfcc2a3433",
    "query": {
        "limit": 10000,
        "metrics": [
            {
                "aggregators": [
                    {
                        "name": "avg",
                        "align_start_time": False,
                        "sampling": {"value": 10, "unit": "minutes"},
                    }
                ],
                "name": "com.onset.sensordata.waterlevel_si",
                "exclude_tags": True,
                "group_by": [],
                "tags": {"dataChannel": ["0942b00d-ca65-460d-9b34-8ca6f3912883"]},
            }
        ],
        "start_absolute": 1764590426563,
        "end_absolute": 1767268826563,
    },
}

for d, _ in DATA_CHANNELS:
    payload["query"]["metrics"][0]["tags"]["dataChannel"][0] = d

    # Make the POST request
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        print(d)
        import pdb; pdb.set_trace()
        print(response.json()["queries"][0]["results"][0]["values"])

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
