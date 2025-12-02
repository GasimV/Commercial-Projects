import requests, json, os, sys, time

# Simple helper to call the FastAPI /train after ensuring the data exists.
print("Ensure synthetic data exists (run scripts/generate_synthetic_data.py if needed).")

# Start local server tip:
print("If the API server is not running yet, start it with:")
print("  python scripts/serve.py")
