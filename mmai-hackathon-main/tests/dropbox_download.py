"""Download a Dropbox folder for CI integration tests.

This small CLI downloads a Dropbox folder as a zip to ``<local_folder>.zip`` and
optionally extracts it beside the destination folder, removing the zip afterwards.

Usage examples
--------------
- Using environment secret (preferred in CI):
  ``python -m tests.dropbox_download "/MMAI25Hackathon" "MMAI25Hackathon" --unzip``
  where ``DROPBOX_ACCESS_TOKEN`` (or ``DROPBOX_TOKEN``) is set in the environment.
- Providing the token explicitly:
  ``python -m tests.dropbox_download "/remote/path" "local_dir" --access_token <TOKEN> --unzip``
"""

import argparse
import os
import zipfile

import dropbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a folder from Dropbox.")
    parser.add_argument("dropbox_folder", type=str, help="Path to the Dropbox folder to download.")
    parser.add_argument("local_folder", type=str, help="Local directory to save downloaded files.")
    parser.add_argument("--unzip", action="store_true", help="Unzip and remove the zip afterwards.")

    # Auth via refresh token (recommended for headless/CI)
    parser.add_argument(
        "--app-key",
        dest="app_key",
        default=os.getenv("DROPBOX_APP_KEY"),
        help="Dropbox app key (or set env DROPBOX_APP_KEY).",
    )
    parser.add_argument(
        "--app-secret",
        dest="app_secret",
        default=os.getenv("DROPBOX_APP_SECRET"),
        help="Dropbox app secret (omit if your refresh token was issued via PKCE).",
    )
    parser.add_argument(
        "--refresh-token",
        dest="refresh_token",
        default=os.getenv("DROPBOX_REFRESH_TOKEN"),
        help="Dropbox OAuth2 refresh token (or set env DROPBOX_REFRESH_TOKEN).",
    )

    args = parser.parse_args()

    # Basic validation
    if not args.app_key:
        parser.error("Missing --app-key (or env DROPBOX_APP_KEY).")
    if not args.refresh_token:
        parser.error("Missing --refresh-token (or env DROPBOX_REFRESH_TOKEN).")
    # app_secret is optional if your refresh token was created with PKCE

    # Create the Dropbox client using refresh token auth
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=args.refresh_token,
        app_key=args.app_key,
        app_secret=args.app_secret,  # may be None if using PKCE
    )

    # Ensure local folder exists
    os.makedirs(args.local_folder, exist_ok=True)
    zip_path = args.local_folder + ".zip"
    dbx.files_download_zip_to_file(zip_path, args.dropbox_folder)

    if args.unzip:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(args.local_folder, ".."))

        print(f"Unzipped to {args.local_folder}")
        os.remove(zip_path)
        print(f"Removed zip file {zip_path}")
