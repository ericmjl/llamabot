"""Convenience wrappers around the Google API."""
import pickle
from hashlib import sha256

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from llamabot.config import llamabot_config_dir


def load_credentials(credentials_file: str, scopes: list):
    """Load credentials for a Google API.

    :param credentials_file: Path to the credentials file.
    :param scopes: List of scopes to request access to.
    :return: Google API credentials.
    """
    creds = None

    # sha256-hash the scopes so that we can use them in the filename.
    scopes_hash = sha256(str(scopes).encode("utf-8")).hexdigest()

    # sha256-hash the contents of credentials_file so that we can use them in the filename.
    with open(credentials_file, "r+") as f:
        credentials_file_contents = f.read()
        # credentials_file_hash = sha256(credentials_file_contents)
        credentials_file_hash = sha256(
            credentials_file_contents.encode("utf-8")
        ).hexdigest()

    token_pickle_path = (
        llamabot_config_dir / f"token-{scopes_hash}-{credentials_file_hash}.pickle"
    )

    # The file token.pickle stores the user's access and refresh tokens,
    # and is created automatically when the authorization flow
    # completes for the first time.
    if token_pickle_path:
        with open(token_pickle_path, "rb") as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, scopes)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_pickle_path, "wb") as token:
            pickle.dump(creds, token)
    return creds


def get_calendar_service(credentials_file: str, scopes: list):
    """Get a Google Calendar service object.

    :param credentials_file: Path to the credentials file.
    :param scopes: List of scopes to request access to.
    :return: Google Calendar service object.
    """
    creds = load_credentials(credentials_file, scopes)
    service = build("calendar", "v3", credentials=creds)
    return service
