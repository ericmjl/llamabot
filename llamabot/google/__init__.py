"""Convenience wrappers around the Google API."""
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from llamabot.config import llamabot_config_dir


def get_calendar_service(credentials_file: str, scopes: list):
    """Get a Google Calendar service object.

    :param credentials_file: Path to the credentials file.
    :param scopes: List of scopes to request access to.
    :return: Google Calendar service object.
    """
    creds = None

    # The file token.pickle stores the user's access and refresh tokens,
    # and is created automatically when the authorization flow
    # completes for the first time.
    token_pickle_path = llamabot_config_dir / "token.pickle"
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

    service = build("calendar", "v3", credentials=creds)
    return service
