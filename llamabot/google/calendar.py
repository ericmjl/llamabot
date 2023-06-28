"""Google Calendar API."""
import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pprint import pprint
from typing import Callable

from llamabot.google.base import GoogleApi

from .utility_functions import capture_errors

API_NAME = "calendar"
API_VERSION = "v3"


@dataclass
class GoogleCalendar(GoogleApi):
    """Google Calendar API."""

    calendar_id: str = field(default=None)

    def __post_init__(self):
        """Post-initialization hook."""
        if self.api_name is None:
            self.api_name = API_NAME
        if self.api_version is None:
            self.api_version = API_VERSION

        if not self.scopes:
            self.scopes = ["https://www.googleapis.com/auth/calendar"]

        super().__post_init__()

    @capture_errors
    def get_events_within(
        self, time_delta: timedelta, filter_funcs: list[Callable] = [lambda x: True]
    ):
        """Get events within a given number of weeks from today.

        :param time_delta: Amount of time to look ahead.
        :param filter_funcs: List of functions to filter events by.
            Each function should have a signature of `func(event) -> bool`.
        :return: List of events.
        """
        # Get the current time
        start_time = datetime.utcnow()

        # Calculate the end time based on the specified number of weeks
        end_time = start_time + time_delta

        # Swap the start and end times if the start time is later than the end time
        if start_time > end_time:
            start_time, end_time = end_time, start_time

        # Convert the time to the required format for the API
        start_time_str = start_time.isoformat() + "Z"
        end_time_str = end_time.isoformat() + "Z"

        # Make the API request to retrieve events for the specified user email
        events_result = (
            self.service.events()
            .list(
                calendarId=self.calendar_id,
                timeMin=start_time_str,
                timeMax=end_time_str,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        # Get the list of events from the response
        events = events_result.get("items", [])

        if not events:
            print("No events found.")
            return []

        # Filter events
        for filter_func in filter_funcs:
            events = list(filter(filter_func, events))

        return [Event(event=event, calendar=self) for event in events]

    def create_event(
        self,
        summary: str,
        start_time: dict,
        end_time: dict,
        description: str,
        attendees: list[str] = [],
    ) -> "Event":
        """Create an event.

        :param summary: Summary of the event. Also known as the title.
        :param start_time: Start time of the event.
            Should have 'dateTime' and 'timeZone' as keys.
        :param end_time: End time of the event.
            Should have 'dateTime' and 'timeZone' as keys.
        :param description: Description of the event.
        :param attendees: List of attendees' email addresses.
        :return: The created event.
        :raises TypeError: if start_time or end_time is not a dictionary with 'dateTime' and
            'timeZone' keys.
        :raises KeyError: if start_time or end_time does not have 'dateTime' or 'timeZone'
            keys.
        """
        # Ensure that start_time and end_time are dictionaries with 'dateTime' and 'timeZone' keys.
        if not isinstance(start_time, dict) or not isinstance(end_time, dict):
            raise TypeError(
                "start_time and end_time must be dictionaries with 'dateTime' and 'timeZone' keys."
            )

        # Ensure that start_time and end_time have 'dateTime' and 'timeZone' keys.
        if "dateTime" not in start_time or "dateTime" not in end_time:
            raise KeyError(
                "start_time and end_time must have 'dateTime' and 'timeZone' keys."
            )

        event = {
            "summary": summary,
            "description": description,
            "start": start_time,
            "end": end_time,
            "attendees": [{"email": attendee} for attendee in attendees],
        }
        event = (
            self.service.events()
            .insert(calendarId=self.calendar_id, body=event, sendUpdates="all")
            .execute()
        )
        return Event(event, self)


# I need a dataclass that wraps an Event and dynamically adds properties based on the top-level dictionary keys.
@dataclass
class Event:
    """Wrapper around a Google Calendar event."""

    event: dict = field(default_factory=dict)
    calendar: GoogleCalendar = field(default=None)

    def __post_init__(self):
        """Post-initialization hook."""
        self.sync_attr()

    def __repr__(self) -> str:
        """Return a string representation of the event.

        :return: String representation of the event.
        """
        output = io.StringIO()
        pprint(vars(self), stream=output)
        ret = "Event(\n" + output.getvalue() + "\n)"
        return ret

    def __getitem__(self, key):
        """Get an attribute from the event.

        :param key: Attribute to get.
        :return: Attribute value.
        """
        return self.event[key]

    def sync_attr(self):
        """Sync the event attributes with the top-level dictionary keys."""
        for k, v in self.event.items():
            setattr(self, k, v)

    def update(self):
        """Update the event."""
        self.event = (
            self.calendar.service.events()
            .update(
                calendarId=self.calendar.calendar_id,
                eventId=self.id,
                body=self.event,
                sendUpdates="all",
            )
            .execute()
        )
        self.sync_attr()

    def invite(self, email: str):
        """Add an attendee to the event.

        :param email: Email address of the attendee.
        """
        if "attendees" not in self.event:
            self.event["attendees"] = []
        self.event["attendees"].append({"email": email})
        self.update()

    def uninvite(self, email: str):
        """Remove an attendee from the event.

        :param email: Email address of the attendee.
        """
        self.event["attendees"] = [a for a in self.attendees if a["email"] != email]
        self.update()

    def set_summary(self, summary: str):
        """Set the summary of the event.

        :param summary: Summary of the event.
        """
        self.event["summary"] = summary
        self.update()

    def set_description(self, description: str):
        """Set the description of the event.

        :param description: Description of the event.
        """
        self.event["description"] = description
        self.update()

    def delete(self):
        """Delete the event."""
        self.calendar.service.events().delete(
            calendarId=self.calendar.calendar_id, eventId=self.id
        ).execute()
