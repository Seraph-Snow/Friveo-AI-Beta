# =============================================================================
# app/tasks/celery_app.py — Celery App + Beat Schedule
# =============================================================================
from celery import Celery
from celery.schedules import crontab
import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "friend_agent",
    broker=REDIS_URL,
    backend=REDIS_URL.replace("/0", "/1"),
    include=["app.tasks.tasks"],  # auto-import tasks
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,

    beat_schedule={
        # Check for upcoming events every 15 minutes
        "check-event-reminders": {
            "task":     "tasks.check_event_reminders",
            "schedule": crontab(minute="*/15"),
        },
        # Write daily journals at 9pm UTC
        "write-daily-journals": {
            "task":     "tasks.write_daily_journals",
            "schedule": crontab(hour=21, minute=0),
        },
        # Compute weekly analytics every Sunday at 10pm UTC
        "compute-weekly-analytics": {
            "task":     "tasks.compute_weekly_analytics",
            "schedule": crontab(hour=22, minute=0, day_of_week=0),
        },
    },
)