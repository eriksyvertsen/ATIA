"""
Usage Analytics Dashboard for ATIA.

This module provides tools for tracking, analyzing and visualizing ATIA usage metrics.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading
import time

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class UsageEvent(BaseModel):
    """
    Represents a single usage event in the system.
    """
    event_type: str  # e.g., "query", "tool_execution", "api_search"
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: str  # Which component generated the event
    duration_ms: Optional[float] = None  # Duration in milliseconds, if applicable
    metadata: Dict[str, Any] = {}  # Additional event-specific data


class ComponentMetrics(BaseModel):
    """
    Metrics for a specific component.
    """
    component: str
    calls: int = 0
    errors: int = 0
    avg_duration_ms: float = 0
    last_duration_ms: float = 0
    last_called: Optional[datetime] = None


class UsageMetrics(BaseModel):
    """
    Overall usage metrics for the system.
    """
    total_queries: int = 0
    total_tool_executions: int = 0
    total_api_searches: int = 0
    total_doc_processes: int = 0
    total_errors: int = 0
    avg_query_duration_ms: float = 0
    avg_tool_execution_duration_ms: float = 0
    components: Dict[str, ComponentMetrics] = {}
    session_start: datetime = datetime.now()


class AnalyticsDashboard:
    """
    Dashboard for tracking and visualizing ATIA usage.
    """

    def __init__(self, storage_dir: str = "data/analytics"):
        """
        Initialize the analytics dashboard.

        Args:
            storage_dir: Directory to store analytics data
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # Create events directory
        self.events_dir = os.path.join(storage_dir, "events")
        os.makedirs(self.events_dir, exist_ok=True)

        # Current metrics
        self.metrics = UsageMetrics()

        # Latest events for real-time display
        self.recent_events: List[UsageEvent] = []
        self.max_recent_events = 100

        # Lock for thread safety
        self.lock = threading.RLock()

        # Start background archiving thread
        self._start_archiving_thread()

    def track_event(self, event: UsageEvent) -> None:
        """
        Track a usage event.

        Args:
            event: The event to track
        """
        with self.lock:
            # Add to recent events
            self.recent_events.append(event)
            if len(self.recent_events) > self.max_recent_events:
                self.recent_events.pop(0)

            # Write to event log
            self._write_event(event)

            # Update metrics
            self._update_metrics(event)

    def get_metrics(self) -> UsageMetrics:
        """
        Get current usage metrics.

        Returns:
            Current usage metrics
        """
        with self.lock:
            return self.metrics

    def get_recent_events(self, limit: int = 10) -> List[UsageEvent]:
        """
        Get recent events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        with self.lock:
            return self.recent_events[-limit:]

    def get_component_metrics(self, component: str) -> Optional[ComponentMetrics]:
        """
        Get metrics for a specific component.

        Args:
            component: Name of the component

        Returns:
            Metrics for the component, or None if not found
        """
        with self.lock:
            return self.metrics.components.get(component)

    def generate_report(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a usage report for a time period.

        Args:
            start_time: Start of the reporting period
            end_time: End of the reporting period

        Returns:
            Report data
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()

        # Load events from the specified time period
        events = self._load_events(start_time, end_time)

        # Generate report data
        report = {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "event_counts": {
                "query": 0,
                "tool_execution": 0,
                "api_search": 0,
                "doc_process": 0,
                "error": 0
            },
            "performance": {
                "avg_query_duration_ms": 0,
                "avg_tool_execution_duration_ms": 0,
                "avg_api_search_duration_ms": 0,
                "avg_doc_process_duration_ms": 0
            },
            "component_usage": {},
            "events_by_day": {}
        }

        # Count events by type
        durations = {
            "query": [],
            "tool_execution": [],
            "api_search": [],
            "doc_process": []
        }

        # Process events
        for event in events:
            # Count by type
            report["event_counts"][event.event_type] = report["event_counts"].get(event.event_type, 0) + 1

            # Track durations
            if event.duration_ms and event.event_type in durations:
                durations[event.event_type].append(event.duration_ms)

            # Track by component
            if event.component not in report["component_usage"]:
                report["component_usage"][event.component] = 0
            report["component_usage"][event.component] += 1

            # Track by day
            day = event.timestamp.date().isoformat()
            if day not in report["events_by_day"]:
                report["events_by_day"][day] = 0
            report["events_by_day"][day] += 1

        # Calculate average durations
        for event_type, duration_list in durations.items():
            if duration_list:
                report["performance"][f"avg_{event_type}_duration_ms"] = sum(duration_list) / len(duration_list)

        return report

    def _write_event(self, event: UsageEvent) -> None:
        """Write an event to storage."""
        try:
            # Generate filename based on date
            date_str = event.timestamp.strftime("%Y-%m-%d")
            events_file = os.path.join(self.events_dir, f"events_{date_str}.jsonl")

            # Write event as JSON line
            with open(events_file, "a") as f:
                f.write(event.json() + "\n")
        except Exception as e:
            logger.error(f"Error writing event: {e}")

    def _update_metrics(self, event: UsageEvent) -> None:
        """Update metrics based on an event."""
        # Update total counts
        if event.event_type == "query":
            self.metrics.total_queries += 1
            if event.duration_ms:
                total_duration = self.metrics.avg_query_duration_ms * (self.metrics.total_queries - 1)
                self.metrics.avg_query_duration_ms = (total_duration + event.duration_ms) / self.metrics.total_queries
        elif event.event_type == "tool_execution":
            self.metrics.total_tool_executions += 1
            if event.duration_ms:
                total_duration = self.metrics.avg_tool_execution_duration_ms * (self.metrics.total_tool_executions - 1)
                self.metrics.avg_tool_execution_duration_ms = (total_duration + event.duration_ms) / self.metrics.total_tool_executions
        elif event.event_type == "api_search":
            self.metrics.total_api_searches += 1
        elif event.event_type == "doc_process":
            self.metrics.total_doc_processes += 1
        elif event.event_type == "error":
            self.metrics.total_errors += 1

        # Update component metrics
        if event.component not in self.metrics.components:
            self.metrics.components[event.component] = ComponentMetrics(component=event.component)

        component_metrics = self.metrics.components[event.component]
        component_metrics.calls += 1
        component_metrics.last_called = event.timestamp

        if event.event_type == "error":
            component_metrics.errors += 1

        if event.duration_ms:
            total_duration = component_metrics.avg_duration_ms * (component_metrics.calls - 1)
            component_metrics.avg_duration_ms = (total_duration + event.duration_ms) / component_metrics.calls
            component_metrics.last_duration_ms = event.duration_ms

    def _load_events(self, start_time: datetime, end_time: datetime) -> List[UsageEvent]:
        """Load events from storage for a time period."""
        events = []

        # Generate list of dates in the range
        current_date = start_time.date()
        end_date = end_time.date()

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            events_file = os.path.join(self.events_dir, f"events_{date_str}.jsonl")

            if os.path.exists(events_file):
                try:
                    with open(events_file, "r") as f:
                        for line in f:
                            try:
                                event = UsageEvent.parse_raw(line)
                                if start_time <= event.timestamp <= end_time:
                                    events.append(event)
                            except:
                                # Skip invalid lines
                                pass
                except Exception as e:
                    logger.error(f"Error loading events from {events_file}: {e}")

            current_date += timedelta(days=1)

        return events

    def _start_archiving_thread(self) -> None:
        """Start a background thread for archiving old events."""
        def archive_worker():
            while True:
                try:
                    self._archive_old_events()
                except Exception as e:
                    logger.error(f"Error archiving events: {e}")
                # Sleep for a day
                time.sleep(86400)

        thread = threading.Thread(target=archive_worker, daemon=True)
        thread.start()

    def _archive_old_events(self, days_to_keep: int = 30) -> None:
        """Archive events older than the specified number of days."""
        cutoff_date = datetime.now().date() - timedelta(days=days_to_keep)

        # Get list of event files
        event_files = [f for f in os.listdir(self.events_dir) if f.startswith("events_")]

        for filename in event_files:
            try:
                # Extract date from filename
                date_str = filename.replace("events_", "").replace(".jsonl", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()

                # If file is older than cutoff, archive it
                if file_date < cutoff_date:
                    # Create archives directory if it doesn't exist
                    archives_dir = os.path.join(self.storage_dir, "archives")
                    os.makedirs(archives_dir, exist_ok=True)

                    # Move file to archives
                    source_path = os.path.join(self.events_dir, filename)
                    dest_path = os.path.join(archives_dir, filename)
                    os.rename(source_path, dest_path)
                    logger.info(f"Archived event file: {filename}")
            except Exception as e:
                logger.error(f"Error archiving file {filename}: {e}")


# Create a dashboard instance
dashboard = AnalyticsDashboard()


def track_agent_query(query: str, duration_ms: float, session_id: Optional[str] = None, 
                    user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Track an agent query event.

    Args:
        query: The query text
        duration_ms: Query processing duration in milliseconds
        session_id: Optional session ID
        user_id: Optional user ID
        metadata: Additional metadata
    """
    if metadata is None:
        metadata = {}

    # Truncate query if too long
    if len(query) > 100:
        query_text = query[:97] + "..."
    else:
        query_text = query

    metadata["query"] = query_text

    event = UsageEvent(
        event_type="query",
        timestamp=datetime.now(),
        user_id=user_id,
        session_id=session_id,
        component="agent_core",
        duration_ms=duration_ms,
        metadata=metadata
    )

    dashboard.track_event(event)


def track_tool_execution(tool_name: str, duration_ms: float, success: bool, 
                       session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Track a tool execution event.

    Args:
        tool_name: Name of the tool executed
        duration_ms: Execution duration in milliseconds
        success: Whether execution was successful
        session_id: Optional session ID
        metadata: Additional metadata
    """
    if metadata is None:
        metadata = {}

    metadata["tool_name"] = tool_name
    metadata["success"] = success

    event = UsageEvent(
        event_type="tool_execution",
        timestamp=datetime.now(),
        session_id=session_id,
        component="tool_executor",
        duration_ms=duration_ms,
        metadata=metadata
    )

    dashboard.track_event(event)


def track_api_search(query: str, num_results: int, duration_ms: float,
                   session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Track an API search event.

    Args:
        query: Search query
        num_results: Number of results found
        duration_ms: Search duration in milliseconds
        session_id: Optional session ID
        metadata: Additional metadata
    """
    if metadata is None:
        metadata = {}

    # Truncate query if too long
    if len(query) > 100:
        query_text = query[:97] + "..."
    else:
        query_text = query

    metadata["query"] = query_text
    metadata["num_results"] = num_results

    event = UsageEvent(
        event_type="api_search",
        timestamp=datetime.now(),
        session_id=session_id,
        component="api_discovery",
        duration_ms=duration_ms,
        metadata=metadata
    )

    dashboard.track_event(event)


def track_doc_processing(url: str, doc_type: str, duration_ms: float, 
                       success: bool, session_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Track a documentation processing event.

    Args:
        url: Documentation URL
        doc_type: Type of documentation
        duration_ms: Processing duration in milliseconds
        success: Whether processing was successful
        session_id: Optional session ID
        metadata: Additional metadata
    """
    if metadata is None:
        metadata = {}

    metadata["url"] = url
    metadata["doc_type"] = doc_type
    metadata["success"] = success

    event = UsageEvent(
        event_type="doc_process",
        timestamp=datetime.now(),
        session_id=session_id,
        component="doc_processor",
        duration_ms=duration_ms,
        metadata=metadata
    )

    dashboard.track_event(event)


def track_error(component: str, error_type: str, error_message: str,
              session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Track an error event.

    Args:
        component: Component where the error occurred
        error_type: Type of error
        error_message: Error message
        session_id: Optional session ID
        metadata: Additional metadata
    """
    if metadata is None:
        metadata = {}

    metadata["error_type"] = error_type
    metadata["error_message"] = error_message

    event = UsageEvent(
        event_type="error",
        timestamp=datetime.now(),
        session_id=session_id,
        component=component,
        metadata=metadata
    )

    dashboard.track_event(event)


def generate_dashboard_html() -> str:
    """
    Generate HTML for a simple dashboard.

    Returns:
        HTML for dashboard
    """
    metrics = dashboard.get_metrics()
    recent_events = dashboard.get_recent_events(10)

    # Calculate uptime
    uptime = datetime.now() - metrics.session_start
    uptime_hours = uptime.total_seconds() / 3600

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ATIA Analytics Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 20px; }}
            .metrics {{ display: flex; flex-wrap: wrap; }}
            .metric {{ flex: 1 0 200px; margin: 10px; text-align: center; }}
            .metric h3 {{ margin-bottom: 5px; }}
            .metric p {{ font-size: 1.5em; font-weight: bold; margin: 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>ATIA Analytics Dashboard</h1>

        <div class="card">
            <h2>Overall Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Total Queries</h3>
                    <p>{metrics.total_queries}</p>
                </div>
                <div class="metric">
                    <h3>Tool Executions</h3>
                    <p>{metrics.total_tool_executions}</p>
                </div>
                <div class="metric">
                    <h3>API Searches</h3>
                    <p>{metrics.total_api_searches}</p>
                </div>
                <div class="metric">
                    <h3>Doc Processes</h3>
                    <p>{metrics.total_doc_processes}</p>
                </div>
                <div class="metric">
                    <h3>Errors</h3>
                    <p>{metrics.total_errors}</p>
                </div>
                <div class="metric">
                    <h3>Uptime (hours)</h3>
                    <p>{uptime_hours:.2f}</p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Performance Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Avg Query Time (ms)</h3>
                    <p>{metrics.avg_query_duration_ms:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Avg Tool Execution (ms)</h3>
                    <p>{metrics.avg_tool_execution_duration_ms:.2f}</p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Recent Events</h2>
            <table>
                <tr>
                    <th>Time</th>
                    <th>Type</th>
                    <th>Component</th>
                    <th>Duration (ms)</th>
                    <th>Details</th>
                </tr>
    """

    for event in reversed(recent_events):
        time_str = event.timestamp.strftime("%H:%M:%S")
        duration = f"{event.duration_ms:.2f}" if event.duration_ms else "-"

        # Create details string from metadata
        details = []
        for key, value in event.metadata.items():
            if isinstance(value, str) and len(value) > 30:
                value = value[:27] + "..."
            details.append(f"{key}: {value}")
        details_str = ", ".join(details)

        html += f"""
                <tr>
                    <td>{time_str}</td>
                    <td>{event.event_type}</td>
                    <td>{event.component}</td>
                    <td>{duration}</td>
                    <td>{details_str}</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <div class="card">
            <h2>Component Usage</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Calls</th>
                    <th>Errors</th>
                    <th>Last Called</th>
                    <th>Avg Duration (ms)</th>
                </tr>
    """

    for component_name, metrics in metrics.components.items():
        last_called = metrics.last_called.strftime("%Y-%m-%d %H:%M:%S") if metrics.last_called else "-"

        html += f"""
                <tr>
                    <td>{component_name}</td>
                    <td>{metrics.calls}</td>
                    <td>{metrics.errors}</td>
                    <td>{last_called}</td>
                    <td>{metrics.avg_duration_ms:.2f}</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <script>
            // Auto-refresh every 30 seconds
            setTimeout(function() {
                location.reload();
            }, 30000);
        </script>
    </body>
    </html>
    """

    return html


def save_dashboard_html(output_path: str = "data/analytics/dashboard.html") -> None:
    """
    Save the dashboard HTML to a file.

    Args:
        output_path: Path to save the dashboard HTML
    """
    html = generate_dashboard_html()

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write HTML to file
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Dashboard saved to {output_path}")


# Auto-update dashboard HTML periodically
def _start_dashboard_update_thread() -> None:
    """Start a background thread to periodically update the dashboard HTML."""
    def update_worker():
        while True:
            try:
                save_dashboard_html()
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
            # Update every 5 minutes
            time.sleep(300)

    thread = threading.Thread(target=update_worker, daemon=True)
    thread.start()

# Start the dashboard update thread
_start_dashboard_update_thread()