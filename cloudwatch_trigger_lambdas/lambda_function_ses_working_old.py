import os
import json
import boto3
import logging
from datetime import datetime, timedelta, timezone
import urllib.parse
import urllib3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.getenv("AWS_REGION", "us-west-2")
ANALYSIS_ENDPOINT = os.getenv(
    "ANALYSIS_ENDPOINT",
    "http://langgraph-alb-1133416885.us-west-2.elb.amazonaws.com/analyze"
)
# --- SES Configuration ---
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "aditya.d@cloudworkmates.com") # e.g., "analysis-noreply@yourdomain.com"
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "aditya.d@cloudworkmates.com") # e.g., "your-team@yourdomain.com"
# --- End SES Configuration ---

HTTP = urllib3.PoolManager()

# clients (explicit region)
CW = boto3.client("cloudwatch", region_name=REGION)
S3 = boto3.client("s3", region_name=REGION) # Keep if needed
# SNS = boto3.client("sns", region_name=REGION) # No longer needed for email
SES = boto3.client("ses", region_name=REGION) # <-- ADDED: SES Client

# --- ts_iso_z, safe_key_component functions remain the same ---
def ts_iso_z(dt: datetime) -> str:
    """Return ISO-style UTC timestamp without +00:00 (Z)."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds') + "Z"

def safe_key_component(s: str) -> str:
    """Make an S3-safe key component: remove colons, spaces -> underscores, and url-quote."""
    if not s:
        return "unknown"
    s2 = s.replace(":", "").replace(" ", "_").replace("/", "_")
    return urllib.parse.quote_plus(s2)

# --- send_to_analysis_endpoint function remains the same ---
def send_to_analysis_endpoint(json_payload_string: str, filename: str):
    """
    Sends the JSON payload string as a multipart/form-data file to the analysis endpoint.
    """
    logger.info("Sending payload to endpoint: %s as file '%s'", ANALYSIS_ENDPOINT, filename)

    fields = {'file': (filename, json_payload_string, 'application/json')}

    try:
        r = HTTP.request(
            'POST',
            ANALYSIS_ENDPOINT,
            fields=fields,
            headers={'Accept': 'application/json'}
        )

        response_body = r.data.decode('utf-8')
        status_code = r.status
        logger.info("Endpoint response status: %d", status_code)

        analysis_result = {
            "status_code": status_code,
            "response_body_raw": response_body,
            "request_filename": filename
        }

        if status_code == 200:
             try:
                 analysis_result["response_body_json"] = json.loads(response_body)
             except json.JSONDecodeError:
                 logger.warning("Endpoint response was status 200 but not valid JSON.")
                 analysis_result["error"] = "json_decode_error"
                 analysis_result["message"] = "Endpoint returned 200 but body was not valid JSON."
        elif status_code >= 400:
             analysis_result["error"] = "http_error"
             analysis_result["message"] = f"Endpoint returned status {status_code}"


        return analysis_result

    except Exception as e:
        logger.exception("Error sending data to endpoint: %s", e)
        return {
            "error": "http_request_failed",
            "message": str(e),
            "request_filename": filename
        }

# --- REPLACED send_sns_notification with send_ses_email ---
def send_ses_email(analysis_data: dict, instance_id: str, region: str, alarm_name: str):
    """
    Formats and sends the analysis result as an HTML email using SES.
    """
    if not SENDER_EMAIL or not RECIPIENT_EMAIL:
        logger.error("SENDER_EMAIL or RECIPIENT_EMAIL environment variables not set. Cannot send email.")
        return {"status": "error", "message": "Sender/Recipient email not configured"}

    subject = f"CPU Alarm Analysis: Instance {instance_id}"
    if alarm_name:
        subject += f" ({alarm_name})"

    html_body_lines = ["<html><body>"]
    text_body_lines = [] # Plain text version for compatibility

    try:
        # Check for analysis errors first
        if "error" in analysis_data:
            subject = f"[ERROR] Analyzing CPU Alarm for {instance_id}"
            html_body_lines.append(f"<h1>Analysis Error for Instance {instance_id}</h1>")
            html_body_lines.append(f"<p><b>Instance:</b> {instance_id} in {region}</p>")
            html_body_lines.append(f"<p><b>Alarm Name:</b> {alarm_name or 'N/A'}</p>")
            html_body_lines.append(f"<p><b>Error Type:</b> {analysis_data.get('error', 'Unknown')}</p>")
            html_body_lines.append(f"<p><b>Error Message:</b> {analysis_data.get('message', 'N/A')}</p>")
            if "request_filename" in analysis_data:
                html_body_lines.append(f"<p><b>Request Filename:</b> {analysis_data['request_filename']}</p>")

            # Simple text version
            text_body_lines.append(f"Analysis failed for instance {instance_id} in {region}.")
            text_body_lines.append(f"Alarm Name: {alarm_name or 'N/A'}")
            text_body_lines.append(f"Error Type: {analysis_data.get('error', 'Unknown')}")
            text_body_lines.append(f"Error Message: {analysis_data.get('message', 'N/A')}")

        # Process successful analysis
        elif "response_body_json" in analysis_data:
            response_json = analysis_data["response_body_json"]

            severity = response_json.get("severity", "UNKNOWN")
            subject = f"[{severity}] {subject}"

            html_body_lines.append(f"<h1>CPU Alarm Analysis Report - {severity}</h1>")
            html_body_lines.append(f"<p><b>Instance:</b> {instance_id} in {region}</p>")
            html_body_lines.append(f"<p><b>Alarm Name:</b> {alarm_name or 'N/A'}</p>")
            html_body_lines.append("<hr>")

            text_body_lines.append(f"Severity: {severity}")
            text_body_lines.append(f"Instance: {instance_id} in {region}")
            text_body_lines.append(f"Alarm Name: {alarm_name or 'N/A'}")
            text_body_lines.append("-" * 20)

            html_body_lines.append("<h2>Summary</h2>")
            html_body_lines.append(f"<p>{response_json.get('summary', 'No summary provided.')}</p>")
            html_body_lines.append("<hr>")
            text_body_lines.append("Summary:")
            text_body_lines.append(response_json.get('summary', 'No summary provided.'))
            text_body_lines.append("-" * 20)

            html_body_lines.append("<h2>Advice</h2>")
            html_body_lines.append(f"<p>{response_json.get('advice', 'No advice provided.')}</p>")
            html_body_lines.append("<hr>")
            text_body_lines.append("Advice:")
            text_body_lines.append(response_json.get('advice', 'No advice provided.'))
            text_body_lines.append("-" * 20)

            # Raw Findings (formatted JSON in HTML)
            raw_findings = response_json.get("raw_findings", {})
            if raw_findings:
                html_body_lines.append("<h2>Key Findings (Raw)</h2>")
                html_body_lines.append("<pre><code>")
                html_body_lines.append(json.dumps(raw_findings, indent=2))
                html_body_lines.append("</code></pre>")
                html_body_lines.append("<hr>")
                # Simple text version
                text_body_lines.append("Key Findings:")
                peak_cpu = raw_findings.get('peak_cpu_value')
                peak_ts = raw_findings.get('peak_timestamp')
                baseline = raw_findings.get('baseline_cpu')
                pattern = raw_findings.get('spike_pattern')
                if peak_cpu is not None and peak_ts:
                     text_body_lines.append(f"- Peak CPU: {peak_cpu:.2f}% at {peak_ts}")
                if baseline is not None:
                     text_body_lines.append(f"- Baseline CPU: {baseline:.2f}%")
                if pattern:
                     text_body_lines.append(f"- Spike Pattern: {pattern}")
                text_body_lines.append("-" * 20)


            # Recommendations
            recommendations = response_json.get("recommendations", [])
            if recommendations:
                html_body_lines.append("<h2>Recommendations</h2><ul>")
                text_body_lines.append("Recommendations:")
                for rec in recommendations:
                    html_body_lines.append(f"<li><b>{rec.get('title', 'Recommendation')}</b> (Priority: {rec.get('priority', 'N/A')}, Effort: {rec.get('effort', 'N/A')})<br/><i>What:</i> {rec.get('what', 'N/A')}<br/><i>Why:</i> {rec.get('why', 'N/A')}</li>")
                    text_body_lines.append(f"- {rec.get('title', 'Recommendation')} (P: {rec.get('priority', 'N/A')})")
                html_body_lines.append("</ul><hr>")
                text_body_lines.append("-" * 20)

            # Diagnostics
            diagnostics = response_json.get("diagnostics", [])
            if diagnostics:
                html_body_lines.append("<h2>Diagnostic Commands</h2><ul>")
                text_body_lines.append("Diagnostic Commands:")
                for cmd in diagnostics:
                    html_body_lines.append(f"<li><code>{cmd}</code></li>")
                    text_body_lines.append(f"- {cmd}")
                html_body_lines.append("</ul><hr>")
                text_body_lines.append("-" * 20)

            # Plots (Embedded Images)
            plots = response_json.get("plots", [])
            if plots:
                html_body_lines.append("<h2>Plots</h2>")
                text_body_lines.append("Plots: (See HTML email for images)")
                for plot in plots:
                    data_uri = plot.get("data_uri")
                    name = plot.get("name", "Plot")
                    if data_uri:
                        # Add some basic styling for responsiveness
                        html_body_lines.append(f"<h3>{name}</h3><img src='{data_uri}' alt='{name}' style='max-width: 100%; height: auto;'><br/><br/>")
                html_body_lines.append("<hr>")

        else:
             subject = f"[ERROR] Analyzing CPU Alarm for {instance_id}"
             html_body_lines.append(f"<h1>Analysis Error for Instance {instance_id}</h1>")
             html_body_lines.append(f"<p>Analysis endpoint response for {instance_id} was unparseable or incomplete.</p>")
             html_body_lines.append(f"<p>Status Code: {analysis_data.get('status_code', 'N/A')}</p>")
             html_body_lines.append("<p>Raw Body Snippet:</p><pre>")
             html_body_lines.append(analysis_data.get('response_body_raw', '')[:500]) # Limit snippet
             html_body_lines.append("</pre>")

             text_body_lines.append(f"Analysis endpoint response for {instance_id} was unparseable or incomplete.")
             text_body_lines.append(f"Status Code: {analysis_data.get('status_code', 'N/A')}")
             text_body_lines.append(f"Raw Body Snippet: {analysis_data.get('response_body_raw', '')[:200]}")


        html_body_lines.append("</body></html>")
        html_body = "\n".join(html_body_lines)
        text_body = "\n".join(text_body_lines)

        # Send using SES
        logger.info("Sending email via SES from %s to %s", SENDER_EMAIL, RECIPIENT_EMAIL)
        response = SES.send_email(
            Source=SENDER_EMAIL,
            Destination={'ToAddresses': [RECIPIENT_EMAIL]}, # Can be a list
            Message={
                'Subject': {
                    'Data': subject,
                    'Charset': 'UTF-8'
                },
                'Body': {
                    'Html': {
                        'Data': html_body,
                        'Charset': 'UTF-8'
                    },
                    'Text': {
                        'Data': text_body,
                        'Charset': 'UTF-8'
                    }
                }
            }
        )
        logger.info("SES SendEmail Response: %s", response)
        return {"status": "success", "message_id": response.get("MessageId")}

    except Exception as e:
        logger.exception("Error formatting or sending SES email: %s", e)
        # Attempt to send a fallback error email if possible
        try:
            fallback_subject = f"[ERROR] Failed to Send Analysis Email for {instance_id}"
            fallback_body = (
                f"An error occurred while trying to send the analysis email for instance {instance_id}.\n"
                f"Error: {str(e)}\n\n"
                f"Original Analysis Status Code: {analysis_data.get('status_code', 'N/A')}"
            )
            SES.send_email(
                Source=SENDER_EMAIL,
                Destination={'ToAddresses': [RECIPIENT_EMAIL]},
                Message={
                    'Subject': {'Data': fallback_subject, 'Charset': 'UTF-8'},
                    'Body': {'Text': {'Data': fallback_body, 'Charset': 'UTF-8'}}
                }
            )
            return {"status": "error", "message": f"Failed to send full email, sent fallback text error. Error: {str(e)}"}
        except Exception as fallback_e:
            logger.exception("Failed to send even fallback SES email: %s", fallback_e)
            return {"status": "error", "message": f"Failed to send email: {str(e)}"}


# --- parse_alarm_from_event and fetch_metric_data functions remain the same ---
def parse_alarm_from_event(event: dict):
    # ... (same as before) ...
    alarm_name = None
    instance_id = None
    namespace = "AWS/EC2"
    metric_name = "CPUUtilization"
    period_seconds = 300 # Default to 5 minutes if not found

    try:
        if "Records" in event and event["Records"]:
            rec = event["Records"][0]
            sns = rec.get("Sns", {})
            msg = sns.get("Message")
            if msg:
                # sometimes Message is a JSON string describing the alarm
                try:
                    msg_obj = json.loads(msg)
                except Exception:
                    msg_obj = None

                if isinstance(msg_obj, dict):
                    alarm_name = msg_obj.get("AlarmName") or msg_obj.get("alarmName") or alarm_name
                    # CloudWatch alarm message may contain trigger->dimensions
                    trigger = msg_obj.get("Trigger") or msg_obj.get("trigger")
                    if trigger:
                        metric_name = trigger.get("MetricName") or metric_name
                        namespace = trigger.get("Namespace") or namespace
                        period_seconds = trigger.get("Period") or period_seconds
                        dims = trigger.get("Dimensions") or []
                        for d in dims:
                            if d.get("name", d.get("Name", "")).lower() == "instanceid":
                                instance_id = d.get("value", d.get("Value"))
                else:
                    # fallback: plain text message
                    alarm_name = alarm_name or sns.get("Subject") or sns.get("MessageId")
        # event may be direct invoke with useful keys
        alarm_name = alarm_name or event.get("alarm_name") or event.get("AlarmName")
        instance_id = instance_id or event.get("instance_id")
        namespace = event.get("namespace", namespace)
        metric_name = event.get("metric_name", metric_name)
        period_seconds = int(event.get("period_seconds", period_seconds))
    except Exception as e:
        logger.warning("Failed to parse event for alarm info: %s", e)

    return alarm_name, instance_id, namespace, metric_name, period_seconds

def fetch_metric_data(namespace: str, metric_name: str, dimensions: list, period_seconds: int, lookback_minutes: int = 30):
    # ... (same as before) ...
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=lookback_minutes)

    # MetricDataQueries form
    query = [{
        "Id": "m1",
        "MetricStat": {
            "Metric": {"Namespace": namespace, "MetricName": metric_name, "Dimensions": dimensions or []},
            "Period": period_seconds,
            "Stat": "Average",
        },
        "ReturnData": True
    }]

    resp = CW.get_metric_data(MetricDataQueries=query, StartTime=start_time, EndTime=end_time)
    # the response contains Results with Timestamps and Values
    datapoints = []
    results = resp.get("MetricDataResults", [])
    if results:
        r = results[0]
        timestamps = r.get("Timestamps", [])
        values = r.get("Values", [])
        # pair and sort by timestamp ascending
        pairs = sorted(zip(timestamps, values), key=lambda x: x[0])
        for ts, val in pairs:
            # ensure ts in UTC and in ISO Z format
            if isinstance(ts, datetime):
                ts_str = ts.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds') + "Z"
            else:
                ts_str = str(ts) # Fallback if not datetime
            datapoints.append({"timestamp": ts_str, "value": val})
    return datapoints, {"start_iso": ts_iso_z(start_time), "end_iso": ts_iso_z(end_time)}

def lambda_handler(event, context):
    logger.info("Received event: %s", json.dumps(event)[:1000])
    # parse inputs
    alarm_name, instance_id, namespace, metric_name, period_seconds = parse_alarm_from_event(event)

    instance_id = instance_id or event.get("instance_id")

    if not instance_id:
         logger.error("Could not determine InstanceId from event.")
         return {
             "statusCode": 400,
             "body": json.dumps({"error": "InstanceId not found in event payload."})
         }

    # --- Rest of the metric fetching and payload creation is the same ---
    dimensions = [{"Name": "InstanceId", "Value": instance_id}]
    lookback_periods_minutes = [5, 10, 15, 30, 60, 120, 180, 360]
    all_metric_data = {}
    for lookback in lookback_periods_minutes:
        logger.info("Fetching metric data for last %d minutes...", lookback)
        range_key = f"last_{lookback}_min"
        try:
            datapoints, time_range = fetch_metric_data(
                namespace, metric_name, dimensions, int(period_seconds), lookback
            )
            all_metric_data[range_key] = {
                "lookback_minutes": lookback,
                "datapoints": datapoints,
                "query_time_range": time_range
            }
        except Exception as e:
            logger.exception("Error fetching metric data for %d min lookback: %s", lookback, e)
            all_metric_data[range_key] = {
                "lookback_minutes": lookback,
                "error": "get_metric_data_failed", "message": str(e)
            }

    alarm_meta = {}
    if alarm_name:
        try:
            alarm_resp = CW.describe_alarms(AlarmNames=[alarm_name])
            metric_alarms = alarm_resp.get("MetricAlarms", [])
            if metric_alarms:
                a = metric_alarms[0]
                alarm_meta = {
                    "alarm_name": a.get("AlarmName"), "alarm_arn": a.get("AlarmArn"),
                    "state": a.get("StateValue"), "state_reason": a.get("StateReason"),
                    "threshold": a.get("Threshold"), "comparison_operator": a.get("ComparisonOperator"),
                    "evaluation_periods": a.get("EvaluationPeriods"), "period_seconds": a.get("Period"),
                    "namespace": a.get("Namespace"), "metric_name": a.get("MetricName"),
                    "dimensions": a.get("Dimensions"), "actions_enabled": a.get("ActionsEnabled"),
                    "alarm_actions": a.get("AlarmActions"), "insufficient_data_actions": a.get("InsufficientDataActions"),
                    "ok_actions": a.get("OKActions")
                }
                period_seconds = alarm_meta.get("period_seconds") or period_seconds
                namespace = alarm_meta.get("namespace") or namespace
                metric_name = alarm_meta.get("metric_name") or metric_name
        except Exception as e:
            logger.warning("Failed to describe alarm '%s': %s", alarm_name, e)

    payload = {
        "instance_id": instance_id, "region": REGION,
        "metric": {
            "namespace": namespace, "metric_name": metric_name, "statistic": "Average",
            "unit": event.get("unit", "Percent"), "period_seconds": int(period_seconds)
        },
        "metric_data_by_range": all_metric_data, "alarm": alarm_meta,
        "generated_at": ts_iso_z(datetime.now(timezone.utc))
    }

    body_json = json.dumps(payload, default=str, indent=2)
    now_for_filename = datetime.now(timezone.utc)
    filename_ts = now_for_filename.strftime("%Y%m%dT%H%M%SZ")
    alarm_comp = safe_key_component(alarm_name or instance_id)
    filename = f"{alarm_comp}_{filename_ts}.json"

    # --- MODIFICATION: Send to Endpoint & then Email via SES ---

    # 1. Send to analysis endpoint
    analysis_result = send_to_analysis_endpoint(body_json, filename)

    # 2. Send SES email based on analysis result
    # Only attempt to send email if analysis didn't fail catastrophically
    # (e.g., HTTP request itself failed). Errors reported by the analysis
    # endpoint *within* the JSON response should still be emailed.
    if analysis_result.get("error") in ["http_request_failed", "json_decode_error"]:
         logger.error("Analysis request failed before getting a usable response. Cannot send detailed email.")
         # Optionally send a very basic failure notification via SES here if desired
         notification_status = {"status": "skipped", "message": "Analysis request failed"}
    else:
        notification_status = send_ses_email(
            analysis_result,
            instance_id,
            REGION,
            alarm_name
        )

    # --- END MODIFICATION ---

    # Create the final response body
    response_body = {
        "analysis_request_status": "success" if "error" not in analysis_result else "error",
        "analysis_http_status": analysis_result.get("status_code", "N/A"),
        "analysis_error": analysis_result.get("error"),
        "analysis_message": analysis_result.get("message"),
        "notification_status": notification_status.get("status", "error"),
        "notification_result": notification_status.get("message_id") or notification_status.get("message")
    }

    final_status_code = 200
    if "error" in analysis_result or notification_status.get("status") == "error":
        logger.warning("Analysis or notification step failed or was skipped. Check response body.")
        # Optionally change to 500 for critical failures
        # final_status_code = 500

    return {
        "statusCode": final_status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(response_body, default=str)
    }