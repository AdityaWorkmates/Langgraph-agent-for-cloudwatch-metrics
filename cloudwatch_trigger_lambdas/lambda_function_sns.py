import os
import json
import boto3
import logging
from datetime import datetime, timedelta, timezone
import urllib.parse
import urllib3

# from dotenv import load_dotenv
# load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.getenv("AWS_REGION", "us-west-2")
# S3_BUCKET = os.getenv("S3_BUCKET", "test-cloudwatch-server-team4-bucket") # No longer strictly needed unless used elsewhere
# PRESIGN_EXPIRATION = int(os.getenv("PRESIGN_EXPIRATION", "3600")) # No longer needed

ANALYSIS_ENDPOINT = os.getenv(
    "ANALYSIS_ENDPOINT",
    "http://langgraph-alb-1133416885.us-west-2.elb.amazonaws.com/analyze"
)
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "arn:aws:sns:us-west-2:565741009456:ec2-cpu-analysis-notifications") # <-- ADDED: Get SNS Topic ARN from environment

HTTP = urllib3.PoolManager()

# clients (explicit region)
CW = boto3.client("cloudwatch", region_name=REGION)
S3 = boto3.client("s3", region_name=REGION) # Keep S3 if needed for other things, remove if not
SNS = boto3.client("sns", region_name=REGION) # <-- ADDED: SNS Client

def ts_iso_z(dt: datetime) -> str:
    """Return ISO-style UTC timestamp without +00:00 (Z)."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds') + "Z"

def safe_key_component(s: str) -> str:
    """Make an S3-safe key component: remove colons, spaces -> underscores, and url-quote."""
    if not s:
        return "unknown"
    s2 = s.replace(":", "").replace(" ", "_").replace("/", "_")
    return urllib.parse.quote_plus(s2)

# --- REMOVED save_payload and save_analysis_response as they are no longer the primary goal ---
# --- You can keep them if needed for other potential workflows ---

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

        # Try to parse response as JSON, but save raw if it fails
        if status_code == 200:
             try:
                 analysis_result["response_body_json"] = json.loads(response_body)
             except json.JSONDecodeError:
                 logger.warning("Endpoint response was status 200 but not valid JSON.")
                 # Mark as error if JSON parsing fails despite 200 status
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

# --- NEW FUNCTION ---
def send_sns_notification(analysis_data: dict, instance_id: str, region: str, alarm_name: str):
    """
    Formats and sends the analysis result as an SNS notification.
    """
    if not SNS_TOPIC_ARN:
        logger.warning("SNS_TOPIC_ARN environment variable not set. Cannot send notification.")
        return {"status": "error", "message": "SNS_TOPIC_ARN not configured"}

    subject = f"CPU Alarm Analysis: Instance {instance_id}"
    if alarm_name:
        subject += f" ({alarm_name})"

    message_lines = []

    try:
        # Check if the analysis itself reported an error first
        if "error" in analysis_data:
             subject = f"[ERROR] Analyzing CPU Alarm for {instance_id}"
             message_lines.append(f"Analysis failed for instance {instance_id} in {region}.")
             message_lines.append(f"Alarm Name: {alarm_name or 'N/A'}")
             message_lines.append(f"Error Type: {analysis_data.get('error', 'Unknown')}")
             message_lines.append(f"Error Message: {analysis_data.get('message', 'N/A')}")
             if "request_filename" in analysis_data:
                message_lines.append(f"Request Filename: {analysis_data['request_filename']}")

        # Proceed if analysis was successful and JSON is present
        elif "response_body_json" in analysis_data:
            response_json = analysis_data["response_body_json"]

            severity = response_json.get("severity", "UNKNOWN")
            subject = f"[{severity}] {subject}" # Prepend severity

            message_lines.append(f"Severity: {severity}")
            message_lines.append(f"Instance: {instance_id} in {region}")
            message_lines.append(f"Alarm Name: {alarm_name or 'N/A'}")
            message_lines.append("-" * 20)

            message_lines.append("Summary:")
            message_lines.append(response_json.get("summary", "No summary provided."))
            message_lines.append("-" * 20)

            message_lines.append("Advice:")
            message_lines.append(response_json.get("advice", "No advice provided."))
            message_lines.append("-" * 20)

            # Add key raw findings if available
            raw_findings = response_json.get("raw_findings", {})
            if raw_findings:
                 message_lines.append("Key Findings:")
                 peak_cpu = raw_findings.get('peak_cpu_value')
                 peak_ts = raw_findings.get('peak_timestamp')
                 baseline = raw_findings.get('baseline_cpu')
                 pattern = raw_findings.get('spike_pattern')
                 if peak_cpu is not None and peak_ts:
                      message_lines.append(f"- Peak CPU: {peak_cpu:.2f}% at {peak_ts}")
                 if baseline is not None:
                      message_lines.append(f"- Baseline CPU: {baseline:.2f}%")
                 if pattern:
                      message_lines.append(f"- Spike Pattern: {pattern}")
                 message_lines.append("-" * 20)


            # List recommendation titles (optional, could make email long)
            recommendations = response_json.get("recommendations", [])
            if recommendations:
                message_lines.append("Recommendations:")
                for i, rec in enumerate(recommendations[:3]): # Limit to first 3 maybe?
                    message_lines.append(f"- {rec.get('title', f'Recommendation {i+1}')} (Priority: {rec.get('priority', 'N/A')})")
                if len(recommendations) > 3:
                     message_lines.append("- ... (See full report for more)")
                message_lines.append("-" * 20)

            message_lines.append("Diagnostics commands and plots are available in the full analysis.")
            # Optionally include a link if the analysis service provides one, or link to CloudWatch logs/metrics

        else:
            # Should not happen if send_to_analysis_endpoint logic is correct, but handle defensively
            subject = f"[ERROR] Analyzing CPU Alarm for {instance_id}"
            message_lines.append(f"Analysis endpoint response for {instance_id} was unparseable or incomplete.")
            message_lines.append(f"Status Code: {analysis_data.get('status_code', 'N/A')}")
            message_lines.append(f"Raw Body Snippet: {analysis_data.get('response_body_raw', '')[:200]}")

        # Join lines into the final message
        message = "\n".join(message_lines)

        # Publish to SNS
        logger.info("Publishing notification to SNS topic: %s", SNS_TOPIC_ARN)
        response = SNS.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        logger.info("SNS Publish Response: %s", response)
        return {"status": "success", "message_id": response.get("MessageId")}

    except Exception as e:
        logger.exception("Error formatting or sending SNS notification: %s", e)
        # Attempt to send a fallback error notification
        try:
            fallback_subject = f"[ERROR] Failed to Send Analysis Notification for {instance_id}"
            fallback_message = (
                f"An error occurred while trying to send the analysis notification for instance {instance_id}.\n"
                f"Error: {str(e)}\n\n"
                f"Original Analysis Status Code: {analysis_data.get('status_code', 'N/A')}"
            )
            SNS.publish(TopicArn=SNS_TOPIC_ARN, Subject=fallback_subject, Message=fallback_message)
            return {"status": "error", "message": f"Failed to send full notification, sent fallback. Error: {str(e)}"}
        except Exception as fallback_e:
            logger.exception("Failed to send even fallback SNS notification: %s", fallback_e)
            return {"status": "error", "message": f"Failed to send notification: {str(e)}"}


def parse_alarm_from_event(event: dict):
    """
    Tries to extract common fields from a CloudWatch Alarm SNS event. Falls back to event fields.
    Returns: (alarm_name, instance_id, namespace, metric_name, period_seconds)
    """
    # ... (rest of the function remains the same) ...
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
    """
    Use get_metric_data to fetch range of datapoints.
    dimensions: list of {Name, Value}
    """
    # ... (rest of the function remains the same) ...
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

    # default instance id if not supplied (use with care in prod)
    # Make sure this default is appropriate or removed for production
    instance_id = instance_id or event.get("instance_id") # Removed hardcoded default

    if not instance_id:
         logger.error("Could not determine InstanceId from event.")
         return {
             "statusCode": 400,
             "body": json.dumps({"error": "InstanceId not found in event payload."})
         }


    # Determine dimensions if instance id available
    dimensions = [{"Name": "InstanceId", "Value": instance_id}]

    lookback_periods_minutes = [5, 10, 15, 30, 60, 120, 180, 360]
    all_metric_data = {}

    for lookback in lookback_periods_minutes:
        logger.info("Fetching metric data for last %d minutes...", lookback)
        range_key = f"last_{lookback}_min"
        try:
            datapoints, time_range = fetch_metric_data(
                namespace,
                metric_name,
                dimensions,
                int(period_seconds),
                lookback
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
                "error": "get_metric_data_failed",
                "message": str(e)
            }

    # Describe alarm metadata (if we have alarm name)
    alarm_meta = {}
    # ... (alarm metadata fetching remains the same) ...
    if alarm_name:
        try:
            alarm_resp = CW.describe_alarms(AlarmNames=[alarm_name])
            metric_alarms = alarm_resp.get("MetricAlarms", [])
            if metric_alarms:
                a = metric_alarms[0]
                alarm_meta = {
                    "alarm_name": a.get("AlarmName"),
                    "alarm_arn": a.get("AlarmArn"),
                    "state": a.get("StateValue"),
                    "state_reason": a.get("StateReason"),
                    "threshold": a.get("Threshold"),
                    "comparison_operator": a.get("ComparisonOperator"),
                    "evaluation_periods": a.get("EvaluationPeriods"),
                    "period_seconds": a.get("Period"), # Use actual alarm period if possible
                    "namespace": a.get("Namespace"),
                    "metric_name": a.get("MetricName"),
                    "dimensions": a.get("Dimensions"),
                    "actions_enabled": a.get("ActionsEnabled"),
                    "alarm_actions": a.get("AlarmActions"),
                    "insufficient_data_actions": a.get("InsufficientDataActions"),
                    "ok_actions": a.get("OKActions")
                }
                # Update period/namespace/metric if fetched from alarm
                period_seconds = alarm_meta.get("period_seconds") or period_seconds
                namespace = alarm_meta.get("namespace") or namespace
                metric_name = alarm_meta.get("metric_name") or metric_name
        except Exception as e:
            logger.warning("Failed to describe alarm '%s': %s", alarm_name, e)


    payload = {
        "instance_id": instance_id,
        "region": REGION,
        "metric": {
            "namespace": namespace,
            "metric_name": metric_name,
            "statistic": "Average",
            "unit": event.get("unit", "Percent"), # Consider getting unit from alarm desc?
            "period_seconds": int(period_seconds)
        },
        "metric_data_by_range": all_metric_data,
        "alarm": alarm_meta,
        "generated_at": ts_iso_z(datetime.now(timezone.utc))
    }

    # Generate the JSON string and a filename for the form
    body_json = json.dumps(payload, default=str, indent=2)

    now_for_filename = datetime.now(timezone.utc)
    filename_ts = now_for_filename.strftime("%Y%m%dT%H%M%SZ")
    alarm_comp = safe_key_component(alarm_name or instance_id)
    filename = f"{alarm_comp}_{filename_ts}.json"

    # --- MODIFICATION: Send to Endpoint & then Notify ---

    # 1. Send to analysis endpoint
    analysis_result = send_to_analysis_endpoint(body_json, filename)

    # 2. Send SNS notification based on analysis result
    notification_status = send_sns_notification(
        analysis_result,
        instance_id,
        REGION,
        alarm_name
    )

    # --- END MODIFICATION ---

    # Create the final response body for the Lambda return
    response_body = {
        "analysis_request_status": "success" if "error" not in analysis_result else "error",
        "analysis_http_status": analysis_result.get("status_code", "N/A"),
        "analysis_error": analysis_result.get("error"), # Include error type if present
        "analysis_message": analysis_result.get("message"), # Include error message if present
        "notification_status": notification_status.get("status", "error"),
        "notification_result": notification_status.get("message_id") or notification_status.get("message")
    }

    # Determine overall status code based on outcomes
    final_status_code = 200
    if "error" in analysis_result or notification_status.get("status") == "error":
        # Consider 500 if the core analysis or notification failed critically
        # Or keep 200 but indicate failure in the body
        logger.warning("Analysis or notification step failed. Check response body.")
        # final_status_code = 500 # Optionally change status code for critical failures


    return {
        "statusCode": final_status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(response_body, default=str)
    }