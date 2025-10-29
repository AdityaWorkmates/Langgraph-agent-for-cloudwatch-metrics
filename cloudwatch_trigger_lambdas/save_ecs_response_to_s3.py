import os
import json
import boto3
import logging
from datetime import datetime, timedelta, timezone
import urllib.parse
import urllib3 # <--- ADDED IMPORT

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET = os.getenv("S3_BUCKET", "test-cloudwatch-server-team4-bucket")
PRESIGN_EXPIRATION = int(os.getenv("PRESIGN_EXPIRATION", "3600")) # seconds
# --- ADDED GLOBALS ---
ANALYSIS_ENDPOINT = os.getenv(
    "ANALYSIS_ENDPOINT", 
    "http://langgraph-alb-1133416885.us-west-2.elb.amazonaws.com/analyze"
)
HTTP = urllib3.PoolManager()
# --- END ADDED GLOBALS ---


# clients (explicit region)
CW = boto3.client("cloudwatch", region_name=REGION)
S3 = boto3.client("s3", region_name=REGION)

def ts_iso_z(dt: datetime) -> str:
    """Return ISO-style UTC timestamp without +00:00 (Z)."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds') + "Z"

def safe_key_component(s: str) -> str:
    """Make an S3-safe key component: remove colons, spaces -> underscores, and url-quote."""
    if not s:
        return "unknown"
    s2 = s.replace(":", "").replace(" ", "_").replace("/", "_")
    return urllib.parse.quote_plus(s2)

def save_payload(payload: dict, alarm_name: str):
    """
    [Original function - No longer called by handler but left for reference if needed]
    """
    now = datetime.now(timezone.utc)
    # use compact timestamp for filenames: 20251015T123045Z
    filename_ts = now.strftime("%Y%m%dT%H%M%SZ")
    alarm_comp = safe_key_component(alarm_name or "unknown_alarm")
    base_key = f"alarms/{alarm_comp}/{filename_ts}"

    json_key = f"{base_key}.json"
    txt_key = f"{base_key}.txt"

    body_json = json.dumps(payload, default=str, indent=2)
    body_txt = f"Alarm: {alarm_name}\nGeneratedAt: {ts_iso_z(now)}\n\n{body_json}"

    # upload
    S3.put_object(Bucket=S3_BUCKET, Key=json_key, Body=body_json.encode("utf-8"), ContentType="application/json")
    S3.put_object(Bucket=S3_BUCKET, Key=txt_key, Body=body_txt.encode("utf-8"), ContentType="text/plain")

    # create presigned urls for quick viewing/downloading
    json_url = S3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": json_key}, ExpiresIn=PRESIGN_EXPIRATION)
    txt_url = S3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": txt_key}, ExpiresIn=PRESIGN_EXPIRATION)

    return {"json_key": json_key, "txt_key": txt_key, "json_url": json_url, "txt_url": txt_url}

# --- NEW FUNCTION ---
def send_to_analysis_endpoint(json_payload_string: str, filename: str):
    """
    Sends the JSON payload string as a multipart/form-data file to the analysis endpoint.
    """
    logger.info("Sending payload to endpoint: %s as file '%s'", ANALYSIS_ENDPOINT, filename)
    
    # This structure emulates: --form 'file=@(filename)'
    fields = {'file': (filename, json_payload_string, 'application/json')}
    
    try:
        r = HTTP.request(
            'POST',
            ANALYSIS_ENDPOINT,
            fields=fields,
            headers={'Accept': 'application/json'} # Good practice
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
        try:
            analysis_result["response_body_json"] = json.loads(response_body)
        except json.JSONDecodeError:
            logger.warning("Endpoint response was not valid JSON.")
            
        return analysis_result

    except Exception as e:
        logger.exception("Error sending data to endpoint: %s", e)
        return {
            "error": "http_request_failed",
            "message": str(e),
            "request_filename": filename
        }

# --- NEW FUNCTION ---
def save_analysis_response(response_data: dict, alarm_name: str, instance_id: str):
    """
    Saves the analysis endpoint's response to S3.
    """
    now = datetime.now(timezone.utc)
    filename_ts = now.strftime("%Y%m%dT%H%M%SZ")
    # Use the same naming convention for the folder
    alarm_comp = safe_key_component(alarm_name or instance_id or "unknown_alarm")
    
    # Use the timestamp to create a base key, and add a suffix for the response
    base_key = f"alarms/{alarm_comp}/{filename_ts}"
    analysis_key = f"{base_key}_analysis_response.json"
    
    body_json = json.dumps(response_data, default=str, indent=2)
    
    try:
        S3.put_object(
            Bucket=S3_BUCKET, 
            Key=analysis_key, 
            Body=body_json.encode("utf-8"), 
            ContentType="application/json"
        )
        
        # create presigned url for the analysis response file
        analysis_url = S3.generate_presigned_url(
            "get_object", 
            Params={"Bucket": S3_BUCKET, "Key": analysis_key}, 
            ExpiresIn=PRESIGN_EXPIRATION
        )
        
        s3_result = {
            "analysis_s3_key": analysis_key,
            "analysis_s3_url": analysis_url
        }
        
    except Exception as e:
        logger.exception("Error saving analysis response to S3: %s", e)
        s3_result = {"error": "s3_put_failed", "message": str(e)}
        
    return s3_result


def parse_alarm_from_event(event: dict):
    """
    Tries to extract common fields from a CloudWatch Alarm SNS event. Falls back to event fields.
    Returns: (alarm_name, instance_id, namespace, metric_name, period_seconds)
    """
    # Typical SNS-wrapped CloudWatch Alarm has Records[0].Sns.Message with JSON string
    alarm_name = None
    instance_id = None
    namespace = "AWS/EC2"
    metric_name = "CPUUtilization"
    period_seconds = 30

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
                ts_str = str(ts)
            datapoints.append({"timestamp": ts_str, "value": val})
    return datapoints, {"start_iso": ts_iso_z(start_time), "end_iso": ts_iso_z(end_time)}

def lambda_handler(event, context):
    logger.info("Received event: %s", json.dumps(event)[:1000])
    # parse inputs
    alarm_name, instance_id, namespace, metric_name, period_seconds = parse_alarm_from_event(event)

    # default instance id if not supplied (use with care in prod)
    instance_id = instance_id or event.get("instance_id") or "i-097bf8bf5e21f9d52"

    # Determine dimensions if instance id available
    dimensions = []
    if instance_id:
        dimensions = [{"Name": "InstanceId", "Value": instance_id}]

    # --- MODIFICATION START ---
    # Define all the lookback periods you want to fetch
    # 6 hours = 360 minutes
    lookback_periods_minutes = [5, 10, 15, 30, 60, 120, 180, 360] 
    
    all_metric_data = {} # Dictionary to store results for all time ranges

    for lookback in lookback_periods_minutes:
        logger.info("Fetching metric data for last %d minutes...", lookback)
        range_key = f"last_{lookback}_min"
        
        try:
            datapoints, time_range = fetch_metric_data(
                namespace, 
                metric_name, 
                dimensions, 
                int(period_seconds), 
                lookback  # Use the lookback period from our list
            )
            
            # Store the successful result
            all_metric_data[range_key] = {
                "lookback_minutes": lookback,
                "datapoints": datapoints,
                "query_time_range": time_range
            }

        except Exception as e:
            logger.exception("Error fetching metric data for %d min lookback: %s", lookback, e)
            # Store the error for this specific range but continue the loop
            all_metric_data[range_key] = {
                "lookback_minutes": lookback,
                "error": "get_metric_data_failed", 
                "message": str(e)
            }
    
    # --- MODIFICATION END ---


    # Describe alarm metadata (if we have alarm name)
    alarm_meta = {}
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
                    "period_seconds": a.get("Period"),
                    "namespace": namespace,
                    "metric_name": metric_name,
                    "dimensions": a.get("Dimensions"),
                    "actions_enabled": a.get("ActionsEnabled"),
                    "alarm_actions": a.get("AlarmActions"),
                    "insufficient_data_actions": a.get("InsufficientDataActions"),
                    "ok_actions": a.get("OKActions")
                }
        except Exception as e:
            logger.warning("Failed to describe alarm '%s': %s", alarm_name, e)

    
    # --- PAYLOAD MODIFICATION ---
    # The payload now contains 'metric_data_by_range' instead of 'datapoints'
    payload = {
        "instance_id": instance_id,
        "region": REGION,
        "metric": {
            "namespace": namespace,
            "metric_name": metric_name,
            "statistic": "Average",
            "unit": event.get("unit", "Percent"),
            "period_seconds": int(period_seconds)
        },
        # This key now holds all the data from the loop
        "metric_data_by_range": all_metric_data, 
        "alarm": alarm_meta,
        "generated_at": ts_iso_z(datetime.now(timezone.utc))
    }
    # --- END PAYLOAD MODIFICATION ---

    # --- MODIFICATION: Send to Endpoint instead of save_payload ---
    
    # 1. Generate the JSON string and a filename for the form
    body_json = json.dumps(payload, default=str, indent=2)
    
    now_for_filename = datetime.now(timezone.utc)
    filename_ts = now_for_filename.strftime("%Y%m%dT%H%M%SZ")
    alarm_comp = safe_key_component(alarm_name or instance_id or "unknown_alarm")
    filename = f"{alarm_comp}_{filename_ts}.json"
    
    # 2. Send to analysis endpoint
    analysis_result = send_to_analysis_endpoint(body_json, filename)
    
    # 3. Save the analysis response to S3
    s3_analysis_result = save_analysis_response(
        analysis_result, 
        alarm_name, 
        instance_id
    )
    
    # --- END MODIFICATION ---

    # Create the final response body for the Lambda return
    response_body = {
        "original_payload_summary": {
            "instance_id": instance_id,
            "alarm_name": alarm_name,
            "metric_name": metric_name,
            "generated_at": payload["generated_at"]
        },
        "analysis_result": analysis_result,
        "s3_analysis_storage": s3_analysis_result
        # You can also include the full 'payload' here if needed,
        # but it can be large and might be redundant if the endpoint has it
        # "original_payload": payload, 
    }
    
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        # Lambda proxy must return a JSON string in `body`
        # Use default=str just in case any datetimes/etc slipped in
        "body": json.dumps(response_body, default=str)
    }