import os
import json
import boto3
import logging
from datetime import datetime, timedelta, timezone
import urllib.parse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET = os.getenv("S3_BUCKET", "test-cloudwatch-server-team4-bucket")
PRESIGN_EXPIRATION = int(os.getenv("PRESIGN_EXPIRATION", "3600"))

CW = boto3.client("cloudwatch", region_name=REGION)
S3 = boto3.client("s3", region_name=REGION)

def ts_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds') + "Z"

def safe_key_component(s: str) -> str:
    if not s:
        return "unknown"
    s2 = s.replace(":", "").replace(" ", "_").replace("/", "_")
    return urllib.parse.quote_plus(s2)

def save_payload(payload: dict, alarm_name: str):
    now = datetime.now(timezone.utc)
    filename_ts = now.strftime("%Y%m%dT%H%M%SZ")
    alarm_comp = safe_key_component(alarm_name or "unknown_alarm")
    base_key = f"alarms/{alarm_comp}/{filename_ts}"

    json_key = f"{base_key}.json"
    txt_key = f"{base_key}.txt"

    body_json = json.dumps(payload, default=str, indent=2)
    body_txt = f"Alarm: {alarm_name}\nGeneratedAt: {ts_iso_z(now)}\n\n{body_json}"

    S3.put_object(Bucket=S3_BUCKET, Key=json_key, Body=body_json.encode("utf-8"), ContentType="application/json")
    S3.put_object(Bucket=S3_BUCKET, Key=txt_key, Body=body_txt.encode("utf-8"), ContentType="text/plain")

    json_url = S3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": json_key}, ExpiresIn=PRESIGN_EXPIRATION)
    txt_url = S3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": txt_key}, ExpiresIn=PRESIGN_EXPIRATION)

    return {"json_key": json_key, "txt_key": txt_key, "json_url": json_url, "txt_url": txt_url}

def parse_alarm_from_event(event: dict):
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
                try:
                    msg_obj = json.loads(msg)
                except Exception:
                    msg_obj = None

                if isinstance(msg_obj, dict):
                    alarm_name = msg_obj.get("AlarmName") or msg_obj.get("alarmName") or alarm_name
                    trigger = msg_obj.get("Trigger") or msg_obj.get("trigger")
                    if trigger:
                        metric_name = trigger.get("MetricName") or metric_name
                        namespace = trigger.get("Namespace") or namespace
                        dims = trigger.get("Dimensions") or []
                        for d in dims:
                            if d.get("name", d.get("Name", "")).lower() == "instanceid":
                                instance_id = d.get("value", d.get("Value"))
                else:
                    alarm_name = alarm_name or sns.get("Subject") or sns.get("MessageId")
        alarm_name = alarm_name or event.get("alarm_name") or event.get("AlarmName")
        instance_id = instance_id or event.get("instance_id")
        namespace = event.get("namespace", namespace)
        metric_name = event.get("metric_name", metric_name)
        period_seconds = int(event.get("period_seconds", period_seconds))
    except Exception as e:
        logger.warning("Failed to parse event for alarm info: %s", e)

    return alarm_name, instance_id, namespace, metric_name, period_seconds

def fetch_metric_data(namespace: str, metric_name: str, dimensions: list, period_seconds: int, lookback_minutes: int = 30):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=lookback_minutes)

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
    datapoints = []
    results = resp.get("MetricDataResults", [])
    if results:
        r = results[0]
        timestamps = r.get("Timestamps", [])
        values = r.get("Values", [])
        pairs = sorted(zip(timestamps, values), key=lambda x: x[0])
        for ts, val in pairs:
            if isinstance(ts, datetime):
                ts_str = ts.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec='seconds') + "Z"
            else:
                ts_str = str(ts)
            datapoints.append({"timestamp": ts_str, "value": val})
    return datapoints, {"start_iso": ts_iso_z(start_time), "end_iso": ts_iso_z(end_time)}

def lambda_handler(event, context):
    logger.info("Received event: %s", json.dumps(event)[:1000])
    alarm_name, instance_id, namespace, metric_name, period_seconds = parse_alarm_from_event(event)

    instance_id = instance_id or event.get("instance_id") or "i-097bf8bf5e21f9d52"

    dimensions = []
    if instance_id:
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
        "metric_data_by_range": all_metric_data, 
        "alarm": alarm_meta,
        "generated_at": ts_iso_z(datetime.now(timezone.utc))
    }

    try:
        s3_result = save_payload(payload, alarm_name or instance_id)
    except Exception as e:
        logger.exception("Error saving payload to S3: %s", e)
        s3_result = {"error": str(e)}

    response_body = {"payload": payload, "s3_result": s3_result}
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(response_body)
    }