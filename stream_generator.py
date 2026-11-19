"""Replay fraudTest.csv as a live stream.
Usage: python stream_transactions.py fraudTest.csv --rate 100 --limit 500 --sink kinesis --stream NAME
Run with --help for all options. Create a file named ATTACK to fire a demo attack."""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BATCH = 500
BATCH_SECONDS = 0.1
ATTACK_FILE = "ATTACK"
FAR_PLACES = [(34.05, -118.24), (25.76, -80.19), (47.61, -122.33), (40.71, -74.01)]


def clean_merchant(name):
    return name[6:] if name.startswith("fraud_") else name


# new name: (CSV column, converter)
FIELDS = {
    "transaction_id": ("trans_num", str),
    "time": ("trans_date_trans_time", str),
    "card_id": ("cc_num", str),
    "merchant": ("merchant", clean_merchant),
    "category": ("category", str),
    "amount": ("amt", float),
    "state": ("state", str),
    "city_population": ("city_pop", int),
    "birth_date": ("dob", str),
    "home_lat": ("lat", float),
    "home_long": ("long", float),
    "merchant_lat": ("merch_lat", float),
    "merchant_long": ("merch_long", float),
}


# ---- CSV ----
def read_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        needed = [old for old, _ in FIELDS.values()] + ["is_fraud"]
        missing = [c for c in needed if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{path} is missing columns: {', '.join(missing)}")
        for row in reader:
            txn = {new: convert(row[old]) for new, (old, convert) in FIELDS.items()}
            yield txn, row["is_fraud"] == "1"


def miles(lat1, long1, lat2, long2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlong = p2 - p1, math.radians(long2 - long1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlong / 2) ** 2
    return 3959 * 2 * math.asin(math.sqrt(a))


# ---- Attack ----
def make_attack(txn, count, minutes):
    far_lat, far_long = max(FAR_PLACES, key=lambda p: miles(txn["home_lat"], txn["home_long"], p[0], p[1]))
    start = datetime.strptime(txn["time"], TIME_FORMAT)
    gap = minutes * 60 / max(count - 1, 1)
    burst = []
    for i in range(count):
        fake = dict(txn)
        fake.update({
            "transaction_id": "attack-" + uuid.uuid4().hex[:12],
            "time": (start + timedelta(seconds=gap * i)).strftime(TIME_FORMAT),
            "merchant": f"Demo Online Shop {random.randint(1, 5)}",
            "category": random.choice(["shopping_net", "misc_net"]),
            "amount": round(random.uniform(300, 1200), 2),
            "merchant_lat": round(far_lat + random.uniform(-0.1, 0.1), 4),
            "merchant_long": round(far_long + random.uniform(-0.1, 0.1), 4),
        })
        burst.append(fake)
    return burst


# ---- Timing ----
def schedule(args):
    start = time.time()
    first_time = None
    attack_done = False

    for n, (txn, is_fraud) in enumerate(read_rows(args.csv_file)):
        if args.limit and n >= args.limit:
            break
        row_time = datetime.strptime(txn["time"], TIME_FORMAT)
        if first_time is None:
            first_time = row_time

        if args.rate:
            send_time = start + n / args.rate
        else:
            send_time = start + (row_time - first_time).total_seconds() / args.speed

        fire = False
        if args.attack_after is not None and not attack_done and time.time() - start >= args.attack_after:
            fire = attack_done = True
        if os.path.exists(ATTACK_FILE):
            os.remove(ATTACK_FILE)
            fire = True
        if fire:
            print(f"[attack] card {txn['card_id']}", file=sys.stderr)
            for fake in make_attack(txn, args.attack_count, args.attack_minutes):
                yield send_time, fake, True, "attack"

        yield send_time, txn, is_fraud, "dataset"


# ---- Sinks ----
class PrintSink:
    def send(self, batch):
        for t in batch:
            print(json.dumps(t), flush=True)

    def close(self):
        pass


class FileSink:
    def __init__(self, path):
        self.f = open(path, "w", buffering=1)

    def send(self, batch):
        for t in batch:
            self.f.write(json.dumps(t) + "\n")

    def close(self):
        self.f.close()


class KinesisSink:
    def __init__(self, stream, region):
        import boto3
        self.client = boto3.client("kinesis", region_name=region)
        self.stream = stream

    def send(self, batch):
        records = [{"Data": json.dumps(t).encode(), "PartitionKey": t["card_id"]} for t in batch]
        for attempt in range(3):
            reply = self.client.put_records(StreamName=self.stream, Records=records)
            if not reply["FailedRecordCount"]:
                return
            records = [r for r, result in zip(records, reply["Records"]) if "ErrorCode" in result]
            time.sleep(0.2 * (attempt + 1))
        print(f"[warning] {len(records)} records not sent", file=sys.stderr)

    def close(self):
        pass


def make_sink(args):
    if args.sink == "print":
        return PrintSink()
    if args.sink == "file":
        return FileSink(args.out)
    if not args.stream:
        sys.exit("--stream is required with --sink kinesis")
    return KinesisSink(args.stream, args.region)


# ---- Main ----
def main():
    p = argparse.ArgumentParser(description="Replay fraudTest.csv as a live stream")
    p.add_argument("csv_file", nargs="?", default="fraudTest.csv")
    speed = p.add_mutually_exclusive_group()
    speed.add_argument("--rate", type=float, help="transactions per second (default 100)")
    speed.add_argument("--speed", type=float, help="times real time, using dataset times")
    p.add_argument("--limit", type=int, default=0, help="max dataset rows (0 = all)")
    p.add_argument("--sink", choices=["print", "file", "kinesis"], default="print")
    p.add_argument("--out", default="transactions.jsonl")
    p.add_argument("--stream", help="Kinesis stream name")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--labels", default="labels.jsonl")
    p.add_argument("--attack-after", type=float, help="seconds until a demo attack")
    p.add_argument("--attack-count", type=int, default=6)
    p.add_argument("--attack-minutes", type=float, default=4)
    p.add_argument("--sample", type=int, help="write N sample transactions and exit")
    args = p.parse_args()

    if args.rate is None and args.speed is None:
        args.rate = 100

    if args.sample:
        rows = []
        for n, (txn, _) in enumerate(read_rows(args.csv_file)):
            if n >= args.sample:
                break
            txn["sent_time_ms"] = int(time.time() * 1000)
            rows.append(txn)
        with open("sample_transactions.json", "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote {len(rows)} transactions to sample_transactions.json")
        return

    sink = make_sink(args)
    labels = open(args.labels, "w", buffering=1)
    batch, batch_start = [], 0.0
    sent = 0
    start = last_report = time.time()

    def flush():
        nonlocal batch, sent
        if batch:
            sink.send(batch)
            sent += len(batch)
            batch = []

    try:
        for send_time, txn, is_fraud, source in schedule(args):
            while True:
                now = time.time()
                wait = send_time - now
                if wait <= 0:
                    break
                if batch:
                    until_batch_due = batch_start + BATCH_SECONDS - now
                    if until_batch_due <= wait:
                        time.sleep(max(until_batch_due, 0))
                        flush()
                        continue
                time.sleep(wait)
                break

            if not batch:
                batch_start = time.time()
            txn["sent_time_ms"] = int(time.time() * 1000)
            batch.append(txn)
            labels.write(json.dumps({"transaction_id": txn["transaction_id"],
                                     "is_fraud": is_fraud, "source": source}) + "\n")

            if len(batch) >= MAX_BATCH or time.time() - batch_start >= BATCH_SECONDS:
                flush()

            if time.time() - last_report >= 5:
                last_report = time.time()
                print(f"[progress] sent {sent} ({sent / (last_report - start):.1f}/s)", file=sys.stderr)
        flush()
    except KeyboardInterrupt:
        flush()
    finally:
        sink.close()
        labels.close()
        elapsed = max(time.time() - start, 1e-9)
        print(f"[done] sent {sent} in {elapsed:.1f}s ({sent / elapsed:.1f}/s)", file=sys.stderr)


if __name__ == "__main__":
    main()