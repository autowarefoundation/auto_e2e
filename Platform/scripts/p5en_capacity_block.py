#!/usr/bin/env python3
"""Search, purchase, and verify p5en Capacity Blocks for training."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Sequence

import boto3

INSTANCE_TYPE = "p5en.48xlarge"
INSTANCE_COUNT = 1
RESERVATION_NAME = "auto-e2e-p5en-capacity-block"
INSTANCE_PLATFORM = "Linux/UNIX"
DEFAULT_MINIMUM_REMAINING_SECONDS = 22 * 60 * 60
SUPPORTED_AVAILABILITY_ZONES = frozenset(
    {"us-west-2a", "us-west-2c"}
)
TERMINAL_BLOCK_STATES = frozenset(
    {"cancelled", "expired", "failed", "payment-failed", "unsupported"}
)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "timestamps must include an explicit UTC offset"
        )
    return parsed.astimezone(timezone.utc)


def _validate_duration(hours: int) -> None:
    if hours % 24 != 0:
        raise ValueError("capacity duration must be a whole number of days")
    days = hours // 24
    if not 1 <= days <= 14:
        raise ValueError(
            "capacity duration must be between 1 and 14 days"
        )


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def search_offerings(
    ec2: Any,
    *,
    duration_hours: int,
    start_after: datetime,
    end_before: datetime,
) -> list[dict[str, Any]]:
    _validate_duration(duration_hours)
    if end_before <= start_after:
        raise ValueError("end-before must be later than start-after")
    request = {
        "InstanceType": INSTANCE_TYPE,
        "InstanceCount": INSTANCE_COUNT,
        "CapacityDurationHours": duration_hours,
        "StartDateRange": start_after,
        "EndDateRange": end_before,
        "AllAvailabilityZones": True,
        "MaxResults": 100,
    }
    offerings: list[dict[str, Any]] = []
    while True:
        response = ec2.describe_capacity_block_offerings(**request)
        offerings.extend(
            offering
            for offering in response.get(
                "CapacityBlockOfferings",
                (),
            )
            if offering["AvailabilityZone"]
            in SUPPORTED_AVAILABILITY_ZONES
        )
        token = response.get("NextToken")
        if not token:
            break
        request["NextToken"] = token
    return sorted(
        offerings,
        key=lambda item: (
            item["StartDate"],
            Decimal(str(item["UpfrontFee"])),
        ),
    )


def _purchase_confirmation(
    offering_id: str,
    upfront_fee: Decimal,
) -> str:
    return f"PURCHASE {offering_id} {upfront_fee}"


def purchase_offering(
    ec2: Any,
    *,
    offering_id: str,
    expected_upfront_fee: Decimal,
    max_upfront_fee: Decimal,
    confirmation: str,
    duration_hours: int,
    start_after: datetime,
    end_before: datetime,
    execute: bool = False,
) -> dict[str, Any]:
    if max_upfront_fee <= 0:
        raise ValueError("maximum upfront fee must be positive")
    if expected_upfront_fee > max_upfront_fee:
        raise ValueError(
            f"expected upfront fee {expected_upfront_fee} exceeds "
            f"maximum {max_upfront_fee}"
        )
    existing = [
        block
        for block in describe_blocks(ec2)
        if block["State"] not in TERMINAL_BLOCK_STATES
    ]
    if existing:
        identifiers = ", ".join(
            sorted(
                block["CapacityReservationId"]
                for block in existing
            )
        )
        raise ValueError(
            "an active or scheduled p5en Capacity Block already exists: "
            f"{identifiers}"
        )
    matching = [
        offering
        for offering in search_offerings(
            ec2,
            duration_hours=duration_hours,
            start_after=start_after,
            end_before=end_before,
        )
        if offering["CapacityBlockOfferingId"] == offering_id
    ]
    if len(matching) != 1:
        raise ValueError("capacity block offering is no longer available")
    actual_fee = Decimal(str(matching[0]["UpfrontFee"]))
    if actual_fee != expected_upfront_fee:
        raise ValueError(
            f"upfront fee changed from {expected_upfront_fee} "
            f"to {actual_fee}"
        )
    if actual_fee > max_upfront_fee:
        raise ValueError(
            f"upfront fee {actual_fee} exceeds maximum "
            f"{max_upfront_fee}"
        )
    required_confirmation = _purchase_confirmation(
        offering_id,
        actual_fee,
    )
    if confirmation != required_confirmation:
        raise ValueError(
            f"confirmation must equal {required_confirmation!r}"
        )
    request = {
        "CapacityBlockOfferingId": offering_id,
        "InstancePlatform": INSTANCE_PLATFORM,
        "DryRun": True,
        "TagSpecifications": [
            {
                "ResourceType": "capacity-reservation",
                "Tags": [
                    {"Key": "Name", "Value": RESERVATION_NAME},
                    {"Key": "managed-by", "Value": "auto-e2e"},
                    {
                        "Key": "purpose",
                        "Value": "distributed-training",
                    },
                    {
                        "Key": "instance-type",
                        "Value": INSTANCE_TYPE,
                    },
                    {
                        "Key": "capacity-block-offering-id",
                        "Value": offering_id,
                    },
                ],
            }
        ],
    }
    try:
        ec2.purchase_capacity_block(**request)
    except Exception as error:
        error_response = getattr(error, "response", {})
        error_code = error_response.get("Error", {}).get("Code")
        if error_code != "DryRunOperation":
            raise
    else:
        raise RuntimeError(
            "PurchaseCapacityBlock DryRun unexpectedly returned success"
        )
    result = {
        "DryRun": True,
        "CapacityBlockOfferingId": offering_id,
        "UpfrontFee": str(actual_fee),
    }
    if not execute:
        return result
    request["DryRun"] = False
    return ec2.purchase_capacity_block(**request)


def describe_blocks(ec2: Any) -> list[dict[str, Any]]:
    paginator = ec2.get_paginator("describe_capacity_reservations")
    reservations: list[dict[str, Any]] = []
    for page in paginator.paginate(
        Filters=[
            {"Name": "tag:Name", "Values": [RESERVATION_NAME]},
            {"Name": "instance-type", "Values": [INSTANCE_TYPE]},
        ]
    ):
        reservations.extend(page.get("CapacityReservations", ()))
    return sorted(
        (
            reservation
            for reservation in reservations
            if reservation.get("ReservationType") == "capacity-block"
        ),
        key=lambda item: item.get("StartDate") or item["CreateDate"],
    )


def wait_until_ready(
    ec2: Any,
    *,
    timeout_seconds: int,
    poll_seconds: int,
    minimum_remaining_seconds: int,
) -> dict[str, Any]:
    if minimum_remaining_seconds <= 0:
        raise ValueError("minimum remaining seconds must be positive")
    deadline = time.monotonic() + timeout_seconds
    while True:
        minimum_end = (
            datetime.now(timezone.utc)
            + timedelta(seconds=minimum_remaining_seconds)
        )
        ready = [
            block
            for block in describe_blocks(ec2)
            if block["State"] == "active"
            and int(block.get("TotalInstanceCount", 0))
            == INSTANCE_COUNT
            and block.get("AvailabilityZone")
            in SUPPORTED_AVAILABILITY_ZONES
            and block.get("EndDate") is not None
            and block["EndDate"] >= minimum_end
        ]
        if ready:
            return ready[0]
        if time.monotonic() >= deadline:
            raise TimeoutError("no active p5en Capacity Block became ready")
        time.sleep(poll_seconds)


def _client(profile: str | None, region: str) -> Any:
    return boto3.Session(
        profile_name=profile,
        region_name=region,
    ).client("ec2")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--region", default="us-west-2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search")
    search.add_argument("--duration-hours", type=int, required=True)
    search.add_argument(
        "--start-after",
        type=_parse_timestamp,
        required=True,
    )
    search.add_argument(
        "--end-before",
        type=_parse_timestamp,
        required=True,
    )

    purchase = subparsers.add_parser("purchase")
    purchase.add_argument("--offering-id", required=True)
    purchase.add_argument(
        "--expected-upfront-fee",
        type=Decimal,
        required=True,
    )
    purchase.add_argument(
        "--max-upfront-fee",
        type=Decimal,
        required=True,
    )
    purchase.add_argument("--confirm", required=True)
    purchase.add_argument("--execute", action="store_true")
    purchase.add_argument("--duration-hours", type=int, required=True)
    purchase.add_argument(
        "--start-after",
        type=_parse_timestamp,
        required=True,
    )
    purchase.add_argument(
        "--end-before",
        type=_parse_timestamp,
        required=True,
    )

    subparsers.add_parser("status")

    wait = subparsers.add_parser("wait-ready")
    wait.add_argument("--timeout-seconds", type=int, default=3600)
    wait.add_argument("--poll-seconds", type=int, default=30)
    wait.add_argument(
        "--minimum-remaining-seconds",
        type=int,
        default=DEFAULT_MINIMUM_REMAINING_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ec2 = _client(args.profile, args.region)
    result: Any
    if args.command == "search":
        result = search_offerings(
            ec2,
            duration_hours=args.duration_hours,
            start_after=args.start_after,
            end_before=args.end_before,
        )
    elif args.command == "purchase":
        result = purchase_offering(
            ec2,
            offering_id=args.offering_id,
            expected_upfront_fee=args.expected_upfront_fee,
            max_upfront_fee=args.max_upfront_fee,
            confirmation=args.confirm,
            duration_hours=args.duration_hours,
            start_after=args.start_after,
            end_before=args.end_before,
            execute=args.execute,
        )
    elif args.command == "status":
        result = describe_blocks(ec2)
    else:
        result = wait_until_ready(
            ec2,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
            minimum_remaining_seconds=args.minimum_remaining_seconds,
        )
    print(json.dumps(result, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
