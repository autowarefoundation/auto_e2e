from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from botocore.exceptions import ClientError

from Platform.scripts.p5en_capacity_block import (
    DEFAULT_MINIMUM_REMAINING_SECONDS,
    RESERVATION_NAME,
    SUPPORTED_AVAILABILITY_ZONES,
    _purchase_confirmation,
    _validate_duration,
    purchase_offering,
    search_offerings,
    wait_until_ready,
)


class _FakeEC2:
    def __init__(self, offerings, reservations=()):
        self.offerings = offerings
        self.reservations = list(reservations)
        self.purchase_requests = []

    def describe_capacity_block_offerings(self, **kwargs):
        return {"CapacityBlockOfferings": list(self.offerings)}

    def purchase_capacity_block(self, **kwargs):
        self.purchase_requests.append(kwargs)
        if kwargs["DryRun"]:
            raise ClientError(
                {
                    "Error": {
                        "Code": "DryRunOperation",
                        "Message": "request would have succeeded",
                    }
                },
                "PurchaseCapacityBlock",
            )
        return {"CapacityReservation": {"State": "scheduled"}}

    def get_paginator(self, operation):
        assert operation == "describe_capacity_reservations"
        reservations = list(self.reservations)

        class _Paginator:
            def paginate(self, **kwargs):
                return [{"CapacityReservations": reservations}]

        return _Paginator()


def _range():
    return (
        datetime(2026, 9, 2, tzinfo=timezone.utc),
        datetime(2026, 9, 9, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("hours", [24, 48, 168, 336])
def test_capacity_block_duration_accepts_supported_windows(hours):
    _validate_duration(hours)


@pytest.mark.parametrize("hours", [1, 360, 480, 504, 672, 4368])
def test_capacity_block_duration_rejects_unsupported_windows(hours):
    with pytest.raises(ValueError):
        _validate_duration(hours)


def test_search_sorts_by_start_then_fee():
    start_after, end_before = _range()
    later = {
        "CapacityBlockOfferingId": "cb-later",
        "AvailabilityZone": "us-west-2a",
        "StartDate": datetime(2026, 9, 4, tzinfo=timezone.utc),
        "UpfrontFee": "100.00",
    }
    cheaper = {
        "CapacityBlockOfferingId": "cb-cheaper",
        "AvailabilityZone": "us-west-2c",
        "StartDate": datetime(2026, 9, 3, tzinfo=timezone.utc),
        "UpfrontFee": "90.00",
    }
    expensive = {
        "CapacityBlockOfferingId": "cb-expensive",
        "AvailabilityZone": "us-west-2a",
        "StartDate": datetime(2026, 9, 3, tzinfo=timezone.utc),
        "UpfrontFee": "110.00",
    }
    result = search_offerings(
        _FakeEC2([later, expensive, cheaper]),
        duration_hours=24,
        start_after=start_after,
        end_before=end_before,
    )
    assert [
        offering["CapacityBlockOfferingId"] for offering in result
    ] == ["cb-cheaper", "cb-expensive", "cb-later"]


def test_search_excludes_zones_without_cluster_subnets():
    start_after, end_before = _range()
    supported_zone = sorted(SUPPORTED_AVAILABILITY_ZONES)[0]
    ec2 = _FakeEC2([
        {
            "CapacityBlockOfferingId": "cb-supported",
            "AvailabilityZone": supported_zone,
            "StartDate": datetime(2026, 9, 3, tzinfo=timezone.utc),
            "UpfrontFee": "100.00",
        },
        {
            "CapacityBlockOfferingId": "cb-no-subnet",
            "AvailabilityZone": "us-west-2d",
            "StartDate": datetime(2026, 9, 3, tzinfo=timezone.utc),
            "UpfrontFee": "90.00",
        },
    ])
    result = search_offerings(
        ec2,
        duration_hours=24,
        start_after=start_after,
        end_before=end_before,
    )
    assert [
        offering["CapacityBlockOfferingId"] for offering in result
    ] == ["cb-supported"]


def test_purchase_requires_exact_fee_and_confirmation():
    start_after, end_before = _range()
    offering = {
        "CapacityBlockOfferingId": "cb-123",
        "AvailabilityZone": "us-west-2a",
        "StartDate": datetime(2026, 9, 3, tzinfo=timezone.utc),
        "UpfrontFee": "1318.0800",
    }
    ec2 = _FakeEC2([offering])
    fee = Decimal("1318.0800")
    result = purchase_offering(
        ec2,
        offering_id="cb-123",
        expected_upfront_fee=fee,
        max_upfront_fee=Decimal("1500.00"),
        confirmation=_purchase_confirmation("cb-123", fee),
        duration_hours=24,
        start_after=start_after,
        end_before=end_before,
        execute=True,
    )
    assert result["CapacityReservation"]["State"] == "scheduled"
    assert [
        request["DryRun"] for request in ec2.purchase_requests
    ] == [True, False]
    tags = ec2.purchase_requests[1]["TagSpecifications"][0]["Tags"]
    assert {"Key": "Name", "Value": RESERVATION_NAME} in tags
    assert {
        "Key": "capacity-block-offering-id",
        "Value": "cb-123",
    } in tags

    with pytest.raises(ValueError, match="upfront fee changed"):
        purchase_offering(
            ec2,
            offering_id="cb-123",
            expected_upfront_fee=Decimal("1.00"),
            max_upfront_fee=Decimal("1500.00"),
            confirmation="wrong",
            duration_hours=24,
            start_after=start_after,
            end_before=end_before,
        )
    wrong_confirmation = _FakeEC2([offering])
    with pytest.raises(ValueError, match="confirmation must equal"):
        purchase_offering(
            wrong_confirmation,
            offering_id="cb-123",
            expected_upfront_fee=fee,
            max_upfront_fee=Decimal("1500.00"),
            confirmation="wrong",
            duration_hours=24,
            start_after=start_after,
            end_before=end_before,
        )
    assert wrong_confirmation.purchase_requests == []


def test_purchase_stops_when_dry_run_returns_success():
    start_after, end_before = _range()
    offering = {
        "CapacityBlockOfferingId": "cb-123",
        "AvailabilityZone": "us-west-2a",
        "StartDate": datetime(2026, 9, 3, tzinfo=timezone.utc),
        "UpfrontFee": "1318.0800",
    }
    ec2 = _FakeEC2([offering])

    def non_raising_purchase(**kwargs):
        ec2.purchase_requests.append(kwargs)
        return {"CapacityReservation": {"State": "scheduled"}}

    ec2.purchase_capacity_block = non_raising_purchase
    fee = Decimal("1318.0800")
    with pytest.raises(
        RuntimeError,
        match="DryRun unexpectedly returned success",
    ):
        purchase_offering(
            ec2,
            offering_id="cb-123",
            expected_upfront_fee=fee,
            max_upfront_fee=Decimal("1500.00"),
            confirmation=_purchase_confirmation("cb-123", fee),
            duration_hours=24,
            start_after=start_after,
            end_before=end_before,
            execute=True,
        )
    assert [
        request["DryRun"] for request in ec2.purchase_requests
    ] == [True]


@pytest.mark.parametrize(
    "state",
    [
        "active",
        "assessing",
        "cancelling",
        "delayed",
        "payment-pending",
        "pending",
        "scheduled",
        "unavailable",
    ],
)
def test_purchase_rejects_duplicate_nonterminal_block(state):
    start_after, end_before = _range()
    ec2 = _FakeEC2(
        [],
        reservations=[
            {
                "CapacityReservationId": "cr-existing",
                "CreateDate": start_after,
                "ReservationType": "capacity-block",
                "State": state,
                "TotalInstanceCount": 1,
            }
        ],
    )
    with pytest.raises(ValueError, match="cr-existing"):
        purchase_offering(
            ec2,
            offering_id="cb-123",
            expected_upfront_fee=Decimal("1318.0800"),
            max_upfront_fee=Decimal("1500.00"),
            confirmation="unused",
            duration_hours=24,
            start_after=start_after,
            end_before=end_before,
        )
    assert ec2.purchase_requests == []


def test_purchase_rejects_duplicate_block_with_unexpected_size():
    start_after, end_before = _range()
    ec2 = _FakeEC2(
        [],
        reservations=[
            {
                "CapacityReservationId": "cr-existing",
                "CreateDate": start_after,
                "ReservationType": "capacity-block",
                "State": "active",
                "TotalInstanceCount": 2,
            }
        ],
    )
    with pytest.raises(ValueError, match="cr-existing"):
        purchase_offering(
            ec2,
            offering_id="cb-123",
            expected_upfront_fee=Decimal("1318.0800"),
            max_upfront_fee=Decimal("1500.00"),
            confirmation="unused",
            duration_hours=24,
            start_after=start_after,
            end_before=end_before,
        )


def test_purchase_enforces_spend_ceiling_and_defaults_to_dry_run():
    start_after, end_before = _range()
    offering = {
        "CapacityBlockOfferingId": "cb-123",
        "AvailabilityZone": "us-west-2a",
        "StartDate": datetime(2026, 9, 3, tzinfo=timezone.utc),
        "UpfrontFee": "1318.0800",
    }
    fee = Decimal("1318.0800")
    with pytest.raises(ValueError, match="exceeds maximum"):
        purchase_offering(
            _FakeEC2([offering]),
            offering_id="cb-123",
            expected_upfront_fee=fee,
            max_upfront_fee=Decimal("1000.00"),
            confirmation=_purchase_confirmation("cb-123", fee),
            duration_hours=24,
            start_after=start_after,
            end_before=end_before,
        )

    ec2 = _FakeEC2([offering])
    result = purchase_offering(
        ec2,
        offering_id="cb-123",
        expected_upfront_fee=fee,
        max_upfront_fee=Decimal("1500.00"),
        confirmation=_purchase_confirmation("cb-123", fee),
        duration_hours=24,
        start_after=start_after,
        end_before=end_before,
    )
    assert result["DryRun"] is True
    assert ec2.purchase_requests[0]["DryRun"] is True
    assert len(ec2.purchase_requests) == 1

    executed = _FakeEC2([offering])
    result = purchase_offering(
        executed,
        offering_id="cb-123",
        expected_upfront_fee=fee,
        max_upfront_fee=Decimal("1500.00"),
        confirmation=_purchase_confirmation("cb-123", fee),
        duration_hours=24,
        start_after=start_after,
        end_before=end_before,
        execute=True,
    )
    assert result["CapacityReservation"]["State"] == "scheduled"
    assert [
        request["DryRun"] for request in executed.purchase_requests
    ] == [True, False]


def test_wait_ready_requires_safe_remaining_window():
    now = datetime.now(timezone.utc)
    ready = {
        "CapacityReservationId": "cr-ready",
        "AvailabilityZone": "us-west-2a",
        "AvailableInstanceCount": 1,
        "CreateDate": now,
        "EndDate": now + timedelta(hours=3),
        "ReservationType": "capacity-block",
        "State": "active",
        "TotalInstanceCount": 1,
    }
    result = wait_until_ready(
        _FakeEC2([], reservations=[ready]),
        timeout_seconds=0,
        poll_seconds=0,
        minimum_remaining_seconds=7200,
    )
    assert result["CapacityReservationId"] == "cr-ready"

    unsupported_zone = dict(ready)
    unsupported_zone["AvailabilityZone"] = "us-west-2d"
    with pytest.raises(TimeoutError):
        wait_until_ready(
            _FakeEC2([], reservations=[unsupported_zone]),
            timeout_seconds=0,
            poll_seconds=0,
            minimum_remaining_seconds=7200,
        )

    too_late = dict(ready)
    too_late["EndDate"] = now + timedelta(hours=1)
    with pytest.raises(TimeoutError):
        wait_until_ready(
            _FakeEC2([], reservations=[too_late]),
            timeout_seconds=0,
            poll_seconds=0,
            minimum_remaining_seconds=7200,
        )

    consumed = dict(ready)
    consumed["AvailableInstanceCount"] = 0
    result = wait_until_ready(
        _FakeEC2([], reservations=[consumed]),
        timeout_seconds=0,
        poll_seconds=0,
        minimum_remaining_seconds=7200,
    )
    assert result["CapacityReservationId"] == "cr-ready"


def test_wait_ready_ignores_same_named_on_demand_reservation():
    now = datetime.now(timezone.utc)
    odcr = {
        "CapacityReservationId": "cr-odcr",
        "AvailableInstanceCount": 1,
        "CreateDate": now,
        "EndDate": now + timedelta(hours=3),
        "ReservationType": "default",
        "State": "active",
        "TotalInstanceCount": 1,
    }
    with pytest.raises(TimeoutError):
        wait_until_ready(
            _FakeEC2([], reservations=[odcr]),
            timeout_seconds=0,
            poll_seconds=0,
            minimum_remaining_seconds=7200,
        )


def test_wait_ready_default_reserves_twenty_two_hours():
    assert DEFAULT_MINIMUM_REMAINING_SECONDS == 22 * 60 * 60
