"""Each app visitor works on their own copy of the simulated month."""
from __future__ import annotations

from headroom.agent.site import ReplaySite
from headroom.agent.tools import GridEventAdjustment, SetRequirement


def test_a_fork_owns_what_a_conversation_changes():
    base = ReplaySite(month="2017-06", n_ports=24)
    before = base.results["headroom"]
    a = base.fork()
    a.now_local = "2017-06-21T18:00"
    a.set_requirement(SetRequirement(label="fleet vans", vehicles=6, kwh_per_vehicle=40,
                                     plug_in_local="2017-06-21T19:00", ready_by_local="2017-06-22T06:00"))
    a.grid_event_adjustment(GridEventAdjustment(
        start_local="2017-06-21T14:00", end_local="2017-06-21T21:00", building_multiplier=1.031,
        source="CAISO Flex Alert archive", reason="test"))
    assert a.requirements and not base.requirements
    assert base.results["headroom"] is before and base.now_local == "2017-06-21T15:00"
    assert base.fork().requirements == [] and base.fork().adjustments == []
