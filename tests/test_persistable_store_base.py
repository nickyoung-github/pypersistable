"""PersistableStoreBase's own behaviour, with its default _serialise and _validate"""

from models import Counterparty, Deal, TradingBook

from pypersistable import PersistableStoreBase


class ListStore(PersistableStoreBase):
    """Records in lists, in the order written: reads only the latest versions"""

    def __init__(self):
        super().__init__()
        self.records: dict[type, list[dict]] = {}

    def _read(self, by_type, effective_time, entry_time):
        ret = {}
        for typ, ids in by_type.items():
            latest = {}
            for record in self.records.get(typ, []):
                if record["_object_id"] in map(str, ids):
                    latest[record["_object_id"]] = record
            ret[typ] = list(latest.values())
        return ret

    def _write(self, by_type, entry_time, expected):
        for typ, records in by_type.items():
            self.records.setdefault(typ, []).extend(records)


def test_round_trip_with_references():
    store = ListStore()
    deal = Deal(
        deal_id="D",
        counterparty=Counterparty(name="Acme"),
        book=TradingBook(name="B"),
    )
    deal.write(store)
    assert Deal.read(store, deal_id="D") == deal
    # References are stored as ids
    assert store.records[Deal][0]["_data"]["counterparty"] == str(
        deal.counterparty.object_id
    )


def test_delete():
    store = ListStore()
    acme = Counterparty(name="Acme")
    acme.write(store)
    acme.del_(store)
    assert store.read([acme.object_id]) == {}
    assert [r["_active"] for r in store.records[Counterparty]] == [True, False]


def test_by_value():
    store = ListStore()
    deal = Deal(
        deal_id="D", counterparty=Counterparty(name="Acme"), book=TradingBook(name="B")
    )
    store.write([deal], by_reference=False)
    assert Deal.read(store, deal_id="D") == deal
    assert list(store.records) == [Deal]
