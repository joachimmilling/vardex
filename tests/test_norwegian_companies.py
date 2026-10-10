"""The norwegian-companies pack's warehouse, built from a few made-up landed files.

Nothing is fetched: every file is landed by the test, as if fetched this morning. The
companies with numbers from 100000008 are fictional; 923609016 is Equinor ASA.
"""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx2
import pytest
import yaml

from vardex.warehouse import build, query

PACK = Path(__file__).parent.parent / "packs" / "norwegian-companies"
NOW = datetime(2026, 10, 5, 7, 0, tzinfo=UTC)
UNITS = """\
organisasjonsnummer,navn,organisasjonsform.kode,naeringskode1.kode,antallAnsatte,\
forretningsadresse.kommunenummer,forretningsadresse.kommune,stiftelsesdato,erIKonsern,\
konkurs,underAvvikling,sisteInnsendteAarsregnskap
923609016,EQUINOR ASA,ASA,06.100,21272,1103,STAVANGER,1972-09-18,true,false,false,2025
100000008,DEMO LAKS AS,AS,03.211,120,4601,BERGEN,2001-05-02,false,false,false,2025
100000016,DEMO DATA AS,AS,62.200,300,0301,OSLO,1990-01-15,false,false,false,2025
100000024,DEMO ARKTIS AS,AS,00.000,,2100,SVALBARD,2019-03-01,false,false,false,
"""


def filing(filing_id: int, kind: str, year: int, currency: str, revenue: int, profit: int):
    """One set of accounts as the accounts register's API returns it, trimmed to what the
    pack reads."""
    return {
        "id": filing_id,
        "journalnr": f"{year + 1}{filing_id:06d}",
        "regnskapstype": kind,
        "regnskapsperiode": {"fraDato": f"{year}-01-01", "tilDato": f"{year}-12-31"},
        "valuta": currency,
        "regnkapsprinsipper": {"regnskapsregler": "regnskapslovenAlminneligRegler"},
        "eiendeler": {"sumEiendeler": 3 * revenue},
        "egenkapitalGjeld": {"egenkapital": {"sumEgenkapital": revenue}},
        "resultatregnskapResultat": {
            "aarsresultat": profit // 2,
            "driftsresultat": {
                "driftsresultat": profit,
                "driftsinntekter": {"sumDriftsinntekter": revenue},
            },
        },
    }


def accounts(*rows: tuple[str, int, list | None]) -> str:
    """JSON lines as Vardex lands them: one per key."""
    return "".join(
        json.dumps({"key": key, "status": status, "body": body, "error": None}) + "\n"
        for key, status, body in rows
    )


LAST_WEEK = accounts(
    ("923609016", 200, [filing(7192429, "KONSERN", 2025, "USD", 106_462_000_000, 1)]),
    ("100000008", 200, [filing(500, "SELSKAP", 2025, "NOK", 1_480_312_000, 353_655_000)]),
)
TODAY = accounts(
    ("923609016", 200, [filing(7192429, "KONSERN", 2025, "USD", 106_462_000_000, 1)]),
    # Demo Laks AS filed corrected accounts for 2025 this week: the newest filing wins
    ("100000008", 200, [filing(501, "SELSKAP", 2025, "NOK", 1_480_312_000, 350_000_000)]),
    ("100000016", 500, None),
)
INDUSTRIES = {
    "codes": [
        {"code": "A", "parentCode": None, "level": "1", "name": "Jordbruk, skogbruk og fiske"},
        {"code": "03", "parentCode": "A", "level": "2", "name": "Fiske, fangst og akvakultur"},
        {"code": "03.211", "parentCode": "03.21", "level": "5", "name": "Akvakultur i hav"},
        {"code": "B", "parentCode": None, "level": "1", "name": "Bergverksdrift og utvinning"},
        {"code": "06", "parentCode": "B", "level": "2", "name": "Utvinning av råolje og gass"},
        {"code": "06.100", "parentCode": "06.10", "level": "5", "name": "Utvinning av råolje"},
        {"code": "K", "parentCode": None, "level": "1", "name": "Telekommunikasjon og IT"},
        {"code": "62", "parentCode": "K", "level": "2", "name": "IT-tjenester"},
        {"code": "62.200", "parentCode": "62.20", "level": "5", "name": "IT-konsulentvirksomhet"},
    ]
}
CHANGES = {
    "correspondenceMaps": [
        # SN2007's 62.020 and 62.030 were merged into SN2025's 62.200
        {"sourceCode": "62.200", "targetCode": "62.020"},
        {"sourceCode": "62.200", "targetCode": "62.030"},
    ]
}
COUNTIES = {
    "codes": [
        {"code": "03", "name": "Oslo", "validToInRequestedRange": "2026-01-01"},
        {"code": "03", "name": "Oslo - Oslove", "validToInRequestedRange": None},
        {"code": "11", "name": "Rogaland", "validToInRequestedRange": None},
        {"code": "46", "name": "Vestland", "validToInRequestedRange": None},
        {"code": "99", "name": "Uoppgitt", "validToInRequestedRange": None},
    ]
}
STATISTICS = """\
"NACE2007","Enhet","Tid","Enheter","Sysselsatte","Oms"
"B","1","2024",900,60000,1500000.0
"06.100","1","2024",60,25000,1100000.5
"62.020","1","2024",6634,35485,117980.0
"62.030","1","2024",166,:,:
"""
RATES = """\
FREQ;BASE_CUR;QUOTE_CUR;UNIT_MULT;TIME_PERIOD;OBS_VALUE
A;USD;NOK;0;2025;10.5
A;SEK;NOK;2;2025;105.2
"""
LANDED = {
    "units/2026-10-05T060000Z.csv": UNITS,
    "accounts/2026-09-28T060000Z.jsonl": LAST_WEEK,
    "accounts/2026-10-05T060000Z.jsonl": TODAY,
    "industries/2026-10-05T060000Z.json": json.dumps(INDUSTRIES),
    "industry_changes/2026-10-05T060000Z.json": json.dumps(CHANGES),
    "counties/2026-10-05T060000Z.json": json.dumps(COUNTIES),
    "industry_statistics/2026-10-05T060000Z.csv": STATISTICS,
    "exchange_rates/2026-10-05T060000Z.csv": RATES,
}


def offline(request: httpx2.Request) -> httpx2.Response:
    raise AssertionError(f"nothing should be fetched, but {request.url} was")


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory) -> Path:
    data = tmp_path_factory.mktemp("data")
    for name, text in LANDED.items():
        path = data / "norwegian-companies" / "landing" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    client = httpx2.Client(transport=httpx2.MockTransport(offline))
    result = build(PACK, data, client, now=NOW)
    assert result.warnings == []
    return result.path


def rows(warehouse: Path, sql: str) -> list[tuple]:
    return query(warehouse, sql).fetchall()


def test_companies_get_counties_and_svalbard(warehouse):
    sql = "select orgnr, county, industry_code from dim_companies order by orgnr"
    assert rows(warehouse, sql) == [
        ("100000008", "Vestland", "03.211"),
        ("100000016", "Oslo", "62.200"),  # the name valid today, without the Sami form
        ("100000024", "Svalbard", None),  # 00.000 means "not stated"
        ("923609016", "Rogaland", "06.100"),
    ]


def test_the_newest_filing_wins_and_keeps_when_it_was_first_seen(warehouse):
    sql = """
        select orgnr, fiscal_year, accounts_type, currency, revenue, operating_profit,
               first_seen_at
        from fct_accounts order by orgnr
    """
    laks, equinor = rows(warehouse, sql)
    assert laks[:6] == ("100000008", 2025, "company", "NOK", 1480312000, 350000000)
    assert laks[6] == datetime(2026, 10, 5, 6, 0)  # the corrected filing arrived today
    assert equinor[2:4] == ("group", "USD")
    assert equinor[6] == datetime(2026, 9, 28, 6, 0)


def test_amounts_are_converted_to_nok(warehouse):
    sql = "select orgnr, nok_per_unit, revenue_nok from fct_accounts order by orgnr"
    assert rows(warehouse, sql) == [
        ("100000008", 1, 1480312000),
        ("923609016", Decimal("10.5"), 1117851000000),  # USD 106,462 million at 10.5
    ]
    sek = "select nok_per_unit from stg_exchange_rates where currency = 'SEK'"
    assert rows(warehouse, sek) == [(Decimal("1.052"),)]  # quoted per 100 kronor, exact


def test_ssb_turnover_is_in_kroner(warehouse):
    sql = "select turnover_nok from fct_industry_statistics where sn2007_code = '06.100'"
    assert rows(warehouse, sql) == [(1100000500000,)]


def test_unchanged_codes_are_added_to_the_change_table(warehouse):
    sql = "select sn2007_code, sn2025_code from bridge_industry_codes order by all"
    assert rows(warehouse, sql) == [
        ("03.211", "03.211"),
        ("06.100", "06.100"),
        ("62.020", "62.200"),
        ("62.030", "62.200"),
    ]


def test_every_metric_has_a_value(warehouse):
    sql = "select sum(operating_profit) / nullif(sum(revenue), 0) from fct_accounts"
    sql += " where currency = 'NOK' and accounts_type = 'company'"
    [(margin,)] = rows(warehouse, sql)
    assert round(margin, 3) == 0.236


def test_every_column_is_documented(warehouse):
    marts = yaml.safe_load((PACK / "warehouse/models/marts/marts.yml").read_text("utf-8"))
    for model in marts["models"]:
        documented = {c["name"] for c in model["columns"] if c.get("description")}
        columns = rows(warehouse, f"select column_name from (describe {model['name']})")
        built = {name for (name,) in columns}
        assert built == documented, f"{model['name']}: describe every column, and only those"
