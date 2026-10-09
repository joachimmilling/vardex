-- One row per filed set of accounts and fetch. The API answers each company with a list of
-- accounts: company and group, for up to three years. Amounts are in whole units of the
-- currency the accounts were filed in.
with filings as (
    select key as orgnr, _fetched_at, unnest(from_json(body, '["json"]')) as a
    from {{ source('raw', 'accounts') }}
    where status = 200
)

select
    orgnr,
    cast(json_extract_string(a, '$.id') as bigint) as filing_id,
    json_extract_string(a, '$.journalnr') as journal_number,
    case json_extract_string(a, '$.regnskapstype')
        when 'SELSKAP' then 'company'
        when 'KONSERN' then 'group'
    end as accounts_type,
    cast(json_extract_string(a, '$.regnskapsperiode.fraDato') as date) as period_start,
    cast(json_extract_string(a, '$.regnskapsperiode.tilDato') as date) as period_end,
    json_extract_string(a, '$.valuta') as currency,
    json_extract_string(a, '$.regnkapsprinsipper.regnskapsregler') as accounting_rules,
    cast(json_extract_string(a, '$.resultatregnskapResultat.driftsresultat.driftsinntekter.sumDriftsinntekter') as decimal(18, 0)) as revenue,
    cast(json_extract_string(a, '$.resultatregnskapResultat.driftsresultat.driftsresultat') as decimal(18, 0)) as operating_profit,
    cast(json_extract_string(a, '$.resultatregnskapResultat.aarsresultat') as decimal(18, 0)) as net_profit,
    cast(json_extract_string(a, '$.eiendeler.sumEiendeler') as decimal(18, 0)) as total_assets,
    cast(json_extract_string(a, '$.egenkapitalGjeld.egenkapital.sumEgenkapital') as decimal(18, 0)) as equity,
    _fetched_at
from filings
