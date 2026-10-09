-- One row per legal entity, with typed columns and English names.
select
    organisasjonsnummer as orgnr,
    navn as name,
    "organisasjonsform.kode" as legal_form,
    -- 00.000 is the register's placeholder for "not stated", not an industry
    nullif("naeringskode1.kode", '00.000') as industry_code,
    try_cast(antallAnsatte as integer) as employees,
    "forretningsadresse.kommunenummer" as municipality_code,
    "forretningsadresse.kommune" as municipality,
    left("forretningsadresse.kommunenummer", 2) as county_code,
    try_cast(stiftelsesdato as date) as founded_on,
    erIKonsern = 'true' as in_group,
    konkurs = 'true' as is_bankrupt,
    underAvvikling = 'true' as is_being_wound_up,
    try_cast(sisteInnsendteAarsregnskap as integer) as last_accounts_year,
    _fetched_at
from {{ source('raw', 'units') }}
