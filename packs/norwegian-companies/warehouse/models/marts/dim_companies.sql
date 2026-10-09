-- One row per legal entity in the register, as it stands today.
select
    units.orgnr,
    units.name,
    units.legal_form,
    units.industry_code,
    units.employees,
    units.municipality_code,
    units.municipality,
    units.county_code,
    counties.name as county,
    units.founded_on,
    units.in_group,
    units.is_bankrupt,
    units.is_being_wound_up,
    units.last_accounts_year
from {{ ref('stg_units') }} as units
left join {{ ref('stg_counties') }} as counties using (county_code)
