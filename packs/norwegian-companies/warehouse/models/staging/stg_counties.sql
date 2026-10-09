-- Counties as they are today. KLASS returns every version of a code in the requested range,
-- so Oslo, renamed in 2026, comes twice; only the version still valid is kept. Official
-- names may add Sami and Kven forms after a dash ("Troms - Romsa - Tromssa"); name keeps
-- the first. Svalbard has its own number, 21, but is in no county, so it is added by hand.
select
    code as county_code,
    split_part(name, ' - ', 1) as name,
    name as official_name
from (select unnest(codes, recursive := true) from {{ source('raw', 'counties') }})
where validToInRequestedRange is null
  and code <> '99'  -- "not stated"

union all

select '21', 'Svalbard', 'Svalbard'
