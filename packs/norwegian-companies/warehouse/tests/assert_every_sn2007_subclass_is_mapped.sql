-- Every SN2007 subclass SSB has figures for must map to at least one SN2025 subclass. A row
-- here is one that does not: its figures would vanish from any join to companies. (The table
-- also lists a few retired codes, such as 55.301, that never have figures; they are ignored.)
select distinct sn2007_code
from {{ ref('fct_industry_statistics') }}
where level = 'subclass'
  and enterprises > 0
  and sn2007_code not in (select sn2007_code from {{ ref('bridge_industry_codes') }})
