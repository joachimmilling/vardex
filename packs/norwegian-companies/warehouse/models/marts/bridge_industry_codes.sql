-- Every SN2007 subclass and the SN2025 subclasses it became. SSB's change table lists only
-- codes that changed, so codes that kept their number and meaning are added here. A code
-- that was split has one row for each new code: joining through it can count a figure twice.
select sn2007_code, sn2025_code
from {{ ref('stg_industry_changes') }}
where sn2007_code is not null  -- a few SN2025 codes, such as 98.100, are new: no SN2007 code

union all

select industry_code as sn2007_code, industry_code as sn2025_code
from {{ ref('dim_industries') }}
where industry_code not in (select sn2025_code from {{ ref('stg_industry_changes') }})
