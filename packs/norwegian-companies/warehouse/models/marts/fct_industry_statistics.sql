-- SSB's figures for all enterprises in Norway, by SN2007 industry and year. Codes come at
-- every level, from section (B) to subclass (10.201), plus a few of SSB's own groupings.
select
    sn2007_code,
    case length(sn2007_code)
        when 1 then 'section'
        when 2 then 'division'
        when 4 then 'group'
        when 5 then 'class'
        when 6 then 'subclass'
        else 'other'
    end as level,
    year,
    enterprises,
    employed,
    turnover_nok
from {{ ref('stg_industry_statistics') }}
