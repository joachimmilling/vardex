-- SSB's figures for all enterprises by SN2007 industry and year. SSB publishes turnover in
-- NOK million; here it is in NOK, like every other amount. ':' means not published.
select
    NACE2007 as sn2007_code,
    cast(Tid as integer) as year,
    try_cast(Enheter as integer) as enterprises,
    try_cast(Sysselsatte as integer) as employed,
    cast(try_cast(Oms as decimal(18, 1)) * 1000000 as decimal(18, 0)) as turnover_nok,
    _fetched_at
from {{ source('raw', 'industry_statistics') }}
