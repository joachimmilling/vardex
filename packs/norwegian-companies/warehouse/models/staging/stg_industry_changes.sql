-- The codes that changed between SN2007 and SN2025: one row per pair. A code that was split
-- has several rows, one for each new code.
select
    targetCode as sn2007_code,
    sourceCode as sn2025_code
from (
    select unnest(correspondenceMaps, recursive := true)
    from {{ source('raw', 'industry_changes') }}
)
