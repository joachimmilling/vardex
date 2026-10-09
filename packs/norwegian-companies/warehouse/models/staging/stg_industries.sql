-- Every code in SN2025, from section (a letter) down to subclass (five digits, such as 03.211).
select
    code,
    parentCode as parent_code,
    cast(level as integer) as level,
    name
from (select unnest(codes, recursive := true) from {{ source('raw', 'industries') }})
