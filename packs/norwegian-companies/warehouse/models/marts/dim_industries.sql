-- One row per SN2025 industry subclass, with its division and section.
with codes as (select * from {{ ref('stg_industries') }})

select
    subclass.code as industry_code,
    subclass.name as industry,
    division.code as division_code,
    division.name as division,
    section.code as section_code,
    section.name as section
from codes as subclass
join codes as division on division.code = left(subclass.code, 2)
join codes as section on section.code = division.parent_code
where subclass.level = 5
