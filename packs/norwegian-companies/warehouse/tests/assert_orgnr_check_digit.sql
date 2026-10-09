-- The last digit of an organisation number is a check digit (modulus 11, weights 3 2 7 6 5 4
-- 3 2). A row here is a number that fails: a typo, or a column read in the wrong format.
with digits as (
    select orgnr, list_transform(range(1, 10), i -> try_cast(orgnr[i] as integer)) as d
    from {{ ref('dim_companies') }}
)

select orgnr
from digits
where not regexp_full_match(orgnr, '[0-9]{9}')
   or (11 - (3 * d[1] + 2 * d[2] + 7 * d[3] + 6 * d[4]
             + 5 * d[5] + 4 * d[6] + 3 * d[7] + 2 * d[8]) % 11) % 11 <> d[9]
