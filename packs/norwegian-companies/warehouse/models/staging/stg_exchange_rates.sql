-- NOK for one unit of each currency, as a yearly average. Norges Bank quotes some currencies
-- per 100 units (UNIT_MULT 2, such as SEK and DKK), so the rate is divided by 10 to the power
-- of UNIT_MULT. The result stays an exact decimal: dividing decimals in DuckDB gives a float.
select
    BASE_CUR as currency,
    cast(TIME_PERIOD as integer) as year,
    cast(
        cast(OBS_VALUE as decimal(18, 6)) / power(10, cast(UNIT_MULT as integer))
        as decimal(18, 6)
    ) as nok_per_unit,
    _fetched_at
from {{ source('raw', 'exchange_rates') }}
