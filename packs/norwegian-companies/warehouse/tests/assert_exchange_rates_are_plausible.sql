-- A check on units. Every currency Vardex converts is worth between 0.5 and 20 kroner. A rate
-- outside that range was read per 100 units, or the wrong way round.
select currency, year, nok_per_unit
from {{ ref('stg_exchange_rates') }}
where nok_per_unit not between 0.5 and 20
