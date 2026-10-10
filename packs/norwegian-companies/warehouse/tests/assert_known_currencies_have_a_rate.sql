-- Every account in a currency Norges Bank quotes, for a year it has published an average
-- for, must have a rate. A row here is one that has none, so its NOK amounts are empty.
select accounts_key, currency, fiscal_year
from {{ ref('fct_accounts') }}
where nok_per_unit is null
  and currency in (select currency from {{ ref('stg_exchange_rates') }})
  and fiscal_year <= (select max(year) from {{ ref('stg_exchange_rates') }})
