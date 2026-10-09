-- A check on units. SSB publishes turnover in NOK million, the register in kroner: mix them up
-- and one side is a million times too big. For industries whose code did not change, the
-- companies in this warehouse can never have ten times the turnover SSB counts for the whole
-- industry. A row here is an industry and year where they do.
with unchanged as (
    select industry_code as code
    from {{ ref('dim_industries') }}
    where industry_code not in (select sn2025_code from {{ ref('stg_industry_changes') }})
),

revenue as (
    select companies.industry_code as code, accounts.fiscal_year as year, sum(accounts.revenue) as revenue
    from {{ ref('fct_accounts') }} as accounts
    join {{ ref('dim_companies') }} as companies using (orgnr)
    where accounts.accounts_type = 'company' and accounts.currency = 'NOK'
    group by all
)

select stats.sn2007_code, stats.year, stats.turnover_nok, revenue.revenue
from {{ ref('fct_industry_statistics') }} as stats
join unchanged on unchanged.code = stats.sn2007_code
join revenue on revenue.code = stats.sn2007_code and revenue.year = stats.year
where stats.turnover_nok > 0 and revenue.revenue > 10 * stats.turnover_nok
