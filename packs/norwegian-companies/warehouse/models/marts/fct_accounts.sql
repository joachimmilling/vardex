-- One row per company, fiscal year and kind of accounts (company or group). When a company
-- files again for the same year, the newest filing wins. first_seen_at is when Vardex first
-- fetched the filing that won, which shows how late it came. Results are also converted to
-- NOK at Norges Bank's average rate for the calendar year the fiscal year ends in.
with filings as (
    select
        *,
        min(_fetched_at) over (partition by filing_id) as first_seen_at,
        row_number() over (
            partition by orgnr, accounts_type, year(period_end)
            order by filing_id desc, _fetched_at desc
        ) as newest
    from {{ ref('stg_accounts') }}
),

converted as (
    select
        filings.*,
        case
            when filings.currency = 'NOK' then 1.0
            else rates.nok_per_unit
        end as nok_per_unit
    from filings
    left join {{ ref('stg_exchange_rates') }} as rates
        on rates.currency = filings.currency and rates.year = year(filings.period_end)
    where filings.newest = 1
)

select
    orgnr || '-' || year(period_end) || '-' || accounts_type as accounts_key,
    orgnr,
    year(period_end) as fiscal_year,
    accounts_type,
    period_start,
    period_end,
    date_diff('month', period_start, period_end + 1) as period_months,
    currency,
    accounting_rules,
    revenue,
    operating_profit,
    net_profit,
    total_assets,
    equity,
    nok_per_unit,
    -- Widen before multiplying: a DECIMAL(18) times a DECIMAL(18, 6) overflows in DuckDB.
    cast(round(cast(revenue as decimal(38, 0)) * nok_per_unit) as decimal(18, 0)) as revenue_nok,
    cast(round(cast(operating_profit as decimal(38, 0)) * nok_per_unit) as decimal(18, 0))
        as operating_profit_nok,
    cast(round(cast(net_profit as decimal(38, 0)) * nok_per_unit) as decimal(18, 0))
        as net_profit_nok,
    filing_id,
    first_seen_at
from converted
