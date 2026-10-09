-- One row per company, fiscal year and kind of accounts (company or group). When a company
-- files again for the same year, the newest filing wins. first_seen_at is when Vardex first
-- fetched the filing that won, which shows how late it came.
with filings as (
    select
        *,
        min(_fetched_at) over (partition by filing_id) as first_seen_at,
        row_number() over (
            partition by orgnr, accounts_type, year(period_end)
            order by filing_id desc, _fetched_at desc
        ) as newest
    from {{ ref('stg_accounts') }}
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
    filing_id,
    first_seen_at
from filings
where newest = 1
