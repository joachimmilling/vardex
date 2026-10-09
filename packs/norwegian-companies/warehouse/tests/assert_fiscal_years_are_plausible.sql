-- A fiscal year is usually 12 months; a first or changed year can be shorter or longer, but
-- never more than two years. A row here is a period that suggests swapped or mistyped dates.
select accounts_key, period_start, period_end, period_months
from {{ ref('fct_accounts') }}
where period_months not between 1 and 24
