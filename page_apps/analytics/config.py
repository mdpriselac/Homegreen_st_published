"""Page-level feature flags for the analytics section."""

import os

# Turnover tab and the lifespan rankings (Fastest Moving / Longest Lasting) share the
# same lifespan data (Kaplan-Meier survival analysis). Shown by default; set
# ANALYTICS_SHOW_TURNOVER=0 (or false/no/off) in the environment to hide them.
SHOW_TURNOVER_TAB = os.environ.get('ANALYTICS_SHOW_TURNOVER', '').strip().lower() not in ('0', 'false', 'no', 'off')
