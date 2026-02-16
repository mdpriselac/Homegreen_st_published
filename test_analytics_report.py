#!/usr/bin/env python3
"""
Test script to fetch data and generate preliminary analytics reports.
Runs outside of Streamlit to validate the data pipeline and print results.
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime
from supabase import create_client

# Load secrets from .streamlit/secrets.toml manually
import tomllib
secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
with open(secrets_path, 'rb') as f:
    secrets = tomllib.load(f)

SUPABASE_URL = secrets['supabase']['url']
SUPABASE_KEY = secrets['supabase']['anon_key']


def fetch_data():
    """Fetch all coffee data from Supabase"""
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Fetch coffee attributes
    print("Fetching coffee attributes...")
    attrs_result = client.table('coffee_attributes').select(
        'coffee_id, country_final, subregion_final, categorized_flavors, '
        'process_type_final, varietal, average_per_lb, cheapest_per_lb, highest_per_lb'
    ).eq('is_cleaned', True).execute()
    attrs_df = pd.DataFrame(attrs_result.data)
    print(f"  -> {len(attrs_df)} coffee attributes rows")

    # Fetch coffees with seller info
    print("Fetching coffees with seller info...")
    coffees_result = client.table('coffees').select(
        'id, name, seller_id, first_observed, last_observed, is_active, sellers(id, name)'
    ).execute()

    coffee_data = []
    for coffee in coffees_result.data:
        seller_info = coffee.get('sellers', {})
        if isinstance(seller_info, list) and len(seller_info) > 0:
            seller_info = seller_info[0]
        elif not isinstance(seller_info, dict):
            seller_info = {}
        coffee_data.append({
            'coffee_id': coffee['id'],
            'coffee_name': coffee['name'],
            'seller_id': coffee.get('seller_id'),
            'seller_name': seller_info.get('name', 'Unknown'),
            'first_observed': coffee.get('first_observed'),
            'last_observed': coffee.get('last_observed'),
            'is_active': coffee.get('is_active'),
        })
    coffees_df = pd.DataFrame(coffee_data)
    print(f"  -> {len(coffees_df)} coffees rows")

    # Merge
    merged = attrs_df.merge(coffees_df, on='coffee_id', how='left')
    print(f"  -> {len(merged)} merged rows")

    return merged


def parse_flavors(val):
    """Parse categorized_flavors JSON field"""
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except:
            try:
                parsed = json.loads(val.replace("'", '"'))
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
    return []


def parse_varietal(val):
    """Parse varietal field"""
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, list):
        return [v.strip() for v in val if v and str(v).strip()]
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        try:
            parsed = json.loads(val.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v and str(v).strip()]
        except:
            pass
        if ',' in val:
            return [v.strip() for v in val.split(',') if v.strip()]
        return [val]
    return []


def normalize_process(val):
    """Normalize process type"""
    if pd.isna(val) or val is None:
        return None
    v = str(val).strip().lower()
    if not v:
        return None
    mapping = {
        'washed': 'Washed', 'fully washed': 'Washed', 'fully-washed': 'Washed',
        'double washed': 'Washed', 'wet process': 'Washed',
        'natural': 'Natural', 'dry process': 'Natural', 'sun dried': 'Natural',
        'honey': 'Honey', 'honey process': 'Honey', 'yellow honey': 'Honey',
        'red honey': 'Honey', 'black honey': 'Honey', 'pulped natural': 'Honey',
        'wet hulled': 'Wet Hulled', 'wet-hulled': 'Wet Hulled', 'giling basah': 'Wet Hulled',
        'anaerobic': 'Anaerobic', 'anaerobic natural': 'Anaerobic',
        'anaerobic washed': 'Anaerobic', 'carbonic maceration': 'Anaerobic',
        'multi-stage fermentation': 'Anaerobic',
        'monsoon': 'Other', 'monsooned': 'Other', 'decaf': 'Other', 'decaffeinated': 'Other',
    }
    if v in mapping:
        return mapping[v]
    for key, canonical in mapping.items():
        if key in v:
            return canonical
    return val.strip()


def prepare_data(merged):
    """Prepare merged data with all parsed fields"""
    df = merged.copy()

    # Parse flavors
    df['flavors_parsed'] = df['categorized_flavors'].apply(parse_flavors)
    df['has_flavors'] = df['flavors_parsed'].apply(lambda x: len(x) > 0)
    df['flavor_families'] = df['flavors_parsed'].apply(
        lambda flavors: list(set(f['family'] for f in flavors if f.get('family')))
    )
    df['flavor_genera'] = df['flavors_parsed'].apply(
        lambda flavors: list(set(f['genus'] for f in flavors if f.get('genus')))
    )

    # Parse varietals
    df['varietals_parsed'] = df['varietal'].apply(parse_varietal)
    df['has_varietal'] = df['varietals_parsed'].apply(lambda x: len(x) > 0)

    # Normalize process
    df['process_clean'] = df['process_type_final'].apply(normalize_process)
    df['has_process'] = df['process_clean'].notna()

    # Parse prices
    for col in ['average_per_lb', 'cheapest_per_lb', 'highest_per_lb']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['has_price'] = df['average_per_lb'].notna() & (df['average_per_lb'] > 0)

    # Parse dates
    df['first_observed'] = pd.to_datetime(df['first_observed'], errors='coerce')
    df['last_observed'] = pd.to_datetime(df['last_observed'], errors='coerce')
    df['lifespan_days'] = (df['last_observed'] - df['first_observed']).dt.days

    return df


def print_separator(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def report_data_completeness(df):
    """Report on data completeness"""
    print_separator("DATA COMPLETENESS REPORT")
    total = len(df)
    print(f"Total coffees: {total}\n")

    fields = [
        ('country_final', 'Country'),
        ('subregion_final', 'Region'),
        ('has_process', 'Process Method'),
        ('has_varietal', 'Varietal'),
        ('has_flavors', 'Flavor Notes'),
        ('has_price', 'Price Data'),
        ('first_observed', 'First Observed Date'),
    ]

    for col, label in fields:
        if col in ['has_process', 'has_varietal', 'has_flavors', 'has_price']:
            count = df[col].sum()
        else:
            count = df[col].notna().sum()
        rate = count / total * 100
        print(f"  {label:25s}: {count:5d} / {total} ({rate:5.1f}%)")


def report_overview(df):
    """Report basic dataset overview"""
    print_separator("DATASET OVERVIEW")
    print(f"Total coffees:     {len(df)}")
    print(f"Active coffees:    {df['is_active'].sum()}")
    print(f"Expired coffees:   {(~df['is_active']).sum()}")
    print(f"Unique countries:  {df['country_final'].nunique()}")
    print(f"Unique sellers:    {df['seller_name'].nunique()}")
    print(f"Unique processes:  {df['process_clean'].nunique()}")

    print(f"\n--- Top 10 Countries by Coffee Count ---")
    country_counts = df['country_final'].value_counts().head(10)
    for country, count in country_counts.items():
        print(f"  {country:25s}: {count}")

    print(f"\n--- Process Method Distribution ---")
    process_counts = df['process_clean'].value_counts()
    for process, count in process_counts.items():
        if pd.notna(process):
            pct = count / len(df) * 100
            print(f"  {str(process):25s}: {count:4d} ({pct:.1f}%)")

    print(f"\n--- Top 15 Varietals ---")
    all_varietals = []
    for vlist in df['varietals_parsed']:
        all_varietals.extend(vlist)
    from collections import Counter
    varietal_counts = Counter(all_varietals).most_common(15)
    for varietal, count in varietal_counts:
        print(f"  {varietal:25s}: {count}")


def report_price_analysis(df):
    """Report price analysis"""
    print_separator("PRICE ANALYSIS")

    priced = df[df['has_price']].copy()
    print(f"Coffees with price data: {len(priced)} ({len(priced)/len(df)*100:.1f}%)\n")

    if priced.empty:
        print("No price data available.")
        return

    prices = priced['average_per_lb']
    print(f"--- Overall Price Statistics ($/lb) ---")
    print(f"  Mean:   ${prices.mean():.2f}")
    print(f"  Median: ${prices.median():.2f}")
    print(f"  Std:    ${prices.std():.2f}")
    print(f"  Min:    ${prices.min():.2f}")
    print(f"  Max:    ${prices.max():.2f}")
    print(f"  Q25:    ${prices.quantile(0.25):.2f}")
    print(f"  Q75:    ${prices.quantile(0.75):.2f}")

    print(f"\n--- Median Price by Country (min 5 coffees) ---")
    for country, group in priced.groupby('country_final'):
        if len(group) >= 5:
            med = group['average_per_lb'].median()
            print(f"  {country:25s}: ${med:.2f}  (n={len(group)})")

    print(f"\n--- Median Price by Process Method (min 5 coffees) ---")
    for process, group in priced.groupby('process_clean'):
        if pd.notna(process) and len(group) >= 5:
            med = group['average_per_lb'].median()
            print(f"  {str(process):25s}: ${med:.2f}  (n={len(group)})")

    # Price-flavor analysis
    print(f"\n--- Price by Flavor Family (median price WITH vs WITHOUT) ---")
    all_families = set()
    for flist in priced['flavor_families']:
        all_families.update(flist)

    flavor_prices = []
    for family in sorted(all_families):
        if not family:
            continue
        has = priced[priced['flavor_families'].apply(lambda x: family in x)]
        lacks = priced[priced['flavor_families'].apply(lambda x: family not in x)]
        if len(has) >= 5 and len(lacks) >= 5:
            med_with = has['average_per_lb'].median()
            med_without = lacks['average_per_lb'].median()
            diff = med_with - med_without
            flavor_prices.append((family, med_with, med_without, diff, len(has)))

    flavor_prices.sort(key=lambda x: x[3], reverse=True)
    for family, with_p, without_p, diff, n in flavor_prices:
        sign = '+' if diff >= 0 else ''
        print(f"  {family:25s}: ${with_p:.2f} vs ${without_p:.2f} ({sign}${diff:.2f})  n={n}")


def report_turnover(df):
    """Report seller turnover and lifespan analysis"""
    print_separator("SELLER TURNOVER & COFFEE LIFESPAN")

    expired = df[
        (df['is_active'] == False) &
        df['lifespan_days'].notna() &
        (df['lifespan_days'] >= 0)
    ].copy()

    print(f"Expired coffees with lifespan data: {len(expired)}")
    print(f"Active coffees: {df['is_active'].sum()}\n")

    if expired.empty:
        print("No expired coffees with date data available.")
        return

    lifespans = expired['lifespan_days']
    print(f"--- Overall Lifespan Statistics (days) ---")
    print(f"  Mean:   {lifespans.mean():.0f}")
    print(f"  Median: {lifespans.median():.0f}")
    print(f"  Std:    {lifespans.std():.0f}")
    print(f"  Min:    {lifespans.min():.0f}")
    print(f"  Max:    {lifespans.max():.0f}")
    print(f"  Q25:    {lifespans.quantile(0.25):.0f}")
    print(f"  Q75:    {lifespans.quantile(0.75):.0f}")

    print(f"\n--- Turnover by Seller ---")
    print(f"  {'Seller':25s} {'Total':>6s} {'Active':>7s} {'Expired':>8s} {'Rate':>6s} {'Med Life':>9s}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*8} {'-'*6} {'-'*9}")

    for seller, group in df.groupby('seller_name'):
        if pd.isna(seller) or not seller or seller == 'Unknown':
            continue
        exp = group[(group['is_active'] == False) & group['lifespan_days'].notna() & (group['lifespan_days'] >= 0)]
        act = group[group['is_active'] == True]
        rate = len(exp) / len(group) if len(group) > 0 else 0
        med = f"{exp['lifespan_days'].median():.0f}" if len(exp) > 0 else "N/A"
        print(f"  {seller:25s} {len(group):6d} {len(act):7d} {len(exp):8d} {rate:5.0%} {med:>9s}")

    print(f"\n--- Median Lifespan by Country (min 3 expired) ---")
    for country, group in expired.groupby('country_final'):
        if pd.isna(country) or len(group) < 3:
            continue
        med = group['lifespan_days'].median()
        print(f"  {country:25s}: {med:.0f} days  (n={len(group)})")

    print(f"\n--- Median Lifespan by Process (min 3 expired) ---")
    for process, group in expired.groupby('process_clean'):
        if pd.isna(process) or len(group) < 3:
            continue
        med = group['lifespan_days'].median()
        print(f"  {str(process):25s}: {med:.0f} days  (n={len(group)})")


def report_flavor_cooccurrence(df):
    """Report flavor co-occurrence patterns"""
    print_separator("FLAVOR CO-OCCURRENCE")

    from collections import Counter
    from itertools import combinations

    coffees_with_flavors = df[df['has_flavors']].copy()
    print(f"Coffees with flavor data: {len(coffees_with_flavors)}\n")

    if coffees_with_flavors.empty:
        return

    # Family level co-occurrence
    pair_counts = Counter()
    family_counts = Counter()
    total = 0

    for _, row in coffees_with_flavors.iterrows():
        families = row['flavor_families']
        if not families or len(families) < 1:
            continue
        total += 1
        unique = sorted(set(families))
        for f in unique:
            family_counts[f] += 1
        for f1, f2 in combinations(unique, 2):
            pair_counts[(f1, f2)] += 1

    print(f"--- Flavor Family Frequencies ---")
    for family, count in family_counts.most_common():
        pct = count / total * 100
        print(f"  {family:25s}: {count:4d} ({pct:.1f}%)")

    print(f"\n--- Top 20 Flavor Family Pairs ---")
    print(f"  {'Pair':45s} {'Count':>6s} {'Rate':>7s}")
    print(f"  {'-'*45} {'-'*6} {'-'*7}")
    for (f1, f2), count in pair_counts.most_common(20):
        rate = count / total * 100
        print(f"  {f1 + ' + ' + f2:45s} {count:6d} {rate:6.1f}%")

    # PMI analysis
    print(f"\n--- Top 15 Surprising Pairs (Highest PMI) ---")
    print(f"  (Pairs that appear together MORE often than expected)")
    import math
    pmi_records = []
    for (f1, f2), pair_count in pair_counts.items():
        p_pair = pair_count / total
        p_f1 = family_counts[f1] / total
        p_f2 = family_counts[f2] / total
        if p_f1 > 0 and p_f2 > 0 and p_pair > 0:
            pmi = math.log2(p_pair / (p_f1 * p_f2))
            pmi_records.append((f1, f2, pmi, pair_count))

    pmi_records.sort(key=lambda x: x[2], reverse=True)
    for f1, f2, pmi, count in pmi_records[:15]:
        print(f"  {f1 + ' + ' + f2:45s} PMI={pmi:+.3f}  (n={count})")

    print(f"\n--- Top 10 Avoiding Pairs (Lowest PMI) ---")
    print(f"  (Pairs that appear together LESS often than expected)")
    for f1, f2, pmi, count in pmi_records[-10:]:
        print(f"  {f1 + ' + ' + f2:45s} PMI={pmi:+.3f}  (n={count})")


def report_cross_feature(df):
    """Report interesting cross-feature relationships"""
    print_separator("CROSS-FEATURE HIGHLIGHTS")

    # Process x Country
    print(f"--- Process Method by Country (top 5 countries) ---")
    top_countries = df['country_final'].value_counts().head(5).index
    for country in top_countries:
        country_df = df[df['country_final'] == country]
        process_dist = country_df['process_clean'].value_counts()
        total = process_dist.sum()
        pcts = ', '.join(f"{p}: {c/total:.0%}" for p, c in process_dist.items() if pd.notna(p))
        print(f"  {country:20s} (n={len(country_df)}): {pcts}")

    # Flavor families by process
    procs_with_flavors = df[df['has_flavors'] & df['has_process']].copy()
    if not procs_with_flavors.empty:
        print(f"\n--- Top Flavor Families by Process Method ---")
        for process, group in procs_with_flavors.groupby('process_clean'):
            if pd.isna(process) or len(group) < 5:
                continue
            from collections import Counter
            all_fams = []
            for flist in group['flavor_families']:
                all_fams.extend(flist)
            top3 = Counter(all_fams).most_common(3)
            top_str = ', '.join(f"{f}({c})" for f, c in top3)
            print(f"  {str(process):25s} (n={len(group)}): {top_str}")

    # Price by varietal (top varietals)
    priced_with_varietal = df[df['has_price'] & df['has_varietal']].copy()
    if not priced_with_varietal.empty:
        print(f"\n--- Median Price by Top Varietals ---")
        from collections import Counter
        varietal_prices = {}
        for _, row in priced_with_varietal.iterrows():
            for v in row['varietals_parsed']:
                if v not in varietal_prices:
                    varietal_prices[v] = []
                varietal_prices[v].append(row['average_per_lb'])

        var_stats = []
        for v, prices in varietal_prices.items():
            if len(prices) >= 5:
                import numpy as np
                var_stats.append((v, np.median(prices), len(prices)))

        var_stats.sort(key=lambda x: x[1], reverse=True)
        for v, med, n in var_stats[:15]:
            print(f"  {v:25s}: ${med:.2f}/lb  (n={n})")


def main():
    print(f"\n{'#'*80}")
    print(f"  COFFEE MARKET ANALYTICS - PRELIMINARY REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}")

    # Fetch and prepare data
    merged = fetch_data()
    df = prepare_data(merged)

    # Run all reports
    report_data_completeness(df)
    report_overview(df)
    report_price_analysis(df)
    report_turnover(df)
    report_flavor_cooccurrence(df)
    report_cross_feature(df)

    print(f"\n{'='*80}")
    print(f"  REPORT COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
