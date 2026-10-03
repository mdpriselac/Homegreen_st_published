#!/usr/bin/env python3
"""
Standalone cache generation script that works outside Streamlit.
Patches st.secrets with values from .streamlit/secrets.toml.
"""

import os
import sys
import tomllib

# Load secrets before importing anything that uses st.secrets
secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
with open(secrets_path, 'rb') as f:
    secrets_data = tomllib.load(f)

# Patch streamlit secrets
import streamlit as st
from streamlit.runtime.secrets import AttrDict

# Build the secrets object
st.secrets._secrets = AttrDict(secrets_data)

# Now import and run
from analytics.frontend.data_cache_generator import FrontendDataCacheGenerator

def main():
    print("Starting full cache generation...")
    generator = FrontendDataCacheGenerator()
    cache_data = generator.generate_full_cache()

    print(f"\nCache generation completed!")
    print(f"  Cache saved to: {generator.cache_dir}")
    print(f"  Unit profiles (10+ coffees): {len(cache_data.get('distinctiveness_profiles', {}))}")
    print(f"  Key findings: {len(cache_data.get('distinctiveness_meta', {}).get('key_findings', []))}")
    print(f"  Flavor families: {len(cache_data.get('flavor_hierarchies', {}).get('families', []))}")
    print(f"  Rankings categories: {len(cache_data.get('rankings_data', {}))}")

    # Report on new caches
    turnover = cache_data.get('turnover_data', {})
    print(f"  Turnover data: {'OK' if turnover.get('lifespan_overview', {}).get('has_data') else 'No data'}")

    price = cache_data.get('price_analysis_data', {})
    print(f"  Price analysis: {'OK' if price.get('overview', {}).get('has_data') else 'No data'}")

    cooccurrence = cache_data.get('cooccurrence_data', {})
    print(f"  Co-occurrence: {'OK' if cooccurrence.get('family_level') else 'No data'}")

    completeness = cache_data.get('data_completeness', {})
    print(f"  Data completeness: {'OK' if completeness.get('has_data') else 'No data'}")

    cross_feature = cache_data.get('cross_feature_data', {})
    has_xf = bool(cross_feature.get('process_by_origin') or cross_feature.get('all_associations'))
    print(f"  Cross-feature: {'OK' if has_xf else 'No data'}")


if __name__ == '__main__':
    main()
