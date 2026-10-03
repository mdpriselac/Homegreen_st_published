"""
Cached Data Loader for Frontend

Fast loading of pre-computed analysis results for the frontend interface.
"""

import copy
import json
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import streamlit as st
from datetime import datetime, timedelta


@st.cache_resource(ttl=3600, show_spinner=False)
def _read_cache_file(path: str, mtime: float) -> Dict[str, Any]:
    """Parse the cache JSON once per file version (keyed on mtime).

    Raises on failure; Streamlit does not memoize exceptions, so a bad or
    missing file is retried on the next call.
    """
    with open(path, 'r') as f:
        return json.load(f)


class CachedDataLoader:
    """Load pre-computed frontend data from cache"""
    
    DEFAULT_CACHE_DIR = "analytics/data/frontend_cache"

    def __init__(self, cache_dir: Optional[str] = None):
        # Precedence: explicit argument, then the ANALYTICS_CACHE_DIR environment
        # variable (absolute path, e.g. to preview a cache generated elsewhere),
        # then the default relative to the project root.
        if cache_dir is None:
            cache_dir = os.environ.get('ANALYTICS_CACHE_DIR') or self.DEFAULT_CACHE_DIR
        # Resolve path relative to project root
        if not os.path.isabs(cache_dir):
            # Get the project root (where this script is running from)
            project_root = Path.cwd()
            self.cache_dir = project_root / cache_dir
        else:
            self.cache_dir = Path(cache_dir)
        
        self._cache = {}
        self._cache_loaded = False
    
    def load_overview_data(_self) -> Dict[str, Any]:
        """Load overview tab data"""
        return _self._load_component('overview_data')
    
    def load_unit_profile(_self, unit_name: str, unit_type: str) -> Optional[Dict[str, Any]]:
        """Load comprehensive profile for a specific unit"""
        unit_profiles = _self._peek_component('distinctiveness_profiles')
        profile_key = f"{unit_type}_{unit_name}"
        return copy.deepcopy(unit_profiles.get(profile_key))
    
    def load_flavor_hierarchies(_self) -> Dict[str, Any]:
        """Load flavor hierarchy data for By Flavor tab"""
        return _self._load_component('flavor_hierarchies')
    
    def load_rankings_data(_self) -> Dict[str, Any]:
        """Load rankings data for Rankings tab"""
        return _self._load_component('rankings_data')
    
    def get_available_units(_self, unit_type: str) -> List[str]:
        """Get list of available units for the specified type"""
        unit_profiles = _self._peek_component('distinctiveness_profiles')  # read-only, result is new

        units = []
        for profile_key, profile in unit_profiles.items():
            if profile.get('unit_type') == unit_type:
                units.append(profile.get('unit_name'))

        return sorted(units)

    def get_unit_sizes(_self, unit_type: str) -> Dict[str, int]:
        """{unit_name: n_coffees} for the available units of a type (for sensible picker defaults)"""
        profiles = _self._peek_component('distinctiveness_profiles')
        return {p.get('unit_name'): int(p.get('n_coffees') or 0)
                for p in profiles.values() if p.get('unit_type') == unit_type}

    def load_distinctiveness_meta(_self) -> Dict[str, Any]:
        """Parameters, counts and key findings of the distinctiveness analysis"""
        return _self._load_component('distinctiveness_meta')

    def load_flavor_unit_rows(_self, unit_type: str, level: str, flavor: str) -> List[Dict[str, Any]]:
        """All tested (unit, flavor) rows for one flavor, as dicts (read-only scan of
        the shared by-flavor table; only the matching rows are copied)."""
        table = _self._peek_component('distinctiveness_by_flavor').get(f"{unit_type}_{level}")
        if not table or not table.get('rows'):
            return []
        cols = table['columns']
        fi = cols.index('flavor')
        return [dict(zip(cols, row)) for row in table['rows'] if row[fi] == flavor]

    def load_turnover_data(_self) -> Dict[str, Any]:
        """Load turnover/lifespan analysis data"""
        return _self._load_component('turnover_data')

    def load_price_analysis_data(_self) -> Dict[str, Any]:
        """Load price analysis data"""
        return _self._load_component('price_analysis_data')

    def load_cooccurrence_data(_self) -> Dict[str, Any]:
        """Load flavor co-occurrence analysis data"""
        return _self._load_component('cooccurrence_data')

    def load_cross_feature_data(_self) -> Dict[str, Any]:
        """Load cross-feature analysis data"""
        return _self._load_component('cross_feature_data')

    def load_interaction_data(_self) -> Dict[str, Any]:
        """Load interaction analysis data"""
        return _self._load_component('interaction_data')

    def load_data_completeness(_self) -> Dict[str, Any]:
        """Load data completeness/coverage rates"""
        return _self._load_component('data_completeness')

    def get_cache_metadata(_self) -> Dict[str, Any]:
        """Get cache metadata including generation time"""
        return _self._load_component('metadata')
    
    def _peek_component(self, component_name: str) -> Dict[str, Any]:
        """Shared (uncopied) component. Internal read-only use only: the
        parsed cache is shared across sessions via st.cache_resource."""
        if not self._cache_loaded:
            self._load_full_cache()
        return self._cache.get(component_name, {})

    def _load_component(self, component_name: str) -> Dict[str, Any]:
        """Load a specific cache component as a private deep copy, so callers
        may mutate it without affecting other sessions or later calls."""
        return copy.deepcopy(self._peek_component(component_name))
    
    def _load_full_cache(self):
        """Load full cache from file.

        On failure the error is shown and nothing is memoized (neither here nor
        in Streamlit's cache), so the next call retries.
        """
        main_cache_file = self.cache_dir / "frontend_cache.json"

        if not main_cache_file.exists():
            st.error("Analytics data is temporarily unavailable (cache file not found).")
            return

        try:
            self._cache = _read_cache_file(str(main_cache_file), main_cache_file.stat().st_mtime)
            self._cache_loaded = True
        except Exception as e:
            self._cache = {}
            self._cache_loaded = False
            st.error(f"Analytics data could not be loaded ({type(e).__name__}). Try reloading the cache from the sidebar.")

    def is_cache_fresh(self, max_age_hours: int = 24) -> bool:
        """Check if cache is fresh enough"""
        metadata = self.get_cache_metadata()
        
        if not metadata or 'generated_at' not in metadata:
            return False
        
        try:
            generated_at = datetime.fromisoformat(metadata['generated_at'])
            age = datetime.now() - generated_at
            return age < timedelta(hours=max_age_hours)
        except:
            return False
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get detailed cache status information"""
        metadata = self.get_cache_metadata()
        
        if not metadata:
            return {
                'exists': False,
                'fresh': False,
                'age_hours': None,
                'components': []
            }
        
        try:
            generated_at = datetime.fromisoformat(metadata['generated_at'])
            age = datetime.now() - generated_at
            age_hours = age.total_seconds() / 3600
        except:
            age_hours = None
        
        return {
            'exists': True,
            'fresh': self.is_cache_fresh(),
            'age_hours': age_hours,
            'generated_at': metadata.get('generated_at'),
            'version': metadata.get('version'),
            'total_units': metadata.get('total_units', {}),
            'components': metadata.get('analysis_components', [])
        }


# Global loader instance
_loader = None

def get_cached_data_loader() -> CachedDataLoader:
    """Get singleton cached data loader"""
    global _loader
    if _loader is None:
        _loader = CachedDataLoader()
    return _loader


def clear_all_caches():
    """Drop the loaded cache so the next access re-reads the file from disk"""
    global _loader
    _loader = None
    _read_cache_file.clear()


# Convenience functions for frontend use
def load_overview_data() -> Dict[str, Any]:
    """Load overview tab data"""
    return get_cached_data_loader().load_overview_data()


def load_unit_profile(unit_name: str, unit_type: str) -> Optional[Dict[str, Any]]:
    """Load unit profile"""
    return get_cached_data_loader().load_unit_profile(unit_name, unit_type)


def get_available_units(unit_type: str) -> List[str]:
    """Get available units for type"""
    return get_cached_data_loader().get_available_units(unit_type)


def load_flavor_hierarchies() -> Dict[str, Any]:
    """Load flavor hierarchies"""
    return get_cached_data_loader().load_flavor_hierarchies()


def load_rankings_data() -> Dict[str, Any]:
    """Load rankings data"""
    return get_cached_data_loader().load_rankings_data()


def get_unit_sizes(unit_type: str) -> Dict[str, int]:
    """Coffees per available unit"""
    return get_cached_data_loader().get_unit_sizes(unit_type)


def load_distinctiveness_meta() -> Dict[str, Any]:
    """Load distinctiveness parameters and key findings"""
    return get_cached_data_loader().load_distinctiveness_meta()


def load_flavor_unit_rows(unit_type: str, level: str, flavor: str) -> List[Dict[str, Any]]:
    """Tested (unit, flavor) rows for one flavor"""
    return get_cached_data_loader().load_flavor_unit_rows(unit_type, level, flavor)


def load_turnover_data() -> Dict[str, Any]:
    """Load turnover data"""
    return get_cached_data_loader().load_turnover_data()


def load_price_analysis_data() -> Dict[str, Any]:
    """Load price analysis data"""
    return get_cached_data_loader().load_price_analysis_data()


def load_cooccurrence_data() -> Dict[str, Any]:
    """Load co-occurrence data"""
    return get_cached_data_loader().load_cooccurrence_data()


def load_cross_feature_data() -> Dict[str, Any]:
    """Load cross-feature data"""
    return get_cached_data_loader().load_cross_feature_data()


def load_interaction_data() -> Dict[str, Any]:
    """Load interaction data"""
    return get_cached_data_loader().load_interaction_data()


def load_data_completeness() -> Dict[str, Any]:
    """Load data completeness"""
    return get_cached_data_loader().load_data_completeness()


def get_cache_status() -> Dict[str, Any]:
    """Get cache status"""
    return get_cached_data_loader().get_cache_status()


def show_cache_status_widget():
    """Show cache status widget in Streamlit sidebar"""
    status = get_cache_status()
    
    with st.sidebar:
        st.subheader("📊 Data Cache Status")
        
        if not status['exists']:
            st.error("❌ Cache not found")
            st.info("The analytics cache has not been generated yet.")
            return
        
        if status['fresh']:
            st.success("✅ Cache is fresh")
        else:
            st.warning("⚠️ Cache may be stale")
        
        if status['age_hours'] is not None:
            if status['age_hours'] < 1:
                age_str = f"{status['age_hours']*60:.0f} minutes ago"
            elif status['age_hours'] < 24:
                age_str = f"{status['age_hours']:.1f} hours ago"
            else:
                age_str = f"{status['age_hours']/24:.1f} days ago"
            
            st.caption(f"Generated: {age_str}")
        
        # Show unit counts
        total_units = status.get('total_units', {})
        if total_units:
            from analytics.constants import unit_plural
            from analytics.processing.distinctiveness import MIN_UNIT_COFFEES
            parts = " · ".join(f"{count} {unit_plural(t) if count != 1 else t}"
                               for t, count in total_units.items())
            st.caption(f"**Profiles ({MIN_UNIT_COFFEES}+ coffees):** {parts}")
        
        # Refresh button
        if st.button("🔄 Reload Cache"):
            # Re-read the cache file from disk
            clear_all_caches()
            st.rerun()