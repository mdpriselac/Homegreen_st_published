"""
Cached Data Loader for Frontend

Fast loading of pre-computed analysis results for the frontend interface.
"""

import json
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import streamlit as st
from datetime import datetime, timedelta


class CachedDataLoader:
    """Load pre-computed frontend data from cache"""
    
    def __init__(self, cache_dir: str = "analytics/data/frontend_cache"):
        # Resolve path relative to project root
        if not os.path.isabs(cache_dir):
            # Get the project root (where this script is running from)
            project_root = Path.cwd()
            self.cache_dir = project_root / cache_dir
        else:
            self.cache_dir = Path(cache_dir)
        
        self._cache = {}
        self._cache_loaded = False
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_overview_data(_self) -> Dict[str, Any]:
        """Load overview tab data"""
        return _self._load_component('overview_data')
    
    @st.cache_data(ttl=3600)
    def load_unit_profile(_self, unit_name: str, unit_type: str) -> Optional[Dict[str, Any]]:
        """Load comprehensive profile for a specific unit"""
        unit_profiles = _self._load_component('unit_profiles')
        profile_key = f"{unit_type}_{unit_name}"
        return unit_profiles.get(profile_key)
    
    @st.cache_data(ttl=3600)
    def load_flavor_hierarchies(_self) -> Dict[str, Any]:
        """Load flavor hierarchy data for By Flavor tab"""
        return _self._load_component('flavor_hierarchies')
    
    @st.cache_data(ttl=3600)
    def load_rankings_data(_self) -> Dict[str, Any]:
        """Load rankings data for Rankings tab"""
        return _self._load_component('rankings_data')
    
    @st.cache_data(ttl=3600)
    def load_comparison_matrices(_self) -> Dict[str, Any]:
        """Load comparison matrices for Compare tab"""
        return _self._load_component('comparison_matrices')
    
    @st.cache_data(ttl=3600)
    def get_available_units(_self, unit_type: str) -> List[str]:
        """Get list of available units for the specified type"""
        unit_profiles = _self._load_component('unit_profiles')
        
        units = []
        for profile_key, profile in unit_profiles.items():
            if profile.get('unit_type') == unit_type:
                units.append(profile.get('unit_name'))
        
        return sorted(units)
    
    @st.cache_data(ttl=3600)
    def get_cache_metadata(_self) -> Dict[str, Any]:
        """Get cache metadata including generation time"""
        return _self._load_component('metadata')
    
    def _load_component(self, component_name: str) -> Dict[str, Any]:
        """Load a specific cache component"""
        if not self._cache_loaded:
            self._load_full_cache()
        
        return self._cache.get(component_name, {})
    
    def _load_full_cache(self):
        """Load full cache from file"""
        main_cache_file = self.cache_dir / "frontend_cache.json"
        
        if not main_cache_file.exists():
            st.error(f"Frontend cache not found at {main_cache_file}")
            if self.cache_dir.exists():
                cache_files = list(self.cache_dir.glob("*.json"))
                st.error(f"Files in cache dir: {[f.name for f in cache_files]}")
            else:
                st.error(f"Cache directory does not exist: {self.cache_dir.absolute()}")
            return
        
        try:
            with open(main_cache_file, 'r') as f:
                self._cache = json.load(f)
            self._cache_loaded = True
            
        except Exception as e:
            st.error(f"Failed to load frontend cache: {e}")
    
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
    """Clear all caches - both the singleton loader and Streamlit's cache"""
    global _loader
    # Reset the singleton loader
    _loader = None
    # Clear Streamlit's cache_data
    st.cache_data.clear()


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
            st.info("Run cache generator to create data cache")
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
            st.caption("**Units cached:**")
            for unit_type, count in total_units.items():
                st.caption(f"• {unit_type.title()}: {count}")
        
        # Refresh button
        if st.button("🔄 Refresh Cache Info"):
            # Clear cache to force reload
            get_cached_data_loader()._cache_loaded = False
            st.rerun()