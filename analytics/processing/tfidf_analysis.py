"""
TF-IDF analysis for coffee flavor distinctiveness

This module implements TF-IDF (Term Frequency-Inverse Document Frequency) analysis
to identify which flavors are most distinctive for each geographic region or seller.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter
import math


class FlavorTFIDFAnalyzer:
    """Analyze flavor distinctiveness using TF-IDF methodology"""
    
    def __init__(self, tfidf_data: Dict[str, Dict[str, Dict[str, List[str]]]]):
        """
        Initialize with TF-IDF format data
        
        Args:
            tfidf_data: Dictionary from coffee_data_extractor.prepare_tfidf_format()
        """
        self.data = tfidf_data
        self.tfidf_scores = {}
        self.idf_scores = {}
    
    def calculate_tf(self, terms: List[str]) -> Dict[str, float]:
        """
        Calculate Term Frequency for a document (unit)
        
        Args:
            terms: List of flavor terms (can include duplicates)
            
        Returns:
            Dictionary of term frequencies
        """
        if not terms:
            return {}
        
        # Count occurrences
        term_counts = Counter(terms)
        total_terms = len(terms)
        
        # Calculate TF (using raw frequency)
        tf_scores = {
            term: count / total_terms 
            for term, count in term_counts.items()
        }
        
        return tf_scores
    
    def calculate_idf(self, documents: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate Inverse Document Frequency across all documents
        
        Args:
            documents: Dictionary mapping document names to term lists
            
        Returns:
            Dictionary of IDF scores for each unique term
        """
        # Total number of documents
        total_docs = len(documents)
        
        # Count documents containing each term
        doc_counts = {}
        all_terms = set()
        
        for doc_terms in documents.values():
            unique_terms = set(doc_terms)
            all_terms.update(unique_terms)
            
            for term in unique_terms:
                doc_counts[term] = doc_counts.get(term, 0) + 1
        
        # Calculate IDF
        idf_scores = {}
        for term in all_terms:
            # Add 1 to avoid division by zero and smooth the values
            idf_scores[term] = math.log(total_docs / (doc_counts.get(term, 0) + 1))
        
        return idf_scores
    
    def calculate_tfidf_scores(self, taxonomy_level: str, unit_type: str) -> pd.DataFrame:
        """
        Calculate TF-IDF scores for all units at a specific taxonomy level
        
        Args:
            taxonomy_level: 'family', 'genus', or 'species'
            unit_type: 'countries', 'regions', or 'sellers'
            
        Returns:
            DataFrame with TF-IDF scores
        """
        # Get documents for this level and type
        documents = self.data[f'{taxonomy_level}_level'][unit_type]
        
        # Calculate IDF scores
        idf_scores = self.calculate_idf(documents)
        
        # Store IDF for later use
        self.idf_scores[f'{taxonomy_level}_{unit_type}'] = idf_scores
        
        # Calculate TF-IDF for each document
        results = []
        
        for unit_name, terms in documents.items():
            # Calculate TF
            tf_scores = self.calculate_tf(terms)
            
            # Calculate TF-IDF
            for term, tf in tf_scores.items():
                tfidf = tf * idf_scores.get(term, 0)
                
                results.append({
                    'unit_name': unit_name,
                    'unit_type': unit_type.rstrip('s'),  # Remove plural
                    'taxonomy_level': taxonomy_level,
                    'flavor': term,
                    'tf_score': tf,
                    'idf_score': idf_scores.get(term, 0),
                    'tfidf_score': tfidf,
                    'term_count': terms.count(term),
                    'total_terms': len(terms)
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Add rankings within each unit
        if not df.empty:
            df['rank_within_unit'] = df.groupby('unit_name')['tfidf_score'].rank(
                ascending=False, method='dense'
            ).astype(int)
        
        return df
    
    def get_distinctive_flavors(self, unit_name: str, unit_type: str, 
                              taxonomy_level: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get the most distinctive flavors for a specific unit
        
        Args:
            unit_name: Name of the unit
            unit_type: Type of unit (singular form)
            taxonomy_level: Taxonomy level
            top_n: Number of top flavors to return
            
        Returns:
            DataFrame with top distinctive flavors
        """
        # Calculate if not already done
        key = f'{taxonomy_level}_{unit_type}s'
        if key not in self.tfidf_scores:
            self.tfidf_scores[key] = self.calculate_tfidf_scores(
                taxonomy_level, f'{unit_type}s'
            )
        
        # Filter for specific unit
        unit_scores = self.tfidf_scores[key][
            self.tfidf_scores[key]['unit_name'] == unit_name
        ].copy()
        
        # Sort by TF-IDF score and get top N
        top_flavors = unit_scores.nlargest(top_n, 'tfidf_score')
        
        return top_flavors
    
    def create_flavor_fingerprints(self, taxonomy_level: str = 'family', 
                                 top_n: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Create flavor fingerprints for all units showing their distinctive flavors
        
        Args:
            taxonomy_level: Taxonomy level to analyze
            top_n: Number of top flavors per unit
            
        Returns:
            Dictionary of flavor fingerprints by unit type
        """
        fingerprints = {}
        
        for unit_type in ['countries', 'regions', 'sellers']:
            # Calculate TF-IDF scores
            scores_df = self.calculate_tfidf_scores(taxonomy_level, unit_type)
            
            # Group by unit
            unit_fingerprints = {}
            for unit_name in scores_df['unit_name'].unique():
                unit_data = scores_df[scores_df['unit_name'] == unit_name]
                
                # Get top flavors
                top_flavors = unit_data.nlargest(top_n, 'tfidf_score')
                
                unit_fingerprints[unit_name] = {
                    'top_flavors': top_flavors[['flavor', 'tfidf_score', 'tf_score']].to_dict('records'),
                    'total_unique_flavors': len(unit_data),
                    'total_flavor_instances': unit_data['term_count'].sum()
                }
            
            fingerprints[unit_type] = unit_fingerprints
        
        return fingerprints
    
    def calculate_similarity_matrix(self, taxonomy_level: str, unit_type: str) -> pd.DataFrame:
        """
        Calculate cosine similarity between units based on TF-IDF vectors
        
        Args:
            taxonomy_level: Taxonomy level to use
            unit_type: Type of units to compare
            
        Returns:
            DataFrame with similarity scores
        """
        # Get TF-IDF scores
        scores_df = self.calculate_tfidf_scores(taxonomy_level, unit_type)
        
        # Pivot to create document-term matrix
        dt_matrix = scores_df.pivot_table(
            index='unit_name',
            columns='flavor',
            values='tfidf_score',
            fill_value=0
        )
        
        # Calculate cosine similarity
        # Normalize vectors
        norms = np.sqrt((dt_matrix ** 2).sum(axis=1))
        normalized = dt_matrix.div(norms, axis=0)
        
        # Calculate similarity matrix
        similarity = normalized.dot(normalized.T)
        
        return similarity
    
    def find_unique_flavors(self, taxonomy_level: str = 'species') -> Dict[str, List[Dict[str, Any]]]:
        """
        Find flavors that are unique or nearly unique to specific units
        
        Args:
            taxonomy_level: Taxonomy level to analyze
            
        Returns:
            Dictionary of unique flavors by unit type
        """
        unique_flavors = {}
        
        for unit_type in ['countries', 'regions', 'sellers']:
            documents = self.data[f'{taxonomy_level}_level'][unit_type]
            
            # Count which units have each flavor
            flavor_units = {}
            for unit_name, terms in documents.items():
                unique_terms = set(terms)
                for term in unique_terms:
                    if term not in flavor_units:
                        flavor_units[term] = []
                    flavor_units[term].append(unit_name)
            
            # Find flavors in only 1-2 units
            unique_list = []
            for flavor, units in flavor_units.items():
                if len(units) <= 2:
                    unique_list.append({
                        'flavor': flavor,
                        'units': units,
                        'exclusivity': 1 / len(units)
                    })
            
            # Sort by exclusivity
            unique_list.sort(key=lambda x: x['exclusivity'], reverse=True)
            unique_flavors[unit_type] = unique_list
        
        return unique_flavors
    
    def compare_units(self, unit1: str, unit2: str, unit_type: str, 
                     taxonomy_level: str = 'family') -> Dict[str, Any]:
        """
        Compare two units to find distinguishing flavors
        
        Args:
            unit1: First unit name
            unit2: Second unit name
            unit_type: Type of units (singular)
            taxonomy_level: Taxonomy level
            
        Returns:
            Comparison results
        """
        # Get documents
        documents = self.data[f'{taxonomy_level}_level'][f'{unit_type}s']
        
        if unit1 not in documents or unit2 not in documents:
            return {'error': 'Unit not found'}
        
        # Get terms for each unit
        terms1 = set(documents[unit1])
        terms2 = set(documents[unit2])
        
        # Calculate TF for each unit
        tf1 = self.calculate_tf(documents[unit1])
        tf2 = self.calculate_tf(documents[unit2])
        
        # Get IDF scores
        idf = self.calculate_idf(documents)
        
        # Find distinctive flavors
        unique_to_1 = terms1 - terms2
        unique_to_2 = terms2 - terms1
        shared = terms1 & terms2
        
        # Calculate distinctiveness scores
        distinctive_1 = []
        for term in terms1:
            tfidf1 = tf1.get(term, 0) * idf.get(term, 0)
            tfidf2 = tf2.get(term, 0) * idf.get(term, 0)
            if tfidf1 > tfidf2 * 1.5:  # At least 50% higher
                distinctive_1.append({
                    'flavor': term,
                    'tfidf_unit1': tfidf1,
                    'tfidf_unit2': tfidf2,
                    'ratio': tfidf1 / (tfidf2 + 0.001)
                })
        
        distinctive_2 = []
        for term in terms2:
            tfidf1 = tf1.get(term, 0) * idf.get(term, 0)
            tfidf2 = tf2.get(term, 0) * idf.get(term, 0)
            if tfidf2 > tfidf1 * 1.5:
                distinctive_2.append({
                    'flavor': term,
                    'tfidf_unit1': tfidf1,
                    'tfidf_unit2': tfidf2,
                    'ratio': tfidf2 / (tfidf1 + 0.001)
                })
        
        # Sort by ratio
        distinctive_1.sort(key=lambda x: x['ratio'], reverse=True)
        distinctive_2.sort(key=lambda x: x['ratio'], reverse=True)
        
        return {
            'unit1': unit1,
            'unit2': unit2,
            'unique_to_unit1': list(unique_to_1),
            'unique_to_unit2': list(unique_to_2),
            'shared_flavors': list(shared),
            'distinctive_for_unit1': distinctive_1[:10],
            'distinctive_for_unit2': distinctive_2[:10],
            'total_flavors_unit1': len(terms1),
            'total_flavors_unit2': len(terms2)
        }


def run_full_tfidf_analysis(tfidf_data: Dict) -> Dict[str, Any]:
    """
    Run complete TF-IDF analysis across all levels
    
    Args:
        tfidf_data: TF-IDF format data from extraction
        
    Returns:
        Dictionary of analysis results
    """
    analyzer = FlavorTFIDFAnalyzer(tfidf_data)
    results = {}
    
    # Calculate TF-IDF scores for all combinations
    for taxonomy_level in ['family', 'genus', 'species']:
        for unit_type in ['countries', 'regions', 'sellers']:
            key = f'{taxonomy_level}_{unit_type}'
            results[f'{key}_scores'] = analyzer.calculate_tfidf_scores(
                taxonomy_level, unit_type
            )
    
    # Create flavor fingerprints
    for taxonomy_level in ['family', 'genus', 'species']:
        results[f'{taxonomy_level}_fingerprints'] = analyzer.create_flavor_fingerprints(
            taxonomy_level, top_n=10
        )
    
    # Find unique flavors
    results['unique_flavors'] = {}
    for taxonomy_level in ['family', 'genus', 'species']:
        results['unique_flavors'][taxonomy_level] = analyzer.find_unique_flavors(
            taxonomy_level
        )
    
    # Store analyzer for further use
    results['analyzer'] = analyzer
    
    return results