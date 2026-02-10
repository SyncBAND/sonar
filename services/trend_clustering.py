"""
SONAR.AI — Trend Clustering Service
====================================

Identifies multiple distinct TRENDS within each taxonomy category.

A CATEGORY is a fixed taxonomy bucket (e.g., 'cybersecurity_ot').
A TREND is a specific pattern/topic within that category (e.g., 'ransomware attacks increasing').

This module:
1. Takes all signals within a category
2. Embeds them using the existing classifier backend
3. Clusters them to find distinct sub-topics
4. Names each cluster based on representative signals
5. Returns multiple trends per category

Example:
    Category: cybersecurity_ot (174 signals)
    └─ Trend 1: "Ransomware targeting energy sector" (62 signals)
    └─ Trend 2: "OT supply chain vulnerabilities" (48 signals)
    └─ Trend 3: "Zero-trust adoption in ICS" (35 signals)
    └─ Trend 4: "NERC CIP compliance updates" (29 signals)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger(__name__)

# Try to import clustering libraries
_SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

# Try to import sentence transformers for embeddings
_SBERT_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    pass


class TrendClusterer:
    """
    Identifies distinct trends within a category by clustering signals.
    
    Methodology:
    1. Embed signal titles + content snippets
    2. Use hierarchical clustering (better for unknown # of clusters)
    3. Determine optimal cluster count using silhouette score
    4. Name clusters using TF-IDF keywords from titles
    """
    
    def __init__(self, 
                 min_signals_per_trend: int = 3,
                 max_trends_per_category: int = 5,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            min_signals_per_trend: Minimum signals for a cluster to be a trend
            max_trends_per_category: Maximum trends to extract per category
            model_name: SentenceTransformer model for embeddings
        """
        self.min_signals_per_trend = min_signals_per_trend
        self.max_trends_per_category = max_trends_per_category
        self.model_name = model_name
        self._embedder = None
        
        if not _SKLEARN_AVAILABLE:
            log.warning("sklearn not available - trend clustering disabled")
        if not _SBERT_AVAILABLE:
            log.warning("sentence-transformers not available - using fallback")
    
    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None and _SBERT_AVAILABLE:
            log.info(f"Loading embedding model: {self.model_name}")
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder
    
    def _get_signal_text(self, signal: Dict) -> str:
        """Extract text from signal for embedding."""
        title = signal.get("title", "") or ""
        content = signal.get("content", "") or ""
        # Use title + first 200 chars of content
        snippet = content[:200] if content else ""
        return f"{title}. {snippet}".strip()
    
    def _embed_signals(self, signals: List[Dict]) -> Optional[np.ndarray]:
        """Embed signals using SentenceTransformer."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        
        texts = [self._get_signal_text(s) for s in signals]
        texts = [t if t else "empty" for t in texts]  # Handle empty
        
        try:
            embeddings = embedder.encode(texts, show_progress_bar=False, 
                                         normalize_embeddings=True)
            return embeddings
        except Exception as e:
            log.error(f"Embedding failed: {e}")
            return None
    
    def _find_optimal_clusters(self, 
                               embeddings: np.ndarray, 
                               max_clusters: int) -> int:
        """Find optimal number of clusters using silhouette score."""
        n_samples = len(embeddings)
        if n_samples < 4:
            return 1
        
        max_k = min(max_clusters, n_samples // self.min_signals_per_trend, n_samples - 1)
        max_k = max(2, max_k)
        
        best_k = 1
        best_score = -1
        
        for k in range(2, max_k + 1):
            try:
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = clusterer.fit_predict(embeddings)
                
                # Check if we have at least 2 clusters with enough samples
                counts = Counter(labels)
                valid_clusters = sum(1 for c in counts.values() if c >= self.min_signals_per_trend)
                if valid_clusters < 2:
                    continue
                
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        
        return best_k
    
    def _extract_cluster_keywords(self, 
                                  signals: List[Dict], 
                                  top_n: int = 5) -> List[str]:
        """Extract representative keywords from signal titles."""
        # Collect all words from titles
        all_words = []
        for s in signals:
            title = s.get("title", "") or ""
            # Basic tokenization - words 4+ chars, lowercase
            words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
            all_words.extend(words)
        
        # Common stopwords to filter
        stopwords = {
            'that', 'this', 'with', 'from', 'have', 'will', 'been', 'were',
            'what', 'when', 'where', 'which', 'while', 'would', 'could',
            'should', 'there', 'their', 'about', 'after', 'before', 'being',
            'between', 'both', 'through', 'during', 'each', 'into', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'more',
            'other', 'some', 'such', 'only', 'same', 'than', 'very', 'just',
            'also', 'most', 'make', 'made', 'many', 'much', 'must', 'like',
            'even', 'back', 'well', 'still', 'take', 'come', 'look', 'want',
            'give', 'first', 'last', 'long', 'great', 'little', 'good',
            'news', 'says', 'year', 'years', 'company', 'companies', 'report',
            'according', 'announces', 'announced', 'research', 'study',
        }
        
        # Count and filter
        word_counts = Counter(all_words)
        keywords = [
            word for word, count in word_counts.most_common(top_n * 3)
            if word not in stopwords and count >= 2
        ][:top_n]
        
        return keywords
    
    def _generate_trend_name(self, 
                             signals: List[Dict], 
                             category: str) -> str:
        """Generate a human-readable trend name from signals."""
        keywords = self._extract_cluster_keywords(signals, top_n=4)
        
        if not keywords:
            return f"{category.replace('_', ' ').title()} Activity"
        
        # Capitalize and join
        name_parts = [kw.title() for kw in keywords[:3]]
        return " ".join(name_parts)
    
    def _get_trend_description(self, 
                               signals: List[Dict],
                               keywords: List[str]) -> str:
        """Generate a brief description of the trend."""
        # Get most recent/representative title
        sorted_signals = sorted(
            signals, 
            key=lambda x: x.get("published_date", "") or "",
            reverse=True
        )
        
        if sorted_signals:
            representative = sorted_signals[0].get("title", "")[:100]
            return f"Pattern around: {representative}"
        
        return f"Pattern involving: {', '.join(keywords[:3])}"
    
    def _keyword_cluster(self, signals: List[Dict], n_clusters: int) -> List[int]:
        """
        Fallback clustering using TF-IDF similarity (no deep learning required).
        
        Uses scikit-learn TF-IDF to cluster signals by title similarity.
        """
        if not signals or not _SKLEARN_AVAILABLE:
            return [0] * len(signals)
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        
        # Extract titles
        titles = [(s.get("title", "") or "signal").strip() or "signal" for s in signals]
        
        try:
            # Vectorize with TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                min_df=1,
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(titles)
            
            # Cluster
            n_clusters = min(n_clusters, len(signals) - 1, tfidf_matrix.shape[0] - 1)
            n_clusters = max(2, n_clusters)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            
            return list(labels)
        except Exception as e:
            log.warning(f"TF-IDF clustering failed: {e}, using simple fallback")
            # Simple fallback - divide evenly
            return [i % n_clusters for i in range(len(signals))]
    
    def cluster_category(self, 
                         category: str,
                         signals: List[Dict]) -> List[Dict[str, Any]]:
        """
        Identify distinct trends within a category.
        
        Args:
            category: Category name (e.g., 'cybersecurity_ot')
            signals: List of signals in this category
            
        Returns:
            List of trend dicts, each with:
                - trend_id: Unique identifier
                - trend_name: Human-readable name
                - category: Parent category
                - signal_count: Number of signals
                - signals: List of signal dicts
                - keywords: Representative keywords
                - description: Brief description
        """
        n_signals = len(signals)
        
        # Too few signals - return as single trend
        if n_signals < self.min_signals_per_trend * 2:
            keywords = self._extract_cluster_keywords(signals)
            return [{
                "trend_id": f"{category}_trend_1",
                "trend_name": self._generate_trend_name(signals, category),
                "category": category,
                "signal_count": n_signals,
                "signals": signals,
                "keywords": keywords,
                "description": self._get_trend_description(signals, keywords),
                "cluster_id": 0,
            }]
        
        # Try ML-based clustering first
        embeddings = self._embed_signals(signals)
        use_ml = embeddings is not None and _SKLEARN_AVAILABLE
        
        if use_ml:
            # Find optimal number of clusters
            n_clusters = self._find_optimal_clusters(
                embeddings, 
                self.max_trends_per_category
            )
            
            if n_clusters > 1:
                # Perform ML clustering
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(embeddings)
            else:
                use_ml = False
        
        if not use_ml:
            # Fallback: keyword-based clustering
            n_clusters = min(self.max_trends_per_category, n_signals // self.min_signals_per_trend)
            n_clusters = max(2, n_clusters)
            labels = self._keyword_cluster(signals, n_clusters)
        
        # Group signals by cluster
        clusters: Dict[int, List[Dict]] = {}
        for i, signal in enumerate(signals):
            cluster_id = labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(signal)
        
        # Convert clusters to trends
        trends = []
        trend_num = 1
        
        for cluster_id, cluster_signals in sorted(
            clusters.items(), 
            key=lambda x: -len(x[1])  # Sort by size descending
        ):
            if len(cluster_signals) < self.min_signals_per_trend:
                continue
            
            keywords = self._extract_cluster_keywords(cluster_signals)
            trend = {
                "trend_id": f"{category}_trend_{trend_num}",
                "trend_name": self._generate_trend_name(cluster_signals, category),
                "category": category,
                "signal_count": len(cluster_signals),
                "signals": cluster_signals,
                "keywords": keywords,
                "description": self._get_trend_description(cluster_signals, keywords),
                "cluster_id": cluster_id,
            }
            trends.append(trend)
            trend_num += 1
            
            if trend_num > self.max_trends_per_category:
                break
        
        # If clustering produced no valid trends, return single trend
        if not trends:
            keywords = self._extract_cluster_keywords(signals)
            return [{
                "trend_id": f"{category}_trend_1",
                "trend_name": self._generate_trend_name(signals, category),
                "category": category,
                "signal_count": n_signals,
                "signals": signals,
                "keywords": keywords,
                "description": self._get_trend_description(signals, keywords),
                "cluster_id": 0,
            }]
        
        return trends
    
    def cluster_all_categories(self, 
                               signals_by_category: Dict[str, List[Dict]]
                               ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Cluster all categories and return trends.
        
        Args:
            signals_by_category: Dict mapping category name to list of signals
            
        Returns:
            Dict mapping category name to list of trend dicts
        """
        all_trends = {}
        total_trends = 0
        
        for category, signals in signals_by_category.items():
            if not signals:
                continue
            
            trends = self.cluster_category(category, signals)
            all_trends[category] = trends
            total_trends += len(trends)
            
            log.info(f"Category '{category}': {len(signals)} signals → {len(trends)} trends")
        
        log.info(f"Total: {total_trends} trends across {len(all_trends)} categories")
        return all_trends


def identify_trends(signals_by_category: Dict[str, List[Dict]],
                    min_signals_per_trend: int = 3,
                    max_trends_per_category: int = 5) -> Tuple[Dict[str, List[Dict]], int]:
    """
    Convenience function to identify trends.
    
    Returns:
        Tuple of (trends_by_category, total_trend_count)
    """
    clusterer = TrendClusterer(
        min_signals_per_trend=min_signals_per_trend,
        max_trends_per_category=max_trends_per_category,
    )
    
    trends = clusterer.cluster_all_categories(signals_by_category)
    total = sum(len(t) for t in trends.values())
    
    return trends, total
