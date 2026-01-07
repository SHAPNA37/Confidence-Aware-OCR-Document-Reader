import numpy as np


class ConfidenceAnalyzer:
    """
    Analyzes OCR confidence and identifies unreliable regions.
    """
    
    def __init__(self, low_threshold=60, high_threshold=80):
        """
        Initialize analyzer with confidence thresholds
        
        Args:
            low_threshold: Below this = low confidence (needs review)
            high_threshold: Above this = high confidence (reliable)
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        print(f" Confidence analyzer ready (thresholds: <{low_threshold}% low, >{high_threshold}% high)")
    

    # Main analysis function
    def analyze(self, ocr_result):
        """
        Categorize words by confidence level
        
        Args:
            ocr_result: Output from ConfidenceOCR.extract_text_with_confidence()
        
        Returns:
            dict with:
                - high_confidence: List of reliable words
                - medium_confidence: List of uncertain words
                - low_confidence: List of words needing review
                - statistics: Overall metrics
        """
        words = ocr_result['words']
        
        if not words:
            return self._empty_result()
        
        # Categorize words
        high = [w for w in words if w['confidence'] >= self.high_threshold]
        medium = [w for w in words if self.low_threshold <= w['confidence'] < self.high_threshold]
        low = [w for w in words if w['confidence'] < self.low_threshold]
        
        total = len(words)
        
        return {
            'high_confidence': {
                'words': high,
                'count': len(high),
                'percentage': (len(high) / total * 100) if total > 0 else 0
            },
            'medium_confidence': {
                'words': medium,
                'count': len(medium),
                'percentage': (len(medium) / total * 100) if total > 0 else 0
            },
            'low_confidence': {
                'words': low,
                'count': len(low),
                'percentage': (len(low) / total * 100) if total > 0 else 0
            },
            'statistics': {
                'total_words': total,
                'avg_confidence': ocr_result['avg_confidence'],
                'needs_review': len(low) > 0,
                'review_percentage': (len(low) / total * 100) if total > 0 else 0
            }
        }

    # Get words needing review

    def get_review_list(self, analysis):
        """
        Extract just the low confidence words for review
        
        Args:
            analysis: Output from analyze()
        
        Returns:
            List of words needing human review
        """
        return analysis['low_confidence']['words']

    # Calculate confidence distribution
    def get_confidence_distribution(self, ocr_result, bins=10):
        """
        Get histogram of confidence scores
        
        Args:
            ocr_result: Output from OCR
            bins: Number of bins for histogram
        
        Returns:
            dict with histogram data
        """
        confidences = [w['confidence'] for w in ocr_result['words']]
        
        if not confidences:
            return {'bins': [], 'counts': []}
        
        hist, bin_edges = np.histogram(confidences, bins=bins, range=(0, 100))
        
        return {
            'bins': bin_edges.tolist(),
            'counts': hist.tolist(),
            'min': min(confidences),
            'max': max(confidences),
            'median': np.median(confidences)
        }

    # Helper function

    def _empty_result(self):
        """Return empty result structure"""
        return {
            'high_confidence': {'words': [], 'count': 0, 'percentage': 0},
            'medium_confidence': {'words': [], 'count': 0, 'percentage': 0},
            'low_confidence': {'words': [], 'count': 0, 'percentage': 0},
            'statistics': {
                'total_words': 0,
                'avg_confidence': 0,
                'needs_review': False,
                'review_percentage': 0
            }
        }

    # Print summary

    def print_summary(self, analysis):
        """
        Print a formatted summary of the analysis
        
        Args:
            analysis: Output from analyze()
        """
        stats = analysis['statistics']
        
        print("=" * 60)
        print(" CONFIDENCE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total words: {stats['total_words']}")
        print(f"Average confidence: {stats['avg_confidence']:.1f}%")
        print()
        print(f"High (≥{self.high_threshold}%):   {analysis['high_confidence']['count']:3} words ({analysis['high_confidence']['percentage']:.1f}%)")
        print(f"  Medium ({self.low_threshold}-{self.high_threshold-1}%): {analysis['medium_confidence']['count']:3} words ({analysis['medium_confidence']['percentage']:.1f}%)")
        print(f" Low (<{self.low_threshold}%):     {analysis['low_confidence']['count']:3} words ({analysis['low_confidence']['percentage']:.1f}%)")
        print()
        
        if stats['needs_review']:
            print(f" {analysis['low_confidence']['count']} words need manual review ({stats['review_percentage']:.1f}% of total)")
        else:
            print(" All words have acceptable confidence!")
        
        print("=" * 60)


# Test code
if __name__ == "__main__":
    from data_loader import SROIEDataLoader
    from ocr_engine import ConfidenceOCR
    
    print(" Testing Confidence Analyzer")
    
    
    # Load sample
    loader = SROIEDataLoader("data/raw/SROIE2019")
    samples = loader.get_sample_ids("train")
    sample = loader.load_complete_sample(samples[0])
    
    print(f" Testing on: {sample['sample_id']}")
    
    # Run OCR
    ocr = ConfidenceOCR()
    ocr_result = ocr.extract_text_with_confidence(sample['image'])
    
    # Analyze confidence
    analyzer = ConfidenceAnalyzer(low_threshold=60, high_threshold=80)
    analysis = analyzer.analyze(ocr_result)
    
    # Print summary
    analyzer.print_summary(analysis)
    
    # Show low confidence words
    low_conf_words = analyzer.get_review_list(analysis)
    if low_conf_words:
        print()
        print(" Low confidence words (first 5):")
        for word in low_conf_words[:5]:
            print(f"   '{word['text']:<15}' → {word['confidence']:>3}%")
    
    print()
    print(" Test passed!")
    print("=" * 60)